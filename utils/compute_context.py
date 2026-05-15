import torch
import json
from transformers import AutoTokenizer, AutoModel
import argparse
from tqdm import tqdm
import os
import yaml

class ContextProcessor:
    def __init__(self, data_dir, model_name='bert-base-uncased', device='cuda'):
        self.data_dir = data_dir
        self.device = self._resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.entity_text = json.load(open(os.path.join(data_dir, 'entity_text.json')))
        self.entity2id = json.load(open(os.path.join(data_dir, 'entity2id.json')))

    def _resolve_device(self, device):
        requested = str(device)
        if requested.startswith('cuda') and not torch.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU for context computation.")
            return 'cpu'
        return requested
        
    def _encode_batch(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token
        
    def _compute_embeddings(self, batch_size=32):
        print("Computing embeddings for all entities...")
        entities = sorted(self.entity2id.keys(), key=lambda x: self.entity2id[x])
        # Fallback to entity ID if text is missing, but it should be there from preprocess
        texts = [self.entity_text.get(e, e) for e in entities]
        all_embeddings = []
        
        # Determine batch size dynamically or use fixed
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding entities"):
            batch_texts = texts[i:i+batch_size]
            emb = self._encode_batch(batch_texts)
            all_embeddings.append(emb.cpu())
            
        return torch.cat(all_embeddings, dim=0)

    def _load_adjacency(self):
        print("Loading graph structure for neighbor context...")
        triples_path = os.path.join(self.data_dir, 'train_triples.pt')
        if not os.path.exists(triples_path):
            raise FileNotFoundError(f"train_triples.pt not found in {self.data_dir}. Run preprocess_data.py first.")
            
        triples = torch.load(triples_path)
        adj = {}

        # Convert to list for iteration
        for h, r, t in triples.tolist():
            if h not in adj:
                adj[h] = {}
            if t not in adj[h]:
                adj[h][t] = r

        return adj

    def _mmr(self, query_emb, candidate_embs, k, lambda_param=0.5):
        """
        Maximal Marginal Relevance selection.
        query_emb: (H)
        candidate_embs: (M, H)
        k: number of items to select
        lambda_param: 0.5 balances relevance and diversity. 1.0 = standard top-k.
        """
        if candidate_embs.size(0) == 0:
            return []
            
        selected_indices = []
        candidate_indices = list(range(candidate_embs.size(0)))
        
        # Ensure tensor is on same device
        query_emb = query_emb.to(candidate_embs.device)
        
        # Precompute similarity of candidates to query
        sim_to_query = torch.matmul(candidate_embs, query_emb)
        
        for _ in range(min(k, len(candidate_indices))):
            if not selected_indices:
                # First step: pick most similar to query
                best_rel_idx = torch.argmax(sim_to_query).item()
                selected_indices.append(best_rel_idx)
            else:
                # MMR step
                # Compute sim Max(Sim(c, s)) for all s in S
                selected_embs = candidate_embs[selected_indices] # (num_sel, H)
                
                # Sim matrix: (M, num_sel)
                sim_to_selected = torch.matmul(candidate_embs, selected_embs.t())
                
                # Max sim for each candidate to ANY selected context node
                max_sim_to_selected, _ = torch.max(sim_to_selected, dim=1)
                
                # MMR score
                # Mask out already selected indices with -inf
                current_scores = lambda_param * sim_to_query - (1 - lambda_param) * max_sim_to_selected
                current_scores[selected_indices] = -float('inf')
                
                best_idx = torch.argmax(current_scores).item()
                selected_indices.append(best_idx)
            
        return selected_indices

    def compute_context_nodes(self, k=10, algorithm='mmr_neighbor', batch_size=64, mmr_lambda=0.5):
        print(f"Computing context nodes using {algorithm}...")
        num_entities = len(self.entity2id)

        if algorithm == 'mmr_neighbor':
            # 1-Hop Neighbor + MMR
            # Solves "Echo Chamber" by enforcing diversity among context nodes
            embeddings = self._compute_embeddings(batch_size) # Keep on CPU RAM mostly
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            adj = self._load_adjacency()
            context_ids = torch.zeros((num_entities, k), dtype=torch.long)
            context_rel_ids = torch.zeros((num_entities, k), dtype=torch.long)
            
            print("Running MMR selection on neighbors...")
            for i in tqdm(range(num_entities), desc="MMR Selection"):
                eid = i
                neighbor_map = adj.get(eid, {})
                neighbors = list(neighbor_map.keys())
                rels = [neighbor_map[n] for n in neighbors]
                
                # If no neighbors, use self (or random, or zero)
                if not neighbors:
                    context_ids[i] = torch.tensor([eid] * k)
                    context_rel_ids[i] = torch.zeros(k, dtype=torch.long)
                    continue
                    
                neighbor_indices = torch.tensor(neighbors, dtype=torch.long)
                neighbor_relations = torch.tensor(rels, dtype=torch.long)
                
                # If neighbors <= k, take all and pad with self
                if len(neighbors) <= k:
                    # Pad
                    needed = k - len(neighbors)
                    padded = torch.cat([neighbor_indices, torch.tensor([eid] * needed)])
                    padded_rels = torch.cat([neighbor_relations, torch.zeros(needed, dtype=torch.long)])
                    context_ids[i] = padded
                    context_rel_ids[i] = padded_rels
                else:
                    # Perform MMR
                    query_emb = embeddings[eid] # (H)
                    cand_embs = embeddings[neighbor_indices] # (Num_N, H)
                    
                    selected_local_indices = self._mmr(query_emb, cand_embs, k, lambda_param=mmr_lambda)
                    selected_global_indices = neighbor_indices[selected_local_indices]
                    selected_relations = neighbor_relations[selected_local_indices]

                    context_ids[i] = selected_global_indices
                    context_rel_ids[i] = selected_relations
        
        elif algorithm == 'random':
            # Randomly sample only from each node's neighbors.
            print("Generating random neighbor context...")
            adj = self._load_adjacency()
            context_ids = torch.zeros((num_entities, k), dtype=torch.long)
            context_rel_ids = torch.zeros((num_entities, k), dtype=torch.long)
            for i in tqdm(range(num_entities)):
                neighbor_map = adj.get(i, {})
                neighbors = list(neighbor_map.keys())
                rels = [neighbor_map[n] for n in neighbors]
                if not neighbors:
                    context_ids[i] = torch.tensor([i] * k)
                    context_rel_ids[i] = torch.zeros(k, dtype=torch.long)
                    continue

                neighbor_indices = torch.tensor(neighbors, dtype=torch.long)
                neighbor_relations = torch.tensor(rels, dtype=torch.long)
                if len(neighbors) >= k:
                    sampled = torch.randperm(len(neighbors))[:k]
                    context_ids[i] = neighbor_indices[sampled]
                    context_rel_ids[i] = neighbor_relations[sampled]
                else:
                    # Keep fixed length by sampling neighbors with replacement.
                    sampled = torch.randint(0, len(neighbors), (k,))
                    context_ids[i] = neighbor_indices[sampled]
                    context_rel_ids[i] = neighbor_relations[sampled]
                 
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Supported: mmr_neighbor, random")

        output_file = os.path.join(self.data_dir, 'context_ids.pt')
        output_rel_file = os.path.join(self.data_dir, 'context_rel_ids.pt')
        torch.save(context_ids, output_file)
        torch.save(context_rel_ids, output_rel_file)
        print(f"Context nodes saved to {output_file}")
        print(f"Context relation ids saved to {output_rel_file}")


def load_config(config_path):
    if not config_path:
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must be a YAML mapping.")
    return config


def pick_value(cli_value, config, keys, default=None):
    if cli_value is not None:
        return cli_value

    if isinstance(keys, (list, tuple)):
        for key in keys:
            if key in config and config[key] is not None:
                return config[key]
        return default

    if keys in config and config[keys] is not None:
        return config[keys]
    return default

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Optional YAML config path')
    parser.add_argument('--data_dir', type=str, default=None, help='Processed dataset directory')
    parser.add_argument('--k', type=int, default=None, help='Number of context neighbors')
    parser.add_argument('--algorithm', type=str, default=None, choices=['mmr_neighbor', 'random'], help='Algorithm for neighbor selection')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size for encoding/similarity')
    parser.add_argument('--mmr_lambda', type=float, default=None, help='Lambda for MMR (0.5 balances relevance and diversity)')
    parser.add_argument('--model_name', type=str, default=None, help='Text encoder model for context computation')
    parser.add_argument('--device', type=str, default=None, help='Device for context computation (e.g., cuda, cuda:0, cpu)')
    args = parser.parse_args()

    config = load_config(args.config)

    data_dir = pick_value(args.data_dir, config, 'data_dir')
    if data_dir is None:
        parser.error("data_dir must be provided via --data_dir or config[data_dir].")

    k = int(pick_value(args.k, config, 'context_k', 10))
    algorithm = pick_value(args.algorithm, config, 'context_algorithm', 'mmr_neighbor')
    batch_size = int(pick_value(args.batch_size, config, 'context_batch_size', 32))
    mmr_lambda = float(pick_value(args.mmr_lambda, config, 'context_mmr_lambda', 0.5))
    model_name = pick_value(args.model_name, config, 'pretrained_model', 'bert-base-uncased')
    device = pick_value(args.device, config, 'device', 'cuda')

    processor = ContextProcessor(data_dir=data_dir, model_name=model_name, device=device)
    processor.compute_context_nodes(
        k=k,
        algorithm=algorithm,
        batch_size=batch_size,
        mmr_lambda=mmr_lambda,
    )
