import torch
import json
import argparse
from tqdm import tqdm
import os
from pathlib import Path

class ContextProcessor:
    def __init__(self, data_dir, device='cuda'):
        self.data_dir = data_dir
        self.device = device
        self.entity2id = json.load(open(os.path.join(data_dir, 'entity2id.json')))
        
    def _load_precomputed_embeddings(self):
        """Load precomputed text embeddings from cache files."""
        print("Loading precomputed text embeddings...")
        
        entity_emb_path = os.path.join(self.data_dir, 'entity_text_embeddings.pt')
        if not os.path.exists(entity_emb_path):
            raise FileNotFoundError(
                f"entity_text_embeddings.pt not found in {self.data_dir}. "
                "Run preprocess_data.py with text embedding precomputation first."
            )
        
        # Load embedding cache (may be a dict or tensor)
        entity_cache = torch.load(entity_emb_path, map_location='cpu')
        if isinstance(entity_cache, dict):
            if 'embeddings' in entity_cache:
                embeddings = entity_cache['embeddings']
            elif 'tensor' in entity_cache:
                embeddings = entity_cache['tensor']
            else:
                raise ValueError(
                    "entity_text_embeddings.pt dict must contain 'embeddings' or 'tensor' key."
                )
        else:
            embeddings = entity_cache
        
        embeddings = embeddings.float().to(self.device)
        return embeddings
        
    def _compute_embeddings(self, batch_size=32):
        """Load precomputed embeddings for all entities."""
        embeddings = self._load_precomputed_embeddings()
        return embeddings

    def _load_adjacency(self):
        print("Loading graph structure for neighbor context...")
        triples_path = os.path.join(self.data_dir, 'train_triples.pt')
        if not os.path.exists(triples_path):
            raise FileNotFoundError(
                f"train_triples.pt not found in {self.data_dir}. "
                "Run preprocess_data.py first."
            )
            
        triples = torch.load(triples_path)
        adj = {}
        
        # Convert to list for iteration
        for h, r, t in triples.tolist():
            if h not in adj: adj[h] = []
            adj[h].append(t)
            
        # Deduplicate neighbors
        for h in adj:
            adj[h] = list(set(adj[h]))
            
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

    def compute_context_nodes(self, k=10, algorithm='dense', batch_size=64, mmr_lambda=0.5):
        print(f"Computing context nodes using {algorithm}...")
        num_entities = len(self.entity2id)
        
        if algorithm == 'dense':
            # Global Dense Retrieval
            embeddings = self._compute_embeddings(batch_size).to(self.device)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            context_ids = torch.zeros((num_entities, k), dtype=torch.long)
            
            print("Computing pairwise similarity...")
            for i in tqdm(range(0, num_entities, batch_size), desc="Mining neighbors"):
                end = min(i + batch_size, num_entities)
                query_emb = embeddings[i:end]
                
                # Sim: (B, H) @ (N, H).T
                try:
                    sim_scores = torch.mm(query_emb, embeddings.t())
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        query_emb = query_emb.cpu()
                        embeddings = embeddings.cpu()
                        sim_scores = torch.mm(query_emb, embeddings.t())
                    else: raise e

                top_vals, top_inds = torch.topk(sim_scores, k + 1, dim=1)
                top_inds = top_inds.cpu()
                
                cleaned_indices = []
                for b_idx in range(len(query_emb)):
                    global_idx = i + b_idx
                    row_inds = top_inds[b_idx]
                    mask = row_inds != global_idx
                    filtered = row_inds[mask]
                    if len(filtered) > k: filtered = filtered[:k]
                    elif len(filtered) < k: filtered = row_inds[:k]
                    cleaned_indices.append(filtered)
                
                context_ids[i:end] = torch.stack(cleaned_indices)
                
        elif algorithm == 'mmr_neighbor':
            # 1-Hop Neighbor + MMR
            # Solves "Echo Chamber" by enforcing diversity among context nodes
            embeddings = self._compute_embeddings(batch_size) # Keep on CPU RAM mostly
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            adj = self._load_adjacency()
            context_ids = torch.zeros((num_entities, k), dtype=torch.long)
            
            print("Running MMR selection on neighbors...")
            for i in tqdm(range(num_entities), desc="MMR Selection"):
                eid = i
                neighbors = adj.get(eid, [])
                
                # If no neighbors, use self (or random, or zero)
                if not neighbors:
                    context_ids[i] = torch.tensor([eid] * k)
                    continue
                    
                neighbor_indices = torch.tensor(neighbors, dtype=torch.long)
                
                # If neighbors <= k, take all and pad with self
                if len(neighbors) <= k:
                    # Pad
                    needed = k - len(neighbors)
                    padded = torch.cat([neighbor_indices, torch.tensor([eid] * needed)])
                    context_ids[i] = padded
                else:
                    # Perform MMR
                    query_emb = embeddings[eid] # (H)
                    cand_embs = embeddings[neighbor_indices] # (Num_N, H)
                    
                    selected_local_indices = self._mmr(query_emb, cand_embs, k, lambda_param=mmr_lambda)
                    selected_global_indices = neighbor_indices[selected_local_indices]
                    
                    context_ids[i] = selected_global_indices
        
        elif algorithm == 'random':
            print("Generating random context...")
            context_ids = torch.zeros((num_entities, k), dtype=torch.long)
            for i in tqdm(range(num_entities)):
                 candidates = torch.randint(0, num_entities, (k,))
                 context_ids[i] = candidates
                 
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Supported: dense, mmr_neighbor, random")

        output_file = os.path.join(self.data_dir, 'context_ids.pt')
        torch.save(context_ids, output_file)
        print(f"Context nodes saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to preprocessed data directory')
    parser.add_argument('--k', type=int, default=10, help='Number of context neighbors per entity')
    parser.add_argument(
        '--algorithm',
        type=str,
        default='dense',
        choices=['dense', 'mmr_neighbor', 'random'],
        help='Algorithm for neighbor selection'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for similarity computation'
    )
    parser.add_argument(
        '--mmr_lambda',
        type=float,
        default=0.5,
        help='Lambda for MMR (0.5 balances relevance and diversity)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    args = parser.parse_args()
    
    processor = ContextProcessor(args.data_dir, device=args.device)
    processor.compute_context_nodes(
        k=args.k,
        algorithm=args.algorithm,
        batch_size=args.batch_size,
        mmr_lambda=args.mmr_lambda
    )
