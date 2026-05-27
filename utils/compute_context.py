import torch
import json
import argparse
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import seed_everything

class ContextProcessor:
    def __init__(self, data_dir, device='cuda'):
        self.data_dir = data_dir
        self.device = device
        self.entity2id = json.load(open(os.path.join(data_dir, 'entity2id.json')))
        self.relation2id = json.load(open(os.path.join(data_dir, 'relation2id.json')))
        
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
        triples_path = os.path.join(self.data_dir, 'train_triples.pt')
        if not os.path.exists(triples_path):
            raise FileNotFoundError(
                f"train_triples.pt not found in {self.data_dir}. "
                "Run preprocess_data.py first."
            )
            
        triples = torch.load(triples_path)
        adj = {}
        
        for h, r, t in triples.tolist():
            if h not in adj:
                adj[h] = []
            adj[h].append((r, t))
            
        # Deduplicate relation-tail edges while preserving tuple info.
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

    def compute_context_nodes(self, k=10, algorithm='mmr_neighbor', batch_size=64, mmr_lambda=0.5):
        print(f"Computing context nodes using {algorithm}...")
        num_entities = len(self.entity2id)
        pad_value = -1
        limit = int(k)
        use_all_neighbors = limit <= 0
        adj = self._load_adjacency()

        if use_all_neighbors:
            max_k = max((len(adj.get(i, [])) for i in range(num_entities)), default=0)
        else:
            max_k = max(limit, 0)

        # Store fixed-width tensors with a sentinel mask value.
        context_entity_ids = torch.full((num_entities, max_k), pad_value, dtype=torch.long)
        context_relation_ids = torch.full((num_entities, max_k), pad_value, dtype=torch.long)
        context_mask = torch.zeros((num_entities, max_k), dtype=torch.bool)

        if algorithm == 'mmr_neighbor':
            # 1-Hop Neighbor + MMR
            # Solves "Echo Chamber" by enforcing diversity among context nodes
            embeddings = self._compute_embeddings(batch_size) # Keep on CPU RAM mostly
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            print("Running MMR selection on neighbors...")
            for i in tqdm(range(num_entities), desc="MMR Selection"):
                eid = i
                neighbors = adj.get(eid, [])  # list[(r, t)]

                # If no real neighbors, keep this row fully masked.
                if not neighbors:
                    continue

                neighbor_rel_ids = torch.tensor([r for r, _ in neighbors], dtype=torch.long)
                neighbor_ent_ids = torch.tensor([t for _, t in neighbors], dtype=torch.long)

                if use_all_neighbors:
                    selected_ent_ids = neighbor_ent_ids
                    selected_rel_ids = neighbor_rel_ids
                elif len(neighbors) <= limit:
                    selected_ent_ids = neighbor_ent_ids
                    selected_rel_ids = neighbor_rel_ids
                else:
                    query_emb = embeddings[eid] # (H)
                    cand_embs = embeddings[neighbor_ent_ids] # (Num_N, H)

                    selected_local_indices = self._mmr(query_emb, cand_embs, limit, lambda_param=mmr_lambda)
                    selected_local_indices = torch.tensor(selected_local_indices, dtype=torch.long)
                    selected_ent_ids = neighbor_ent_ids[selected_local_indices]
                    selected_rel_ids = neighbor_rel_ids[selected_local_indices]

                count = min(selected_ent_ids.numel(), max_k)
                if count > 0:
                    context_entity_ids[i, :count] = selected_ent_ids[:count]
                    context_relation_ids[i, :count] = selected_rel_ids[:count]
                    context_mask[i, :count] = True

        elif algorithm == 'random':
            # Randomly sample only from each node's relation-aware neighbors.
            print("Generating random neighbor context...")
            for i in tqdm(range(num_entities)):
                neighbors = adj.get(i, [])  # list[(r, t)]
                if not neighbors:
                    continue

                neighbor_rel_ids = torch.tensor([r for r, _ in neighbors], dtype=torch.long)
                neighbor_ent_ids = torch.tensor([t for _, t in neighbors], dtype=torch.long)

                if use_all_neighbors:
                    selected_ent_ids = neighbor_ent_ids
                    selected_rel_ids = neighbor_rel_ids
                elif len(neighbors) >= limit:
                    sampled = torch.randperm(len(neighbors))[:limit]
                    selected_ent_ids = neighbor_ent_ids[sampled]
                    selected_rel_ids = neighbor_rel_ids[sampled]
                else:
                    selected_ent_ids = neighbor_ent_ids
                    selected_rel_ids = neighbor_rel_ids

                count = min(selected_ent_ids.numel(), max_k)
                if count > 0:
                    context_entity_ids[i, :count] = selected_ent_ids[:count]
                    context_relation_ids[i, :count] = selected_rel_ids[:count]
                    context_mask[i, :count] = True
                 
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Supported: mmr_neighbor, random")

        # Save one compact artifact.
        output_file = os.path.join(self.data_dir, 'context_neighbors.pt')
        torch.save(
            {
                'entity_ids': context_entity_ids,
                'relation_ids': context_relation_ids,
                'mask': context_mask,
                'pad_value': pad_value,
                'k_requested': limit,
                'k_effective': max_k,
                'algorithm': algorithm,
            },
            output_file,
        )

        print(f"Context neighbors saved to {output_file}")
        print(f"  pad_value={pad_value}, k_effective={max_k}, algorithm={algorithm}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to preprocessed data directory')
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Max number of context neighbors per entity. Use k<=0 to keep all real neighbors (variable count).',
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='mmr_neighbor',
        choices=['mmr_neighbor', 'random'],
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
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible context selection',
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    
    processor = ContextProcessor(args.data_dir, device=args.device)
    processor.compute_context_nodes(
        k=args.k,
        algorithm=args.algorithm,
        batch_size=args.batch_size,
        mmr_lambda=args.mmr_lambda
    )
