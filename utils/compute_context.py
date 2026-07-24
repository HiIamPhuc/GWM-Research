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
            
        # Deduplicate relation-tail edges with deterministic ordering.
        for h in adj:
            adj[h] = sorted(set(adj[h]))
            
        return adj

    @staticmethod
    def _select_relation_diverse_neighbors(neighbors, limit):
        """Select relation coverage first, then fill remaining slots randomly."""
        if limit <= 0 or len(neighbors) <= limit:
            return list(neighbors)

        by_relation = {}
        for relation_id, entity_id in neighbors:
            by_relation.setdefault(relation_id, []).append(
                (relation_id, entity_id)
            )

        relation_ids = sorted(by_relation)
        relation_order = torch.randperm(len(relation_ids)).tolist()
        selected = []
        for relation_index in relation_order[:limit]:
            candidates = by_relation[relation_ids[relation_index]]
            candidate_index = int(torch.randint(len(candidates), (1,)).item())
            selected.append(candidates[candidate_index])

        if len(selected) < limit:
            selected_set = set(selected)
            remaining = [
                edge for edge in neighbors
                if edge not in selected_set
            ]
            fill_count = min(limit - len(selected), len(remaining))
            if fill_count:
                fill_indices = torch.randperm(len(remaining))[:fill_count]
                selected.extend(remaining[index] for index in fill_indices.tolist())

        return selected

    def compute_context_nodes(self, k=10):
        print("Computing relation-diverse context memory...")
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
        context_direction_ids = torch.zeros((num_entities, max_k), dtype=torch.long)
        context_mask = torch.zeros((num_entities, max_k), dtype=torch.bool)
        relation_direction_lookup = torch.zeros(len(self.relation2id), dtype=torch.long)
        for relation, relation_id in self.relation2id.items():
            if relation.endswith('_inv'):
                relation_direction_lookup[int(relation_id)] = 1

        for i in tqdm(range(num_entities), desc="Relation-diverse context"):
            neighbors = adj.get(i, [])  # list[(r, t)]
            if not neighbors:
                continue

            if use_all_neighbors:
                selected = neighbors
            else:
                selected = self._select_relation_diverse_neighbors(
                    neighbors,
                    limit,
                )

            selected_rel_ids = torch.tensor(
                [relation_id for relation_id, _ in selected],
                dtype=torch.long,
            )
            selected_ent_ids = torch.tensor(
                [entity_id for _, entity_id in selected],
                dtype=torch.long,
            )

            count = min(selected_ent_ids.numel(), max_k)
            if count > 0:
                context_entity_ids[i, :count] = selected_ent_ids[:count]
                context_relation_ids[i, :count] = selected_rel_ids[:count]
                context_direction_ids[i, :count] = relation_direction_lookup[
                    selected_rel_ids[:count]
                ]
                context_mask[i, :count] = True

        # Save one compact artifact.
        output_file = os.path.join(self.data_dir, 'context_neighbors.pt')
        torch.save(
            {
                'entity_ids': context_entity_ids,
                'relation_ids': context_relation_ids,
                'direction_ids': context_direction_ids,
                'mask': context_mask,
                'pad_value': pad_value,
                'k_requested': limit,
                'k_effective': max_k,
                'algorithm': 'relation_diverse',
            },
            output_file,
        )

        print(f"Context neighbors saved to {output_file}")
        print(
            f"  pad_value={pad_value}, k_effective={max_k}, "
            "algorithm=relation_diverse"
        )

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
    processor.compute_context_nodes(k=args.k)
