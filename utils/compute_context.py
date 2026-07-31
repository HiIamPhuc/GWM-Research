import torch
import json
import argparse
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import seed_everything

class ContextProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, 'entity2id.json'), 'r', encoding='utf-8',) as f:
            self.entity2id = json.load(f)

    def _load_adjacency(self):
        triples_path = os.path.join(self.data_dir, 'train_triples.pt')
        triples = torch.load(triples_path)
        adj = {}
        
        for h, r, t in triples.tolist():
            adj.setdefault(h, []).append((r, t))
            
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

    @staticmethod
    def _select_relation_pair_diverse_paths(paths, limit):
        if limit <= 0 or len(paths) <= limit:
            return list(paths)

        by_relation_pair = {}
        for path in paths:
            relation_pair = (path[0], path[2])
            by_relation_pair.setdefault(relation_pair, []).append(path)

        relation_pairs = sorted(by_relation_pair)
        pair_order = torch.randperm(len(relation_pairs)).tolist()
        selected = []
        for pair_index in pair_order[:limit]:
            candidates = by_relation_pair[relation_pairs[pair_index]]
            candidate_index = int(torch.randint(len(candidates), (1,)).item())
            selected.append(candidates[candidate_index])

        if len(selected) < limit:
            selected_set = set(selected)
            remaining = [path for path in paths if path not in selected_set]
            fill_count = min(limit - len(selected), len(remaining))
            if fill_count:
                fill_indices = torch.randperm(len(remaining))[:fill_count]
                selected.extend(
                    remaining[index] for index in fill_indices.tolist()
                )

        return selected

    def _select_two_hop_paths(self, head_id, first_edges, adjacency, limit):
        if limit <= 0 or not first_edges:
            return []

        candidate_limit = max(limit * 4, limit)
        second_edges_per_first = max(
            1,
            (candidate_limit + len(first_edges) - 1) // len(first_edges),
        )
        candidates = []
        for first_relation, intermediate_entity in first_edges:
            second_edges = [
                edge
                for edge in adjacency.get(intermediate_entity, [])
                if edge[1] != head_id
            ]
            selected_seconds = self._select_relation_diverse_neighbors(
                second_edges,
                second_edges_per_first,
            )
            candidates.extend(
                (
                    first_relation,
                    intermediate_entity,
                    second_relation,
                    final_entity,
                )
                for second_relation, final_entity in selected_seconds
            )

        candidates = sorted(set(candidates))
        return self._select_relation_pair_diverse_paths(candidates, limit)

    def compute_context_nodes(self, k=10, path_k=4):
        print("Computing relation-diverse context memory...")
        num_entities = len(self.entity2id)
        pad_value = -1
        limit = int(k)
        path_limit = max(int(path_k), 0)
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
        path_intermediate_entity_ids = torch.full(
            (num_entities, path_limit),
            pad_value,
            dtype=torch.long,
        )
        path_final_entity_ids = torch.full(
            (num_entities, path_limit),
            pad_value,
            dtype=torch.long,
        )
        path_first_relation_ids = torch.full(
            (num_entities, path_limit),
            pad_value,
            dtype=torch.long,
        )
        path_second_relation_ids = torch.full(
            (num_entities, path_limit),
            pad_value,
            dtype=torch.long,
        )
        path_mask = torch.zeros(
            (num_entities, path_limit),
            dtype=torch.bool,
        )

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
                context_mask[i, :count] = True

            selected_paths = self._select_two_hop_paths(
                i,
                selected,
                adj,
                path_limit,
            )
            path_count = min(len(selected_paths), path_limit)
            if path_count:
                selected_paths = selected_paths[:path_count]
                path_first_relation_ids[i, :path_count] = torch.tensor(
                    [path[0] for path in selected_paths],
                    dtype=torch.long,
                )
                path_intermediate_entity_ids[i, :path_count] = torch.tensor(
                    [path[1] for path in selected_paths],
                    dtype=torch.long,
                )
                path_second_relation_ids[i, :path_count] = torch.tensor(
                    [path[2] for path in selected_paths],
                    dtype=torch.long,
                )
                path_final_entity_ids[i, :path_count] = torch.tensor(
                    [path[3] for path in selected_paths],
                    dtype=torch.long,
                )
                path_mask[i, :path_count] = True

        output_file = os.path.join(self.data_dir, 'context_neighbors.pt')
        torch.save(
            {
                'entity_ids': context_entity_ids,
                'relation_ids': context_relation_ids,
                'mask': context_mask,
            },
            output_file,
        )
        path_output_file = os.path.join(self.data_dir, 'context_paths.pt')
        torch.save(
            {
                'intermediate_entity_ids': path_intermediate_entity_ids,
                'final_entity_ids': path_final_entity_ids,
                'first_relation_ids': path_first_relation_ids,
                'second_relation_ids': path_second_relation_ids,
                'mask': path_mask,
            },
            path_output_file,
        )

        print(f"Context neighbors saved to {output_file}")
        print(f"  context width={max_k}")
        print(f"Two-hop paths saved to {path_output_file}")
        print(f"  path width={path_limit}")

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
        '--path_k',
        type=int,
        default=4,
        help='Max number of relation-pair-diverse two-hop paths per entity.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible context selection',
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    
    processor = ContextProcessor(args.data_dir)
    processor.compute_context_nodes(k=args.k, path_k=args.path_k)
