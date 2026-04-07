import os

import torch
from torch.utils.data import Dataset


class GWMDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Dataset for GWM.
        Loads triples, context IDs, and optional adjacency for GAT aggregation.
        """
        self.data_dir = data_dir
        self.split = split

        triples_path = os.path.join(data_dir, f'{split}_triples.pt')
        if not os.path.exists(triples_path) and split == 'valid':
            # Fallback for WN18RR naming if needed.
            triples_path = os.path.join(data_dir, 'dev_triples.pt')
        self.triples = torch.load(triples_path)

        context_path = os.path.join(data_dir, 'context_ids.pt')
        if os.path.exists(context_path):
            self.context_ids = torch.load(context_path)
        else:
            print(f"Warning: {context_path} not found. Context will be zeros.")
            self.context_ids = None

        adj_path = os.path.join(data_dir, 'adjacency_matrix.pt')
        if os.path.exists(adj_path):
            self.adjacency = torch.load(adj_path)
        else:
            self.adjacency = None

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]

        if self.context_ids is not None:
            ctx_ids = self.context_ids[h]
        else:
            ctx_ids = torch.zeros(10, dtype=torch.long)

        sample = {
            'h_id': h.long(),
            'r_id': r.long(),
            't_id': t.long(),
            'context_ids': ctx_ids.long(),
        }

        if self.adjacency is not None:
            # Build local subgraph adjacency among context nodes: (K, K)
            ctx_ids_list = ctx_ids.tolist()
            edges_matrix = torch.zeros((len(ctx_ids_list), len(ctx_ids_list)), dtype=torch.bool)
            for i, node_i in enumerate(ctx_ids_list):
                for j, node_j in enumerate(ctx_ids_list):
                    edges_matrix[i, j] = self.adjacency[node_i, node_j]
            sample['context_edges'] = edges_matrix

        return sample


class CollateFN:
    """
    ID-only collator; text embeddings are loaded from precomputed caches.
    """

    def __call__(self, batch):
        h_ids = torch.stack([b['h_id'] for b in batch])
        r_ids = torch.stack([b['r_id'] for b in batch])
        t_ids = torch.stack([b['t_id'] for b in batch])
        context_ids = torch.stack([b['context_ids'] for b in batch])

        result = {
            'h_batch': {'id': h_ids},
            'r_batch': {'id': r_ids},
            't_batch': {'id': t_ids},
            'context_batch': {'id': context_ids},
        }

        if 'context_edges' in batch[0]:
            result['context_edges'] = torch.stack([b['context_edges'] for b in batch])

        return result
