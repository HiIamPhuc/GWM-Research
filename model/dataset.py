import json
import os

import torch
from torch.utils.data import Dataset


class TrainTruthIndex:
    """Build query-aware in-batch truth masks from training triples only."""

    def __init__(self, train_triples):
        train_triples = torch.as_tensor(train_triples, dtype=torch.long)
        if train_triples.dim() != 2 or train_triples.size(1) != 3:
            raise ValueError("Training triples must have shape (N, 3).")

        self.query_tails = {}
        for h, r, t in train_triples.tolist():
            self.query_tails.setdefault((h, r), set()).add(t)

    def build_in_batch_truth_mask(
        self,
        head_ids,
        relation_ids,
        candidate_tail_ids,
        device=None,
    ):
        head_ids = torch.as_tensor(head_ids, dtype=torch.long).reshape(-1).cpu()
        relation_ids = torch.as_tensor(relation_ids, dtype=torch.long).reshape(-1).cpu()
        candidate_tail_ids = torch.as_tensor(
            candidate_tail_ids,
            dtype=torch.long,
        ).reshape(-1).cpu()

        batch_size = head_ids.numel()
        if relation_ids.numel() != batch_size:
            raise ValueError("head_ids and relation_ids must have equal lengths.")
        if candidate_tail_ids.numel() != batch_size:
            raise ValueError("In-batch loss requires one candidate tail per query.")

        candidate_columns = {}
        for column, tail_id in enumerate(candidate_tail_ids.tolist()):
            candidate_columns.setdefault(tail_id, []).append(column)

        truth_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool)
        for row, (head_id, relation_id) in enumerate(
            zip(head_ids.tolist(), relation_ids.tolist())
        ):
            for tail_id in self.query_tails.get((head_id, relation_id), ()):
                columns = candidate_columns.get(tail_id)
                if columns:
                    truth_mask[row, columns] = True

        truth_mask.fill_diagonal_(True)
        return truth_mask.to(device) if device is not None else truth_mask


class GWMDataset(Dataset):
    """ID-only knowledge graph triples for the structural LSTM baseline."""

    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split

        with open(os.path.join(data_dir, 'entity2id.json'), 'r', encoding='utf-8') as f:
            self.num_entities = len(json.load(f))
        with open(os.path.join(data_dir, 'relation2id.json'), 'r', encoding='utf-8') as f:
            self.num_relations = len(json.load(f))

        triples_path = os.path.join(data_dir, f'{split}_triples.pt')
        if split == 'valid' and not os.path.exists(triples_path):
            triples_path = os.path.join(data_dir, 'dev_triples.pt')
        if not os.path.exists(triples_path):
            raise FileNotFoundError(f"Triple tensor not found: {triples_path}")

        self.triples = torch.load(triples_path, map_location='cpu').long()
        if self.triples.dim() != 2 or self.triples.size(1) != 3:
            raise ValueError(
                f"Expected triples with shape (N, 3), got {tuple(self.triples.shape)}"
            )

    def __len__(self):
        return int(self.triples.size(0))

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        return {
            'h_id': h.long(),
            'r_id': r.long(),
            't_id': t.long(),
        }


class CollateFN:
    """Collate structural IDs without text or graph-context features."""

    def __call__(self, batch):
        return {
            'h_batch': {'id': torch.stack([item['h_id'] for item in batch])},
            'r_batch': {'id': torch.stack([item['r_id'] for item in batch])},
            't_batch': {'id': torch.stack([item['t_id'] for item in batch])},
        }
