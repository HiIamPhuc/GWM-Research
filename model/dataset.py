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
    """Knowledge graph triples with fixed relation-aware head context."""

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

        context_path = os.path.join(data_dir, 'context_neighbors.pt')
        if not os.path.exists(context_path):
            raise FileNotFoundError(
                f"Context tensor not found: {context_path}. "
                "Run utils/compute_context.py first."
            )

        context = torch.load(context_path, map_location='cpu')
        self.context_entity_ids = context['entity_ids'].long()
        self.context_relation_ids = context['relation_ids'].long()
        self.context_mask = context['mask'].bool()
        self.context_k_requested = int(context.get('k_requested', -1))
        self.context_k_effective = int(
            context.get('k_effective', self.context_entity_ids.size(1))
        )
        self.context_algorithm = str(context.get('algorithm', 'unknown'))

        expected_shape = self.context_entity_ids.shape
        if (
            self.context_entity_ids.dim() != 2
            or self.context_relation_ids.shape != expected_shape
            or self.context_mask.shape != expected_shape
        ):
            raise ValueError(
                "Context entity IDs, relation IDs, and mask must share "
                "the same rank-2 shape."
            )
        if self.context_entity_ids.size(0) != self.num_entities:
            raise ValueError(
                "Context artifact row count must equal the entity vocabulary size."
            )

        valid_entities = self.context_entity_ids[self.context_mask]
        valid_relations = self.context_relation_ids[self.context_mask]
        if valid_entities.numel() and (
            valid_entities.min() < 0
            or valid_entities.max() >= self.num_entities
        ):
            raise ValueError("Context artifact contains invalid entity IDs.")
        if valid_relations.numel() and (
            valid_relations.min() < 0
            or valid_relations.max() >= self.num_relations
        ):
            raise ValueError("Context artifact contains invalid relation IDs.")

    def __len__(self):
        return int(self.triples.size(0))

    def make_item(self, h, r, t):
        h = torch.as_tensor(h, dtype=torch.long)
        r = torch.as_tensor(r, dtype=torch.long)
        t = torch.as_tensor(t, dtype=torch.long)
        h_idx = int(h.item())

        context_entity_ids = self.context_entity_ids[h_idx]
        context_relation_ids = self.context_relation_ids[h_idx]
        context_mask = self.context_mask[h_idx].clone()

        # A training query must never receive its answer edge as context.
        answer_edge = (
            context_entity_ids.eq(int(t.item()))
            & context_relation_ids.eq(int(r.item()))
        )
        context_mask &= ~answer_edge

        return {
            'h_id': h.long(),
            'r_id': r.long(),
            't_id': t.long(),
            'context_entity_ids': context_entity_ids.long(),
            'context_relation_ids': context_relation_ids.long(),
            'context_mask': context_mask,
        }

    def __getitem__(self, idx):
        return self.make_item(*self.triples[idx])


class CollateFN:
    """Collate triples and fixed-width context rows for Transformer memory."""

    def __call__(self, batch):
        for item in batch:
            entity_ids = item['context_entity_ids']
            relation_ids = item['context_relation_ids']
            mask = item['context_mask'].bool()
            if not (
                entity_ids.dim() == relation_ids.dim() == mask.dim() == 1
            ):
                raise ValueError("Each context row must be one-dimensional.")
            if not (
                entity_ids.numel() == relation_ids.numel() == mask.numel()
            ):
                raise ValueError("Context entity, relation, and mask lengths differ.")

        return {
            'h_batch': {'id': torch.stack([item['h_id'] for item in batch])},
            'r_batch': {'id': torch.stack([item['r_id'] for item in batch])},
            't_batch': {'id': torch.stack([item['t_id'] for item in batch])},
            'context_batch': {
                'id': torch.stack([
                    item['context_entity_ids'].long() for item in batch
                ]),
                'rel_id': torch.stack([
                    item['context_relation_ids'].long() for item in batch
                ]),
                'mask': torch.stack([
                    item['context_mask'].bool() for item in batch
                ]),
            },
        }
