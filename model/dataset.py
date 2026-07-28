import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class GWMDataset(Dataset):
    """Knowledge graph triples with fixed relation-aware head context."""

    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split

        triples_path = os.path.join(data_dir, f'{split}_triples.pt')
        self.triples = torch.load(triples_path, map_location='cpu').long()

        context_path = os.path.join(data_dir, 'context_neighbors.pt')
        context = torch.load(context_path, map_location='cpu')
        self.context_entity_ids = context['entity_ids'].long()
        self.context_relation_ids = context['relation_ids'].long()
        self.context_mask = context['mask'].bool()

    def __len__(self):
        return int(self.triples.size(0))

    def make_item(self, h, r, t):
        h_idx = int(h)

        context_entity_ids = self.context_entity_ids[h_idx]
        context_relation_ids = self.context_relation_ids[h_idx]
        context_mask = self.context_mask[h_idx].clone()

        # A training query must never receive its answer edge as context.
        answer_edge = (
            context_entity_ids.eq(t)
            & context_relation_ids.eq(r)
        )
        context_mask &= ~answer_edge

        return {
            'h_id': h,
            'r_id': r,
            't_id': t,
            'context_entity_ids': context_entity_ids,
            'context_relation_ids': context_relation_ids,
            'context_mask': context_mask,
        }

    def __getitem__(self, idx):
        return self.make_item(*self.triples[idx])


class CollateFN:
    """Collate triples and fixed-width context rows for Transformer memory."""

    def __call__(self, batch):
        return {
            'h_batch': {'id': torch.stack([item['h_id'] for item in batch])},
            'r_batch': {'id': torch.stack([item['r_id'] for item in batch])},
            't_batch': {'id': torch.stack([item['t_id'] for item in batch])},
            'context_batch': {
                'id': torch.stack([item['context_entity_ids'] for item in batch]),
                'rel_id': torch.stack([item['context_relation_ids'] for item in batch]),
                'mask': torch.stack([item['context_mask'] for item in batch]),
            },
        }


class TrainTruthIndex:
    """Known training tails used to filter false negatives."""

    def __init__(self, triples):
        tails_by_query = defaultdict(set)
        for h_id, r_id, t_id in triples.tolist():
            tails_by_query[(h_id, r_id)].add(t_id)
        self.tails_by_query = tails_by_query

    def alternate_positive_indices(
        self,
        head_ids,
        relation_ids,
        target_ids,
        device,
    ):
        rows = []
        columns = []
        for row, (h_id, r_id, target_id) in enumerate(zip(
            head_ids.tolist(),
            relation_ids.tolist(),
            target_ids.tolist(),
        )):
            for tail_id in self.tails_by_query[(h_id, r_id)]:
                if tail_id != target_id:
                    rows.append(row)
                    columns.append(tail_id)

        return (
            torch.tensor(rows, dtype=torch.long, device=device),
            torch.tensor(columns, dtype=torch.long, device=device),
        )
