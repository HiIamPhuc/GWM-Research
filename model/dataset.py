import os

import torch
from torch.utils.data import Dataset


class GWMDataset(Dataset):
    """Knowledge graph triples with fixed relation-aware head context."""

    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split

        triples_path = os.path.join(data_dir, f'{split}_triples.pt')
        self.triples = torch.load(triples_path, map_location='cpu').long()
        self.positive_tails = None
        if split == 'train':
            positive_tails = {}
            for h, r, t in self.triples.tolist():
                positive_tails.setdefault((h, r), set()).add(t)
            self.positive_tails = {
                query: torch.tensor(sorted(tails), dtype=torch.long)
                for query, tails in positive_tails.items()
            }

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

        positive_tail_ids = None
        if self.positive_tails is not None:
            positive_tail_ids = self.positive_tails[(int(h), int(r))]

        # A training query must never receive any target edge as context.
        answer_entities = t.unsqueeze(0)
        if positive_tail_ids is not None:
            answer_entities = positive_tail_ids
        answer_edge = (
            context_entity_ids.unsqueeze(-1).eq(answer_entities).any(dim=-1)
            & context_relation_ids.eq(r)
        )
        context_mask &= ~answer_edge

        item = {
            'h_id': h,
            'r_id': r,
            't_id': t,
            'context_entity_ids': context_entity_ids,
            'context_relation_ids': context_relation_ids,
            'context_mask': context_mask,
        }
        if positive_tail_ids is not None:
            item['positive_tail_ids'] = positive_tail_ids
        return item

    def __getitem__(self, idx):
        return self.make_item(*self.triples[idx])


class CollateFN:
    """Collate triples and fixed-width context rows for Transformer memory."""

    def __call__(self, batch):
        collated = {
            'h_batch': {'id': torch.stack([item['h_id'] for item in batch])},
            'r_batch': {'id': torch.stack([item['r_id'] for item in batch])},
            't_batch': {'id': torch.stack([item['t_id'] for item in batch])},
            'context_batch': {
                'id': torch.stack([item['context_entity_ids'] for item in batch]),
                'rel_id': torch.stack([item['context_relation_ids'] for item in batch]),
                'mask': torch.stack([item['context_mask'] for item in batch]),
            },
        }
        if 'positive_tail_ids' in batch[0]:
            max_positives = max(item['positive_tail_ids'].numel() for item in batch)
            positive_ids = torch.zeros(
                len(batch),
                max_positives,
                dtype=torch.long,
            )
            positive_mask = torch.zeros(
                len(batch),
                max_positives,
                dtype=torch.bool,
            )
            for row, item in enumerate(batch):
                count = item['positive_tail_ids'].numel()
                positive_ids[row, :count] = item['positive_tail_ids']
                positive_mask[row, :count] = True
            collated['positive_tail_batch'] = {
                'id': positive_ids,
                'mask': positive_mask,
            }
        return collated
