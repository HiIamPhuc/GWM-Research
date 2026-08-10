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

        # The path branch must reason beyond direct query edges. Removing every
        # observed (h, r, *) edge also prevents other known positives from
        # becoming one-hop shortcuts for multi-answer queries.
        path_hop1_mask = self.context_mask[h_idx].clone()
        path_hop1_mask &= ~context_relation_ids.eq(r)
        safe_hop1_ids = context_entity_ids.masked_fill(~path_hop1_mask, 0)
        path_hop2_entity_ids = self.context_entity_ids[safe_hop1_ids]
        path_hop2_relation_ids = self.context_relation_ids[safe_hop1_ids]
        path_hop2_mask = self.context_mask[safe_hop1_ids].clone()
        path_hop2_mask &= path_hop1_mask.unsqueeze(-1)
        path_hop2_mask &= path_hop2_entity_ids.ne(h_idx)

        return {
            'h_id': h,
            'r_id': r,
            't_id': t,
            'context_entity_ids': context_entity_ids,
            'context_relation_ids': context_relation_ids,
            'context_mask': context_mask,
            'path_hop1_entity_ids': context_entity_ids,
            'path_hop1_relation_ids': context_relation_ids,
            'path_hop1_mask': path_hop1_mask,
            'path_hop2_entity_ids': path_hop2_entity_ids,
            'path_hop2_relation_ids': path_hop2_relation_ids,
            'path_hop2_mask': path_hop2_mask,
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
            'path_batch': {
                'hop1_id': torch.stack([
                    item['path_hop1_entity_ids'] for item in batch
                ]),
                'hop1_rel_id': torch.stack([
                    item['path_hop1_relation_ids'] for item in batch
                ]),
                'hop1_mask': torch.stack([
                    item['path_hop1_mask'] for item in batch
                ]),
                'hop2_id': torch.stack([
                    item['path_hop2_entity_ids'] for item in batch
                ]),
                'hop2_rel_id': torch.stack([
                    item['path_hop2_relation_ids'] for item in batch
                ]),
                'hop2_mask': torch.stack([
                    item['path_hop2_mask'] for item in batch
                ]),
            },
        }
