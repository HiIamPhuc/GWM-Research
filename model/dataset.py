import os

import torch
from torch.utils.data import Dataset


class GWMDataset(Dataset):
    """Knowledge graph triples with fixed one- and two-hop head context."""

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

        paths_path = os.path.join(data_dir, 'context_paths.pt')
        paths = torch.load(paths_path, map_location='cpu')
        self.path_intermediate_entity_ids = paths[
            'intermediate_entity_ids'
        ].long()
        self.path_final_entity_ids = paths['final_entity_ids'].long()
        self.path_first_relation_ids = paths['first_relation_ids'].long()
        self.path_second_relation_ids = paths['second_relation_ids'].long()
        self.path_mask = paths['mask'].bool()

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

        path_intermediate_entity_ids = self.path_intermediate_entity_ids[h_idx]
        path_final_entity_ids = self.path_final_entity_ids[h_idx]
        path_first_relation_ids = self.path_first_relation_ids[h_idx]
        path_second_relation_ids = self.path_second_relation_ids[h_idx]
        path_mask = self.path_mask[h_idx].clone()
        answer_first_edge = (
            path_intermediate_entity_ids.eq(t)
            & path_first_relation_ids.eq(r)
        )
        path_mask &= ~answer_first_edge

        return {
            'h_id': h,
            'r_id': r,
            't_id': t,
            'context_entity_ids': context_entity_ids,
            'context_relation_ids': context_relation_ids,
            'context_mask': context_mask,
            'path_intermediate_entity_ids': path_intermediate_entity_ids,
            'path_final_entity_ids': path_final_entity_ids,
            'path_first_relation_ids': path_first_relation_ids,
            'path_second_relation_ids': path_second_relation_ids,
            'path_mask': path_mask,
        }

    def __getitem__(self, idx):
        return self.make_item(*self.triples[idx])


class CollateFN:
    """Collate triples and fixed-width world-state context."""

    def __call__(self, batch):
        return {
            'h_batch': {'id': torch.stack([item['h_id'] for item in batch])},
            'r_batch': {'id': torch.stack([item['r_id'] for item in batch])},
            't_batch': {'id': torch.stack([item['t_id'] for item in batch])},
            'context_batch': {
                'id': torch.stack([item['context_entity_ids'] for item in batch]),
                'rel_id': torch.stack([item['context_relation_ids'] for item in batch]),
                'mask': torch.stack([item['context_mask'] for item in batch]),
                'path_intermediate_id': torch.stack(
                    [item['path_intermediate_entity_ids'] for item in batch]
                ),
                'path_final_id': torch.stack(
                    [item['path_final_entity_ids'] for item in batch]
                ),
                'path_first_rel_id': torch.stack(
                    [item['path_first_relation_ids'] for item in batch]
                ),
                'path_second_rel_id': torch.stack(
                    [item['path_second_relation_ids'] for item in batch]
                ),
                'path_mask': torch.stack(
                    [item['path_mask'] for item in batch]
                ),
            },
        }
