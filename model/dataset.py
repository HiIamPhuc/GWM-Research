import json
import os

import torch
from torch.utils.data import Dataset


class TrainTruthIndex:
    """Known training tails used to filter false in-batch negatives."""

    def __init__(self, train_triples):
        self.query_tails = {}
        for h, r, t in train_triples.tolist():
            self.query_tails.setdefault((h, r), set()).add(t)

    def build_in_batch_truth_mask(
        self, head_ids, relation_ids, candidate_tail_ids, device=None
    ):
        heads = head_ids.tolist()
        relations = relation_ids.tolist()
        tails = candidate_tail_ids.tolist()
        tail_columns = {}
        for column, tail in enumerate(tails):
            tail_columns.setdefault(tail, []).append(column)

        mask = torch.zeros(len(heads), len(tails), dtype=torch.bool, device=device)
        for row, query in enumerate(zip(heads, relations)):
            for tail in self.query_tails.get(query, ()):
                mask[row, tail_columns.get(tail, [])] = True
        mask.fill_diagonal_(True)
        return mask


class GWMDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split

        with open(os.path.join(data_dir, 'entity2id.json'), encoding='utf-8') as file:
            self.num_entities = len(json.load(file))
        with open(os.path.join(data_dir, 'relation2id.json'), encoding='utf-8') as file:
            self.num_relations = len(json.load(file))

        self.triples = torch.load(
            os.path.join(data_dir, f'{split}_triples.pt'),
            map_location='cpu',
        ).long()
        context = torch.load(
            os.path.join(data_dir, 'context_neighbors.pt'),
            map_location='cpu',
        )
        self.context_entity_ids = context['entity_ids'].long()
        self.context_relation_ids = context['relation_ids'].long()
        self.context_mask = context['mask'].bool()

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        head, relation, tail = self.triples[index]
        context_entities = self.context_entity_ids[head]
        context_relations = self.context_relation_ids[head]
        context_mask = self.context_mask[head].clone()
        answer_edge = context_entities.eq(tail) & context_relations.eq(relation)

        return {
            'h_id': head,
            'r_id': relation,
            't_id': tail,
            'context_entity_ids': context_entities,
            'context_relation_ids': context_relations,
            'context_mask': context_mask & ~answer_edge,
        }


class CollateFN:
    def __call__(self, batch):
        heads = torch.stack([item['h_id'] for item in batch])
        relations = torch.stack([item['r_id'] for item in batch])
        tails = torch.stack([item['t_id'] for item in batch])

        entity_chunks = []
        relation_chunks = []
        batch_chunks = []
        for batch_index, item in enumerate(batch):
            mask = item['context_mask']
            entities = item['context_entity_ids'][mask]
            context_relations = item['context_relation_ids'][mask]
            if entities.numel():
                entity_chunks.append(entities)
                relation_chunks.append(context_relations)
                batch_chunks.append(torch.full_like(entities, batch_index))

        if entity_chunks:
            context_entities = torch.cat(entity_chunks)
            context_relations = torch.cat(relation_chunks)
            context_batch = torch.cat(batch_chunks)
        else:
            context_entities = torch.empty(0, dtype=torch.long)
            context_relations = torch.empty(0, dtype=torch.long)
            context_batch = torch.empty(0, dtype=torch.long)

        return {
            'h_batch': {'id': heads},
            'r_batch': {'id': relations},
            't_batch': {'id': tails},
            'context_batch': {
                'id': context_entities,
                'rel_id': context_relations,
                'batch_index': context_batch,
            },
        }
