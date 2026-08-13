import torch
from torch.utils.data import Dataset
import json
import os


class TrainTruthIndex:
    def __init__(self, train_triples):
        train_triples = torch.as_tensor(train_triples, dtype=torch.long)
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
        candidate_tail_ids = torch.as_tensor(candidate_tail_ids, dtype=torch.long).reshape(-1).cpu()

        batch_size = head_ids.numel()
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
        if device is not None:
            truth_mask = truth_mask.to(device)
        return truth_mask


class GWMDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split

        with open(os.path.join(data_dir, 'entity2id.json'), 'r', encoding='utf-8') as f:
            self.num_entities = len(json.load(f))
        with open(os.path.join(data_dir, 'relation2id.json'), 'r', encoding='utf-8') as f:
            self.num_relations = len(json.load(f))
        
        triples_path = os.path.join(data_dir, f'{split}_triples.pt')
        self.triples = torch.load(triples_path, map_location='cpu').long()
        context_pack_path = os.path.join(data_dir, 'context_neighbors.pt')
        context_pack = torch.load(context_pack_path, map_location='cpu')
        self.context_entity_ids = context_pack['entity_ids'].long()
        self.context_relation_ids = context_pack['relation_ids'].long()
        self.context_mask = context_pack['mask'].bool()

    def __len__(self):
        return len(self.triples)
        
    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        h_idx = int(h.item())

        ctx_entity_ids = self.context_entity_ids[h_idx]
        ctx_relation_ids = self.context_relation_ids[h_idx]
        ctx_mask = self.context_mask[h_idx].clone()
        target_edge = ctx_entity_ids.eq(t) & ctx_relation_ids.eq(r)
        ctx_mask &= ~target_edge

        return {
            'h_id': h.long(),
            'r_id': r.long(),
            't_id': t.long(),
            'context_entity_ids': ctx_entity_ids.long(),
            'context_relation_ids': ctx_relation_ids.long(),
            'context_mask': ctx_mask.bool(),
        }

class CollateFN:
    def __call__(self, batch):
        h_ids = torch.stack([b['h_id'] for b in batch])
        r_ids = torch.stack([b['r_id'] for b in batch])
        t_ids = torch.stack([b['t_id'] for b in batch])

        # Build ragged context representation: flattened edges + edge->sample index.
        context_entity_chunks = []
        context_relation_chunks = []
        context_batch_chunks = []
        for sample_idx, item in enumerate(batch):
            ent_ids = item['context_entity_ids']
            rel_ids = item['context_relation_ids']
            mask = item['context_mask'].bool()

            valid_ent = ent_ids[mask]
            valid_rel = rel_ids[mask]

            if valid_ent.numel() > 0:
                context_entity_chunks.append(valid_ent.long())
                context_relation_chunks.append(valid_rel.long())
                context_batch_chunks.append(torch.full((valid_ent.numel(),), sample_idx, dtype=torch.long))

        if context_entity_chunks:
            context_entity_ids = torch.cat(context_entity_chunks, dim=0)
            context_relation_ids = torch.cat(context_relation_chunks, dim=0)
            context_batch_index = torch.cat(context_batch_chunks, dim=0)
        else:
            context_entity_ids = torch.zeros(0, dtype=torch.long)
            context_relation_ids = torch.zeros(0, dtype=torch.long)
            context_batch_index = torch.zeros(0, dtype=torch.long)
        
        return {
            'h_batch': {'id': h_ids},
            'r_batch': {'id': r_ids},
            't_batch': {'id': t_ids},
            'context_batch': {
                'id': context_entity_ids,
                'rel_id': context_relation_ids,
                'batch_index': context_batch_index,
            },
        }
