import torch
from torch.utils.data import Dataset
import json
import os

class GWMDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Dataset for GWM.
        Loads triples and relation-aware context edge tensors.
        """
        self.data_dir = data_dir
        self.split = split

        with open(os.path.join(data_dir, 'relation2id.json'), 'r', encoding='utf-8') as f:
            self.num_relations = len(json.load(f))
        
        # Load triples
        triples_path = os.path.join(data_dir, f'{split}_triples.pt')
        if not os.path.exists(triples_path):
             # Fallback for WN18RR dev vs valid naming if needed, but preprocess handles it
             if split == 'valid' and not os.path.exists(triples_path):
                 triples_path = os.path.join(data_dir, 'dev_triples.pt')
                 
        self.triples = torch.load(triples_path)
        
        # Load compact relation-aware context artifact.
        context_pack_path = os.path.join(data_dir, 'context_neighbors.pt')

        self.context_entity_ids = None
        self.context_relation_ids = None
        self.context_mask = None
        self.context_pad_value = -1

        if os.path.exists(context_pack_path):
            context_pack = torch.load(context_pack_path)
            self.context_entity_ids = context_pack['entity_ids'].long()
            self.context_relation_ids = context_pack['relation_ids'].long()
            self.context_mask = context_pack['mask'].bool()
            self.context_pad_value = int(context_pack.get('pad_value', -1))
        else:
            raise FileNotFoundError(
                "Error: context files not found "
                "(expected context_neighbors.pt)."
            )

    def __len__(self):
        return len(self.triples)
        
    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        h_idx = int(h.item())

        # Retrieve context row for this head.
        if self.context_entity_ids is not None:
            ctx_entity_ids = self.context_entity_ids[h_idx]
            ctx_relation_ids = self.context_relation_ids[h_idx]
            ctx_mask = self.context_mask[h_idx]
        else:
            # Dummy fallback with zero neighbors.
            ctx_entity_ids = torch.zeros(0, dtype=torch.long)
            ctx_relation_ids = torch.zeros(0, dtype=torch.long)
            ctx_mask = torch.zeros(0, dtype=torch.bool)

        return {
            'h_id': h.long(),
            'r_id': r.long(),
            't_id': t.long(),
            'context_entity_ids': ctx_entity_ids.long(),
            'context_relation_ids': ctx_relation_ids.long(),
            'context_mask': ctx_mask.bool(),
        }

class CollateFN:
    """
    ID-only collator; text embeddings are loaded from precomputed caches.
    """
    def __init__(self):
        pass
        
    def __call__(self, batch):
        h_ids = torch.stack([b['h_id'] for b in batch])
        r_ids = torch.stack([b['r_id'] for b in batch])
        t_ids = torch.stack([b['t_id'] for b in batch])

        max_context_len = max((b['context_entity_ids'].numel() for b in batch), default=0)
        context_entity_ids_seq = torch.full((len(batch), max_context_len), -1, dtype=torch.long)
        context_relation_ids_seq = torch.full((len(batch), max_context_len), -1, dtype=torch.long)
        context_mask_seq = torch.zeros((len(batch), max_context_len), dtype=torch.bool)

        for sample_idx, item in enumerate(batch):
            ent_ids = item['context_entity_ids'].long().reshape(-1)
            rel_ids = item['context_relation_ids'].long().reshape(-1)
            mask = item['context_mask'].bool().reshape(-1)

            seq_len = min(ent_ids.numel(), rel_ids.numel(), mask.numel(), max_context_len)
            if seq_len > 0:
                context_entity_ids_seq[sample_idx, :seq_len] = ent_ids[:seq_len]
                context_relation_ids_seq[sample_idx, :seq_len] = rel_ids[:seq_len]
                context_mask_seq[sample_idx, :seq_len] = mask[:seq_len]

        # Build ragged context representation: flattened edges + edge->sample index.
        context_entity_chunks = []
        context_relation_chunks = []
        context_batch_chunks = []
        for sample_idx, item in enumerate(batch):
            ent_ids = item['context_entity_ids']
            rel_ids = item['context_relation_ids']
            mask = item['context_mask'].bool()

            if ent_ids.dim() == 1:
                valid_ent = ent_ids[mask] if mask.numel() == ent_ids.numel() else ent_ids
                valid_rel = rel_ids[mask] if mask.numel() == rel_ids.numel() else rel_ids
            else:
                # Fallback safety; flatten unusual shapes.
                valid_ent = ent_ids.reshape(-1)
                valid_rel = rel_ids.reshape(-1)

            # Extra guard for sentinel padding values.
            valid_pair_mask = (valid_ent >= 0) & (valid_rel >= 0)
            valid_ent = valid_ent[valid_pair_mask]
            valid_rel = valid_rel[valid_pair_mask]

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
                'sequence_id': context_entity_ids_seq,
                'sequence_rel_id': context_relation_ids_seq,
                'sequence_mask': context_mask_seq,
                'mask': context_mask_seq,
            },
        }
