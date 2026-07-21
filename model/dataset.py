import torch
from torch.utils.data import Dataset
import json
import os


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
    def __init__(self, data_dir, split='train'):
        """
        Dataset for GWM.
        Loads triples and relation-aware context edge tensors.
        """
        self.data_dir = data_dir
        self.split = split

        with open(os.path.join(data_dir, 'entity2id.json'), 'r', encoding='utf-8') as f:
            self.num_entities = len(json.load(f))
        with open(os.path.join(data_dir, 'relation2id.json'), 'r', encoding='utf-8') as f:
            relation2id = json.load(f)
        self.num_relations = len(relation2id)
        self.relation_direction_lookup = torch.zeros(
            self.num_relations,
            dtype=torch.long,
        )
        for relation, relation_id in relation2id.items():
            if relation.endswith('_inv'):
                self.relation_direction_lookup[int(relation_id)] = 1
        
        # Load triples
        triples_path = os.path.join(data_dir, f'{split}_triples.pt')
        if not os.path.exists(triples_path):
            if split == 'valid':
                triples_path = os.path.join(data_dir, 'dev_triples.pt')
        if not os.path.exists(triples_path):
            raise FileNotFoundError(f"Triple tensor not found: {triples_path}")
                 
        self.triples = torch.load(triples_path, map_location='cpu').long()
        if self.triples.dim() != 2 or self.triples.size(1) != 3:
            raise ValueError(
                f"Expected triples with shape (N, 3), got {tuple(self.triples.shape)}"
            )

        self.num_input_triples = int(self.triples.size(0))
        
        # Load compact relation-aware context artifact.
        context_pack_path = os.path.join(data_dir, 'context_neighbors.pt')

        self.context_entity_ids = None
        self.context_relation_ids = None
        self.context_direction_ids = None
        self.context_mask = None
        self.context_pad_value = -1

        if os.path.exists(context_pack_path):
            context_pack = torch.load(context_pack_path, map_location='cpu')
            self.context_entity_ids = context_pack['entity_ids'].long()
            self.context_relation_ids = context_pack['relation_ids'].long()
            stored_directions = context_pack.get('direction_ids')
            if stored_directions is not None:
                self.context_direction_ids = stored_directions.long()
            else:
                safe_relation_ids = self.context_relation_ids.clamp_min(0)
                self.context_direction_ids = self.relation_direction_lookup[
                    safe_relation_ids
                ]
            self.context_mask = context_pack['mask'].bool()
            self.context_pad_value = int(context_pack.get('pad_value', -1))
            expected_shape = self.context_entity_ids.shape
            if (
                self.context_entity_ids.dim() != 2
                or self.context_relation_ids.shape != expected_shape
                or self.context_direction_ids.shape != expected_shape
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
            valid_directions = self.context_direction_ids[self.context_mask]
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
            if valid_directions.numel() and not torch.all(
                (valid_directions == 0) | (valid_directions == 1)
            ):
                raise ValueError("Context artifact contains invalid direction IDs.")
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
            ctx_direction_ids = self.context_direction_ids[h_idx]
            ctx_mask = self.context_mask[h_idx].clone()

            target_edge = (
                ctx_entity_ids.eq(int(t.item()))
                & ctx_relation_ids.eq(int(r.item()))
            )
            ctx_mask &= ~target_edge
        else:
            # Dummy fallback with zero neighbors.
            ctx_entity_ids = torch.zeros(0, dtype=torch.long)
            ctx_relation_ids = torch.zeros(0, dtype=torch.long)
            ctx_direction_ids = torch.zeros(0, dtype=torch.long)
            ctx_mask = torch.zeros(0, dtype=torch.bool)

        return {
            'h_id': h.long(),
            'r_id': r.long(),
            't_id': t.long(),
            'context_entity_ids': ctx_entity_ids.long(),
            'context_relation_ids': ctx_relation_ids.long(),
            'context_direction_ids': ctx_direction_ids.long(),
            'context_mask': ctx_mask.bool(),
        }

class CollateFN:
    """
    ID-only collator; text embeddings are loaded from precomputed caches.
    """
    def __call__(self, batch):
        h_ids = torch.stack([b['h_id'] for b in batch])
        r_ids = torch.stack([b['r_id'] for b in batch])
        t_ids = torch.stack([b['t_id'] for b in batch])

        # Build ragged context representation: flattened edges + edge->sample index.
        context_entity_chunks = []
        context_relation_chunks = []
        context_direction_chunks = []
        context_batch_chunks = []
        for sample_idx, item in enumerate(batch):
            ent_ids = item['context_entity_ids']
            rel_ids = item['context_relation_ids']
            direction_ids = item['context_direction_ids']
            mask = item['context_mask'].bool()

            if (
                ent_ids.dim() != 1
                or rel_ids.dim() != 1
                or direction_ids.dim() != 1
                or mask.dim() != 1
            ):
                raise ValueError("Each context row must be one-dimensional.")
            if not (
                ent_ids.numel()
                == rel_ids.numel()
                == direction_ids.numel()
                == mask.numel()
            ):
                raise ValueError("Context entity, relation, direction, and mask lengths differ.")

            valid_ent = ent_ids[mask]
            valid_rel = rel_ids[mask]
            valid_direction = direction_ids[mask]

            # Extra guard for sentinel padding values.
            valid_pair_mask = (valid_ent >= 0) & (valid_rel >= 0)
            valid_ent = valid_ent[valid_pair_mask]
            valid_rel = valid_rel[valid_pair_mask]
            valid_direction = valid_direction[valid_pair_mask]

            if valid_ent.numel() > 0:
                context_entity_chunks.append(valid_ent.long())
                context_relation_chunks.append(valid_rel.long())
                context_direction_chunks.append(valid_direction.long())
                context_batch_chunks.append(torch.full((valid_ent.numel(),), sample_idx, dtype=torch.long))

        if context_entity_chunks:
            context_entity_ids = torch.cat(context_entity_chunks, dim=0)
            context_relation_ids = torch.cat(context_relation_chunks, dim=0)
            context_direction_ids = torch.cat(context_direction_chunks, dim=0)
            context_batch_index = torch.cat(context_batch_chunks, dim=0)
        else:
            context_entity_ids = torch.zeros(0, dtype=torch.long)
            context_relation_ids = torch.zeros(0, dtype=torch.long)
            context_direction_ids = torch.zeros(0, dtype=torch.long)
            context_batch_index = torch.zeros(0, dtype=torch.long)
        
        return {
            'h_batch': {'id': h_ids},
            'r_batch': {'id': r_ids},
            't_batch': {'id': t_ids},
            'context_batch': {
                'id': context_entity_ids,
                'rel_id': context_relation_ids,
                'direction_id': context_direction_ids,
                'batch_index': context_batch_index,
            },
        }
