import json
import os
from collections import defaultdict

import torch


ARITY_THRESHOLD = 1.5


def _swap_arity(arity):
    if arity == '1-N':
        return 'N-1'
    if arity == 'N-1':
        return '1-N'
    return arity


def build_relation_direction_mapping(relation2id):
    """Map expanded relation IDs to shared base IDs and directions."""
    relations = sorted(relation2id.items(), key=lambda item: item[1])
    base_relations = [
        relation for relation, _ in relations
        if not relation.endswith('_inv')
    ]
    base_relation2id = {
        relation: base_id
        for base_id, relation in enumerate(base_relations)
    }

    full_to_base = [0] * len(relations)
    directions = [0] * len(relations)
    for relation, relation_id in relations:
        is_inverse = relation.endswith('_inv')
        base_relation = relation[:-4] if is_inverse else relation
        full_to_base[relation_id] = base_relation2id[base_relation]
        directions[relation_id] = int(is_inverse)

    return {
        'num_base_relations': len(base_relations),
        'full_to_base': full_to_base,
        'directions': directions,
    }


def build_relation_arity_mapping(data_dir, relation2id):
    """Classify full directional relation IDs from training triples only."""
    id2relation = {
        int(relation_id): relation
        for relation, relation_id in relation2id.items()
    }
    tails_by_hr = defaultdict(set)
    heads_by_rt = defaultdict(set)
    train_triples = torch.load(
        os.path.join(data_dir, 'train_triples.pt'),
        map_location='cpu',
    )
    for h, r, t in train_triples.tolist():
        relation_id = int(r)
        if id2relation[relation_id].endswith('_inv'):
            continue
        tails_by_hr[(int(h), relation_id)].add(int(t))
        heads_by_rt[(relation_id, int(t))].add(int(h))

    tails_per_relation = defaultdict(list)
    heads_per_relation = defaultdict(list)
    for (_, relation_id), tails in tails_by_hr.items():
        tails_per_relation[relation_id].append(len(tails))
    for (relation_id, _), heads in heads_by_rt.items():
        heads_per_relation[relation_id].append(len(heads))

    base_arities = {}
    for relation, relation_id in relation2id.items():
        if relation.endswith('_inv'):
            continue
        relation_id = int(relation_id)
        tail_counts = tails_per_relation[relation_id]
        head_counts = heads_per_relation[relation_id]
        avg_tails = sum(tail_counts) / len(tail_counts) if tail_counts else 0.0
        avg_heads = sum(head_counts) / len(head_counts) if head_counts else 0.0
        if avg_tails < ARITY_THRESHOLD and avg_heads < ARITY_THRESHOLD:
            arity = '1-1'
        elif avg_tails >= ARITY_THRESHOLD and avg_heads < ARITY_THRESHOLD:
            arity = '1-N'
        elif avg_tails < ARITY_THRESHOLD and avg_heads >= ARITY_THRESHOLD:
            arity = 'N-1'
        else:
            arity = 'N-N'
        base_arities[relation] = arity

    arities = ['1-1'] * len(relation2id)
    for relation, relation_id in relation2id.items():
        is_inverse = relation.endswith('_inv')
        base_relation = relation[:-4] if is_inverse else relation
        arity = base_arities[base_relation]
        arities[int(relation_id)] = _swap_arity(arity) if is_inverse else arity
    return arities


def attach_relation_direction_mapping(config, data_dir):
    with open(
        os.path.join(data_dir, 'relation2id.json'),
        'r',
        encoding='utf-8',
    ) as f:
        relation2id = json.load(f)
        mapping = build_relation_direction_mapping(relation2id)

    relation_arities = build_relation_arity_mapping(data_dir, relation2id)
    relation_slot_counts = [
        config.num_next_state_slots
        if arity in ('1-N', 'N-N') else 1
        for arity in relation_arities
    ]

    config.num_base_relations = mapping['num_base_relations']
    config.relation_base_ids = mapping['full_to_base']
    config.relation_directions = mapping['directions']
    config.relation_arities = relation_arities
    config.relation_slot_counts = relation_slot_counts
    mapping['arities'] = relation_arities
    mapping['slot_counts'] = relation_slot_counts
    return mapping
