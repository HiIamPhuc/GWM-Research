import json
import os


def build_relation_direction_mapping(relation2id):
    """Map full relation IDs to shared base IDs and inverse indicators."""
    relations = sorted(relation2id.items(), key=lambda item: item[1])
    base_relations = [
        relation
        for relation, _ in relations
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


def attach_relation_direction_mapping(config, data_dir):
    with open(
        os.path.join(data_dir, 'relation2id.json'),
        'r',
        encoding='utf-8',
    ) as f:
        mapping = build_relation_direction_mapping(json.load(f))

    config.num_base_relations = mapping['num_base_relations']
    config.relation_base_ids = mapping['full_to_base']
    config.relation_directions = mapping['directions']
    return mapping
