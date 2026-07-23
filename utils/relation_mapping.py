import json
from pathlib import Path


RELATION_DIRECTION_MAP_FILE = 'relation_direction_map.json'


def build_relation_direction_mapping(relation2id, require_inverse=True):
    """Map expanded relation IDs to shared base IDs and directions."""
    if not relation2id:
        raise ValueError("relation2id must contain at least one relation.")

    normalized = {str(name): int(relation_id) for name, relation_id in relation2id.items()}
    relation_ids = sorted(normalized.values())
    expected_ids = list(range(len(normalized)))
    if relation_ids != expected_ids:
        raise ValueError(
            "Relation IDs must be contiguous and start at zero. "
            f"Expected {expected_ids[:10]}, got {relation_ids[:10]}."
        )

    base_relations = [
        relation
        for relation, _ in sorted(normalized.items(), key=lambda item: item[1])
        if not relation.endswith('_inv')
    ]
    if not base_relations:
        raise ValueError("No forward relations were found in relation2id.")

    base_relation2id = {
        relation: base_id
        for base_id, relation in enumerate(base_relations)
    }
    full_to_base = [0] * len(normalized)
    directions = [0] * len(normalized)

    for relation, full_id in normalized.items():
        is_inverse = relation.endswith('_inv')
        base_relation = relation[:-4] if is_inverse else relation
        if base_relation not in base_relation2id:
            raise ValueError(
                f"Inverse relation {relation!r} has no forward relation entry."
            )

        full_to_base[full_id] = base_relation2id[base_relation]
        directions[full_id] = int(is_inverse)

    if require_inverse:
        missing = [
            relation
            for relation in base_relations
            if relation + '_inv' not in normalized
        ]
        if missing:
            preview = ', '.join(missing[:10])
            raise ValueError(
                "Shared forward/inverse parameterization requires an inverse "
                f"entry for every relation. Missing: {preview}"
            )

    return {
        'version': 1,
        'num_relations': len(normalized),
        'num_base_relations': len(base_relations),
        'full_to_base': full_to_base,
        'directions': directions,
        'base_relation2id': base_relation2id,
    }


def save_relation_direction_mapping(output_dir, relation2id, require_inverse=True):
    mapping = build_relation_direction_mapping(
        relation2id,
        require_inverse=require_inverse,
    )
    output_path = Path(output_dir) / RELATION_DIRECTION_MAP_FILE
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)
    return mapping


def load_relation_direction_mapping(data_dir, require_inverse=True):
    """Load persisted metadata, deriving it for older processed datasets."""
    data_path = Path(data_dir)
    with open(data_path / 'relation2id.json', 'r', encoding='utf-8') as f:
        relation2id = json.load(f)

    expected = build_relation_direction_mapping(
        relation2id,
        require_inverse=require_inverse,
    )
    mapping_path = data_path / RELATION_DIRECTION_MAP_FILE
    if not mapping_path.exists():
        return expected

    with open(mapping_path, 'r', encoding='utf-8') as f:
        stored = json.load(f)

    compared_keys = (
        'num_relations',
        'num_base_relations',
        'full_to_base',
        'directions',
        'base_relation2id',
    )
    mismatched = [
        key for key in compared_keys
        if stored.get(key) != expected.get(key)
    ]
    if mismatched:
        raise ValueError(
            f"{mapping_path} is inconsistent with relation2id.json for: "
            + ', '.join(mismatched)
        )
    return stored


def attach_relation_direction_mapping(config, data_dir):
    mapping = load_relation_direction_mapping(data_dir, require_inverse=True)
    config.num_base_relations = int(mapping['num_base_relations'])
    config.relation_base_ids = list(mapping['full_to_base'])
    config.relation_directions = list(mapping['directions'])
    return mapping
