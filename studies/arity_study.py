"""Arity-based evaluation study for saved GWM predictions.

The relation arity classification follows the common OpenKE protocol:

* avg tails per (head, relation) < 1.5 and avg heads per (relation, tail) < 1.5 -> 1-1
* avg tails per (head, relation) >= 1.5 and avg heads per (relation, tail) < 1.5 -> 1-N
* avg tails per (head, relation) < 1.5 and avg heads per (relation, tail) >= 1.5 -> N-1
* otherwise -> N-N

For bidirectional GWM evaluation, backward predictions are assigned a
direction-aware query arity by swapping 1-N and N-1. This reflects that
ranking heads through an inverse relation reverses the query multiplicity.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import torch


ARITY_ORDER = ('1-1', '1-N', 'N-1', 'N-N')


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _relation_maps(data_dir):
    relation2id = _load_json(os.path.join(data_dir, 'relation2id.json'))
    id2relation = {int(idx): relation for relation, idx in relation2id.items()}
    return relation2id, id2relation


def _is_inverse_relation(relation_name):
    return relation_name.endswith('_inv')


def _base_relation_name(relation_name):
    return relation_name[:-4] if _is_inverse_relation(relation_name) else relation_name


def _swap_arity(arity):
    if arity == '1-N':
        return 'N-1'
    if arity == 'N-1':
        return '1-N'
    return arity


def _iter_original_triples(data_dir, id2relation, splits):
    for split in splits:
        path = os.path.join(data_dir, f'{split}_triples.pt')
        if not os.path.exists(path):
            continue

        triples = torch.load(path, map_location='cpu')
        for h, r, t in triples.tolist():
            relation_name = id2relation[int(r)]
            if _is_inverse_relation(relation_name):
                continue
            yield int(h), int(r), int(t)


def classify_relation_arities(data_dir, splits=('train', 'valid', 'test')):
    relation2id, id2relation = _relation_maps(data_dir)

    tails_by_hr = defaultdict(set)
    heads_by_rt = defaultdict(set)
    support = defaultdict(int)

    for h, r, t in _iter_original_triples(data_dir, id2relation, splits):
        tails_by_hr[(h, r)].add(t)
        heads_by_rt[(r, t)].add(h)
        support[r] += 1

    hr_by_relation = defaultdict(list)
    rt_by_relation = defaultdict(list)

    for (_, r), tails in tails_by_hr.items():
        hr_by_relation[r].append(len(tails))
    for (r, _), heads in heads_by_rt.items():
        rt_by_relation[r].append(len(heads))

    rows = []
    arity_by_relation = {}

    for relation_name, relation_id in sorted(relation2id.items(), key=lambda item: item[1]):
        relation_id = int(relation_id)
        if _is_inverse_relation(relation_name):
            continue

        avg_tails = (
            sum(hr_by_relation[relation_id]) / len(hr_by_relation[relation_id])
            if hr_by_relation[relation_id]
            else 0.0
        )
        avg_heads = (
            sum(rt_by_relation[relation_id]) / len(rt_by_relation[relation_id])
            if rt_by_relation[relation_id]
            else 0.0
        )

        if avg_tails < 1.5 and avg_heads < 1.5:
            arity = '1-1'
        elif avg_tails >= 1.5 and avg_heads < 1.5:
            arity = '1-N'
        elif avg_tails < 1.5 and avg_heads >= 1.5:
            arity = 'N-1'
        else:
            arity = 'N-N'

        arity_by_relation[relation_id] = arity
        rows.append({
            'relation_id': relation_id,
            'relation': relation_name,
            'arity': arity,
            'avg_tails_per_head_relation': avg_tails,
            'avg_heads_per_relation_tail': avg_heads,
            'support_triples': support[relation_id],
            'num_head_relation_pairs': len(hr_by_relation[relation_id]),
            'num_relation_tail_pairs': len(rt_by_relation[relation_id]),
        })

    return rows, arity_by_relation, relation2id, id2relation


def _new_metric_state():
    return {
        'count': 0,
        'rank_sum': 0.0,
        'rr_sum': 0.0,
        'hits1': 0,
        'hits3': 0,
        'hits10': 0,
    }


def _add_rank(state, rank):
    state['count'] += 1
    state['rank_sum'] += rank
    state['rr_sum'] += 1.0 / rank
    state['hits1'] += int(rank <= 1)
    state['hits3'] += int(rank <= 3)
    state['hits10'] += int(rank <= 10)


def _finalize_state(state):
    count = state['count']
    if count == 0:
        return {
            'count': 0,
            'mrr': 0.0,
            'mr': 0.0,
            'hits1': 0.0,
            'hits3': 0.0,
            'hits10': 0.0,
        }
    return {
        'count': count,
        'mrr': state['rr_sum'] / count,
        'mr': state['rank_sum'] / count,
        'hits1': state['hits1'] / count,
        'hits3': state['hits3'] / count,
        'hits10': state['hits10'] / count,
    }


def _read_predictions(path, direction, id2relation, relation2id, arity_by_relation):
    if path is None or not os.path.exists(path):
        return []

    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            relation_id = int(record['r'])
            relation_name = id2relation[relation_id]
            base_relation = _base_relation_name(relation_name)
            base_relation_id = int(relation2id[base_relation])
            base_arity = arity_by_relation.get(base_relation_id, 'unknown')
            query_arity = _swap_arity(base_arity) if direction == 'backward' else base_arity

            rows.append({
                'direction': direction,
                'h': int(record['h']),
                'r': relation_id,
                't': int(record['t']),
                'rank': int(record['rank']),
                'relation': relation_name,
                'base_relation_id': base_relation_id,
                'base_relation': base_relation,
                'base_arity': base_arity,
                'query_arity': query_arity,
            })
    return rows


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _metric_rows(prediction_rows):
    states = defaultdict(_new_metric_state)

    for row in prediction_rows:
        rank = row['rank']
        direction = row['direction']
        arity = row['query_arity']

        _add_rank(states[(direction, arity)], rank)
        _add_rank(states[(direction, 'all')], rank)
        _add_rank(states[('micro', arity)], rank)
        _add_rank(states[('micro', 'all')], rank)

    rows = []
    for direction in ('forward', 'backward', 'micro'):
        for arity in (*ARITY_ORDER, 'all'):
            metrics = _finalize_state(states[(direction, arity)])
            rows.append({
                'direction': direction,
                'arity': arity,
                **metrics,
            })
    return rows


def _relation_metric_rows(prediction_rows):
    states = defaultdict(_new_metric_state)
    meta = {}

    for row in prediction_rows:
        key = (row['base_relation_id'], row['direction'])
        meta[key] = {
            'relation_id': row['base_relation_id'],
            'relation': row['base_relation'],
            'base_arity': row['base_arity'],
            'direction': row['direction'],
            'query_arity': row['query_arity'],
        }
        _add_rank(states[key], row['rank'])

    rows = []
    for key in sorted(states):
        rows.append({
            **meta[key],
            **_finalize_state(states[key]),
        })
    return rows


def run_arity_study(
    data_dir,
    output_dir,
    forward_predictions_path=None,
    backward_predictions_path=None,
    splits=('train', 'valid', 'test'),
):
    output_dir = Path(output_dir)
    study_dir = output_dir / 'arity_study'
    study_dir.mkdir(parents=True, exist_ok=True)

    if forward_predictions_path is None:
        forward_predictions_path = output_dir / 'predictions_test.jsonl'
    if backward_predictions_path is None:
        backward_predictions_path = output_dir / 'predictions_test_backward.jsonl'

    relation_rows, arity_by_relation, relation2id, id2relation = classify_relation_arities(
        data_dir=data_dir,
        splits=splits,
    )

    prediction_rows = []
    prediction_rows.extend(
        _read_predictions(
            forward_predictions_path,
            direction='forward',
            id2relation=id2relation,
            relation2id=relation2id,
            arity_by_relation=arity_by_relation,
        )
    )
    prediction_rows.extend(
        _read_predictions(
            backward_predictions_path,
            direction='backward',
            id2relation=id2relation,
            relation2id=relation2id,
            arity_by_relation=arity_by_relation,
        )
    )

    if not prediction_rows:
        raise FileNotFoundError(
            "No prediction rows found. Run evaluate.py with prediction saving first."
        )

    metric_rows = _metric_rows(prediction_rows)
    relation_metric_rows = _relation_metric_rows(prediction_rows)

    _write_csv(
        study_dir / 'relation_arity.csv',
        relation_rows,
        [
            'relation_id',
            'relation',
            'arity',
            'avg_tails_per_head_relation',
            'avg_heads_per_relation_tail',
            'support_triples',
            'num_head_relation_pairs',
            'num_relation_tail_pairs',
        ],
    )
    _write_csv(
        study_dir / 'arity_metrics.csv',
        metric_rows,
        ['direction', 'arity', 'count', 'mrr', 'mr', 'hits1', 'hits3', 'hits10'],
    )
    _write_csv(
        study_dir / 'relation_direction_metrics.csv',
        relation_metric_rows,
        [
            'relation_id',
            'relation',
            'base_arity',
            'direction',
            'query_arity',
            'count',
            'mrr',
            'mr',
            'hits1',
            'hits3',
            'hits10',
        ],
    )
    _write_csv(
        study_dir / 'predictions_with_arity.csv',
        prediction_rows,
        [
            'direction',
            'h',
            'r',
            't',
            'rank',
            'relation',
            'base_relation_id',
            'base_relation',
            'base_arity',
            'query_arity',
        ],
    )

    summary = {
        'data_dir': str(data_dir),
        'output_dir': str(output_dir),
        'splits_used_for_arity': list(splits),
        'num_prediction_rows': len(prediction_rows),
        'files': {
            'relation_arity': str(study_dir / 'relation_arity.csv'),
            'arity_metrics': str(study_dir / 'arity_metrics.csv'),
            'relation_direction_metrics': str(study_dir / 'relation_direction_metrics.csv'),
            'predictions_with_arity': str(study_dir / 'predictions_with_arity.csv'),
        },
    }
    with open(study_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Run arity-based study from saved predictions.')
    parser.add_argument('--data_dir', required=True, help='Processed data directory.')
    parser.add_argument('--output_dir', required=True, help='Training/evaluation artifact directory.')
    parser.add_argument('--forward_predictions', default=None, help='Path to forward prediction JSONL.')
    parser.add_argument('--backward_predictions', default=None, help='Path to backward prediction JSONL.')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'valid', 'test'],
        help='Processed splits used to classify relation arity.',
    )
    args = parser.parse_args()

    summary = run_arity_study(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        forward_predictions_path=args.forward_predictions,
        backward_predictions_path=args.backward_predictions,
        splits=args.splits,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
