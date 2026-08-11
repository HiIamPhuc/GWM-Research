"""Arity-based evaluation study for saved GWM predictions.

The relation arity classification follows the common OpenKE protocol:

* avg tails per (head, relation) < 1.5 and avg heads per (relation, tail) < 1.5 -> 1-1
* avg tails per (head, relation) >= 1.5 and avg heads per (relation, tail) < 1.5 -> 1-N
* avg tails per (head, relation) < 1.5 and avg heads per (relation, tail) >= 1.5 -> N-1
* otherwise -> N-N

The study uses the saved main-protocol predictions for original test triples.
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


def _read_predictions(path, id2relation, arity_by_relation):
    if path is None or not os.path.exists(path):
        return []

    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            relation_id = int(record['r'])
            relation_name = id2relation[relation_id]
            arity = arity_by_relation.get(relation_id, 'unknown')

            rows.append({
                'h': int(record['h']),
                'r': relation_id,
                't': int(record['t']),
                'rank': int(record['rank']),
                'relation': relation_name,
                'arity': arity,
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
        _add_rank(states[row['arity']], rank)
        _add_rank(states['all'], rank)

    rows = []
    for arity in (*ARITY_ORDER, 'all'):
        rows.append({'arity': arity, **_finalize_state(states[arity])})
    return rows


def _relation_metric_rows(prediction_rows):
    states = defaultdict(_new_metric_state)
    meta = {}

    for row in prediction_rows:
        key = row['r']
        meta[key] = {
            'relation_id': row['r'],
            'relation': row['relation'],
            'arity': row['arity'],
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
    predictions_path=None,
    splits=('train', 'valid', 'test'),
):
    output_dir = Path(output_dir)
    study_dir = output_dir / 'arity_study'
    study_dir.mkdir(parents=True, exist_ok=True)

    if predictions_path is None:
        predictions_path = output_dir / 'predictions_test.jsonl'

    relation_rows, arity_by_relation, _, id2relation = classify_relation_arities(
        data_dir=data_dir,
        splits=splits,
    )

    prediction_rows = _read_predictions(
        predictions_path,
        id2relation=id2relation,
        arity_by_relation=arity_by_relation,
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
        study_dir / 'arity_metrics.csv', metric_rows,
        ['arity', 'count', 'mrr', 'mr', 'hits1', 'hits3', 'hits10'],
    )
    _write_csv(
        study_dir / 'relation_metrics.csv',
        relation_metric_rows,
        [
            'relation_id',
            'relation',
            'arity',
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
            'h',
            'r',
            't',
            'rank',
            'relation',
            'arity',
        ],
    )

    summary = {
        'evaluation_protocol': 'main',
        'data_dir': str(data_dir),
        'output_dir': str(output_dir),
        'splits_used_for_arity': list(splits),
        'num_prediction_rows': len(prediction_rows),
        'files': {
            'relation_arity': str(study_dir / 'relation_arity.csv'),
            'arity_metrics': str(study_dir / 'arity_metrics.csv'),
            'relation_metrics': str(study_dir / 'relation_metrics.csv'),
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
    parser.add_argument('--predictions', default=None, help='Path to prediction JSONL.')
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
        predictions_path=args.predictions,
        splits=args.splits,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
