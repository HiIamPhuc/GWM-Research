import argparse
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.dataset import CollateFN, GWMDataset
from model.model import GWM
from utils.eval import (
    build_bidirectional_hr_map_for_filtering,
    build_entity_loader,
    compute_bidirectional_filtered_ranking_metrics,
    encode_all_entities_as_targets,
)
from utils.relation_mapping import attach_relation_direction_mapping


def get_config(args):
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    if args.data_dir:
        config_dict['data_dir'] = args.data_dir
    if args.output_dir:
        config_dict['output_dir'] = args.output_dir

    class Config:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                setattr(self, key, value)

    return Config(config_dict)


def _format_direction(metrics):
    return {
        'mrr': metrics['MRR'],
        'mr': metrics['MR'],
        'hits1': metrics['Hits@1'],
        'hits3': metrics['Hits@3'],
        'hits10': metrics['Hits@10'],
        'count': metrics['count'],
    }


def evaluate(args):
    config = get_config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open(os.path.join(config.data_dir, 'entity2id.json')) as f:
        config.num_entities = len(json.load(f))
    with open(os.path.join(config.data_dir, 'relation2id.json')) as f:
        config.num_relations = len(json.load(f))
    attach_relation_direction_mapping(config, config.data_dir)

    print("Loading model...")
    model = GWM(config).to(device)
    checkpoint_path = os.path.join(config.output_dir, 'best_checkpoint.pt')
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    print("Encoding all entities as targets...")
    entity_loader = build_entity_loader(
        data_dir=config.data_dir,
        batch_size=config.candidate_batch_size,
        num_workers=4,
    )
    all_entity_embeddings = encode_all_entities_as_targets(
        model=model,
        entity_loader=entity_loader,
        device=device,
    )
    print(f"Encoded {all_entity_embeddings.size(0)} entities.")

    print("Evaluating on test set...")
    test_loader = DataLoader(
        GWMDataset(config.data_dir, split='test'),
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=CollateFN(),
        num_workers=4,
    )
    hr_map = build_bidirectional_hr_map_for_filtering(config.data_dir, splits=['train', 'valid', 'test'])
    metrics = compute_bidirectional_filtered_ranking_metrics(
        model=model,
        data_loader=test_loader,
        all_entity_embeddings=all_entity_embeddings,
        hr_map=hr_map,
        device=device,
        desc="Evaluating",
        save_predictions_path=os.path.join(
            config.output_dir,
            'predictions_test.jsonl',
        ),
    )

    print("\n--- Bidirectional Evaluation Results (test) ---")
    print(f"MRR       : {metrics['MRR']:.4f}")
    print(f"Hits@1    : {metrics['Hits@1']:.4f}")
    print(f"Hits@3    : {metrics['Hits@3']:.4f}")
    print(f"Hits@10   : {metrics['Hits@10']:.4f}")
    for name in ('forward', 'backward'):
        direction = metrics[name]
        print(
            f"{name.title():<9}: MRR {direction['MRR']:.4f} | "
            f"Hits@10 {direction['Hits@10']:.4f} | "
            f"count {direction['count']}"
        )
    print("-------------------------------")

    results = {
        'mrr': metrics['MRR'],
        'hits1': metrics['Hits@1'],
        'hits3': metrics['Hits@3'],
        'hits10': metrics['Hits@10'],
        'forward': _format_direction(metrics['forward']),
        'backward': _format_direction(metrics['backward']),
        'micro': _format_direction(metrics['micro']),
    }
    with open(os.path.join(config.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir')
    evaluate(parser.parse_args())
