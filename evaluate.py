import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from model.dataset import CollateFN, GWMDataset
from studies.ablation_models import build_model
from utils.config import load_config
from utils.eval import (
    build_entity_loader,
    compute_filtered_ranking_metrics,
    encode_all_entities_as_targets,
    load_hr_map_for_filtering,
)


def metric_record(metrics):
    return {
        'mrr': metrics['MRR'],
        'mr': metrics['MR'],
        'hits1': metrics['Hits@1'],
        'hits3': metrics['Hits@3'],
        'hits10': metrics['Hits@10'],
        'count': metrics['count'],
    }


def evaluate(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = build_model(config).to(device)
    checkpoint = torch.load(
        os.path.join(config.output_dir, 'best_checkpoint.pt'),
        map_location=device,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    entity_loader = build_entity_loader(
        config.data_dir,
        config.candidate_batch_size,
        num_workers=4,
    )
    candidates = encode_all_entities_as_targets(model, entity_loader, device)

    test_loader = DataLoader(
        GWMDataset(config.data_dir, 'test'),
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=CollateFN(),
        num_workers=4,
        pin_memory=device.type == 'cuda',
    )
    filter_map = load_hr_map_for_filtering(
        config.data_dir,
        fallback_splits=['train', 'valid', 'test'],
    )
    metrics = compute_filtered_ranking_metrics(
        model,
        test_loader,
        candidates,
        filter_map,
        device,
        desc='Test',
        save_predictions_path=os.path.join(config.output_dir, 'predictions_test.jsonl'),
        topk=10,
    )

    results = {
        'evaluation_protocol': 'main',
        'split': 'test',
        'main': metric_record(metrics),
    }
    with open(os.path.join(config.output_dir, 'evaluation_results.json'), 'w') as file:
        json.dump(results, file, indent=2)

    print(
        f"MRR {results['main']['mrr']:.4f} | "
        f"H@1 {results['main']['hits1']:.4f} | "
        f"H@3 {results['main']['hits3']:.4f} | "
        f"H@10 {results['main']['hits10']:.4f}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir')
    args = parser.parse_args()
    evaluate(load_config(args.config, args.data_dir, args.output_dir))
