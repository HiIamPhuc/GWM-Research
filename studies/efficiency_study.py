"""Measure training artifacts and full-candidate inference efficiency."""

import argparse
import json
import os
import platform
import sys
import time

import torch
from torch.utils.data import DataLoader

from model.dataset import CollateFN, GWMDataset
from studies.ablation_models import build_model
from utils.config import load_config
from utils.eval import build_entity_loader, encode_all_entities_as_targets


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def move_query_batch(batch, device):
    return {
        name: {key: value.to(device) for key, value in batch[name].items()}
        for name in ('h_batch', 'r_batch', 'context_batch')
    }


def benchmark(config, repeats):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config).to(device)
    checkpoint_path = os.path.join(config.output_dir, 'best_checkpoint.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    entity_loader = build_entity_loader(
        config.data_dir,
        config.candidate_batch_size,
        num_workers=2,
    )
    test_loader = DataLoader(
        GWMDataset(config.data_dir, 'test'),
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=CollateFN(),
        num_workers=2,
        pin_memory=device.type == 'cuda',
    )

    synchronize(device)
    start = time.perf_counter()
    candidates = encode_all_entities_as_targets(model, entity_loader, device)
    synchronize(device)
    candidate_seconds = time.perf_counter() - start

    first_batch = move_query_batch(next(iter(test_loader)), device)
    with torch.no_grad():
        query = model(
            first_batch['h_batch'],
            first_batch['r_batch'],
            first_batch['context_batch'],
        )
        _ = query @ candidates.t()
    synchronize(device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    query_count = len(test_loader.dataset) * repeats
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            for cpu_batch in test_loader:
                batch = move_query_batch(cpu_batch, device)
                query = model(
                    batch['h_batch'],
                    batch['r_batch'],
                    batch['context_batch'],
                )
                _ = query @ candidates.t() / model.temperature
    synchronize(device)
    inference_seconds = time.perf_counter() - start

    training_log_path = os.path.join(config.output_dir, 'training_log.json')
    with open(training_log_path, encoding='utf-8') as file:
        training_log = json.load(file)
    train_seconds = [epoch['train_seconds'] for epoch in training_log]

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    results = {
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'software': {
            'python': sys.version.split()[0],
            'pytorch': torch.__version__,
            'cuda': torch.version.cuda,
            'platform': platform.platform(),
        },
        'model_variant': getattr(config, 'model_variant', 'fused'),
        'seed': config.seed,
        'parameters': {
            'total': total_parameters,
            'trainable': trainable_parameters,
        },
        'checkpoint_mb': os.path.getsize(checkpoint_path) / (1024 ** 2),
        'training': {
            'epochs': len(training_log),
            'total_seconds': sum(train_seconds),
            'mean_epoch_seconds': sum(train_seconds) / len(train_seconds),
            'best_validation_mrr': checkpoint['best_mrr'],
        },
        'inference': {
            'entities': candidates.size(0),
            'queries': query_count,
            'repeats': repeats,
            'query_batch_size': config.eval_batch_size,
            'candidate_batch_size': config.candidate_batch_size,
            'candidate_encoding_seconds': candidate_seconds,
            'candidate_matrix_mb': candidates.numel() * candidates.element_size() / (1024 ** 2),
            'full_candidate_seconds': inference_seconds,
            'queries_per_second': query_count / inference_seconds,
            'milliseconds_per_query': inference_seconds * 1000 / query_count,
            'peak_gpu_memory_mb': (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == 'cuda'
                else None
            ),
        },
    }

    output_path = os.path.join(config.output_dir, 'efficiency_results.json')
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    benchmark(
        load_config(args.config, args.data_dir, args.output_dir),
        args.repeats,
    )
