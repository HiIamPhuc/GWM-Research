import argparse
import json
import math
import os
import time

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import CollateFN, GWMDataset, TrainTruthIndex
from studies.ablation_models import build_model
from utils.config import load_config
from utils.early_stopping import EarlyStopping
from utils.eval import (
    build_entity_loader,
    compute_filtered_ranking_metrics,
    encode_all_entities_as_targets,
    load_hr_map_for_filtering,
)
from utils.seed import make_torch_generator, make_worker_init_fn, seed_everything


def save_config(config, model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    values = vars(config) | {
        'validation_protocol': 'main',
        'deterministic_training': True,
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'model_parameters': {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
        }
    }
    with open(os.path.join(config.output_dir, 'training_config.json'), 'w') as file:
        json.dump(values, file, indent=2)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_mrr, stopper):
    torch.save(
        {
            'epoch': epoch,
            'best_mrr': best_mrr,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'early_stopping_state': {
                'best_value': stopper.best_value,
                'counter': stopper.counter,
                'should_stop': stopper.should_stop,
            },
        },
        path,
    )


def make_scheduler(optimizer, config, steps_per_epoch):
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)
    min_ratio = config.min_lr / config.learning_rate

    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_ratio + (1 - min_ratio) * cosine

    return LambdaLR(optimizer, schedule)


def move_batch(batch, device):
    return {
        name: {
            key: value.to(device, non_blocking=True)
            for key, value in values.items()
        }
        for name, values in batch.items()
    }


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def train(config):
    os.makedirs(config.output_dir, exist_ok=True)
    seed_everything(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    collate = CollateFN()
    train_dataset = GWMDataset(config.data_dir, 'train')
    valid_dataset = GWMDataset(config.data_dir, 'valid')
    truth_index = TrainTruthIndex(train_dataset.triples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        pin_memory=device.type == 'cuda',
        drop_last=True,
        generator=make_torch_generator(config.seed),
        worker_init_fn=make_worker_init_fn(config.seed),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
        pin_memory=device.type == 'cuda',
        worker_init_fn=make_worker_init_fn(config.seed),
    )
    entity_loader = build_entity_loader(
        config.data_dir,
        config.candidate_batch_size,
        num_workers=2,
    )
    filter_map = load_hr_map_for_filtering(config.data_dir, fallback_splits=['train'])

    model = build_model(config).to(device)
    if model.requires_text_embeddings:
        model.load_text_embeddings(
            os.path.join(config.data_dir, 'entity_text_embeddings.pt'),
            os.path.join(config.data_dir, 'relation_text_embeddings.pt'),
        )
    save_config(config, model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = make_scheduler(optimizer, config, len(train_loader))
    stopper = EarlyStopping(config.early_stopping_patience)
    history = []
    best_mrr = float('-inf')
    training_start = time.perf_counter()

    for epoch in range(config.num_epochs):
        model.train()
        synchronize(device)
        epoch_start = time.perf_counter()
        total_loss = 0.0
        gate_sums = {}
        gate_counts = {}

        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1} [train]')
        for cpu_batch in progress:
            truth_mask = truth_index.build_in_batch_truth_mask(
                cpu_batch['h_batch']['id'],
                cpu_batch['r_batch']['id'],
                cpu_batch['t_batch']['id'],
            ).to(device)
            batch = move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            query = model(batch['h_batch'], batch['r_batch'], batch['context_batch'])
            target = model.encode_target(batch['t_batch'])
            loss, _ = model.compute_loss(query, target, truth_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            loss_value = loss.item()
            total_loss += loss_value
            for name, value in model.pop_gate_stats().items():
                gate_sums[name] = gate_sums.get(name, 0.0) + value
                gate_counts[name] = gate_counts.get(name, 0) + 1
            progress.set_postfix(loss=f'{loss_value:.4f}')

        synchronize(device)
        train_seconds = time.perf_counter() - epoch_start
        train_loss = total_loss / len(train_loader)
        gate_stats = {
            f'train_{name}': gate_sums[name] / gate_counts[name]
            for name in gate_sums
        }

        synchronize(device)
        validation_start = time.perf_counter()
        candidates = encode_all_entities_as_targets(model, entity_loader, device)
        metrics = compute_filtered_ranking_metrics(
            model,
            valid_loader,
            candidates,
            filter_map,
            device,
            desc='Validation',
        )
        synchronize(device)
        validation_seconds = time.perf_counter() - validation_start
        val_mrr = metrics['MRR']
        epoch_log = {
            'epoch': epoch + 1,
            'validation_protocol': 'main',
            'train_loss': train_loss,
            'train_seconds': train_seconds,
            'validation_seconds': validation_seconds,
            'epoch_seconds': train_seconds + validation_seconds,
            'val_mrr': val_mrr,
            'val_mr': metrics['MR'],
            'val_hits1': metrics['Hits@1'],
            'val_hits3': metrics['Hits@3'],
            'val_hits10': metrics['Hits@10'],
            **gate_stats,
        }
        history.append(epoch_log)
        with open(os.path.join(config.output_dir, 'training_log.json'), 'w') as file:
            json.dump(history, file, indent=2)

        print(
            f"Epoch {epoch + 1} | loss {train_loss:.4f} | "
            f"MRR {val_mrr:.4f} | H@10 {metrics['Hits@10']:.4f} | "
            f"train {train_seconds:.1f}s | validation {validation_seconds:.1f}s"
        )

        should_stop = stopper(val_mrr)
        if val_mrr > best_mrr:
            best_mrr = val_mrr
            save_checkpoint(os.path.join(config.output_dir, 'best_checkpoint.pt'), model, optimizer, scheduler, epoch, best_mrr, stopper)
        save_checkpoint(os.path.join(config.output_dir, 'latest_checkpoint.pt'), model, optimizer, scheduler, epoch, best_mrr, stopper)
        if should_stop:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    print(f'Total training time: {time.perf_counter() - training_start:.2f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir')
    args = parser.parse_args()
    train(load_config(args.config, args.data_dir, args.output_dir))
