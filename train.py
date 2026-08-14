import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.dataset import CollateFN, GWMDataset
from model.model import GWM
from utils.early_stopping import EarlyStopping
from utils.eval import (
    build_bidirectional_hr_map_for_filtering,
    build_entity_loader,
    compute_bidirectional_filtered_ranking_metrics,
    encode_all_entities_as_targets,
)
from utils.relation_mapping import attach_relation_direction_mapping
from utils.seed import make_torch_generator, make_worker_init_fn, seed_everything


ARCHITECTURE = (
    'structural_pairre_role_conditioned_'
    'world_memory_transition_decoder'
)
TRAINING_OBJECTIVE = 'triple_level_full_entity_cross_entropy'


def get_config(args):
    with open(args.config, 'r') as f:
        values = yaml.safe_load(f)
    if args.data_dir:
        values['data_dir'] = args.data_dir
    if args.output_dir:
        values['output_dir'] = args.output_dir
    return SimpleNamespace(**values)


def sync_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_mrr, early_stopping):
    torch.save(
        {
            'architecture': ARCHITECTURE,
            'training_objective': TRAINING_OBJECTIVE,
            'epoch': epoch,
            'best_mrr': best_mrr,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'early_stopping_state': {
                'best_value': early_stopping.best_value,
                'counter': early_stopping.counter,
                'should_stop': early_stopping.should_stop,
            },
        },
        path,
    )


def build_scheduler(optimizer, config, steps_per_epoch):
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)
    min_lr_ratio = config.min_lr / config.learning_rate

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def train(args):
    config = get_config(args)
    os.makedirs(config.output_dir, exist_ok=True)

    seed_everything(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading data from {config.data_dir}...")
    train_dataset = GWMDataset(config.data_dir, split='train')
    valid_dataset = GWMDataset(config.data_dir, split='valid')
    config.context_k = train_dataset.context_entity_ids.size(1)
    with open(os.path.join(config.data_dir, 'entity2id.json')) as f:
        config.num_entities = len(json.load(f))
    with open(os.path.join(config.data_dir, 'relation2id.json')) as f:
        config.num_relations = len(json.load(f))
    relation_mapping = attach_relation_direction_mapping(config, config.data_dir)

    print("Initializing model...")
    model = GWM(config)
    model = model.to(device)
    collate_fn = CollateFN()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=device.type == 'cuda',
        drop_last=True,
        generator=make_torch_generator(config.seed),
        worker_init_fn=make_worker_init_fn(config.seed),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=device.type == 'cuda',
        worker_init_fn=make_worker_init_fn(config.seed),
    )
    hr_map = build_bidirectional_hr_map_for_filtering(config.data_dir, splits=['train', 'valid'])
    entity_loader = build_entity_loader(
        config.data_dir,
        batch_size=config.candidate_batch_size,
        num_workers=2,
    )

    print(
        f"Loaded {len(train_dataset)} triples and "
        f"{relation_mapping['num_base_relations']} shared relations."
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    training_config = vars(config).copy()
    training_config['model_parameters'] = {
        'total': parameter_count,
        'trainable': trainable_parameter_count,
        'frozen': parameter_count - trainable_parameter_count,
    }
    training_config['architecture'] = ARCHITECTURE
    training_config['training_objective'] = TRAINING_OBJECTIVE
    training_config['cli_args'] = vars(args)
    with open(os.path.join(config.output_dir, 'training_config.json'), 'w', encoding='utf-8') as f:
        json.dump(training_config, f, indent=2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    )
    scheduler = build_scheduler(
        optimizer,
        config,
        optimizer_steps_per_epoch,
    )
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode='max')

    log_path = os.path.join(config.output_dir, 'training_log.json')
    history = []
    best_mrr = float('-inf')
    train_start = time.perf_counter()
    print("Starting training...")

    for epoch in range(config.num_epochs):
        sync_device(device)
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
        optimizer.zero_grad()
        for step, batch in enumerate(progress):
            h_batch = {key: value.to(device) for key, value in batch['h_batch'].items()}
            r_batch = {key: value.to(device) for key, value in batch['r_batch'].items()}
            target_ids = batch['t_batch']['id'].to(device)
            context_batch = {key: value.to(device) for key, value in batch['context_batch'].items()}

            query = model(
                h_batch,
                r_batch,
                context_batch,
            )
            loss = model.compute_loss(
                query,
                r_batch['id'],
                target_ids,
            )
            accumulation_group_start = (
                step // config.gradient_accumulation_steps
            ) * config.gradient_accumulation_steps
            accumulation_group_size = min(
                config.gradient_accumulation_steps,
                len(train_loader) - accumulation_group_start,
            )
            scaled_loss = loss / accumulation_group_size
            scaled_loss.backward()
            should_step = (
                (step + 1) % config.gradient_accumulation_steps == 0
                or step + 1 == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        sync_device(device)
        train_seconds = time.perf_counter() - epoch_start
        train_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1} Train Loss: {train_loss:.4f} | "
            f"Train Time: {train_seconds:.2f}s"
        )

        model.eval()
        candidates = encode_all_entities_as_targets(model, entity_loader, device)
        metrics = compute_bidirectional_filtered_ranking_metrics(
            model=model,
            data_loader=valid_loader,
            all_entity_embeddings=candidates,
            hr_map=hr_map,
            device=device,
            desc="Validation",
        )
        micro = metrics['micro']
        print(
            f"Epoch {epoch + 1} Val | MRR: {micro['MRR']:.4f} | "
            f"MR: {micro['MR']:.2f} | Hits@1: {micro['Hits@1']:.4f} | "
            f"Hits@3: {micro['Hits@3']:.4f} | "
            f"Hits@10: {micro['Hits@10']:.4f}"
        )
        print(
            "Val Directions | "
            f"Forward MRR: {metrics['forward']['MRR']:.4f} | "
            f"Backward MRR: {metrics['backward']['MRR']:.4f}"
        )

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_mrr': micro['MRR'],
            'val_mr': micro['MR'],
            'val_hits1': micro['Hits@1'],
            'val_hits3': micro['Hits@3'],
            'val_hits10': micro['Hits@10'],
            'val_forward_mrr': metrics['forward']['MRR'],
            'val_backward_mrr': metrics['backward']['MRR'],
        })
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=2)

        is_best = micro['MRR'] > best_mrr
        best_mrr = max(best_mrr, micro['MRR'])
        should_stop = early_stopping(micro['MRR'])
        if is_best:
            save_checkpoint(
                os.path.join(config.output_dir, 'best_checkpoint.pt'),
                model,
                optimizer,
                scheduler,
                epoch,
                best_mrr,
                early_stopping,
            )
        save_checkpoint(
            os.path.join(config.output_dir, 'latest_checkpoint.pt'),
            model,
            optimizer,
            scheduler,
            epoch,
            best_mrr,
            early_stopping,
        )
        if should_stop:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    sync_device(device)
    history.append({
        'event': 'training_complete',
        'total_train_seconds': time.perf_counter() - train_start,
        'epochs_completed': len(history),
    })
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir')
    train(parser.parse_args())
