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
    'head_centered_world_state_transformer_'
    'next_state_dot_'
    'masked_reconstruction'
)
TRAINING_OBJECTIVE = (
    'triple_level_full_entity_cross_entropy_with_'
    'masked_state_reconstruction'
)


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
    model = GWM(config).to(device)
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
    training_config = vars(config).copy()
    training_config['model_parameters'] = parameter_count
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
    scheduler = build_scheduler(optimizer, config, len(train_loader))
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
        total_kg_loss = 0.0
        total_state_loss = 0.0
        state_examples = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for batch in progress:
            h_batch = {key: value.to(device) for key, value in batch['h_batch'].items()}
            r_batch = {key: value.to(device) for key, value in batch['r_batch'].items()}
            target_ids = batch['t_batch']['id'].to(device)
            context_batch = {key: value.to(device) for key, value in batch['context_batch'].items()}

            optimizer.zero_grad()
            query = model(
                h_batch,
                r_batch,
                context_batch,
            )
            kg_loss = model.compute_loss(query, target_ids)

            eligible = context_batch['mask'].any(dim=1)
            reconstruct = (
                torch.rand(eligible.size(0), device=device)
                < config.state_reconstruction_ratio
            ) & eligible
            state_loss = kg_loss.new_zeros(())
            selected_count = int(reconstruct.sum().item())
            if selected_count:
                state_h_batch = {
                    'id': h_batch['id'][reconstruct],
                }
                state_context_batch = {
                    key: value[reconstruct]
                    for key, value in context_batch.items()
                }
                reconstructed_heads = model.encode_masked_world_state(
                    state_h_batch,
                    state_context_batch,
                )
                state_loss = model.compute_state_reconstruction_loss(
                    reconstructed_heads,
                    state_h_batch['id'],
                )

            loss = (
                kg_loss
                + config.state_reconstruction_weight * state_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_kg_loss += kg_loss.item()
            total_state_loss += state_loss.item() * selected_count
            state_examples += selected_count
            progress.set_postfix(
                loss=loss.item(),
                state=state_loss.item(),
            )

        sync_device(device)
        train_seconds = time.perf_counter() - epoch_start
        train_loss = total_loss / len(train_loader)
        train_kg_loss = total_kg_loss / len(train_loader)
        train_state_loss = (
            total_state_loss / state_examples
            if state_examples else 0.0
        )
        print(
            f"Epoch {epoch + 1} Train Loss: {train_loss:.4f} | "
            f"KGC: {train_kg_loss:.4f} | "
            f"State: {train_state_loss:.4f} | "
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
            'train_kg_loss': train_kg_loss,
            'train_state_loss': train_state_loss,
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
