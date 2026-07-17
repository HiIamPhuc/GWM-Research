import os
import math
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
import json
from torch.optim.lr_scheduler import LambdaLR

# Need to set PYTHONPATH or import relatively if structure is respected
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.dataset import CollateFN, GWMDataset
from studies.ablation_models import build_model
from utils.seed import make_torch_generator, make_worker_init_fn, seed_everything
from utils.eval import (
    build_bidirectional_hr_map_for_filtering,
    build_entity_loader,
    compute_bidirectional_filtered_ranking_metrics,
    encode_all_entities_as_targets,
    load_inverse_relation_ids,
)
from utils.early_stopping import EarlyStopping

def _to_serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    return str(value)


def _get_model_parameter_info(model):
    total_params = 0
    trainable_params = 0
    param_details = []

    for name, param in model.named_parameters():
        numel = int(param.numel())
        total_params += numel
        if param.requires_grad:
            trainable_params += numel

        param_details.append({
            'name': name,
            'shape': list(param.shape),
            'numel': numel,
            'requires_grad': bool(param.requires_grad),
            'dtype': str(param.dtype).replace('torch.', ''),
        })

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'parameters': param_details,
    }


def _update_metric_sums(sums, counts, values):
    for key, value in values.items():
        if value is None:
            continue
        sums[key] = sums.get(key, 0.0) + float(value)
        counts[key] = counts.get(key, 0) + 1


def _average_metric_sums(sums, counts, prefix=''):
    return {
        f'{prefix}{key}': sums[key] / counts[key]
        for key in sorted(sums)
        if counts.get(key, 0) > 0
    }


def save_training_config(config, output_dir, args=None, model=None):
    """Persist the effective training config used for this run."""
    config_dict = {k: _to_serializable(v) for k, v in vars(config).items()}
    if args is not None:
        config_dict['cli_args'] = {k: _to_serializable(v) for k, v in vars(args).items()}
    if model is not None:
        config_dict['model_parameters'] = _get_model_parameter_info(model)

    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2)


def get_config(args):
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Override with args
    if args.data_dir: config_dict['data_dir'] = args.data_dir
    if args.output_dir: config_dict['output_dir'] = args.output_dir
    
    # Convert to SimpleNamespace (object with attributes)
    class Config:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)
    
    return Config(config_dict)

def _sync_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_mrr, early_stopping):
    torch.save(
        {
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

def train(args):
    # Load Config
    config = get_config(args)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    seed = int(getattr(config, 'seed', 42))
    seed_everything(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Dataset
    print(f"Loading data from {config.data_dir}...")
    train_dataset = GWMDataset(config.data_dir, split='train')
    print(f"Loaded {len(train_dataset)} training triples.")
    
    # Infer input dimensions from dataset
    # e.g., number of entities/relations for embedding layers
    # Load vocabulary sizes
    with open(os.path.join(config.data_dir, 'entity2id.json')) as f:
        num_ent = len(json.load(f))
    with open(os.path.join(config.data_dir, 'relation2id.json')) as f:
        num_rel = len(json.load(f))
        
    config.num_entities = num_ent
    config.num_relations = num_rel
    config.inverse_relation_ids = load_inverse_relation_ids(config.data_dir)
    
    # Init Model
    print("Initializing model...")
    model = build_model(config).to(device)
    decoder_name = getattr(model, 'decoder_name', 'dot')
    config.training_objective = 'full_entity_cross_entropy'

    # Collater
    collate_fn = CollateFN()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
        generator=make_torch_generator(seed),
        worker_init_fn=make_worker_init_fn(seed),
    )

    if getattr(model, 'requires_text_embeddings', True):
        entity_emb_path = os.path.join(config.data_dir, 'entity_text_embeddings.pt')
        relation_emb_path = os.path.join(config.data_dir, 'relation_text_embeddings.pt')
        if not os.path.exists(entity_emb_path) or not os.path.exists(relation_emb_path):
            raise FileNotFoundError(
                "Missing precomputed text embedding cache files. "
                "Expected entity_text_embeddings.pt and relation_text_embeddings.pt in data_dir."
            )

        model.load_text_embeddings(
            entity_source=entity_emb_path,
            relation_source=relation_emb_path,
            freeze=True,
        )
        print("Loaded text embeddings into text embedding tables...")
    else:
        print("Skipping precomputed text embeddings for structure-only ablation.")
    
    # Save effective config (including inferred dimensions, CLI overrides, and model params).
    save_training_config(config, config.output_dir, args=args, model=model)

    base_lr = float(config.learning_rate)
    weight_decay = float(getattr(config, 'weight_decay', 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
    )

    total_steps = max(1, config.num_epochs * len(train_loader))
    warmup_ratio = float(getattr(config, 'warmup_ratio', 0.0))
    warmup_steps = min(int(total_steps * warmup_ratio), total_steps)
    min_lr = float(getattr(config, 'min_lr', 0.0))
    min_lr_ratio = 0.0 if base_lr <= 0 else max(min_lr / base_lr, 0.0)

    def lr_lambda(step_index):
        if total_steps <= 1:
            return 1.0
        if warmup_steps > 0 and step_index < warmup_steps:
            return float(step_index + 1) / float(max(1, warmup_steps))

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(max((step_index - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    grad_clip_norm = float(getattr(config, 'grad_clip_norm', 1.0))
    print(
        f"Training objective: Full-Entity Cross-Entropy | "
        f"Decoder: {decoder_name} | Optimizer: AdamW | Scheduler: cosine"
    )
    
    # Validation Loader
    if os.path.exists(os.path.join(config.data_dir, 'valid_triples.pt')):
        print("Loading validation data...")
        valid_dataset = GWMDataset(config.data_dir, split='valid')
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=(device.type == 'cuda'),
            drop_last=False,
            worker_init_fn=make_worker_init_fn(seed),
        )
    else:
        valid_loader = None

    # Build filtered-ranking structures for validation. Bidirectional
    # validation constructs inverse queries in memory; preprocessing remains
    # unchanged.
    hr_map = None
    all_entity_embeddings = None
    entity_loader = None
    if valid_loader is not None:
        hr_map = build_bidirectional_hr_map_for_filtering(
            config.data_dir,
            splits=['train', 'valid'],
        )

        candidate_batch_size = int(getattr(config, 'candidate_batch_size', min(int(config.batch_size), 256)))
        entity_loader = build_entity_loader(
            data_dir=config.data_dir,
            batch_size=candidate_batch_size,
            num_workers=2,
        )
    
    print("Starting training...")
    train_start_time = time.perf_counter()
    best_mrr = float('-inf')
    
    early_stopping = EarlyStopping(
        patience=getattr(config, 'early_stopping_patience', getattr(config, 'early_stopping', 10)),
        mode='max'  # Maximize MRR
    )

    # Simple JSON Logger
    log_path = os.path.join(config.output_dir, 'training_log.json')
    history = []
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.perf_counter()
        _sync_device(device)
        model.train()
        total_loss = 0
        gate_stat_sums = {}
        gate_stat_counts = {}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch in pbar:
            # Move batch to device (handle nested dicts)
            h_batch = {k: v.to(device) for k, v in batch['h_batch'].items()}
            r_batch = {k: v.to(device) for k, v in batch['r_batch'].items()}
            t_batch = {k: v.to(device) for k, v in batch['t_batch'].items()}
            context_batch = {k: v.to(device) for k, v in batch['context_batch'].items()}

            optimizer.zero_grad()

            scores = model.score_all_entities(
                h_batch,
                r_batch,
                context_batch,
            )
            loss = model.compute_loss(scores, t_batch['id'])
            gate_stats = model.pop_gate_stats()

            if not torch.isfinite(loss):
                print("Warning: non-finite loss detected; skipping batch to avoid corrupting model weights.")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            _update_metric_sums(gate_stat_sums, gate_stat_counts, gate_stats)
            pbar.set_postfix({'loss': loss.item()})

        _sync_device(device)
        epoch_train_seconds = time.perf_counter() - epoch_start_time
            
        avg_train_loss = total_loss / len(train_loader)
        avg_gate_stats = _average_metric_sums(
            gate_stat_sums,
            gate_stat_counts,
            prefix='train_',
        )

        print(
            f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} | "
            f"Objective: Full-Entity Cross-Entropy | "
            f"Decoder: {decoder_name} | "
            f"Candidates/Query: {config.num_entities} | "
            f"Train Time: {epoch_train_seconds:.2f}s"
        )
        
        # Validation
        eval_every = getattr(config, 'eval_every', 1)
        if valid_loader and (epoch + 1) % eval_every == 0:
            model.eval()

            all_entity_embeddings = encode_all_entities_as_targets(
                model=model,
                entity_loader=entity_loader,
                device=device,
            )

            directional_val_metrics = compute_bidirectional_filtered_ranking_metrics(
                model=model,
                data_loader=valid_loader,
                all_entity_embeddings=all_entity_embeddings,
                hr_map=hr_map,
                device=device,
                desc="Validation",
            )
            val_metrics = directional_val_metrics['micro']

            val_mrr = val_metrics['MRR']
            val_h1 = val_metrics['Hits@1']
            val_h3 = val_metrics['Hits@3']
            val_h10 = val_metrics['Hits@10']
            val_mr = val_metrics['MR']
            
            print(
                f"Epoch {epoch+1} Val | "
                f"MRR: {val_mrr:.4f} | MR: {val_mr:.2f} | "
                f"Hits@1: {val_h1:.4f} | Hits@3: {val_h3:.4f} | Hits@10: {val_h10:.4f}"
            )
            print(
                f"Val Directions | "
                f"Forward MRR: {directional_val_metrics['forward']['MRR']:.4f} | "
                f"Backward MRR: {directional_val_metrics['backward']['MRR']:.4f}"
            )
            
            # Log metrics
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_mrr': val_mrr, 
                'val_mr': val_mr,
                'val_hits1': val_h1,
                'val_hits3': val_h3,
                'val_hits10': val_h10,
                'val_forward_mrr': directional_val_metrics['forward']['MRR'],
                'val_backward_mrr': directional_val_metrics['backward']['MRR'],
            }
            epoch_log.update(avg_gate_stats)
            history.append(epoch_log)
            with open(log_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            is_best = val_mrr > best_mrr
            if is_best:
                best_mrr = val_mrr
            
            # Check early stopping
            should_stop = early_stopping(val_mrr)
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
            if should_stop:
                print(f"\n✓ Early stopping triggered at epoch {epoch + 1}")
                print(f"  Best MRR: {early_stopping.best_value:.4f}")
                print(f"  No improvement for {early_stopping.patience} epochs")
        else:
            should_stop = False
             # Log train only
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
            }
            epoch_log.update(avg_gate_stats)
            history.append(epoch_log)
            with open(log_path, 'w') as f:
                  json.dump(history, f, indent=2)
        
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
            break

    _sync_device(device)
    total_train_seconds = time.perf_counter() - train_start_time
    print(f"Total training time: {total_train_seconds:.2f}s")
    history.append({
        'event': 'training_complete',
        'total_train_seconds': total_train_seconds,
        'epochs_completed': len(history),
    })
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    
    args = parser.parse_args()
    train(args)
