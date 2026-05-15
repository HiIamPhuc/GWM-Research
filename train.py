import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
import json
from transformers import get_linear_schedule_with_warmup

# Need to set PYTHONPATH or import relatively if structure is respected
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.model import GWM
from model.dataset import GWMDataset, CollateFN
from utils.eval import (
    build_entity_loader,
    compute_filtered_ranking_metrics,
    encode_all_entities_as_targets,
    load_hr_map_for_filtering,
)
from utils.early_stopping import EarlyStopping


def _format_param_count(count):
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.2f}K"
    return str(count)


def print_model_parameter_info(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen_params = total_params - trainable_params

    module_stats = {}
    for name, param in model.named_parameters():
        module_name = name.split('.')[0]
        stats = module_stats.setdefault(module_name, {'total': 0, 'trainable': 0})
        count = param.numel()
        stats['total'] += count
        if param.requires_grad:
            stats['trainable'] += count

    print("Model parameter summary:")
    print(f"  Total params     : {_format_param_count(total_params)} ({total_params:,})")
    print(f"  Trainable params : {_format_param_count(trainable_params)} ({trainable_params:,})")
    print(f"  Frozen params    : {_format_param_count(frozen_params)} ({frozen_params:,})")

    print("  By top-level module:")
    for module_name in sorted(module_stats.keys()):
        module_total = module_stats[module_name]['total']
        module_trainable = module_stats[module_name]['trainable']
        module_frozen = module_total - module_trainable
        print(
            f"    - {module_name:<22} "
            f"total={_format_param_count(module_total):>8} ({module_total:,}) | "
            f"trainable={_format_param_count(module_trainable):>8} ({module_trainable:,}) | "
            f"frozen={_format_param_count(module_frozen):>8} ({module_frozen:,})"
        )

def get_config(args):
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Override with args
    if args.data_dir:
        config_dict['data_dir'] = args.data_dir
    if args.output_dir:
        config_dict['output_dir'] = args.output_dir

    # Convert to SimpleNamespace (object with attributes)
    class Config:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)

    return Config(config_dict), config_dict


def save_config_snapshot(config_dict, output_dir, source_path=None):
    if source_path is not None:
        config_dict = dict(config_dict)
        config_dict['config_source'] = source_path
    yaml_path = os.path.join(output_dir, 'config.yaml')
    json_path = os.path.join(output_dir, 'config.json')
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)
    with open(json_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def train(args):
    # Load Config
    config, config_dict = get_config(args)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    save_config_snapshot(config_dict, config.output_dir, source_path=args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Dataset
    print(f"Loading data from {config.data_dir}...")
    train_dataset = GWMDataset(config.data_dir, split='train')
    
    # Infer input dimensions from dataset
    # e.g., number of entities/relations for embedding layers
    # Load vocabulary sizes
    with open(os.path.join(config.data_dir, 'entity2id.json')) as f:
        num_ent = len(json.load(f))
    with open(os.path.join(config.data_dir, 'relation2id.json')) as f:
        num_rel = len(json.load(f))
        
    config.num_entities = num_ent
    config.num_relations = num_rel
    
    # Init Model
    print("Initializing model...")
    model = GWM(config).to(device)

    structural_entity_source = os.path.join(config.data_dir, 'structural_entities.pt')
    structural_relation_source = os.path.join(config.data_dir, 'structural_relations.pt')

    if os.path.exists(structural_entity_source) and os.path.exists(structural_relation_source):
        freeze_structural = bool(getattr(config, 'freeze_structural_priors', False))
        print("Loading precomputed structural priors...")
        model.load_precomputed_structural_cache(
            entity_source=structural_entity_source,
            relation_source=structural_relation_source,
            freeze=freeze_structural,
        )
    else:
        print(
            "Warning: Structural priors not found; using randomly initialized structural embeddings."
        )

    print_model_parameter_info(model)
    
    # Collater
    collate_fn = CollateFN(model.tokenizer)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=True # Important for In-Batch Negatives stability
    )

    # Use separate learning rates for LoRA/text encoder, structural embeddings, and the rest.
    base_lr = float(config.learning_rate)
    text_encoder_lr = float(getattr(config, 'text_encoder_lr', base_lr * 0.1))
    structural_lr = float(getattr(config, 'structural_lr', base_lr * 0.1))
    weight_decay = float(getattr(config, 'weight_decay', 0.01))
    max_grad_norm = float(getattr(config, 'max_grad_norm', 1.0))

    structural_params = [model.entity_embeddings.weight, model.relation_embeddings.weight]
    structural_param_ids = {id(p) for p in structural_params}

    text_encoder_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in structural_param_ids:
            continue
        if name.startswith('text_encoder.'):
            text_encoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if text_encoder_params:
        param_groups.append({
            'params': text_encoder_params,
            'lr': text_encoder_lr,
            'weight_decay': weight_decay,
            'name': 'text_encoder',
        })
    if structural_params:
        param_groups.append({
            'params': structural_params,
            'lr': structural_lr,
            'weight_decay': weight_decay,
            'name': 'structural',
        })
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'base',
        })

    optimizer = torch.optim.AdamW(param_groups)

    text_encoder_param_count = sum(param.numel() for param in text_encoder_params)
    structural_param_count = sum(param.numel() for param in structural_params)
    other_param_count = sum(param.numel() for param in other_params)
    print(
        "Optimizer parameter groups: "
        f"text_encoder={_format_param_count(text_encoder_param_count)} ({text_encoder_param_count:,}), "
        f"structural={_format_param_count(structural_param_count)} ({structural_param_count:,}), "
        f"others={_format_param_count(other_param_count)} ({other_param_count:,})"
    )

    total_training_steps = max(1, len(train_loader) * int(config.num_epochs))
    warmup_steps = getattr(config, 'warmup_steps', None)
    if warmup_steps is None:
        warmup_ratio = float(getattr(config, 'warmup_ratio', 0.1))
        warmup_steps = int(total_training_steps * warmup_ratio)
    warmup_steps = max(0, min(int(warmup_steps), total_training_steps - 1))

    use_scheduler = bool(getattr(config, 'use_scheduler', True))
    scheduler = None
    if use_scheduler:
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
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
            drop_last=False
        )
    else:
        valid_loader = None

    # Build filtered-ranking structures for standard validation
    hr_map = None
    all_entity_embeddings = None
    entity_loader = None
    if valid_loader is not None:
        hr_map = load_hr_map_for_filtering(
            config.data_dir,
            preferred_ground_truth_file='ground_truth_train.json',
            fallback_splits=['train']
        )

        candidate_batch_size = int(getattr(config, 'candidate_batch_size', min(int(config.batch_size), 256)))
        entity_loader = build_entity_loader(
            model=model,
            data_dir=config.data_dir,
            batch_size=candidate_batch_size,
            num_workers=2,
            max_length=getattr(config, 'max_length', 512),
        )
    
    print("Starting training...")
    best_mrr = 0.0
    
    early_stopping = EarlyStopping(
        patience=getattr(config, 'early_stopping_patience', 10),
        mode='max'  # Maximize MRR
    )
    
    # Simple JSON Logger
    log_path = os.path.join(config.output_dir, 'training_log.json')
    history = []
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        if hasattr(model, 'reset_alpha_stats'):
            model.reset_alpha_stats()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
        for batch in pbar:
            # Move batch to device (handle nested dicts)
            h_batch = {k: v.to(device) for k, v in batch['h_batch'].items()}
            r_batch = {k: v.to(device) for k, v in batch['r_batch'].items()}
            t_batch = {k: v.to(device) for k, v in batch['t_batch'].items()}
            context_batch = {k: v.to(device) for k, v in batch['context_batch'].items()}
            
            optimizer.zero_grad()
            
            # Forward: Query Vector (from head, relation, context)
            query_vector = model(h_batch, r_batch, context_batch)
            
            # Forward: Target Vector (Symmetric Fused Tail)
            t_fused = model.encode_target(t_batch)
            
            # Loss: In-Batch Negatives
            loss, _ = model.compute_loss(query_vector, t_fused)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            lr_report = {}
            for group in optimizer.param_groups:
                name = group.get('name', 'group')
                lr_report[name] = f"{group['lr']:.2e}"

            pbar.set_postfix({
                'loss': loss.item(),
                **lr_report,
            })
            
        avg_train_loss = total_loss / len(train_loader)
        train_alpha = model.get_alpha_mean(reset=True) if hasattr(model, 'get_alpha_mean') else None
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        if train_alpha is not None:
            print(f"Epoch {epoch+1} Train Alpha (text weight): {train_alpha:.4f}")
        
        # Validation
        eval_every = getattr(config, 'eval_every', 1)
        if valid_loader and (epoch + 1) % eval_every == 0:
            model.eval()

            if hasattr(model, 'reset_alpha_stats'):
                model.reset_alpha_stats()

            all_entity_embeddings = encode_all_entities_as_targets(
                model=model,
                entity_loader=entity_loader,
                device=device,
                desc="Encoding Validation Candidates",
            )

            val_metrics = compute_filtered_ranking_metrics(
                model=model,
                data_loader=valid_loader,
                all_entity_embeddings=all_entity_embeddings,
                hr_map=hr_map,
                device=device,
                desc="Filtered Validation",
            )

            val_mrr = val_metrics['MRR']
            val_h1 = val_metrics['Hits@1']
            val_h3 = val_metrics['Hits@3']
            val_h10 = val_metrics['Hits@10']
            val_mr = val_metrics['MR']
            val_alpha = model.get_alpha_mean(reset=True) if hasattr(model, 'get_alpha_mean') else None
            
            print(
                f"Epoch {epoch+1} Val (Filtered) | "
                f"MRR: {val_mrr:.4f} | MR: {val_mr:.2f} | "
                f"Hits@1: {val_h1:.4f} | Hits@3: {val_h3:.4f} | Hits@10: {val_h10:.4f}"
            )
            if val_alpha is not None:
                print(f"Epoch {epoch+1} Val Alpha (text weight): {val_alpha:.4f}")
            
            # Log metrics
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_mrr': val_mrr, 
                'val_mr': val_mr,
                'val_hits1': val_h1,
                'val_hits3': val_h3,
                'val_hits10': val_h10
            }
            if train_alpha is not None:
                epoch_log['train_alpha'] = train_alpha
            if val_alpha is not None:
                epoch_log['val_alpha'] = val_alpha
            history.append(epoch_log)
            with open(log_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            if val_mrr > best_mrr:
                best_mrr = val_mrr
                torch.save(model.state_dict(), os.path.join(config.output_dir, 'best_checkpoint.pt'))
            
            # Check early stopping
            if early_stopping(val_mrr):
                print(f"\n✓ Early stopping triggered at epoch {epoch + 1}")
                print(f"  Best MRR: {early_stopping.best_value:.4f}")
                print(f"  No improvement for {early_stopping.patience} epochs")
                break
        else:
             # Log train only
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss
            }
            if train_alpha is not None:
                epoch_log['train_alpha'] = train_alpha
            history.append(epoch_log)
            with open(log_path, 'w') as f:
                  json.dump(history, f, indent=2)
        
        # Save Checkpoint
        torch.save(model.state_dict(), os.path.join(config.output_dir, 'latest_checkpoint.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    
    args = parser.parse_args()
    train(args)
