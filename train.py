import os
import math
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

from model.model import GWM
from model.dataset import GWMDataset, CollateFN
from utils.eval import (
    build_entity_loader,
    compute_filtered_ranking_metrics,
    encode_all_entities_as_targets,
    load_hr_map_for_filtering,
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


def save_training_config(config, output_dir, args=None):
    """Persist the effective training config used for this run."""
    config_dict = {k: _to_serializable(v) for k, v in vars(config).items()}
    if args is not None:
        config_dict['cli_args'] = {k: _to_serializable(v) for k, v in vars(args).items()}

    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2)

    print(f"Saved training config to: {config_path}")


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

def train(args):
    # Load Config
    config = get_config(args)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

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
    
    # Save effective config (including inferred dimensions and CLI overrides).
    save_training_config(config, config.output_dir, args=args)
    
    # Init Model
    print("Initializing model...")
    model = GWM(config).to(device)
    
    # Collater
    collate_fn = CollateFN()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(device.type == 'cuda'),
        drop_last=True # Important for In-Batch Negatives stability
    )

    entity_emb_path = os.path.join(config.data_dir, 'entity_text_embeddings.pt')
    relation_emb_path = os.path.join(config.data_dir, 'relation_text_embeddings.pt')
    if not os.path.exists(entity_emb_path) or not os.path.exists(relation_emb_path):
        raise FileNotFoundError(
            "Missing precomputed text embedding cache files. "
            "Expected entity_text_embeddings.pt and relation_text_embeddings.pt in data_dir."
        )

    print("Loading precomputed text embeddings into text embedding tables...")
    model.load_embeddings(
        entity_source=entity_emb_path,
        relation_source=relation_emb_path,
        kind='text',
        freeze=True,
    )
    print("Text embeddings ready. Training uses ID-only batches with context IDs.")

    # Load Structural Prior Caches (if specified in config)
    structural_entity_source = os.path.join(config.data_dir, 'structural_entities.pt')
    structural_relation_source = os.path.join(config.data_dir, 'structural_relations.pt')
    if structural_entity_source and structural_relation_source:
        print("Loading precomputed structural priors (e.g. RotatE)...")
        if os.path.exists(structural_entity_source) and os.path.exists(structural_relation_source):
            model.load_embeddings(
                entity_source=structural_entity_source,
                relation_source=structural_relation_source,
                kind='structural',
                freeze=True,
            )
        else:
            print(f"Warning: Structural priors not found at {structural_entity_source} or {structural_relation_source}")
    
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
            data_dir=config.data_dir,
            batch_size=candidate_batch_size,
            num_workers=2,
        )
    
    print("Starting training...")
    best_mrr = 0.0
    
    early_stopping = EarlyStopping(
        patience=getattr(config, 'early_stopping_patience', getattr(config, 'early_stopping', 10)),
        mode='max'  # Maximize MRR
    )

    # Simple JSON Logger
    log_path = os.path.join(config.output_dir, 'training_log.json')
    history = []
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        total_sigreg = 0.0
        sigreg_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch in pbar:
            # Move batch to device (handle nested dicts)
            h_batch = {k: v.to(device) for k, v in batch['h_batch'].items()}
            r_batch = {k: v.to(device) for k, v in batch['r_batch'].items()}
            t_batch = {k: v.to(device) for k, v in batch['t_batch'].items()}
            context_batch = {k: v.to(device) for k, v in batch['context_batch'].items()}

            # Forward: Query Vector (from head, relation, context)
            query_vector = model(h_batch, r_batch, context_batch)
            
            # Forward: Target Vector (Symmetric Fused Tail)
            t_fused = model.encode_target(t_batch)

            optimizer.zero_grad()

            loss_text, loss_struct, _ = model.compute_loss_components(
                query_vector,
                t_fused,
                relation_ids=r_batch['id'],
            )
            main_loss = loss_text + loss_struct
            aux_losses = model.compute_auxiliary_losses()
            if aux_losses is not None:
                main_loss = main_loss + model.multi_step_weight * aux_losses['total_aux']
            sigreg_loss = model.compute_sigreg_loss(query_vector)
            if sigreg_loss is None:
                loss = main_loss
            else:
                loss = main_loss + model.sigreg_weight * sigreg_loss
                total_sigreg += sigreg_loss.item()
                sigreg_batches += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_loss / len(train_loader)
        avg_sigreg = None
        if sigreg_batches > 0:
            avg_sigreg = total_sigreg / sigreg_batches

        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        if avg_sigreg is not None:
            print(f"Epoch {epoch+1} Train SIGReg: {avg_sigreg:.6f}")
        
        # Validation
        eval_every = getattr(config, 'eval_every', 1)
        if valid_loader and (epoch + 1) % eval_every == 0:
            model.eval()

            all_entity_embeddings = encode_all_entities_as_targets(
                model=model,
                entity_loader=entity_loader,
                device=device,
            )

            save_eval_predictions = bool(getattr(config, 'save_eval_predictions', False))
            eval_topk = int(getattr(config, 'eval_topk', 50))
            predictions_dir = getattr(config, 'eval_predictions_dir', None)
            predictions_path = None
            if save_eval_predictions:
                if predictions_dir is None:
                    predictions_dir = os.path.join(config.output_dir, 'predictions')
                predictions_path = os.path.join(predictions_dir, f'val_epoch_{epoch + 1}.jsonl')

            val_metrics = compute_filtered_ranking_metrics(
                model=model,
                data_loader=valid_loader,
                all_entity_embeddings=all_entity_embeddings,
                hr_map=hr_map,
                device=device,
                desc="Filtered Validation",
                save_predictions_path=predictions_path,
                topk=eval_topk,
            )

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
            if avg_sigreg is not None:
                epoch_log['train_sigreg'] = avg_sigreg
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
            if avg_sigreg is not None:
                epoch_log['train_sigreg'] = avg_sigreg
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
