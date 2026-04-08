import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
import json

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

    cache_device = getattr(config, 'text_cache_device', 'cpu')
    print(f"Loading precomputed text embedding cache to {cache_device}...")
    model.load_precomputed_text_embedding_cache(
        entity_source=entity_emb_path,
        relation_source=relation_emb_path,
        cache_device=cache_device,
    )
    print("Text cache ready. Training uses ID-only batches with context IDs.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate))
    
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
            context_edges = batch.get('context_edges', None)
            if context_edges is not None:
                context_edges = context_edges.to(device)
                query_vector = model(h_batch, r_batch, context_batch, context_edges)
            else:
                query_vector = model(h_batch, r_batch, context_batch)
            
            # Forward: Target Vector (Symmetric Fused Tail)
            t_fused = model.encode_target(t_batch)
            
            # Loss: In-Batch Negatives
            loss, _ = model.compute_loss(query_vector, t_fused)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
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
            )

            val_metrics = compute_filtered_ranking_metrics(
                model=model,
                data_loader=valid_loader,
                all_entity_embeddings=all_entity_embeddings,
                hr_map=hr_map,
                device=device,
                desc="Validation",
            )

            val_mrr = val_metrics['MRR']
            val_h1 = val_metrics['Hits@1']
            val_h3 = val_metrics['Hits@3']
            val_h10 = val_metrics['Hits@10']
            val_mr = val_metrics['MR']
            val_alpha = model.get_alpha_mean(reset=True) if hasattr(model, 'get_alpha_mean') else None
            
            metrics = (
                f"Epoch {epoch+1} Val (Filtered) | "
                f"MRR: {val_mrr:.4f} | MR: {val_mr:.2f} | "
                f"Hits@1: {val_h1:.4f} | Hits@3: {val_h3:.4f} | Hits@10: {val_h10:.4f} | "
            )
            if val_alpha is not None:
                metrics += f"Val Alpha (text weight): {val_alpha:.4f}"
            
            print(metrics)

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
