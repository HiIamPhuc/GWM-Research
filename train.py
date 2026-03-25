import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
from pathlib import Path

# Need to set PYTHONPATH or import relatively if structure is respected
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.model import GWM
from model.dataset import GWMDataset, CollateFN

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
    import json
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
    collate_fn = CollateFN(model.tokenizer)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=True # Important for In-Batch Negatives stability
    )
    
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
            drop_last=True
        )
    else:
        valid_loader = None
    
    print("Starting training...")
    best_mrr = 0.0
    
    # Simple JSON Logger
    log_path = os.path.join(config.output_dir, 'training_log.json')
    history = []
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        if valid_loader:
            model.eval()
            val_loss = 0
            
            # Additional Metrics
            hits1 = 0
            hits3 = 0
            hits10 = 0
            mrr = 0
            total_examples = 0
            
            with torch.no_grad():
                for batch in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]"):
                    h_batch = {k: v.to(device) for k, v in batch['h_batch'].items()}
                    r_batch = {k: v.to(device) for k, v in batch['r_batch'].items()}
                    t_batch = {k: v.to(device) for k, v in batch['t_batch'].items()}
                    context_batch = {k: v.to(device) for k, v in batch['context_batch'].items()}
                    
                    query_vector = model(h_batch, r_batch, context_batch)
                    t_fused = model.encode_target(t_batch)
                    
                    # Compute Loss
                    loss, scores = model.compute_loss(query_vector, t_fused)
                    val_loss += loss.item()
                    
                    # Compute Metrics
                    # scores: (B, B)
                    # target_score: diag
                    target_scores = scores.diag() # (B,)
                    
                    # Count how many are greater
                    rankings = (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1
                    
                    hits1 += (rankings <= 1).sum().item()
                    hits3 += (rankings <= 3).sum().item()
                    hits10 += (rankings <= 10).sum().item()
                    mrr += (1.0 / rankings.float()).sum().item()
                    total_examples += rankings.size(0)
            
            avg_val_loss = val_loss / len(valid_loader)
            
            # Compute Averages
            val_mrr = mrr / total_examples
            val_h1 = hits1 / total_examples
            val_h3 = hits3 / total_examples
            val_h10 = hits10 / total_examples
            
            print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f} | MRR: {val_mrr:.4f} | Hits@10: {val_h10:.4f}")
            
            # Log metrics
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_mrr': val_mrr, 
                'val_hits1': val_h1,
                'val_hits3': val_h3,
                'val_hits10': val_h10
            }
            history.append(epoch_log)
            with open(log_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            if val_mrr > best_mrr:
                best_mrr = val_mrr
                torch.save(model.state_dict(), os.path.join(config.output_dir, 'best_checkpoint.pt'))
        else:
             # Log train only
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss
            }
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
