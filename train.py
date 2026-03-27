import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import yaml
import json
from pathlib import Path

# Need to set PYTHONPATH or import relatively if structure is respected
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.model import GWM
from model.dataset import GWMDataset, CollateFN


class EntityDataset(Dataset):
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, 'entity_text.json'), 'r') as f:
            self.entity_text = json.load(f)

        with open(os.path.join(data_dir, 'entity2id.json'), 'r') as f:
            self.entity2id = json.load(f)

        self.num_entities = len(self.entity2id)
        self.texts = [self.entity_text.get(str(i), "") for i in range(self.num_entities)]

    def __len__(self):
        return self.num_entities

    def __getitem__(self, idx):
        return {
            'id': idx,
            'text': self.texts[idx]
        }


def load_all_triples(data_dir):
    all_triples = set()
    for split in ['train', 'valid', 'test']:
        path = os.path.join(data_dir, f'{split}_triples.pt')
        if os.path.exists(path):
            triples = torch.load(path)
            for h, r, t in triples:
                all_triples.add((h.item(), r.item(), t.item()))
    return all_triples


def compute_filtered_ranking_metrics(model, data_loader, all_entity_embeddings, hr_map, device):
    hits1, hits3, hits10, mrr, mr = 0, 0, 0, 0.0, 0.0
    total = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Filtered Validation"):
            h_batch = {k: v.to(device) for k, v in batch['h_batch'].items()}
            r_batch = {k: v.to(device) for k, v in batch['r_batch'].items()}
            context_batch = {k: v.to(device) for k, v in batch['context_batch'].items()}

            t_ids = batch['t_batch']['id'].to(device)
            h_ids = batch['h_batch']['id'].cpu().numpy()
            r_ids = batch['r_batch']['id'].cpu().numpy()

            query_vector = model(h_batch, r_batch, context_batch)
            scores = torch.mm(query_vector, all_entity_embeddings.t())

            # Filter out other true tails for (h, r)
            for i in range(scores.size(0)):
                h_id = h_ids[i]
                r_id = r_ids[i]
                true_t = t_ids[i].item()

                filter_mask_indices = list(hr_map.get((h_id, r_id), []))
                if true_t in filter_mask_indices:
                    filter_mask_indices.remove(true_t)

                if filter_mask_indices:
                    scores[i, filter_mask_indices] = -float('inf')

            target_scores = scores.gather(1, t_ids.unsqueeze(1))
            ranks = (scores > target_scores).sum(dim=1) + 1

            hits1 += (ranks <= 1).sum().item()
            hits3 += (ranks <= 3).sum().item()
            hits10 += (ranks <= 10).sum().item()
            mrr += (1.0 / ranks.float()).sum().item()
            mr += ranks.float().sum().item()
            total += ranks.size(0)

    return {
        'MRR': mrr / total,
        'MR': mr / total,
        'Hits@1': hits1 / total,
        'Hits@3': hits3 / total,
        'Hits@10': hits10 / total
    }

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

    if not config.finetune_text_encoder:
        print("Precomputing frozen text embeddings (one-time cache)...")
        cache_batch_size = int(getattr(config, 'text_cache_batch_size', 128))
        model.build_text_embedding_cache(
            entity_text_map=train_dataset.entity_text,
            relation_text_map=train_dataset.relation_text,
            device=device,
            batch_size=cache_batch_size,
            max_entity_length=512,
            max_relation_length=128
        )
        print("Frozen text cache ready. Training will reuse cached text embeddings.")
    
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
            drop_last=False
        )
    else:
        valid_loader = None

    # Build filtered-ranking structures for standard validation
    hr_map = None
    all_entity_embeddings = None
    entity_loader = None
    if valid_loader is not None:
        all_triples = load_all_triples(config.data_dir)
        hr_map = {}
        for h, r, t in all_triples:
            if (h, r) not in hr_map:
                hr_map[(h, r)] = set()
            hr_map[(h, r)].add(t)

        entity_dataset = EntityDataset(config.data_dir)
        candidate_batch_size = int(getattr(config, 'candidate_batch_size', min(int(config.batch_size), 256)))

        def entity_collate(batch):
            ids = [x['id'] for x in batch]

            # In frozen-text mode, encode_target can use cached text by ID only.
            if model.use_text_cache and not config.finetune_text_encoder:
                return {'id': torch.tensor(ids)}

            texts = [x['text'] for x in batch]
            inputs = model.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
            return {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'id': torch.tensor(ids)
            }

        entity_loader = DataLoader(
            entity_dataset,
            batch_size=candidate_batch_size,
            shuffle=False,
            collate_fn=entity_collate,
            num_workers=2
        )

        # If text encoder is frozen, candidate text embeddings are static.
        if not config.finetune_text_encoder:
            print("Encoding all entities once for filtered validation...")
            all_chunks = []
            model.eval()
            with torch.no_grad():
                for batch in tqdm(entity_loader, desc="Encoding Validation Candidates"):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    all_chunks.append(model.encode_target(batch).cpu())
            all_entity_embeddings = torch.cat(all_chunks, dim=0).to(device)
    
    print("Starting training...")
    best_mrr = 0.0
    
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

            # In finetuning mode, candidate embeddings change every epoch.
            if config.finetune_text_encoder:
                all_chunks = []
                with torch.no_grad():
                    for batch in tqdm(entity_loader, desc="Encoding Validation Candidates"):
                        batch = {k: v.to(device) for k, v in batch.items()}
                        all_chunks.append(model.encode_target(batch).cpu())
                all_entity_embeddings = torch.cat(all_chunks, dim=0).to(device)

            val_metrics = compute_filtered_ranking_metrics(
                model=model,
                data_loader=valid_loader,
                all_entity_embeddings=all_entity_embeddings,
                hr_map=hr_map,
                device=device
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
