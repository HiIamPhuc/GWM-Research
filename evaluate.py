import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import yaml
import json
import sys

# Paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.model import GWM
from model.dataset import GWMDataset, CollateFN

class EntityDataset(Dataset):
    def __init__(self, data_dir):
        # Load entity map
        with open(os.path.join(data_dir, 'entity_text.json'), 'r') as f:
            self.entity_text = json.load(f) # id_str -> text
        
        # Load entity2id to ensure correct ordering 0..N-1
        with open(os.path.join(data_dir, 'entity2id.json'), 'r') as f:
            self.entity2id = json.load(f)
            
        # Create list of (id, text) sorted by id
        # The IDs in entity2id are effectively 0..N-1 based on implementation convention
        # We trust entity2id.items() or just range(len(entity2id))
        self.num_entities = len(self.entity2id)
        
        # Precompute texts list for indexing
        # keys in entity_text matches str(id)
        self.texts = [self.entity_text.get(str(i), "") for i in range(self.num_entities)]

    def __len__(self):
        return self.num_entities

    def __getitem__(self, idx):
        return {
            'id': idx,
            'text': self.texts[idx]
        }

def get_config(args):
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    if args.data_dir: config_dict['data_dir'] = args.data_dir
    if args.output_dir: config_dict['output_dir'] = args.output_dir
    
    class Config:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)
    return Config(config_dict)

def load_all_triples(data_dir):
    """Load all triples for filtering."""
    all_triples = set()
    for split in ['train', 'valid', 'test']:
        path = os.path.join(data_dir, f'{split}_triples.pt')
        if os.path.exists(path):
            triples = torch.load(path)
            for h, r, t in triples:
                all_triples.add((h.item(), r.item(), t.item()))
    return all_triples

def evaluate(args):
    config = get_config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use conservative defaults for evaluation to avoid OOM on large configs.
    eval_batch_size = int(getattr(config, 'eval_batch_size', min(int(config.batch_size), 128)))
    candidate_batch_size = int(getattr(config, 'candidate_batch_size', min(eval_batch_size * 2, 256)))
    text_cache_batch_size = int(getattr(config, 'text_cache_batch_size', 128))

    # 1. Load Model
    print("Loading model...")
    # Get num entities/relations
    with open(os.path.join(config.data_dir, 'entity2id.json')) as f:
        config.num_entities = len(json.load(f))
    with open(os.path.join(config.data_dir, 'relation2id.json')) as f:
        config.num_relations = len(json.load(f))

    model = GWM(config).to(device)
    
    # Load Checkpoint
    checkpoint_path = os.path.join(config.output_dir, 'best_checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}, trying latest...")
        checkpoint_path = os.path.join(config.output_dir, 'latest_checkpoint.pt')
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint found. Evaluating initialized model (random).")

    if not config.finetune_text_encoder:
        print("Precomputing frozen text embeddings for evaluation...")
        with open(os.path.join(config.data_dir, 'entity_text.json'), 'r') as f:
            entity_text_map = json.load(f)
        with open(os.path.join(config.data_dir, 'relation_text.json'), 'r') as f:
            relation_text_map = json.load(f)

        model.build_text_embedding_cache(
            entity_text_map=entity_text_map,
            relation_text_map=relation_text_map,
            device=device,
            batch_size=text_cache_batch_size,
            max_entity_length=512,
            max_relation_length=128
        )
        print("Frozen text cache ready for evaluation.")

    model.eval()

    # 2. Encode All Candidates (Target Embeddings)
    print("Encoding all entities as targets...")
    entity_dataset = EntityDataset(config.data_dir)
    
    # Custom Collate for Entities
    tokenizer = model.tokenizer
    def entity_collate(batch):
        ids = [x['id'] for x in batch]

        # In frozen-text mode, encode_target can use cached text via IDs only.
        if model.use_text_cache and not config.finetune_text_encoder:
            return {
                'id': torch.tensor(ids)
            }

        texts = [x['text'] for x in batch]
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
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
        num_workers=4
    )

    all_entity_embeddings = []
    with torch.no_grad():
        for batch in tqdm(entity_loader, desc="Encoding Entities"):
            batch = {k: v.to(device) for k, v in batch.items()}
            # Encode symmetrically
            emb = model.encode_target(batch)
            all_entity_embeddings.append(emb.cpu())
            
    all_entity_embeddings = torch.cat(all_entity_embeddings, dim=0).to(device) # (N, H)
    print(f"Encoded {all_entity_embeddings.size(0)} entities.")

    # 3. Evaluation Loop
    split = 'test'
    print(f"Evaluating on {split} set...")
    if not os.path.exists(os.path.join(config.data_dir, f'{split}_triples.pt')):
        print(f"Test triples not found, using 'valid' set.")
        split = 'valid'

    test_dataset = GWMDataset(config.data_dir, split=split)
    collate_fn = CollateFN(model.tokenizer)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=eval_batch_size,
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=4
    )

    # Filtering Setup
    all_triples = load_all_triples(config.data_dir)
    # Map (h, r) -> set of true tails
    hr_map = {}
    for h, r, t in all_triples:
        if (h, r) not in hr_map: hr_map[(h, r)] = set()
        hr_map[(h, r)].add(t)

    hits1, hits3, hits10, mrr = 0, 0, 0, 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move to device
            h_batch = {k: v.to(device) for k, v in batch['h_batch'].items()}
            r_batch = {k: v.to(device) for k, v in batch['r_batch'].items()}
            context_batch = {k: v.to(device) for k, v in batch['context_batch'].items()}
            
            # Ground Truth Tails
            t_ids = batch['t_batch']['id'].to(device)
            h_ids = batch['h_batch']['id'].cpu().numpy()
            r_ids = batch['r_batch']['id'].cpu().numpy()
            
            # Forward Query
            query_vector = model(h_batch, r_batch, context_batch) # (B, H)
            
            # Score against ALL entities
            # (B, N)
            scores = torch.mm(query_vector, all_entity_embeddings.t())
            
            # Filter Scores
            # For each sample in batch
            for i in range(scores.size(0)):
                h_id = h_ids[i]
                r_id = r_ids[i]
                true_t = t_ids[i].item()
                
                # Get all known true tails for this h,r
                filter_mask_indices = list(hr_map.get((h_id, r_id), []))
                # Remove current true tail from filter (we want to rank it!)
                if true_t in filter_mask_indices:
                    filter_mask_indices.remove(true_t)
                
                # Apply mask: Set scores of other true tails to -inf
                if filter_mask_indices:
                    scores[i, filter_mask_indices] = -float('inf')

            # Ranking
            # We want rank of true_t
            # scores[i, true_t] is the score of the target
            target_scores = scores.gather(1, t_ids.unsqueeze(1)) # (B, 1)
            
            # Rank: count how many have score > target_score
            # Add 1 for 1-based rank
            ranks = (scores > target_scores).sum(dim=1) + 1
            
            hits1 += (ranks <= 1).sum().item()
            hits3 += (ranks <= 3).sum().item()
            hits10 += (ranks <= 10).sum().item()
            mrr += (1.0 / ranks.float()).sum().item()
            total += ranks.size(0)

    # Final Metrics
    final_mrr = mrr / total
    final_h1 = hits1 / total
    final_h3 = hits3 / total
    final_h10 = hits10 / total

    print(f"\n--- Evaluation Results ({split}) ---")
    print(f"MRR       : {final_mrr:.4f}")
    print(f"Hits@1    : {final_h1:.4f}")
    print(f"Hits@3    : {final_h3:.4f}")
    print(f"Hits@10   : {final_h10:.4f}")
    print("-------------------------------")
    
    # Save results
    results = {
        'mrr': final_mrr,
        'hits1': final_h1,
        'hits3': final_h3,
        'hits10': final_h10
    }
    with open(os.path.join(config.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to yaml config')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    args = parser.parse_args()
    evaluate(args)
