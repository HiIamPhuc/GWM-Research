import json
import torch
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

def load_triples(file_path):
    """Load triples from a text file."""
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            triples.append((h, r, t))
    return triples

def create_vocabularies(train_triples, valid_triples, test_triples, add_inverse=True):
    """Create entity and relation mappings."""
    entities = set()
    relations = set()
    
    all_triples = train_triples + valid_triples + test_triples
    for h, r, t in all_triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)
    
    if add_inverse:
        original_rels = list(relations)
        for r in original_rels:
            relations.add(r + '_inv')
            
    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}
    
    return entity2id, relation2id

def process_text_fb15k237(data_dir, entity2id, relation2id):
    """
    Process text for FB15k-237.
    Uses mid2description.txt (primary) and mid2name.txt (fallback).
    """
    data_path = Path(data_dir)
    entity_text = {}
    relation_text = {}
    
    # helper to clean text
    def clean_text(text):
        return text.strip().replace('"', '').replace('@en', '')

    # Load Descriptions
    mid2desc = {}
    desc_file = data_path / 'mid2description.txt'
    if desc_file.exists():
        with open(desc_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    mid, desc = parts[0], parts[1]
                    mid2desc[mid] = clean_text(desc)
    
    # Load Names
    mid2name = {}
    name_file = data_path / 'mid2name.txt'
    if name_file.exists():
        with open(name_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    mid, name = parts[0], parts[1]
                    mid2name[mid] = name
                    
    # Map Entities
    for entity, eid in entity2id.items():
        if entity in mid2desc:
            entity_text[str(eid)] = mid2desc[entity]
        elif entity in mid2name:
            entity_text[str(eid)] = mid2name[entity]
        else:
            entity_text[str(eid)] = f"Entity {entity}"
            
    # Map Relations
    for relation, rid in relation2id.items():
        if relation.endswith('_inv'):
            base_rel = relation[:-4]
            relation_text[str(rid)] = 'inverse of ' + base_rel
        else:
            relation_text[str(rid)] = relation
            
    return entity_text, relation_text

def process_text_wn18rr(data_dir, entity2id, relation2id):
    """
    Process text for WN18RR using entity2text.txt and relation2text.txt.
    """
    data_path = Path(data_dir)
    entity_text = {}
    relation_text = {}
    
    # helper for loading text map
    def load_text_map(filename):
        text_map = {}
        path = data_path / filename
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        text_map[parts[0]] = parts[1]
        return text_map

    e_map = load_text_map('entity2text.txt')
    r_map = load_text_map('relation2text.txt')
    
    # Map Entities
    for entity, eid in entity2id.items():
        if entity in e_map:
            entity_text[str(eid)] = e_map[entity]
        else:
             # Fallback to heuristic
            clean_ent = entity.split('.')[0].replace('_', ' ')
            entity_text[str(eid)] = clean_ent
        
    # Map Relations
    for relation, rid in relation2id.items():
        if relation.endswith('_inv'):
            base_rel = relation[:-4]
            if base_rel in r_map:
                 relation_text[str(rid)] = 'inverse of ' + r_map[base_rel]
            else:
                 relation_text[str(rid)] = 'inverse of ' + base_rel.replace('_', ' ').strip()
        else:
            if relation in r_map:
                relation_text[str(rid)] = r_map[relation]
            else:
                 relation_text[str(rid)] = relation.replace('_', ' ').strip()
            
    return entity_text, relation_text

def process_dataset(data_dir, output_dir, dataset_name, embedding_model='all-MiniLM-L6-v2', add_inverse=True):
    """
    Process raw dataset into training files.
    1. Reads train/valid/test.txt
    2. Generates entity2id, relation2id
    3. Saves triples as tensors
    4. Saves entity/relation text descriptions
    5. Saves ground_truth for evaluation
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing data from {data_path} for {dataset_name}...")
    
    # 1. Load Triples
    train_triples = load_triples(data_path / 'train.txt')
    valid_triples = load_triples(data_path / 'valid.txt')
    test_triples = load_triples(data_path / 'test.txt')
    
    # 2. Vocabularies
    entity2id, relation2id = create_vocabularies(train_triples, valid_triples, test_triples, add_inverse)
    
    # Save Vocabs
    with open(out_path / 'entity2id.json', 'w') as f:
        json.dump(entity2id, f, indent=2)
    with open(out_path / 'relation2id.json', 'w') as f:
        json.dump(relation2id, f, indent=2)
        
    # 3. Convert Triples to IDs
    def triples_to_ids(triples, add_inv=False):
        ids = []
        for h, r, t in triples:
            h_id, r_id, t_id = entity2id[h], relation2id[r], entity2id[t]
            ids.append((h_id, r_id, t_id))
            if add_inv:
                r_inv_id = relation2id[r + '_inv']
                ids.append((t_id, r_inv_id, h_id))
        return torch.tensor(ids, dtype=torch.long)

    train_tensor = triples_to_ids(train_triples, add_inv=add_inverse)
    valid_tensor = triples_to_ids(valid_triples, add_inv=False)
    test_tensor = triples_to_ids(test_triples, add_inv=False)
    
    torch.save(train_tensor, out_path / 'train_triples.pt')
    torch.save(valid_tensor, out_path / 'valid_triples.pt')
    torch.save(test_tensor, out_path / 'test_triples.pt')
    
    # 4. Process Text Descriptions
    print(f"Generating descriptions for {dataset_name}...")
    if 'fb15k' in dataset_name.lower():
        # Requires mid2description.txt
        entity_text_dict, relation_text_dict = process_text_fb15k237(data_dir, entity2id, relation2id)
    elif 'wn18' in dataset_name.lower():
        entity_text_dict, relation_text_dict = process_text_wn18rr(data_dir, entity2id, relation2id)
    else:
        raise ValueError(f"Error: Unknown dataset {dataset_name}. Please provide text descriptions for this dataset.")

    with open(out_path / 'entity_text.json', 'w') as f:
        json.dump(entity_text_dict, f, indent=2)
    with open(out_path / 'relation_text.json', 'w') as f:
        json.dump(relation_text_dict, f, indent=2)

    # 5. Ground Truth for Filtered Eval
    ground_truth = {}
    all_triples_ids = torch.cat([train_tensor, valid_tensor, test_tensor])
    for h, r, t in all_triples_ids.tolist():
        key = f"{h},{r}"
        if key not in ground_truth:
            ground_truth[key] = []
        ground_truth[key].append(t)
        
    with open(out_path / 'ground_truth.json', 'w') as f:
        json.dump(ground_truth, f)
        
    print("Data processing complete.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g., fb15k-237, wn18rr)')
    args = parser.parse_args()
    
    process_dataset(args.data_dir, args.output_dir, args.dataset)
