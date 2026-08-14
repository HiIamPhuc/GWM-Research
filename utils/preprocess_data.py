import json
import torch
from pathlib import Path
from tqdm import tqdm
import re

def load_triples(file_path):
    """Load triples from a text file."""
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            triples.append((h, r, t))
    return triples

def load_counted_id_map(file_path):
    """Load files with first-line count followed by token<TAB>id rows."""
    token_to_id = {}
    id_to_token = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            token, raw_id = line.strip().split('\t')
            idx = int(raw_id)
            token_to_id[token] = idx
            id_to_token[idx] = token

    return token_to_id, id_to_token

def load_nell995_triples(file_path, id_to_entity, id_to_relation):
    """
    Load NELL-995 split files.

    Format: first line is count; each triple line is `head_id tail_id relation_id`.
    """
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            h_id, t_id, r_id = map(int, line.strip().split())
            triples.append((
                id_to_entity[h_id],
                id_to_relation[r_id],
                id_to_entity[t_id],
            ))

    return triples

def load_nell995_dataset(data_dir, add_inverse=True):
    data_path = Path(data_dir)
    entity2id, id_to_entity = load_counted_id_map(data_path / 'entity2id.txt')
    relation2id, id_to_relation = load_counted_id_map(data_path / 'relation2id.txt')

    if add_inverse:
        num_original_relations = len(relation2id)
        for relation, rid in list(relation2id.items()):
            relation2id[relation + '_inv'] = rid + num_original_relations

    train_triples = load_nell995_triples(data_path / 'train.txt', id_to_entity, id_to_relation)
    valid_triples = load_nell995_triples(data_path / 'valid.txt', id_to_entity, id_to_relation)
    test_triples = load_nell995_triples(data_path / 'test.txt', id_to_entity, id_to_relation)
    return train_triples, valid_triples, test_triples, entity2id, relation2id

def load_ordered_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def load_umls_triples(file_path):
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            triples.append(tuple(line.strip().split('\t')))
    return triples

def load_umls_dataset(data_dir, add_inverse=True):
    data_path = Path(data_dir)
    entities = load_ordered_tokens(data_path / 'entities.txt')
    relations = load_ordered_tokens(data_path / 'relations.txt')

    entity2id = {entity: idx for idx, entity in enumerate(entities)}
    relation2id = {relation: idx for idx, relation in enumerate(relations)}
    if add_inverse:
        num_original_relations = len(relation2id)
        for relation, rid in list(relation2id.items()):
            relation2id[relation + '_inv'] = rid + num_original_relations

    train_triples = load_umls_triples(data_path / 'train.tsv')
    valid_triples = load_umls_triples(data_path / 'valid.tsv')
    test_triples = load_umls_triples(data_path / 'test.tsv')

    return train_triples, valid_triples, test_triples, entity2id, relation2id

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

def process_text_nell995(data_dir, entity2id, relation2id):
    """Create readable text from NELL concept identifiers."""
    entity_text = {}
    relation_text = {}

    def clean_entity(entity):
        text = entity
        if text.startswith('concept_'):
            text = text[len('concept_'):]
        text = text.replace('_', ' ')
        return text.strip()

    def clean_relation(relation):
        text = relation
        if text.endswith('_inv'):
            base = clean_relation(text[:-4])
            return 'inverse of ' + base
        if text.startswith('concept:'):
            text = text[len('concept:'):]
        text = text.replace('_', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    for entity, eid in entity2id.items():
        entity_text[str(eid)] = clean_entity(entity)

    for relation, rid in relation2id.items():
        relation_text[str(rid)] = clean_relation(relation)

    return entity_text, relation_text

def process_text_umls(data_dir, entity2id, relation2id):
    data_path = Path(data_dir)
    entity_text = {}
    relation_text = {}

    def load_text_map(filename):
        path = data_path / filename
        text_map = {}
        if not path.exists():
            return text_map
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t', 1)
                if len(parts) == 2:
                    text_map[parts[0]] = parts[1].strip()
        return text_map

    # Prefer longer descriptions when present; otherwise use concise labels.
    e_map = load_text_map('entity2textlong.txt')
    if not e_map:
        e_map = load_text_map('entity2text.txt')
    r_map = load_text_map('relation2text.txt')

    for entity, eid in entity2id.items():
        entity_text[str(eid)] = e_map.get(entity, entity.replace('_', ' '))

    for relation, rid in relation2id.items():
        if relation.endswith('_inv'):
            base_relation = relation[:-4]
            base_text = r_map.get(base_relation, base_relation.replace('_', ' '))
            relation_text[str(rid)] = 'inverse of ' + base_text
        else:
            relation_text[str(rid)] = r_map.get(relation, relation.replace('_', ' '))

    return entity_text, relation_text

def triples_to_ids(triples, entity2id, relation2id, add_inverse=False):
    ids = []
    for h, r, t in triples:
        h_id, r_id, t_id = entity2id[h], relation2id[r], entity2id[t]
        ids.append((h_id, r_id, t_id))
        if add_inverse:
            r_inv_id = relation2id[r + '_inv']
            ids.append((t_id, r_inv_id, h_id))
    return torch.tensor(ids, dtype=torch.long)

def precompute_entity_text_embeddings(
    entity_text_dict,
    num_entities,
    pretrained_model='bert-base-uncased',
    batch_size=128,
    max_entity_length=256,
    device=None,
):
    """Encode entity descriptions once for residual semantic fusion."""
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    text_encoder = AutoModel.from_pretrained(pretrained_model).to(device)
    text_encoder.eval()

    def encode_ordered_texts(text_dict, size, max_length, desc):
        texts = [text_dict.get(str(i), f"Token {i}") for i in range(size)]
        all_emb = []
        with torch.no_grad():
            for start in tqdm(range(0, size, batch_size), desc=desc, leave=False):
                chunk = texts[start:start + batch_size]
                encoded = tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt',
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                outputs = text_encoder(**encoded)
                all_emb.append(outputs.last_hidden_state[:, 0, :].detach().cpu())
        return torch.cat(all_emb, dim=0).contiguous()

    embeddings = encode_ordered_texts(
        entity_text_dict,
        num_entities,
        max_entity_length,
        desc='Encoding entity text',
    )

    del text_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings


def precompute_and_save_entity_text_embeddings(
    data_dir,
    output_dir,
    dataset_name,
    entity2id,
    relation2id,
    pretrained_model='bert-base-uncased',
    text_batch_size=128,
    max_entity_length=256,
    text_device=None,
):
    """Build and cache frozen entity-description embeddings."""
    dataset_key = dataset_name.lower()
    if 'fb15k' in dataset_key:
        entity_text, _ = process_text_fb15k237(
            data_dir,
            entity2id,
            relation2id,
        )
    elif 'wn18' in dataset_key:
        entity_text, _ = process_text_wn18rr(
            data_dir,
            entity2id,
            relation2id,
        )
    elif 'nell' in dataset_key:
        entity_text, _ = process_text_nell995(
            data_dir,
            entity2id,
            relation2id,
        )
    elif 'umls' in dataset_key:
        entity_text, _ = process_text_umls(
            data_dir,
            entity2id,
            relation2id,
        )
    else:
        raise ValueError(f"Text preprocessing is unavailable for {dataset_name}.")

    output_dir = Path(output_dir)
    with open(output_dir / 'entity_text.json', 'w') as f:
        json.dump(entity_text, f, indent=2)
    entity_embeddings = precompute_entity_text_embeddings(
        entity_text_dict=entity_text,
        num_entities=len(entity2id),
        pretrained_model=pretrained_model,
        batch_size=text_batch_size,
        max_entity_length=max_entity_length,
        device=text_device,
    )
    torch.save(
        {
            'embeddings': entity_embeddings,
            'model_name': pretrained_model,
            'embedding_dim': entity_embeddings.size(1),
        },
        output_dir / 'entity_text_embeddings.pt',
    )


def process_dataset(
    data_dir,
    output_dir,
    dataset_name,
    add_inverse=True,
):
    """
    Process raw dataset into training files.
    1. Reads train/valid/test.txt
    2. Generates entity2id, relation2id
    3. Saves triples as tensors
    4. Saves ground_truth for evaluation
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing data from {data_path} for {dataset_name}...")
    
    # 1. Load triples and vocabularies
    dataset_key = dataset_name.lower()
    if 'nell' in dataset_key:
        train_triples, valid_triples, test_triples, entity2id, relation2id = load_nell995_dataset(
            data_path,
            add_inverse=add_inverse,
        )
    elif 'umls' in dataset_key:
        train_triples, valid_triples, test_triples, entity2id, relation2id = load_umls_dataset(
            data_path,
            add_inverse=add_inverse,
        )
    else:
        train_triples = load_triples(data_path / 'train.txt')
        valid_triples = load_triples(data_path / 'valid.txt')
        test_triples = load_triples(data_path / 'test.txt')
        entity2id, relation2id = create_vocabularies(
            train_triples,
            valid_triples,
            test_triples,
            add_inverse,
        )
    
    # Save Vocabs
    with open(out_path / 'entity2id.json', 'w') as f:
        json.dump(entity2id, f, indent=2)
    with open(out_path / 'relation2id.json', 'w') as f:
        json.dump(relation2id, f, indent=2)
    train_tensor = triples_to_ids(train_triples, entity2id, relation2id, add_inverse=add_inverse)
    valid_tensor = triples_to_ids(valid_triples, entity2id, relation2id, add_inverse=False)
    test_tensor = triples_to_ids(test_triples, entity2id, relation2id, add_inverse=False)
    
    torch.save(train_tensor, out_path / 'train_triples.pt')
    torch.save(valid_tensor, out_path / 'valid_triples.pt')
    torch.save(test_tensor, out_path / 'test_triples.pt')

    # 4. Ground Truth for Filtered Eval
    # Standard filtered KGC ranking removes every other known true answer,
    # including facts from train, validation, and test.
    def build_ground_truth(*triple_tensors):
        gt = {}
        for tensor in triple_tensors:
            for h, r, t in tensor.tolist():
                key = f"{h},{r}"
                if key not in gt:
                    gt[key] = set()
                gt[key].add(t)
        return {k: sorted(list(v)) for k, v in gt.items()}

    ground_truth_all = build_ground_truth(train_tensor, valid_tensor, test_tensor)

    # Keep the existing filenames for compatibility. All of them now contain
    # the complete known-fact map required by filtered validation/test ranking.
    with open(out_path / 'ground_truth.json', 'w') as f:
        json.dump(ground_truth_all, f)
        
    print("Data processing complete.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g., fb15k-237, wn18rr)')
    args = parser.parse_args()
    
    process_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
    )
