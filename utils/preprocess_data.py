import json
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

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

class _TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def _get_base_model(model):
    if hasattr(model, 'base_model'):
        return model.base_model
    base_prefix = getattr(model, 'base_model_prefix', None)
    if base_prefix and hasattr(model, base_prefix):
        return getattr(model, base_prefix)
    return model


def _train_lora_adapter(
    model,
    tokenizer,
    texts,
    device,
    max_length,
    batch_size,
    epochs,
    lr,
    mlm_probability,
):
    from transformers import DataCollatorForLanguageModeling

    dataset = _TextDataset(texts)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    def _collate(batch):
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        return collator(encoded)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch in tqdm(loader, desc=f"LoRA MLM epoch {epoch + 1}/{epochs}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def precompute_text_embeddings(
    entity_text_dict,
    relation_text_dict,
    num_entities,
    num_relations,
    pretrained_model='bert-base-uncased',
    batch_size=128,
    max_entity_length=256,
    max_relation_length=64,
    device=None,
    lora_pretrain=False,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_modules=None,
    lora_epochs=1,
    lora_lr=5e-5,
    lora_mlm_probability=0.15,
    lora_output_dir=None,
):
    """
    Encode entity/relation text once and return dense embedding tensors.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if lora_pretrain and (LoraConfig is None or get_peft_model is None):
        raise RuntimeError(
            "peft is required for LoRA. Install with: pip install peft"
        )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    text_encoder = AutoModelForMaskedLM.from_pretrained(pretrained_model).to(device)

    if lora_target_modules is None:
        lora_target_modules = ["query", "value"]

    if lora_pretrain:
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        text_encoder = get_peft_model(text_encoder, lora_config)

        all_texts = list(entity_text_dict.values()) + list(relation_text_dict.values())
        max_length = max(max_entity_length, max_relation_length)
        _train_lora_adapter(
            model=text_encoder,
            tokenizer=tokenizer,
            texts=all_texts,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            epochs=lora_epochs,
            lr=lora_lr,
            mlm_probability=lora_mlm_probability,
        )

        if lora_output_dir is not None:
            Path(lora_output_dir).mkdir(parents=True, exist_ok=True)
            text_encoder.save_pretrained(lora_output_dir)

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
                base_model = _get_base_model(text_encoder)
                outputs = base_model(**encoded)
                all_emb.append(outputs.last_hidden_state[:, 0, :].detach().cpu())
        return torch.cat(all_emb, dim=0).contiguous()

    entity_embeddings = encode_ordered_texts(
        entity_text_dict,
        num_entities,
        max_entity_length,
        desc='Encoding entity text',
    )
    relation_embeddings = encode_ordered_texts(
        relation_text_dict,
        num_relations,
        max_relation_length,
        desc='Encoding relation text',
    )

    del text_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return entity_embeddings, relation_embeddings

def process_dataset(
    data_dir,
    output_dir,
    dataset_name,
    add_inverse=True,
    pretrained_model='bert-base-uncased',
    text_batch_size=128,
    max_entity_length=256,
    max_relation_length=64,
    text_device=None,
    lora_pretrain=False,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_target_modules=None,
    lora_epochs=1,
    lora_lr=5e-5,
    lora_mlm_probability=0.15,
):
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

    print("Encoding and caching text embeddings...")
    entity_text_embeddings, relation_text_embeddings = precompute_text_embeddings(
        entity_text_dict=entity_text_dict,
        relation_text_dict=relation_text_dict,
        num_entities=len(entity2id),
        num_relations=len(relation2id),
        pretrained_model=pretrained_model,
        batch_size=text_batch_size,
        max_entity_length=max_entity_length,
        max_relation_length=max_relation_length,
        device=text_device,
        lora_pretrain=lora_pretrain,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        lora_epochs=lora_epochs,
        lora_lr=lora_lr,
        lora_mlm_probability=lora_mlm_probability,
        lora_output_dir=str(Path(output_dir) / "lora_adapter") if lora_pretrain else None,
    )

    torch.save(
        {
            'embeddings': entity_text_embeddings,
            'model_name': pretrained_model,
            'embedding_dim': int(entity_text_embeddings.size(1)),
        },
        out_path / 'entity_text_embeddings.pt'
    )
    torch.save(
        {
            'embeddings': relation_text_embeddings,
            'model_name': pretrained_model,
            'embedding_dim': int(relation_text_embeddings.size(1)),
        },
        out_path / 'relation_text_embeddings.pt'
    )

    # 5. Ground Truth for Filtered Eval
    # Save split-aware maps to ensure fair ranking protocols:
    # - validation: filter with train only
    # - test: filter with train + valid
    def build_ground_truth(*triple_tensors):
        gt = {}
        for tensor in triple_tensors:
            for h, r, t in tensor.tolist():
                key = f"{h},{r}"
                if key not in gt:
                    gt[key] = set()
                gt[key].add(t)
        return {k: sorted(list(v)) for k, v in gt.items()}

    ground_truth_train = build_ground_truth(train_tensor)
    ground_truth_train_valid = build_ground_truth(train_tensor, valid_tensor)
    ground_truth_all = build_ground_truth(train_tensor, valid_tensor, test_tensor)

    # Backward-compatible legacy file
    with open(out_path / 'ground_truth.json', 'w') as f:
        json.dump(ground_truth_all, f)

    # Split-aware files used by train/evaluate ranking
    with open(out_path / 'ground_truth_train.json', 'w') as f:
        json.dump(ground_truth_train, f)
    with open(out_path / 'ground_truth_train_valid.json', 'w') as f:
        json.dump(ground_truth_train_valid, f)
        
    print("Data processing complete.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g., fb15k-237, wn18rr)')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased')
    parser.add_argument('--text_batch_size', type=int, default=128)
    parser.add_argument('--max_entity_length', type=int, default=256)
    parser.add_argument('--max_relation_length', type=int, default=64)
    parser.add_argument('--text_device', type=str, default=None, help='cpu or cuda; defaults to auto')
    parser.add_argument('--lora_pretrain', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_target_modules', type=str, default=None, help='Comma-separated module names')
    parser.add_argument('--lora_epochs', type=int, default=1)
    parser.add_argument('--lora_lr', type=float, default=5e-5)
    parser.add_argument('--lora_mlm_probability', type=float, default=0.15)
    args = parser.parse_args()

    target_modules = None
    if args.lora_target_modules:
        target_modules = [m.strip() for m in args.lora_target_modules.split(',') if m.strip()]
    
    process_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        pretrained_model=args.pretrained_model,
        text_batch_size=args.text_batch_size,
        max_entity_length=args.max_entity_length,
        max_relation_length=args.max_relation_length,
        text_device=args.text_device,
        lora_pretrain=args.lora_pretrain,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=target_modules,
        lora_epochs=args.lora_epochs,
        lora_lr=args.lora_lr,
        lora_mlm_probability=args.lora_mlm_probability,
    )
