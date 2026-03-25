import torch
from torch.utils.data import Dataset
import json
import os

class GWMDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Dataset for GWM.
        Loads triples, text maps, and context IDs.
        """
        self.data_dir = data_dir
        self.split = split
        
        # Load triples
        triples_path = os.path.join(data_dir, f'{split}_triples.pt')
        if not os.path.exists(triples_path):
             # Fallback for WN18RR dev vs valid naming if needed, but preprocess handles it
             if split == 'valid' and not os.path.exists(triples_path):
                 triples_path = os.path.join(data_dir, 'dev_triples.pt')
                 
        self.triples = torch.load(triples_path)
        
        # Load text maps
        with open(os.path.join(data_dir, 'entity_text.json'), 'r') as f:
            self.entity_text = json.load(f)
            
        with open(os.path.join(data_dir, 'relation_text.json'), 'r') as f:
            self.relation_text = json.load(f)
            
        # Load context IDs (Precomputed neighbors)
        context_path = os.path.join(data_dir, 'context_ids.pt')
        if os.path.exists(context_path):
            self.context_ids = torch.load(context_path)
        else:
            print(f"Warning: {context_path} not found. Context will be zeros.")
            # Create dummy context if missing
            self.context_ids = None

    def __len__(self):
        return len(self.triples)
        
    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        
        # Convert IDs to text
        # Keys in json are strings
        h_str, r_str, t_str = str(h.item()), str(r.item()), str(t.item())
        
        h_text = self.entity_text.get(h_str, f"Entity {h_str}")
        r_text = self.relation_text.get(r_str, f"Relation {r_str}")
        t_text = self.entity_text.get(t_str, f"Entity {t_str}")
        
        # Retrieve context
        if self.context_ids is not None:
            ctx_ids = self.context_ids[h]
        else:
            ctx_ids = torch.zeros(10, dtype=torch.long) # Dummy
            
        # Context Texts
        ctx_texts = []
        for cid in ctx_ids:
            c_str = str(cid.item())
            ctx_texts.append(self.entity_text.get(c_str, f"Entity {c_str}"))
            
        return {
            'h_id': h,
            'r_id': r,
            't_id': t,
            'h_text': h_text,
            'r_text': r_text,
            't_text': t_text,
            'context_ids': ctx_ids,
            'context_texts': ctx_texts
        }

class CollateFN:
    """
    Collator to handle dynamic padding of text.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        h_texts = [b['h_text'] for b in batch]
        r_texts = [b['r_text'] for b in batch]
        t_texts = [b['t_text'] for b in batch]
        
        # Context Texts: List of Lists -> Flatten
        context_texts_flat = []
        
        # We need to know K (context size) to un-flatten later or reshaping
        # Assuming fixed K per batch
        
        for b in batch:
            context_texts_flat.extend(b['context_texts']) # [B*K]
        
        # Tokenize (Max length constraints for efficiency)
        h_enc = self.tokenizer(h_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        r_enc = self.tokenizer(r_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        t_enc = self.tokenizer(t_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        
        # Tokenize Context (Use shorter length for context to save memory, e.g. 16/24 tokens)
        ctx_enc = self.tokenizer(context_texts_flat, padding=True, truncation=True, return_tensors='pt', max_length=24)
        
        h_ids = torch.stack([b['h_id'] for b in batch])
        r_ids = torch.stack([b['r_id'] for b in batch])
        t_ids = torch.stack([b['t_id'] for b in batch])
        context_ids = torch.stack([b['context_ids'] for b in batch])
        
        return {
            'h_batch': {'input_ids': h_enc['input_ids'], 'attention_mask': h_enc['attention_mask'], 'id': h_ids},
            'r_batch': {'input_ids': r_enc['input_ids'], 'attention_mask': r_enc['attention_mask'], 'id': r_ids},
            't_batch': {'input_ids': t_enc['input_ids'], 'attention_mask': t_enc['attention_mask'], 'id': t_ids},
            'context_batch': {'input_ids': ctx_enc['input_ids'], 'attention_mask': ctx_enc['attention_mask'], 'id': context_ids}
        }
