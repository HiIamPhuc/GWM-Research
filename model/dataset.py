import torch
from torch.utils.data import Dataset
import os

class GWMDataset(Dataset):
    def __init__(self, data_dir, split='train', require_context=True):
        """
        Dataset for GWM.
        Loads triples and context IDs.
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
        
        # Load context IDs (Precomputed neighbors)
        context_path = os.path.join(data_dir, 'context_ids.pt')
        if os.path.exists(context_path):
            self.context_ids = torch.load(context_path)
        else:
            if require_context:
                raise FileNotFoundError(
                    f"Missing required context file: {context_path}. "
                    "Run utils/compute_context.py to generate context_ids.pt and context_mask.pt."
                )
            print(f"Warning: {context_path} not found. Context IDs will be zeros with invalid mask.")
            self.context_ids = None

        # Optional context validity mask: 1 for valid context token, 0 for padding.
        context_mask_path = os.path.join(data_dir, 'context_mask.pt')
        if os.path.exists(context_mask_path):
            self.context_mask = torch.load(context_mask_path)
        else:
            self.context_mask = None
            if self.context_ids is not None:
                print(
                    f"Warning: {context_mask_path} not found. "
                    "Using all-ones context mask (treat all context positions as valid)."
                )

    def __len__(self):
        return len(self.triples)
        
    def __getitem__(self, idx):
        h, r, t = self.triples[idx]

        # Retrieve context
        if self.context_ids is not None:
            ctx_ids = self.context_ids[h]
            if self.context_mask is not None:
                ctx_mask = self.context_mask[h]
            else:
                ctx_mask = torch.ones_like(ctx_ids, dtype=torch.bool)
        else:
            ctx_ids = torch.zeros(10, dtype=torch.long) # Dummy
            ctx_mask = torch.zeros_like(ctx_ids, dtype=torch.bool)

        return {
            'h_id': h.long(),
            'r_id': r.long(),
            't_id': t.long(),
            'context_ids': ctx_ids.long(),
            'context_mask': ctx_mask.bool(),
        }

class CollateFN:
    """
    ID-only collator; text embeddings are loaded from precomputed caches.
    """
    def __init__(self):
        pass
        
    def __call__(self, batch):
        h_ids = torch.stack([b['h_id'] for b in batch])
        r_ids = torch.stack([b['r_id'] for b in batch])
        t_ids = torch.stack([b['t_id'] for b in batch])
        context_ids = torch.stack([b['context_ids'] for b in batch])
        context_mask = torch.stack([b['context_mask'] for b in batch])
        
        return {
            'h_batch': {'id': h_ids},
            'r_batch': {'id': r_ids},
            't_batch': {'id': t_ids},
            'context_batch': {
                'id': context_ids,
                'mask': context_mask,
            },
        }
