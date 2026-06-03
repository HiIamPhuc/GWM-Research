"""
RerankerCacheDataset — PyTorch Dataset that serves pre-computed stage-1 retriever
outputs for standalone reranker training.

Expected cache files (produced by utils/build_retriever_cache.py):
    entity_cache.pt      — {'entity_text': (N, D_t), 'entity_struct': (N, D_s)}
    train_query_cache.pt — {'rel_text': (M, D_t), 'rel_struct': (M, D_s),
                             'cand_idx': (M, K+1), 'true_t_id': (M,)}

The 'cand_idx' tensor already has the true tail at the last column (see cache builder).
Labels are therefore always K (the final position index) for every example.

No GWM forward pass is needed at training time.
"""

import os
import torch
from torch.utils.data import Dataset


class RerankerCacheDataset(Dataset):
    """
    Loads a pre-built retriever cache and serves one (relation, candidates, label) tuple
    per training triple.

    Parameters
    ----------
    cache_dir   : directory containing entity_cache.pt and <split>_query_cache.pt
    split       : 'train' or 'valid'
    device      : if provided, entity embeddings are pinned to this device once at load time
    """

    def __init__(self, cache_dir: str, split: str = 'train', device=None):
        entity_path = os.path.join(cache_dir, 'entity_cache.pt')
        query_path  = os.path.join(cache_dir, f'{split}_query_cache.pt')

        if not os.path.exists(entity_path):
            raise FileNotFoundError(
                f"Entity cache not found: {entity_path}\n"
                "Run utils/build_retriever_cache.py first."
            )
        if not os.path.exists(query_path):
            raise FileNotFoundError(
                f"Query cache not found: {query_path}\n"
                f"Run utils/build_retriever_cache.py with --splits {split} first."
            )

        entity_cache = torch.load(entity_path, map_location='cpu')
        query_cache  = torch.load(query_path,  map_location='cpu')

        # Entity embeddings — keep on CPU, move to device per-batch in collate / DataLoader
        self.entity_text   = entity_cache['entity_text'].float()    # (N, D_t)
        self.entity_struct = entity_cache['entity_struct'].float()  # (N, D_s)

        # Query-level tensors
        self.rel_text   = query_cache['rel_text'].float()    # (M, D_t)
        self.rel_struct = query_cache['rel_struct'].float()  # (M, D_s)
        self.cand_idx   = query_cache['cand_idx'].long()     # (M, K+1)
        self.true_t_id  = query_cache['true_t_id'].long()   # (M,)

        # Label: true tail is always the last candidate slot
        self.K = self.cand_idx.size(1)

    def __len__(self):
        return self.rel_text.size(0)

    def __getitem__(self, idx):
        cand_idx = self.cand_idx[idx]               # (K+1,)
        cand_text   = self.entity_text[cand_idx]    # (K+1, D_t)
        cand_struct = self.entity_struct[cand_idx]  # (K+1, D_s)

        return {
            'rel_text':    self.rel_text[idx],      # (D_t,)
            'rel_struct':  self.rel_struct[idx],    # (D_s,)
            'cand_text':   cand_text,               # (K+1, D_t)
            'cand_struct': cand_struct,             # (K+1, D_s)
            'label':       torch.tensor(self.K - 1, dtype=torch.long),  # always last
        }


def reranker_collate(batch):
    """Stack a list of __getitem__ dicts into batched tensors."""
    return {
        'rel_text':    torch.stack([x['rel_text']    for x in batch]),    # (B, D_t)
        'rel_struct':  torch.stack([x['rel_struct']  for x in batch]),    # (B, D_s)
        'cand_text':   torch.stack([x['cand_text']   for x in batch]),    # (B, K+1, D_t)
        'cand_struct': torch.stack([x['cand_struct'] for x in batch]),    # (B, K+1, D_s)
        'label':       torch.stack([x['label']       for x in batch]),    # (B,)
    }
