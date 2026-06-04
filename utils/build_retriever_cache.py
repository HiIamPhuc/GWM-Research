"""
build_retriever_cache.py  —  Stage-1 output cache builder for the GWM reranker.

Runs a frozen GWM retriever over the train (and optionally valid) split and
saves per-triple retrieval artefacts to disk.  Stage-2 reranker training loads
these artefacts directly — no GWM forward pass required.

Cache files produced
--------------------
<output_dir>/
    entity_cache.pt          -- {'entity_text': (N,D_t), 'entity_struct': (N,D_s)}
    train_query_cache.pt     -- {'rel_text':..., 'rel_struct':..., 'cand_idx':..., 'true_t_id':...}
    valid_query_cache.pt     -- same, for the validation split

Cache tensor shapes (per split)
--------------------------------
    rel_text   : (M, D_t)   relation/query text embedding for each triple
    rel_struct : (M, D_s)
    cand_idx   : (M, K)     top-K fused-score candidate indices (entity ids)
                             the last column is always the true tail id
    true_t_id  : (M,)       ground-truth tail entity id (for loss construction)

Usage
-----
    python utils/build_retriever_cache.py \\
        --config  configs/wn18rr_stage2_rerank.yaml \\
        --splits  train valid

The config must contain:
    init_checkpoint   : path to the trained stage-1 retriever checkpoint
    data_dir          : dataset directory (same as stage-1)
    output_dir        : where to write the cache files
    reranker_train_topk, batch_size, eval_batch_size, text_emb_*, struct_emb_* ...
"""

import os
import sys
import json
import argparse

import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import GWM
from model.dataset import GWMDataset, CollateFN
from utils.eval import build_entity_loader, encode_all_entities_as_targets


# ---------------------------------------------------------------------------
# Config helpers (shared with train.py)
# ---------------------------------------------------------------------------

def _load_config(config_path, data_dir_override=None, output_dir_override=None):
    with open(config_path, 'r') as f:
        d = yaml.safe_load(f)
    if data_dir_override:
        d['data_dir'] = data_dir_override
    if output_dir_override:
        d['output_dir'] = output_dir_override

    class Config:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)

    return Config(d)


# ---------------------------------------------------------------------------
# Cache building
# ---------------------------------------------------------------------------

@torch.no_grad()
def _build_query_cache(model, data_loader, all_t_text, all_t_struct, topk, device):
    """
    Returns: dict with rel_text, rel_struct, cand_idx, true_t_id (all CPU tensors).
    Candidate list per query = top-K fused indices + forced true tail (last column).
    """
    rel_text_list   = []
    rel_struct_list = []
    cand_idx_list   = []
    true_t_list     = []

    model.eval()
    for batch in tqdm(data_loader, desc="Building query cache"):
        h_batch      = {k: v.to(device) for k, v in batch['h_batch'].items()}
        r_batch      = {k: v.to(device) for k, v in batch['r_batch'].items()}
        ctx_batch    = {k: v.to(device) for k, v in batch['context_batch'].items()}
        true_t       = batch['t_batch']['id'].to(device)   # (B,)

        q_text, q_struct, rel_text, rel_struct, _, _ = model(h_batch, r_batch, ctx_batch)

        # Fused scores via alpha_mlp
        alpha     = model.alpha_mlp(torch.cat([rel_text, rel_struct], dim=-1))
        balance   = alpha.squeeze(-1)
        if getattr(model, 'balance_floor', 0.0) > 0.0:
            balance = balance.clamp(min=model.balance_floor, max=1.0 - model.balance_floor)
        alpha_eff = balance.unsqueeze(-1)

        s_text   = torch.mm(q_text,   all_t_text.t())   / model.temperature
        s_struct = torch.mm(q_struct, all_t_struct.t()) / model.temperature
        fused    = alpha_eff * s_text + (1.0 - alpha_eff) * s_struct  # (B, N)

        k = max(1, min(topk, fused.size(1) - 1))
        top_idx  = torch.topk(fused, k=k, dim=1).indices        # (B, K)
        true_col = true_t.unsqueeze(1)                           # (B, 1)

        # Deduplicate — ensure the forced positive is not already in top_idx,
        # otherwise CE gets conflicting supervision.
        pre_mask = top_idx.eq(true_col)
        if pre_mask.any():
            replacement = (top_idx + 1) % fused.size(1)
            top_idx     = torch.where(pre_mask, replacement, top_idx)

        cand = torch.cat([top_idx, true_col], dim=1)  # (B, K+1)

        rel_text_list.append(rel_text.cpu())
        rel_struct_list.append(rel_struct.cpu())
        cand_idx_list.append(cand.cpu())
        true_t_list.append(true_t.cpu())

    return {
        'rel_text':   torch.cat(rel_text_list,   dim=0),
        'rel_struct': torch.cat(rel_struct_list, dim=0),
        'cand_idx':   torch.cat(cand_idx_list,   dim=0),
        'true_t_id':  torch.cat(true_t_list,     dim=0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build retriever output cache for stage-2 reranker training.")
    parser.add_argument('--config',     required=True, help='Path to stage-2 YAML config')
    parser.add_argument('--data_dir',   default=None)
    parser.add_argument('--output_dir', default=None,
                        help='Override cache output directory (default: config.output_dir)')
    parser.add_argument('--splits',     nargs='+', default=['train', 'valid'],
                        choices=['train', 'valid', 'test'])
    parser.add_argument('--force_rebuild', action='store_true',
                        help='Rebuild entity/query cache files even if they already exist.')
    args = parser.parse_args()

    config = _load_config(args.config, args.data_dir, args.output_dir)

    retriever_ckpt = getattr(config, 'init_checkpoint', None)
    if not retriever_ckpt:
        raise ValueError("Config must specify 'init_checkpoint' (path to stage-1 retriever).")
    if not os.path.exists(retriever_ckpt):
        raise FileNotFoundError(f"Retriever checkpoint not found: {retriever_ckpt}")

    cache_dir = getattr(config, 'cache_dir', config.output_dir)
    os.makedirs(cache_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load vocab sizes
    # -----------------------------------------------------------------------
    with open(os.path.join(config.data_dir, 'entity2id.json')) as f:
        config.num_entities  = len(json.load(f))
    with open(os.path.join(config.data_dir, 'relation2id.json')) as f:
        config.num_relations = len(json.load(f))

    # -----------------------------------------------------------------------
    # Load frozen retriever
    # -----------------------------------------------------------------------
    print(f"Loading retriever from {retriever_ckpt} ...")
    model = GWM(config).to(device)
    state = torch.load(retriever_ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # -----------------------------------------------------------------------
    # Encode all entities once and save
    # -----------------------------------------------------------------------
    entity_cache_path = os.path.join(cache_dir, 'entity_cache.pt')
    if (not args.force_rebuild) and os.path.exists(entity_cache_path):
        print(f"Entity cache already exists at {entity_cache_path}, loading ...")
        entity_cache = torch.load(entity_cache_path, map_location='cpu')
        all_t_text = entity_cache['entity_text'].to(device)
        all_t_struct = entity_cache['entity_struct'].to(device)
    else:
        print("Encoding all entities ...")
        num_workers = int(getattr(config, 'num_workers', 2))
        entity_loader = build_entity_loader(config.data_dir, batch_size=int(getattr(config, 'eval_batch_size', 512)), num_workers=num_workers)
        all_t_text, all_t_struct = encode_all_entities_as_targets(model, entity_loader, device)
        torch.save({'entity_text': all_t_text.cpu(), 'entity_struct': all_t_struct.cpu()}, entity_cache_path)
        print(f"Entity cache saved → {entity_cache_path}")

    # -----------------------------------------------------------------------
    # Build query cache per split
    # -----------------------------------------------------------------------
    topk        = int(getattr(config, 'reranker_train_topk', 30))
    batch_size  = int(getattr(config, 'batch_size', 512))
    num_workers = int(getattr(config, 'num_workers', 2))

    for split in args.splits:
        out_path = os.path.join(cache_dir, f'{split}_query_cache.pt')
        if (not args.force_rebuild) and os.path.exists(out_path):
            print(f"[{split}] Cache already exists at {out_path}, skipping.")
            continue

        print(f"[{split}] Building query cache (topk={topk}) ...")
        dataset    = GWMDataset(config.data_dir, split=split)
        collate_fn = CollateFN()
        loader     = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            drop_last=False,
        )

        cache = _build_query_cache(model, loader, all_t_text, all_t_struct, topk, device)
        torch.save(cache, out_path)
        print(f"[{split}] Cache saved ({len(cache['true_t_id'])} triples) → {out_path}")

    print("Done.")


if __name__ == '__main__':
    main()
