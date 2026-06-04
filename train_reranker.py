import os
import math
import time
import json
import argparse
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.reranker import GWMReranker
from model.reranker_dataset import RerankerCacheDataset, reranker_collate
from model.model import GWM
from model.dataset import GWMDataset, CollateFN
from utils.seed import seed_everything
from utils.eval import (
    build_entity_loader,
    compute_filtered_ranking_metrics,
    encode_all_entities_as_targets,
    load_hr_map_for_filtering,
)
from utils.early_stopping import EarlyStopping


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_serializable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_to_serializable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_serializable(x) for k, x in v.items()}
    return str(v)


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def get_config(args):
    with open(args.config, 'r') as f:
        d = yaml.safe_load(f)
    if args.data_dir:
        d['data_dir'] = args.data_dir
    if args.output_dir:
        d['output_dir'] = args.output_dir

    class Config:
        def __init__(self, cfg):
            for k, v in cfg.items():
                setattr(self, k, v)

    return Config(d)


def _lr_lambda_fn(warmup_steps, total_steps, min_lr_ratio):
    def fn(step):
        if total_steps <= 1:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        decay = max(1, total_steps - warmup_steps)
        progress = min(max((step - warmup_steps) / decay, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_reranker(args):
    config = get_config(args)
    os.makedirs(config.output_dir, exist_ok=True)

    seed = int(getattr(config, 'seed', 42))
    seed_everything(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Verify cache exists
    # ------------------------------------------------------------------
    cache_dir   = getattr(config, 'cache_dir', config.output_dir)
    train_cache = os.path.join(cache_dir, 'train_query_cache.pt')
    if not os.path.exists(train_cache):
        raise FileNotFoundError(
            f"Training cache not found: {train_cache}\n"
            "Run:  python utils/build_retriever_cache.py --config <cfg> --splits train valid"
        )

    # ------------------------------------------------------------------
    # Datasets and loaders (pure cache — no GWM forward in training loop)
    # ------------------------------------------------------------------
    print("Loading training cache ...")
    train_dataset = RerankerCacheDataset(cache_dir, split='train')
    train_loader  = DataLoader(
        train_dataset,
        batch_size=int(getattr(config, 'rr_batch_size', 256)),
        shuffle=True,
        collate_fn=reranker_collate,
        num_workers=int(getattr(config, 'num_workers', 2)),
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    print(f"Train cache: {len(train_dataset)} triples, {len(train_loader)} batches/epoch")

    has_valid = os.path.exists(os.path.join(cache_dir, 'valid_query_cache.pt'))

    # ------------------------------------------------------------------
    # Load frozen GWM retriever (used only for end-to-end validation)
    # ------------------------------------------------------------------
    retriever_ckpt = getattr(config, 'init_checkpoint', None)
    if not retriever_ckpt:
        raise ValueError("Config must specify 'init_checkpoint'.")
    if not os.path.exists(retriever_ckpt):
        raise FileNotFoundError(f"Retriever checkpoint not found: {retriever_ckpt}")

    with open(os.path.join(config.data_dir, 'entity2id.json')) as f:
        config.num_entities  = len(json.load(f))
    with open(os.path.join(config.data_dir, 'relation2id.json')) as f:
        config.num_relations = len(json.load(f))

    print(f"Loading frozen retriever from {retriever_ckpt} ...")
    gwm = GWM(config).to(device)
    gwm.load_state_dict(torch.load(retriever_ckpt, map_location=device), strict=True)
    for p in gwm.parameters():
        p.requires_grad_(False)
    gwm.eval()

    # ------------------------------------------------------------------
    # Standalone reranker
    # ------------------------------------------------------------------
    print("Initialising standalone GWMReranker ...")
    reranker = GWMReranker(config).to(device)


    valid_loader  = None
    hr_map        = None
    entity_loader = None
    if has_valid:
        print("Loading validation data ...")
        valid_dataset = GWMDataset(config.data_dir, split='valid')
        valid_loader  = DataLoader(
            valid_dataset,
            batch_size=int(getattr(config, 'rr_eval_batch_size', getattr(config, 'eval_batch_size', config.batch_size))),
            shuffle=False,
            collate_fn=CollateFN(),
            num_workers=int(getattr(config, 'num_workers', 2)),
            pin_memory=(device.type == 'cuda'),
            drop_last=False,
        )
        hr_map = load_hr_map_for_filtering(
            config.data_dir,
            preferred_ground_truth_file='ground_truth_train.json',
            fallback_splits=['train'],
        )
        entity_loader = build_entity_loader(
            data_dir=config.data_dir,
            batch_size=int(getattr(config, 'rr_candidate_batch_size',
                                   getattr(config, 'rr_eval_batch_size', getattr(config, 'eval_batch_size', 512)))),
            num_workers=int(getattr(config, 'num_workers', 2)),
        )

    # ------------------------------------------------------------------
    # Optimiser + LR schedule
    # ------------------------------------------------------------------
    base_lr      = float(config.learning_rate)
    weight_decay = float(getattr(config, 'rr_weight_decay', 0.0))
    optimizer    = torch.optim.AdamW(reranker.parameters(), lr=base_lr, weight_decay=weight_decay)
    grad_clip    = float(getattr(config, 'rr_grad_clip_norm', 1.0))
    num_epochs   = int(getattr(config, 'rr_num_epochs', 1000))
    total_steps  = max(1, num_epochs * len(train_loader))
    warmup_steps = min(int(total_steps * float(getattr(config, 'rr_warmup_ratio', 0.0))), total_steps)
    min_lr       = float(getattr(config, 'rr_min_lr', 0.0))
    min_lr_ratio = 0.0 if base_lr <= 0 else max(min_lr / base_lr, 0.0)

    scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda_fn(warmup_steps, total_steps, min_lr_ratio))

    early_stop = EarlyStopping(
        patience=int(getattr(config, 'rr_early_stopping_patience', 5)),
        mode='max',
    )

    config_record = {k: _to_serializable(v) for k, v in vars(config).items()}
    with open(os.path.join(config.output_dir, 'training_config.json'), 'w') as f:
        json.dump(config_record, f, indent=2)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("Stage-2 reranker training (cache-based) ...")
    log_path    = os.path.join(config.output_dir, 'training_log.json')
    history     = []
    history.append({
        'event': 'stage_start',
        'stage': 'reranker',
        'cache_dir': cache_dir,
    })
    best_mrr    = 0.0
    train_start = time.perf_counter()
    eval_every  = int(getattr(config, 'rr_eval_every', 1))

    for epoch in range(num_epochs):
        t0 = time.perf_counter()
        _sync(device)
        reranker.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Reranker]")
        for batch in pbar:
            rel_text    = batch['rel_text'].to(device)
            rel_struct  = batch['rel_struct'].to(device)
            cand_text   = batch['cand_text'].to(device)
            cand_struct = batch['cand_struct'].to(device)
            label       = batch['label'].to(device)

            optimizer.zero_grad()
            logits = reranker.score(rel_text, rel_struct, cand_text, cand_struct)
            logits = logits / reranker.temperature
            loss   = F.cross_entropy(logits, label)

            if not torch.isfinite(loss):
                print("Non-finite loss — skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(reranker.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        _sync(device)
        avg_train_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch+1} | train_loss={avg_train_loss:.4f} | time={time.perf_counter()-t0:.1f}s")

        epoch_log = {'epoch': epoch + 1, 'stage': 'reranker', 'train_loss': avg_train_loss}

        if valid_loader is not None and (epoch + 1) % eval_every == 0:
            reranker.eval()
            all_ent_emb = encode_all_entities_as_targets(gwm, entity_loader, device)

            val_metrics = compute_filtered_ranking_metrics(
                model=gwm,
                data_loader=valid_loader,
                all_entity_embeddings=all_ent_emb,
                hr_map=hr_map,
                device=device,
                desc="Validation",
                topk=int(getattr(config, 'eval_topk', 50)),
                rerank_topk=int(getattr(config, 'reranker_eval_topk', 100)),
                reranker=reranker,
            )

            val_mrr = val_metrics['MRR']
            print(
                f"Epoch {epoch+1} Val | "
                f"MRR={val_mrr:.4f} | MR={val_metrics['MR']:.1f} | "
                f"H@1={val_metrics['Hits@1']:.4f} | "
                f"H@3={val_metrics['Hits@3']:.4f} | "
                f"H@10={val_metrics['Hits@10']:.4f}"
            )

            epoch_log.update({
                'stage': 'reranker',
                'val_mrr':    val_mrr,
                'val_mr':     val_metrics['MR'],
                'val_hits1':  val_metrics['Hits@1'],
                'val_hits3':  val_metrics['Hits@3'],
                'val_hits10': val_metrics['Hits@10'],
            })

            if val_mrr > best_mrr:
                best_mrr = val_mrr
                torch.save(reranker.state_dict(),
                           os.path.join(config.output_dir, 'reranker_best.pt'))
                print(f"  -> New best MRR={best_mrr:.4f}, saved reranker_best.pt")

            if early_stop(val_mrr):
                print(f"\nEarly stopping at epoch {epoch+1}. Best MRR={early_stop.best_value:.4f}")
                history.append(epoch_log)
                break

        history.append(epoch_log)
        torch.save(reranker.state_dict(),
                   os.path.join(config.output_dir, 'reranker_latest.pt'))
        with open(log_path, 'w') as f:
            json.dump(history, f, indent=2)

    _sync(device)
    total_sec = time.perf_counter() - train_start
    print(f"Training complete. Total time: {total_sec:.1f}s  Best val MRR: {best_mrr:.4f}")
    history.append({'event': 'training_complete', 'stage': 'reranker', 'total_seconds': total_sec})
    with open(log_path, 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Stage-2 standalone reranker training from cached retriever outputs."
    )
    parser.add_argument('--config',     required=True, help='YAML config file')
    parser.add_argument('--data_dir',   default=None)
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    train_reranker(args)
