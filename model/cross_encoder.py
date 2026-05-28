"""Cross-encoder reranker utilities for KG tail reranking.

Provides a small `CrossEncoderRanker` wrapper around a Hugging Face encoder,
scoring helpers, and a minimal training loop helper suitable for prototyping
in the notebook or a lightweight training script.
"""
from typing import List, Tuple, Iterable, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CrossEncoderRanker(nn.Module):
    """Simple cross-encoder that scores (query, candidate) text pairs.

    The encoder can be any Hugging Face model that returns `last_hidden_state`
    and optionally `pooler_output`. The linear `scorer` projects the pooled
    representation to a single logit.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", freeze_base: bool = False):
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = getattr(self.encoder.config, "hidden_size", None)
        if hidden is None:
            raise ValueError("Could not determine encoder hidden size")
        self.scorer = nn.Linear(hidden, 1)
        if freeze_base:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # Prefer pooler_output when available, else use CLS token
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]
        logits = self.scorer(pooled).squeeze(-1)
        return logits


def get_tokenizer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def _to_device(batch: dict, device: Optional[torch.device]):
    if device is None:
        return batch
    return {k: v.to(device) for k, v in batch.items()}


def score_pairs(
    model: CrossEncoderRanker,
    tokenizer: AutoTokenizer,
    pairs: Iterable[Tuple[str, str]],
    batch_size: int = 64,
    device: Optional[torch.device] = None,
    max_length: int = 128,
) -> List[float]:
    """Score an iterable of (query_text, candidate_text) pairs and return logits.

    The function batches tokenization and runs the model in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    scores: List[float] = []
    pairs = list(pairs)
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            queries = [q for q, c in batch_pairs]
            cands = [c for q, c in batch_pairs]
            enc = tokenizer(queries, cands, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            enc = _to_device(enc, device)
            logits = model(enc["input_ids"], enc["attention_mask"])  # (B,)
            scores.extend(logits.detach().cpu().tolist())
    return scores


def train_epoch(
    model: CrossEncoderRanker,
    tokenizer: AutoTokenizer,
    dataloader,  # yields (queries:list[str], cands:list[str], labels:Tensor)
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device] = None,
    max_length: int = 128,
    criterion: Optional[nn.Module] = None,
) -> float:
    """Train for one epoch over a dataloader that yields batches of text pairs.

    Expects each batch to be (queries, cands, labels) where queries and cands
    are lists of strings of equal length and labels is a tensor of 0/1s.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    model.train()
    total_loss = 0.0
    n = 0
    for batch in dataloader:
        queries, cands, labels = batch
        enc = tokenizer(queries, cands, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = _to_device(enc, device)
        labels = labels.to(device).float()
        logits = model(enc["input_ids"], enc["attention_mask"])  # (B,)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        n += labels.size(0)
    return total_loss / max(1, n)


__all__ = ["CrossEncoderRanker", "get_tokenizer", "score_pairs", "train_epoch"]
