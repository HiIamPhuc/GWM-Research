"""
GWMReranker — standalone cross-encoder reranker trained on cached GWM retriever outputs.

Usage pattern
-------------
Stage 2 (training):
    reranker = GWMReranker(config)
    # Trained purely from cached (query, relation, candidate) embeddings.

Evaluation (end-to-end):
    gwm      = GWM(config)            # loads retriever checkpoint
    reranker = GWMReranker(config)    # loads reranker checkpoint
    # GWM retrieves top-K, reranker re-scores them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GWMReranker(nn.Module):
    """
    Relation-conditioned cross-encoder that re-scores a fixed candidate list.

    Inputs at forward time are *pre-computed, L2-normalised* embeddings produced
    by a frozen GWM retriever — no GWM components live here.

    Feature vector per candidate (6 * proj_dim):
        [q_t, c_t, q_t ⊙ c_t,  q_s, c_s, q_s ⊙ c_s]
    """

    def __init__(self, config):
        super().__init__()
        text_dim   = int(getattr(config, 'text_emb_dim'))
        struct_dim = int(getattr(config, 'struct_emb_dim'))
        proj_dim   = int(getattr(config, 'reranker_proj_dim', 64))
        hidden_dim = int(getattr(config, 'reranker_hidden_dim', 128))
        dropout    = float(getattr(config, 'reranker_dropout',
                                   getattr(config, 'dropout', 0.1)))

        self.text_proj   = nn.Linear(text_dim,   proj_dim)
        self.struct_proj = nn.Linear(struct_dim, proj_dim)

        feat_dim = proj_dim * 6
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.temperature = float(getattr(config, 'rr_temperature', 1.0))
        self.train_topk  = int(getattr(config, 'reranker_train_topk', 30))

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score(self, relation_text, relation_struct, cand_text, cand_struct):
        """
        relation_text   : (B, D_t)
        relation_struct : (B, D_s)
        cand_text       : (B, K, D_t)
        cand_struct     : (B, K, D_s)
        returns         : (B, K) raw logits
        """
        q_t = self.text_proj(relation_text).unsqueeze(1)    # (B,1,P)
        q_s = self.struct_proj(relation_struct).unsqueeze(1)
        c_t = self.text_proj(cand_text)                     # (B,K,P)
        c_s = self.struct_proj(cand_struct)

        feat = torch.cat(
            [q_t.expand_as(c_t), c_t, q_t * c_t,
             q_s.expand_as(c_s), c_s, q_s * c_s],
            dim=-1,
        )
        return self.head(feat).squeeze(-1)                  # (B, K)

    def forward(self, relation_text, relation_struct, cand_text, cand_struct):
        """Alias for score() — matches CrossModalReranker interface."""
        return self.score(relation_text, relation_struct, cand_text, cand_struct)

    # ------------------------------------------------------------------
    # Convenience: rerank from a flat entity table + index tensor
    # ------------------------------------------------------------------

    def rerank_with_indices(
        self,
        relation_text,
        relation_struct,
        all_t_text,
        all_t_struct,
        candidate_indices,
    ):
        """
        relation_text/struct : (B, D)
        all_t_text/struct    : (N, D)  — full entity table (pre-computed)
        candidate_indices    : (B, K)  — indices into the entity table
        returns              : (B, K)  scores
        """
        cand_text   = all_t_text[candidate_indices]    # (B, K, D_t)
        cand_struct = all_t_struct[candidate_indices]  # (B, K, D_s)
        return self.score(relation_text, relation_struct, cand_text, cand_struct)

    # ------------------------------------------------------------------
    # Loss (used during stage-2 training)
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        relation_text,
        relation_struct,
        cand_text,
        cand_struct,
        fused_scores,
        true_tail_local_idx,
    ):
        """
        Build top-K candidates from *fused_scores* (already computed by frozen GWM),
        guarantee the true tail is in the list, and compute cross-entropy.

        relation_text/struct   : (B, D)
        cand_text/struct       : (B, N_cand, D)  — ALL candidate embeddings for this batch
        fused_scores           : (B, N_cand)      — pre-computed retrieval scores
        true_tail_local_idx    : (B,) long         — index within N_cand of the true tail

        returns : scalar CE loss
        """
        B, N = fused_scores.shape
        k = max(1, min(self.train_topk, N))

        top_idx   = torch.topk(fused_scores, k=k, dim=1).indices      # (B, K)
        true_idx  = true_tail_local_idx.unsqueeze(1)                   # (B, 1)
        cand_idx  = torch.cat([top_idx, true_idx], dim=1)             # (B, K+1)

        # Deduplicate: if true tail already appears in top_idx, replace those
        # slots to avoid conflicting CE supervision.
        pre       = cand_idx[:, :-1]
        dup_mask  = pre.eq(true_idx)
        if dup_mask.any():
            replacement = (pre + 1) % N
            pre      = torch.where(dup_mask, replacement, pre)
            cand_idx = torch.cat([pre, true_idx], dim=1)

        # Gather embeddings for chosen candidate indices
        cand_t = cand_text[torch.arange(B, device=cand_idx.device).unsqueeze(1), cand_idx]
        cand_s = cand_struct[torch.arange(B, device=cand_idx.device).unsqueeze(1), cand_idx]

        rerank_scores = self.score(relation_text, relation_struct, cand_t, cand_s)
        rerank_scores = rerank_scores / self.temperature

        # True tail is always placed at the last position
        labels = torch.full(
            (B,), rerank_scores.size(1) - 1,
            device=rerank_scores.device, dtype=torch.long,
        )
        return F.cross_entropy(rerank_scores, labels)
