import torch
import torch.nn as nn
import torch.nn.functional as F


def filtered_in_batch_contrastive_loss(scores, truth_mask=None):
    """Single-positive InfoNCE that ignores other known in-batch truths."""

    batch_size = scores.size(0)
    if scores.dim() != 2 or scores.size(1) != batch_size:
        raise ValueError("In-batch contrastive scores must have shape (B, B).")

    if truth_mask is None:
        truth_mask = torch.eye(
            batch_size,
            dtype=torch.bool,
            device=scores.device,
        )
    elif truth_mask.shape != scores.shape:
        raise ValueError("truth_mask must have the same shape as scores.")
    else:
        truth_mask = truth_mask.to(device=scores.device, dtype=torch.bool)

    diagonal = torch.eye(batch_size, dtype=torch.bool, device=scores.device)
    denominator_mask = (~truth_mask) | diagonal
    filtered_scores = scores.masked_fill(~denominator_mask, float('-inf'))
    labels = torch.arange(batch_size, device=scores.device)
    return F.cross_entropy(filtered_scores, labels, reduction='none')


class GWM(nn.Module):
    """Context-free structural model with a two-token Transformer transition."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = int(getattr(config, 'struct_emb_dim'))
        self.temperature = float(getattr(config, 'temperature', 0.07))

        self.struct_ent_embs = nn.Embedding(
            int(config.num_entities),
            self.embedding_dim,
        )
        self.struct_rel_embs = nn.Embedding(
            int(config.num_relations),
            self.embedding_dim,
        )
        transformer_heads = int(getattr(config, 'transformer_heads', 4))
        if self.embedding_dim % transformer_heads != 0:
            raise ValueError(
                "struct_emb_dim must be divisible by transformer_heads. "
                f"Got {self.embedding_dim} and {transformer_heads}."
            )

        transformer_layers = int(getattr(config, 'transformer_layers', 1))
        ffn_multiplier = int(getattr(config, 'transformer_ffn_multiplier', 2))
        transformer_dropout = float(getattr(config, 'transformer_dropout', 0.1))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=transformer_heads,
            dim_feedforward=ffn_multiplier * self.embedding_dim,
            dropout=transformer_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )
        self.token_roles = nn.Parameter(torch.empty(2, self.embedding_dim))
        self.transition_projection = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
        )
        self.output_norm = nn.LayerNorm(self.embedding_dim)
        self.register_buffer(
            'transition_mask',
            torch.tensor([[False, True], [False, False]]),
            persistent=False,
        )
        nn.init.normal_(self.token_roles, mean=0.0, std=0.02)

    def encode_query(self, h_batch, r_batch, context_batch=None):
        head = self.struct_ent_embs(h_batch['id'])
        relation = self.struct_rel_embs(r_batch['id'])
        tokens = torch.stack([head, relation], dim=1)
        tokens = tokens + self.token_roles.unsqueeze(0)
        encoded = self.transformer(tokens, mask=self.transition_mask)
        transition_delta = self.transition_projection(encoded[:, 1])
        query = self.output_norm(head + transition_delta)
        return F.normalize(query, p=2, dim=-1)

    def forward(self, h_batch, r_batch, context_batch=None):
        return self.encode_query(h_batch, r_batch, context_batch)

    def encode_target(self, t_batch):
        target = self.struct_ent_embs(t_batch['id'])
        return F.normalize(target, p=2, dim=-1)

    def compute_loss(self, query_vectors, target_vectors, truth_mask=None):
        scores = torch.mm(query_vectors, target_vectors.t()) / self.temperature
        loss = filtered_in_batch_contrastive_loss(
            scores,
            truth_mask=truth_mask,
        ).mean()
        return loss, scores

    def score_all_entities(
        self,
        h_batch,
        r_batch,
        context_batch=None,
        candidate_vectors=None,
    ):
        query_vectors = self.encode_query(h_batch, r_batch, context_batch)
        if candidate_vectors is None:
            entity_ids = torch.arange(
                int(self.config.num_entities),
                device=query_vectors.device,
                dtype=torch.long,
            )
            candidate_vectors = self.encode_target({'id': entity_ids})

        return torch.mm(query_vectors, candidate_vectors.t()) / self.temperature
