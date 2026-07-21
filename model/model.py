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
    """Basic structural baseline: [head, relation] -> LSTM -> tail retrieval."""

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

        dynamics_layers = int(getattr(config, 'dynamics_layers', 1))
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=dynamics_layers,
            batch_first=True,
        )

    def encode_query(self, h_batch, r_batch):
        head = self.struct_ent_embs(h_batch['id'])
        relation = self.struct_rel_embs(r_batch['id'])
        sequence = torch.stack([head, relation], dim=1)
        _, (hidden, _) = self.lstm(sequence)
        return F.normalize(hidden[-1], p=2, dim=-1)

    def forward(self, h_batch, r_batch):
        return self.encode_query(h_batch, r_batch)

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
        candidate_vectors=None,
    ):
        query_vectors = self.encode_query(h_batch, r_batch)
        if candidate_vectors is None:
            entity_ids = torch.arange(
                int(self.config.num_entities),
                device=query_vectors.device,
                dtype=torch.long,
            )
            candidate_vectors = self.encode_target({'id': entity_ids})

        return torch.mm(query_vectors, candidate_vectors.t()) / self.temperature
