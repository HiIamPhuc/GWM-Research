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
    """Structural LSTM with a relation-gated contextual head."""

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
        self.context_gate_scale = nn.Parameter(torch.zeros(self.embedding_dim))
        self.context_gate_bias = nn.Parameter(torch.zeros(self.embedding_dim))
        self.context_strength = nn.Parameter(torch.tensor(0.0))

        dynamics_layers = int(getattr(config, 'dynamics_layers', 1))
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=dynamics_layers,
            batch_first=True,
        )

    def _mean_context(self, context_batch, batch_size):
        context_entity_ids = context_batch['id']
        context_relation_ids = context_batch['rel_id']
        context_batch_index = context_batch['batch_index']

        if not (
            context_entity_ids.dim()
            == context_relation_ids.dim()
            == context_batch_index.dim()
            == 1
        ):
            raise ValueError("Ragged context tensors must be one-dimensional.")
        if not (
            context_entity_ids.numel()
            == context_relation_ids.numel()
            == context_batch_index.numel()
        ):
            raise ValueError("Ragged context tensors must have equal lengths.")
        if context_batch_index.numel() and (
            context_batch_index.min() < 0
            or context_batch_index.max() >= batch_size
        ):
            raise ValueError("Context batch index is outside the current batch.")

        state = self.struct_ent_embs.weight.new_zeros(
            batch_size,
            self.embedding_dim,
        )
        counts = self.struct_ent_embs.weight.new_zeros(batch_size, 1)
        if context_entity_ids.numel() == 0:
            return state

        context_entities = self.struct_ent_embs(context_entity_ids)
        context_relations = self.struct_rel_embs(context_relation_ids)
        composed_facts = context_entities * context_relations
        state.index_add_(0, context_batch_index, composed_facts)
        counts.index_add_(
            0,
            context_batch_index,
            counts.new_ones(context_batch_index.numel(), 1),
        )
        return state / counts.clamp_min(1.0)

    def _contextualize_head(self, head, relation, context):
        gate = torch.sigmoid(
            self.context_gate_scale * relation + self.context_gate_bias
        )
        strength = torch.tanh(self.context_strength)
        return head + strength * gate * context

    @torch.no_grad()
    def context_stats(self):
        relation_gates = torch.sigmoid(
            self.context_gate_scale * self.struct_rel_embs.weight
            + self.context_gate_bias
        )
        return {
            'context_strength': torch.tanh(self.context_strength).item(),
            'context_gate_mean': relation_gates.mean().item(),
            'context_gate_std': relation_gates.std(unbiased=False).item(),
        }

    def encode_query(self, h_batch, r_batch, context_batch):
        head = self.struct_ent_embs(h_batch['id'])
        relation = self.struct_rel_embs(r_batch['id'])
        context = self._mean_context(
            context_batch,
            batch_size=head.size(0),
        )
        contextual_head = self._contextualize_head(
            head,
            relation,
            context,
        )
        sequence = torch.stack([contextual_head, relation], dim=1)
        _, (hidden, _) = self.lstm(sequence)
        return F.normalize(hidden[-1], p=2, dim=-1)

    def forward(self, h_batch, r_batch, context_batch):
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
        context_batch,
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
