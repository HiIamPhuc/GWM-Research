import torch
import torch.nn as nn
import torch.nn.functional as F


def load_embedding_cache(path):
    cache = torch.load(path, map_location='cpu')
    return cache['embeddings'] if isinstance(cache, dict) else cache


class ContextAggregator(nn.Module):
    """Mean-pool relation-composed context facts around the head entity."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)

    def _aggregate(self, messages, batch_index, batch_size, reference):
        pooled = torch.zeros_like(reference)
        pooled.index_add_(0, batch_index, messages)

        counts = reference.new_zeros(batch_size, 1)
        counts.index_add_(0, batch_index, reference.new_ones(messages.size(0), 1))
        return pooled / counts.clamp_min(1)

    def forward(self, head, context_entities, context_relations, batch_index):
        facts = context_entities * context_relations
        context = self._aggregate(facts, batch_index, head.size(0), head)
        return self.norm(head + context)


class MLPAdapter(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.fc1(self.norm(x))
        x = self.dropout(self.act(x))
        return residual + self.fc2(x)


class GatedFusion(nn.Module):
    def __init__(self, text_dim, struct_dim, fusion_dim, dropout):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.struct_projection = nn.Linear(struct_dim, fusion_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, structure):
        text = self.text_projection(text)
        structure = self.struct_projection(structure)
        gate = self.gate(torch.cat([text, structure], dim=-1))
        fused = gate * text + (1 - gate) * structure
        return self.output_norm(self.dropout(fused)), gate


class RoleAwareTransition(nn.Module):
    """Run a role-ordered transition and expand it into successor modes."""

    def __init__(self, hidden_dim, num_layers, num_modes, dropout):
        super().__init__()
        self.num_modes = num_modes
        self.role_embeddings = nn.Parameter(torch.empty(2, hidden_dim))
        self.mode_embeddings = nn.Parameter(torch.empty(num_modes, hidden_dim))
        self.h0_projection = nn.Linear(hidden_dim, hidden_dim)
        self.c0_projection = nn.Linear(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.film = nn.Linear(hidden_dim, num_modes * hidden_dim * 2)
        self.mixture = nn.Linear(hidden_dim * 2, num_modes)
        self.mode_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.role_embeddings, std=0.02)
        nn.init.normal_(self.mode_embeddings, std=0.02)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, world_state, head, relation):
        h0 = torch.tanh(self.h0_projection(world_state))
        c0 = torch.tanh(self.c0_projection(world_state))
        h0 = h0.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
        c0 = c0.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
        sequence = torch.stack(
            [head + self.role_embeddings[0], relation + self.role_embeddings[1]],
            dim=1,
        )
        _, (hidden, _) = self.lstm(sequence, (h0, c0))
        base_state = hidden[-1]

        gamma, beta = self.film(relation).view(
            relation.size(0), self.num_modes, 2, relation.size(1)
        ).unbind(dim=2)
        modes = (1 + torch.tanh(gamma)) * base_state.unsqueeze(1)
        modes = self.mode_norm(modes + beta + self.mode_embeddings.unsqueeze(0))
        mixture_logits = self.mixture(torch.cat([base_state, relation], dim=-1))
        return modes, mixture_logits


class GWM(nn.Module):
    requires_text_embeddings = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.temperature = config.temperature
        self.fusion_dim = config.fusion_dim
        self.num_successor_modes = config.num_successor_modes

        self.text_ent_embs = nn.Embedding(config.num_entities, config.text_emb_dim)
        self.text_rel_embs = nn.Embedding(config.num_relations, config.text_emb_dim)
        self.struct_ent_embs = nn.Embedding(config.num_entities, config.struct_emb_dim)
        self.struct_rel_embs = nn.Embedding(config.num_relations, config.struct_emb_dim)

        self.text_adapter = MLPAdapter(config.text_emb_dim, config.adapter_dropout)
        self.struct_adapter = MLPAdapter(config.struct_emb_dim, config.adapter_dropout)
        self.entity_fusion = GatedFusion(
            config.text_emb_dim,
            config.struct_emb_dim,
            config.fusion_dim,
            config.dropout,
        )
        self.relation_fusion = GatedFusion(
            config.text_emb_dim,
            config.struct_emb_dim,
            config.fusion_dim,
            config.dropout,
        )

        self.fused_context_aggregator = ContextAggregator(config.fusion_dim)
        self.transition = RoleAwareTransition(
            config.fusion_dim,
            config.dynamics_layers,
            config.num_successor_modes,
            config.dropout,
        )
        self.fused_output_projection = nn.Linear(config.fusion_dim, config.fusion_dim)
        self._last_gate_stats = {}

    def load_text_embeddings(self, entity_path, relation_path, freeze=True):
        self.text_ent_embs.weight.data.copy_(load_embedding_cache(entity_path))
        self.text_rel_embs.weight.data.copy_(load_embedding_cache(relation_path))
        self.text_ent_embs.weight.requires_grad = not freeze
        self.text_rel_embs.weight.requires_grad = not freeze

    def _fuse_entity(self, ids):
        text = self.text_adapter(self.text_ent_embs(ids))
        structure = self.struct_adapter(self.struct_ent_embs(ids))
        return self.entity_fusion(text, structure)

    def _fuse_relation(self, ids):
        text = self.text_adapter(self.text_rel_embs(ids))
        structure = self.struct_adapter(self.struct_rel_embs(ids))
        return self.relation_fusion(text, structure)

    def _record_gates(self, entity_gate=None, relation_gate=None):
        if entity_gate is not None and entity_gate.numel():
            self._last_gate_stats['entity_gate'] = entity_gate.detach().mean().item()
        if relation_gate is not None and relation_gate.numel():
            self._last_gate_stats['relation_gate'] = relation_gate.detach().mean().item()

    def pop_gate_stats(self):
        stats = self._last_gate_stats
        self._last_gate_stats = {}
        return stats

    def forward(self, h_batch, r_batch, context_batch):
        self._last_gate_stats = {}
        head, head_gate = self._fuse_entity(h_batch['id'])
        relation, relation_gate = self._fuse_relation(r_batch['id'])
        context_entities, context_entity_gate = self._fuse_entity(context_batch['id'])
        context_relations, context_relation_gate = self._fuse_relation(
            context_batch['rel_id']
        )

        self._record_gates(
            torch.cat([head_gate, context_entity_gate]),
            torch.cat([relation_gate, context_relation_gate]),
        )
        world_state = self.fused_context_aggregator(
            head,
            context_entities,
            context_relations,
            context_batch['batch_index'],
        )
        modes, mixture_logits = self.transition(world_state, head, relation)
        mixture_weights = F.softmax(mixture_logits, dim=-1)
        self._last_gate_stats['successor_entropy'] = (
            -(mixture_weights * mixture_weights.clamp_min(1e-9).log()).sum(dim=-1)
            .mean()
            .detach()
            .item()
        )
        self._last_gate_stats['successor_top_weight'] = (
            mixture_weights.max(dim=-1).values.mean().detach().item()
        )
        return {
            'modes': F.normalize(self.fused_output_projection(modes), dim=-1),
            'mixture_logits': mixture_logits,
        }

    def encode_target(self, t_batch):
        target, gate = self._fuse_entity(t_batch['id'])
        self._record_gates(entity_gate=gate)
        return F.normalize(self.fused_output_projection(target), dim=-1)

    @staticmethod
    def _filtered_in_batch_contrastive_loss(scores, truth_mask):
        diagonal = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(truth_mask & ~diagonal, float('-inf'))
        labels = torch.arange(scores.size(0), device=scores.device)
        return F.cross_entropy(scores, labels, reduction='none')

    def compute_loss(self, query_vectors, target_vectors, truth_mask):
        scores = self.score_candidates(query_vectors, target_vectors)
        loss = self._filtered_in_batch_contrastive_loss(scores, truth_mask).mean()
        return loss, scores

    def score_candidates(self, query, candidates):
        mode_scores = torch.einsum('bkd,nd->bkn', query['modes'], candidates)
        log_weights = F.log_softmax(query['mixture_logits'], dim=-1).unsqueeze(-1)
        return torch.logsumexp(log_weights + mode_scores / self.temperature, dim=1)
