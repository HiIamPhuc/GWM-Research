import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextAggregator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)

    def _aggregate(self, messages, batch_index, batch_size, reference):
        aggregated = torch.zeros_like(reference)
        if messages.numel() == 0:
            return aggregated

        aggregated.index_add_(0, batch_index, messages)
        counts = torch.zeros(batch_size, 1, device=reference.device, dtype=reference.dtype)
        counts.index_add_(0, batch_index, torch.ones(messages.size(0), 1, device=reference.device, dtype=reference.dtype))

        return aggregated / counts.clamp_min(1.0)

    def forward(self, head_feat, nbr_entity_feat, nbr_relation_feat, nbr_batch_index):
        composed_facts = nbr_entity_feat * nbr_relation_feat
        agg = self._aggregate(
            composed_facts,
            nbr_batch_index,
            head_feat.size(0),
            head_feat,
        )
        return self.norm(head_feat + agg)


class GatedFusion(nn.Module):
    def __init__(self, text_dim, struct_dim, fusion_dim, dropout=0.0):
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

    def forward(self, text_features, struct_features):
        text_projected = self.text_projection(text_features)
        struct_projected = self.struct_projection(struct_features)
        gate = self.gate(torch.cat([text_projected, struct_projected], dim=-1))
        fused = gate * text_projected + (1.0 - gate) * struct_projected
        fused = self.dropout(fused)
        return self.output_norm(fused), gate


class GWM(nn.Module):
    requires_text_embeddings = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = float(getattr(config, 'dropout'))

        # 1. Text Components (Entity/Relation Embeddings)
        self.text_emb_dim = int(getattr(config, 'text_emb_dim'))
        self.text_ent_embs = nn.Embedding(config.num_entities, self.text_emb_dim)
        self.text_rel_embs = nn.Embedding(config.num_relations, self.text_emb_dim)

        # 2. Structural Components (Entity/Relation Embeddings)
        self.struct_emb_dim = int(getattr(config, 'struct_emb_dim'))
        self.struct_ent_embs = nn.Embedding(config.num_entities, self.struct_emb_dim)
        self.struct_rel_embs = nn.Embedding(config.num_relations, self.struct_emb_dim)

        # 3. Early Fusion and Shared Dynamics
        self.fusion_dim = int(getattr(config, 'fusion_dim'))
        self.entity_fusion = GatedFusion(
            self.text_emb_dim,
            self.struct_emb_dim,
            self.fusion_dim,
            dropout=self.dropout,
        )
        self.relation_fusion = GatedFusion(
            self.text_emb_dim,
            self.struct_emb_dim,
            self.fusion_dim,
            dropout=self.dropout,
        )

        self.fused_context_aggregator = ContextAggregator(hidden_dim=self.fusion_dim)
        self.fused_h0_projection = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.fused_c0_projection = nn.Linear(self.fusion_dim, self.fusion_dim)

        dynamics_layers = int(getattr(config, 'dynamics_layers', 1))
        self.fused_lstm = nn.LSTM(
            input_size=self.fusion_dim,
            hidden_size=self.fusion_dim,
            num_layers=dynamics_layers,
            batch_first=True,
            dropout=self.dropout if dynamics_layers > 1 else 0.0,
        )
        self.fused_output_projection = nn.Linear(
            self.fusion_dim,
            self.fusion_dim,
        )

        self.temperature = float(getattr(config, 'temperature'))
        self._last_gate_stats = {}

    def reset_gate_stats(self):
        self._last_gate_stats = {}

    def _record_gate_stats(self, name, gate):
        if gate.numel() == 0:
            return

        self._last_gate_stats[name] = gate.detach().float().mean().item()

    def pop_gate_stats(self):
        stats = dict(self._last_gate_stats)
        self.reset_gate_stats()
        return stats
        
    def _prepare_context_batch(self, context_batch):
        return (
            context_batch['id'],
            context_batch['rel_id'],
            context_batch['batch_index'],
        )

    # def _run_dynamics(self, world_state, head_emb, relation_emb, mixer, lstm, h0_proj, c0_proj):
    def _run_dynamics(self, world_state, head_emb, relation_emb):
        h_0 = torch.tanh(self.fused_h0_projection(world_state))
        c_0 = torch.tanh(self.fused_c0_projection(world_state))

        # Prepare initial LSTM states
        num_layers = self.fused_lstm.num_layers
        h_0_lstm = h_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
        c_0_lstm = c_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous()

        # Run LSTM over the sequence
        _, (h_n, _) = self.fused_lstm(
            torch.stack([head_emb, relation_emb], dim=1),
            (h_0_lstm, c_0_lstm),
        )
        query_vector = h_n[-1]
        return query_vector

    def _load_text_embedding_tensor(self, source):
        if isinstance(source, str):
            loaded = torch.load(source, map_location='cpu')
        else:
            loaded = source.detach().cpu()

        if isinstance(loaded, dict):
            loaded = loaded['embeddings']
        return loaded.float().contiguous()

    def load_text_embeddings(self, entity_source, relation_source, freeze=True):
        entity_cache = self._load_text_embedding_tensor(
            source=entity_source,
        )
        relation_cache = self._load_text_embedding_tensor(
            source=relation_source,
        )

        self.text_ent_embs.weight.data.copy_(entity_cache)
        self.text_rel_embs.weight.data.copy_(relation_cache)

        if freeze:
            self.text_ent_embs.weight.requires_grad = False
            self.text_rel_embs.weight.requires_grad = False

    @staticmethod
    def _filtered_in_batch_contrastive_loss(
        scores,
        truth_mask=None,
    ):
        batch_size = scores.size(0)
        if truth_mask is None:
            truth_mask = torch.eye(
                batch_size, dtype=torch.bool, device=scores.device
            )
        else:
            truth_mask = truth_mask.to(
                device=scores.device, dtype=torch.bool
            )

        diagonal = torch.eye(
            batch_size, dtype=torch.bool, device=scores.device
        )
        # Keep the sampled diagonal target and all false candidates. Other
        # known training truths are ignored instead of treated as negatives.
        denominator_mask = (~truth_mask) | diagonal
        filtered_scores = scores.masked_fill(
            ~denominator_mask, float('-inf')
        )
        labels = torch.arange(batch_size, device=scores.device)
        return F.cross_entropy(
            filtered_scores,
            labels,
            reduction='none',
        )

    def forward(self, h_batch, r_batch, context_batch):
        self.reset_gate_stats()
        h_text = self.text_ent_embs(h_batch['id'])
        r_text = self.text_rel_embs(r_batch['id'])
        h_struct = self.struct_ent_embs(h_batch['id'])
        r_struct = self.struct_rel_embs(r_batch['id'])
        h_fused, h_gate = self.entity_fusion(h_text, h_struct)
        r_fused, r_gate = self.relation_fusion(r_text, r_struct)
        self._record_gate_stats('entity_gate', h_gate)
        self._record_gate_stats('relation_gate', r_gate)

        flat_context_entity_ids, flat_context_relation_ids, context_batch_index = self._prepare_context_batch(context_batch)
        ctx_ent_text = self.text_ent_embs(flat_context_entity_ids)
        ctx_rel_text = self.text_rel_embs(flat_context_relation_ids)
        ctx_ent_struct = self.struct_ent_embs(flat_context_entity_ids)
        ctx_rel_struct = self.struct_rel_embs(flat_context_relation_ids)
        ctx_ent_fused, ctx_ent_gate = self.entity_fusion(ctx_ent_text, ctx_ent_struct)
        ctx_rel_fused, ctx_rel_gate = self.relation_fusion(ctx_rel_text, ctx_rel_struct)
        self._record_gate_stats('entity_gate', ctx_ent_gate)
        self._record_gate_stats('relation_gate', ctx_rel_gate)

        world_state = self.fused_context_aggregator(
            head_feat=h_fused,
            nbr_entity_feat=ctx_ent_fused,
            nbr_relation_feat=ctx_rel_fused,
            nbr_batch_index=context_batch_index,
        )
        query = self._run_dynamics(
            world_state,
            h_fused,
            r_fused,
        )
        return F.normalize(self.fused_output_projection(query), p=2, dim=1)

    def encode_target(self, t_batch):
        t_text = self.text_ent_embs(t_batch['id'])
        t_struct = self.struct_ent_embs(t_batch['id'])
        t_fused, t_gate = self.entity_fusion(t_text, t_struct)
        self._record_gate_stats('entity_gate', t_gate)
        return F.normalize(
            self.fused_output_projection(t_fused),
            p=2,
            dim=1,
        )

    def compute_loss(
        self,
        query_vectors,
        target_vectors,
        truth_mask=None,
    ):
        scores = torch.mm(query_vectors, target_vectors.t()) / self.temperature
        loss = self._filtered_in_batch_contrastive_loss(
            scores,
            truth_mask=truth_mask,
        ).mean()
        return loss, scores
