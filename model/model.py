import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ContextAggregator(nn.Module):
    """Pool relation-composed facts, add the head residual, and normalize."""

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
        """
        head_feat: (B, H)
        nbr_entity_feat: (E, H)
        nbr_relation_feat: (E, H)
        nbr_batch_index: (E,) long, edge -> head index in batch
        """
        
        composed_facts = nbr_entity_feat * nbr_relation_feat
        agg = self._aggregate(
            composed_facts,
            nbr_batch_index,
            head_feat.size(0),
            head_feat,
        )
        return self.norm(head_feat + agg)


class MLPAdapter(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + x


class GatedFusion(nn.Module):
    """Project two modalities into one space and combine them feature-wise."""

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


class ConvTransEDecoder(nn.Module):
    """ConvTransE decoder shared with the temporal GWM architecture."""

    def __init__(self, embedding_dim, dropout=0.0, channels=50, kernel_size=3):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.bn0 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(self.embedding_dim)
        self.conv = nn.Conv1d(
            2,
            channels,
            kernel_size,
            stride=1,
            padding=int(math.floor(kernel_size / 2)),
        )
        self.fc = nn.Linear(self.embedding_dim * channels, self.embedding_dim)

    def forward(self, query_vectors, relation_vectors, candidate_vectors):
        batch_size = query_vectors.size(0)
        stacked_inputs = torch.stack([query_vectors, relation_vectors], dim=1)
        x = self.bn0(stacked_inputs)
        x = self.dropout1(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        x = self.dropout3(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        return torch.mm(x, torch.tanh(candidate_vectors).t())


class GWM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = float(getattr(config, 'dropout'))
        self.adapter_dropout = float(getattr(config, 'adapter_dropout', self.dropout))

        # 1. Text Components (Entity/Relation Embeddings)
        self.text_emb_dim = int(getattr(config, 'text_emb_dim'))
        self.text_ent_embs = nn.Embedding(config.num_entities, self.text_emb_dim)
        self.text_rel_embs = nn.Embedding(config.num_relations, self.text_emb_dim)

        self.text_adapter = MLPAdapter(self.text_emb_dim,
            int(getattr(config, 'text_adapter_dim')),
            dropout=self.adapter_dropout
            )

        # 2. Structural Components (Entity/Relation Embeddings)
        self.struct_emb_dim = int(getattr(config, 'struct_emb_dim'))
        self.struct_ent_embs = nn.Embedding(config.num_entities, self.struct_emb_dim)
        self.struct_rel_embs = nn.Embedding(config.num_relations, self.struct_emb_dim)

        self.struct_adapter = MLPAdapter(
            self.struct_emb_dim, 
            int(getattr(config, 'struct_adapter_dim')),
            dropout=self.adapter_dropout
            )

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
        self.decoder_name = str(getattr(config, 'decoder', 'dot')).lower()
        if self.decoder_name == 'convtranse':
            self.decoder = ConvTransEDecoder(
                embedding_dim=self.fusion_dim,
                dropout=self.dropout,
                channels=int(getattr(config, 'convtranse_channels', 50)),
                kernel_size=int(getattr(config, 'convtranse_kernel_size', 3)),
            )
        elif self.decoder_name in {'dot', 'contrastive'}:
            self.decoder = None
        else:
            raise ValueError(f"Unsupported decoder: {self.decoder_name}")
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
        context_entity_ids = context_batch['id']
        context_relation_ids = context_batch.get('rel_id')
        context_batch_index = context_batch.get('batch_index')
        if context_relation_ids is None or context_batch_index is None:
            raise ValueError(
                "context_batch requires 'id', 'rel_id', and 'batch_index'."
            )
        if (
            context_entity_ids.dim() != 1
            or context_relation_ids.dim() != 1
            or context_batch_index.dim() != 1
        ):
            raise ValueError("Ragged context tensors must all be one-dimensional.")
        if not (
            context_entity_ids.numel()
            == context_relation_ids.numel()
            == context_batch_index.numel()
        ):
            raise ValueError("Ragged context tensors must have equal lengths.")
        return context_entity_ids, context_relation_ids, context_batch_index

    # def _run_dynamics(self, world_state, head_emb, relation_emb, mixer, lstm, h0_proj, c0_proj):
    def _run_dynamics(self, world_state, head_emb, relation_emb, lstm, h0_proj, c0_proj):
        """
        Run recurrent dynamics over a sequence of steps.

        world_state: (B, fusion_dim) used to initialise h0/c0
        head_emb: (B, D) head embedding for this path
        relation_emb: (B, D_rel) relation embedding for this path
        """
        h_0 = torch.tanh(h0_proj(world_state))
        c_0 = torch.tanh(c0_proj(world_state))

        # Prepare initial LSTM states
        num_layers = lstm.num_layers
        h_0_lstm = h_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
        c_0_lstm = c_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous()

        # Run LSTM over the sequence
        _, (h_n, _) = lstm(torch.stack([head_emb, relation_emb], dim=1), (h_0_lstm, c_0_lstm))
        query_vector = h_n[-1]
        return query_vector

    def _load_text_embedding_tensor(self, source, expected_rows, expected_dim, name):
        if isinstance(source, str):
            loaded = torch.load(source, map_location='cpu')
        elif torch.is_tensor(source):
            loaded = source.detach().cpu()
        else:
            raise TypeError(f"Unsupported {name} cache source: {type(source)}")

        if isinstance(loaded, dict):
            if 'embeddings' in loaded:
                loaded = loaded['embeddings']
            elif 'tensor' in loaded:
                loaded = loaded['tensor']
            else:
                raise ValueError(f"{name} cache dict must contain 'embeddings' or 'tensor'.")

        if not torch.is_tensor(loaded):
            raise TypeError(f"{name} cache must resolve to a torch.Tensor.")

        loaded = loaded.float().contiguous()
        if loaded.dim() != 2:
            raise ValueError(f"{name} cache must be rank-2. Got shape {tuple(loaded.shape)}")
        if loaded.size(0) != expected_rows:
            raise ValueError(
                f"{name} cache row count mismatch. Expected {expected_rows}, got {loaded.size(0)}"
            )
        if loaded.size(1) != expected_dim:
            raise ValueError(
                f"{name} cache dimension mismatch. Expected {expected_dim}, got {loaded.size(1)}"
            )
        return loaded      

    def load_text_embeddings(self, entity_source, relation_source, freeze=True):
        entity_cache = self._load_text_embedding_tensor(
            source=entity_source,
            expected_rows=self.text_ent_embs.num_embeddings,
            expected_dim=self.text_emb_dim,
            name='text_entity',
        )
        relation_cache = self._load_text_embedding_tensor(
            source=relation_source,
            expected_rows=self.text_rel_embs.num_embeddings,
            expected_dim=self.text_emb_dim,
            name='text_relation',
        )

        if entity_cache.size(1) != relation_cache.size(1):
            raise ValueError(
                "text_entity and text_relation embeddings must share the same embedding dimension. "
                f"Got {entity_cache.size(1)} and {relation_cache.size(1)}"
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
        if scores.dim() != 2 or scores.size(1) != batch_size:
            raise ValueError(
                "In-batch contrastive scores must have shape (B, B)."
            )

        if truth_mask is None:
            truth_mask = torch.eye(
                batch_size, dtype=torch.bool, device=scores.device
            )
        else:
            if truth_mask.shape != scores.shape:
                raise ValueError(
                    "truth_mask must have the same shape as scores."
                )
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

    def encode_query(self, h_batch, r_batch, context_batch):
        self.reset_gate_stats()
        h_text = self.text_adapter(self.text_ent_embs(h_batch['id']))
        r_text = self.text_adapter(self.text_rel_embs(r_batch['id']))
        h_struct = self.struct_adapter(self.struct_ent_embs(h_batch['id']))
        r_struct = self.struct_adapter(self.struct_rel_embs(r_batch['id']))
        h_fused, h_gate = self.entity_fusion(h_text, h_struct)
        r_fused, r_gate = self.relation_fusion(r_text, r_struct)
        self._record_gate_stats('entity_gate', h_gate)
        self._record_gate_stats('relation_gate', r_gate)

        flat_context_entity_ids, flat_context_relation_ids, context_batch_index = self._prepare_context_batch(context_batch)
        ctx_ent_text = self.text_adapter(self.text_ent_embs(flat_context_entity_ids))
        ctx_rel_text = self.text_adapter(self.text_rel_embs(flat_context_relation_ids))
        ctx_ent_struct = self.struct_adapter(self.struct_ent_embs(flat_context_entity_ids))
        ctx_rel_struct = self.struct_adapter(self.struct_rel_embs(flat_context_relation_ids))
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
            self.fused_lstm,
            self.fused_h0_projection,
            self.fused_c0_projection,
        )
        query = F.normalize(self.fused_output_projection(query), p=2, dim=1)
        return query, r_fused

    def forward(self, h_batch, r_batch, context_batch):
        query, _ = self.encode_query(h_batch, r_batch, context_batch)
        return query

    def encode_target(self, t_batch):
        t_text = self.text_adapter(self.text_ent_embs(t_batch['id']))
        t_struct = self.struct_adapter(self.struct_ent_embs(t_batch['id']))
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

    def score_all_entities(
        self,
        h_batch,
        r_batch,
        context_batch,
        candidate_vectors=None,
    ):
        query_vectors, relation_vectors = self.encode_query(
            h_batch,
            r_batch,
            context_batch,
        )
        if candidate_vectors is None:
            entity_ids = torch.arange(
                int(self.config.num_entities),
                device=query_vectors.device,
                dtype=torch.long,
            )
            candidate_vectors = self.encode_target({'id': entity_ids})

        if self.decoder_name == 'convtranse':
            return self.decoder(
                query_vectors,
                relation_vectors,
                candidate_vectors,
            )
        return torch.mm(query_vectors, candidate_vectors.t()) / self.temperature

    @staticmethod
    def compute_full_softmax_loss(scores, target_ids, truth_mask=None):
        if scores.dim() != 2:
            raise ValueError("Full-softmax scores must have shape (B, |E|).")

        target_ids = torch.as_tensor(
            target_ids, dtype=torch.long, device=scores.device
        ).reshape(-1)
        if target_ids.numel() != scores.size(0):
            raise ValueError("target_ids must contain one entity ID per score row.")

        if truth_mask is not None:
            if truth_mask.shape != scores.shape:
                raise ValueError("truth_mask must have the same shape as scores.")
            truth_mask = truth_mask.to(device=scores.device, dtype=torch.bool)

            # Other known truths are ignored, while the sampled target remains
            # in the softmax denominator and receives the positive gradient.
            ignored_truths = truth_mask.clone()
            ignored_truths.scatter_(1, target_ids.unsqueeze(1), False)
            scores = scores.masked_fill(ignored_truths, float('-inf'))

        return F.cross_entropy(scores, target_ids)
