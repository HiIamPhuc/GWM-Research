import torch
import torch.nn as nn
import torch.nn.functional as F


class CompGCN(nn.Module):
    """A lightweight CompGCN for head-centric aggregation."""

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lin_self = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, head_feat, nbr_entity_feat, nbr_relation_feat, nbr_batch_index):
        """
        head_feat: (B, H)
        nbr_entity_feat: (E, H)
        nbr_relation_feat: (E, H)
        nbr_batch_index: (E,) long, edge -> head index in batch
        """
        B = head_feat.size(0)
        composed = nbr_entity_feat * nbr_relation_feat  # (E, H)
        msg = self.lin_msg(composed)  # (E, H)

        agg = torch.zeros_like(head_feat)
        if msg.numel() > 0:
            agg.index_add_(0, nbr_batch_index, msg)

        denom = torch.zeros(B, 1, device=head_feat.device, dtype=head_feat.dtype)
        if msg.numel() > 0:
            ones = torch.ones(msg.size(0), 1, device=head_feat.device, dtype=head_feat.dtype)
            denom.index_add_(0, nbr_batch_index, ones)
        denom = denom.clamp(min=1.0)
        agg = agg / denom

        self_state = self.lin_self(head_feat)
        neighbor_delta = self.dropout(self.lin_out(agg))
        return self.norm(self_state + neighbor_delta)


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


class DynamicsMixer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.mixer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, head_emb, relation_emb):
        return self.mixer(torch.cat([head_emb, relation_emb], dim=-1))


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
        return self.output_norm(self.dropout(fused)), gate


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
        if self.fusion_dim <= 0:
            raise ValueError("fusion_dim must be a positive integer.")
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

        self.fused_compgcn = CompGCN(
            hidden_dim=self.fusion_dim,
            dropout=self.dropout,
        )
        self.fused_h0_projection = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.fused_c0_projection = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.fused_dynamics_mixer = DynamicsMixer(
            self.fusion_dim,
            dropout=self.dropout,
        )

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
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")

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

    def _run_dynamics(self, world_state, head_emb, relation_emb, mixer, lstm, h0_proj, c0_proj):
        """
        Run recurrent dynamics over a sequence of steps.

        world_state: (B, compgcn_dim) used to initialise h0/c0
        head_emb: (B, D) head embedding for this path
        relation_emb: (B, D_rel) relation embedding for this path
        """
        h_0 = torch.tanh(h0_proj(world_state))
        c_0 = torch.tanh(c0_proj(world_state))

        # Prepare initial LSTM states
        num_layers = lstm.num_layers
        h_0_lstm = h_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
        c_0_lstm = c_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous()

        # Relation-conditioned dynamics mixer on the step inputs
        mixed_step = mixer(head_emb, relation_emb).unsqueeze(1)

        # Run LSTM over the sequence
        _, (h_n, _) = lstm(mixed_step, (h_0_lstm, c_0_lstm))
        query_vector = h_n[-1]
        return query_vector

    def _load_embedding_tensor(self, source, expected_rows, name):
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
        return loaded      

    def load_embeddings(self, entity_source, relation_source, kind='text', freeze=False):
        if kind == 'text':
            entity_table = self.text_ent_embs
            relation_table = self.text_rel_embs
            expected_dim = self.text_emb_dim
            entity_name = 'text_entity'
            relation_name = 'text_relation'
        elif kind == 'structural':
            entity_table = self.struct_ent_embs
            relation_table = self.struct_rel_embs
            expected_dim = self.struct_emb_dim
            entity_name = 'structural_entity'
            relation_name = 'structural_relation'
        else:
            raise ValueError(f"Unsupported embedding kind: {kind}")

        entity_cache = self._load_embedding_tensor(
            source=entity_source,
            expected_rows=entity_table.num_embeddings,
            name=entity_name,
        )
        relation_cache = self._load_embedding_tensor(
            source=relation_source,
            expected_rows=relation_table.num_embeddings,
            name=relation_name,
        )

        if entity_cache.size(1) != relation_cache.size(1):
            raise ValueError(
                f"{entity_name} and {relation_name} embeddings must share the same embedding dimension. "
                f"Got {entity_cache.size(1)} and {relation_cache.size(1)}"
            )

        if entity_cache.size(1) != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {expected_dim}, got {entity_cache.size(1)}"
            )

        entity_table.weight.data.copy_(entity_cache)
        relation_table.weight.data.copy_(relation_cache)

        if freeze:
            entity_table.weight.requires_grad = False
            relation_table.weight.requires_grad = False

    @staticmethod
    def _multi_positive_contrastive_loss(scores, target_ids=None):
        batch_size = scores.size(0)
        if target_ids is None:
            positive_mask = torch.eye(
                batch_size, dtype=torch.bool, device=scores.device
            )
        else:
            target_ids = target_ids.reshape(-1)
            if target_ids.numel() != batch_size:
                raise ValueError(
                    "target_ids must contain one entity ID per score row."
                )
            positive_mask = target_ids[:, None].eq(target_ids[None, :])

        positive_scores = scores.masked_fill(~positive_mask, float('-inf'))
        return -(
            torch.logsumexp(positive_scores, dim=1)
            - torch.logsumexp(scores, dim=1)
        )

    def forward(self, h_batch, r_batch, context_batch):
        h_text = self.text_adapter(self.text_ent_embs(h_batch['id']))
        r_text = self.text_adapter(self.text_rel_embs(r_batch['id']))
        h_struct = self.struct_adapter(self.struct_ent_embs(h_batch['id']))
        r_struct = self.struct_adapter(self.struct_rel_embs(r_batch['id']))
        h_fused, _ = self.entity_fusion(h_text, h_struct)
        r_fused, _ = self.relation_fusion(r_text, r_struct)

        flat_context_entity_ids, flat_context_relation_ids, context_batch_index = self._prepare_context_batch(context_batch)
        ctx_ent_text = self.text_adapter(self.text_ent_embs(flat_context_entity_ids))
        ctx_rel_text = self.text_adapter(self.text_rel_embs(flat_context_relation_ids))
        ctx_ent_struct = self.struct_adapter(self.struct_ent_embs(flat_context_entity_ids))
        ctx_rel_struct = self.struct_adapter(self.struct_rel_embs(flat_context_relation_ids))
        ctx_ent_fused, _ = self.entity_fusion(ctx_ent_text, ctx_ent_struct)
        ctx_rel_fused, _ = self.relation_fusion(ctx_rel_text, ctx_rel_struct)

        world_state = self.fused_compgcn(
            head_feat=h_fused,
            nbr_entity_feat=ctx_ent_fused,
            nbr_relation_feat=ctx_rel_fused,
            nbr_batch_index=context_batch_index,
        )
        query = self._run_dynamics(
            world_state,
            h_fused,
            r_fused,
            self.fused_dynamics_mixer,
            self.fused_lstm,
            self.fused_h0_projection,
            self.fused_c0_projection,
        )
        return F.normalize(self.fused_output_projection(query), p=2, dim=1)

    def encode_target(self, t_batch):
        t_text = self.text_adapter(self.text_ent_embs(t_batch['id']))
        t_struct = self.struct_adapter(self.struct_ent_embs(t_batch['id']))
        t_fused, _ = self.entity_fusion(t_text, t_struct)
        return F.normalize(
            self.fused_output_projection(t_fused),
            p=2,
            dim=1,
        )

    def compute_loss(self, query_vectors, target_vectors, target_ids=None):
        scores = torch.mm(query_vectors, target_vectors.t()) / self.temperature
        loss = self._multi_positive_contrastive_loss(
            scores,
            target_ids,
        ).mean()
        return loss, scores

