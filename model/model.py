import math
import torch
import torch.nn as nn


class CompGCNLayer(nn.Module):
    """A lightweight CompGCN-style layer for head-centric aggregation."""

    def __init__(self, hidden_dim, comp_op='sub'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.comp_op = comp_op
        self.lin_self = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_out = nn.Linear(hidden_dim, hidden_dim)

    def _compose(self, entity_feat, relation_feat):
        if self.comp_op == 'sub':
            return entity_feat - relation_feat
        if self.comp_op == 'mult':
            return entity_feat * relation_feat
        raise ValueError(f"Unsupported compgcn_op: {self.comp_op}. Use 'sub' or 'mult'.")

    def forward(self, head_feat, nbr_entity_feat, nbr_relation_feat, nbr_batch_index):
        """
        head_feat: (B, H)
        nbr_entity_feat: (E, H)
        nbr_relation_feat: (E, H)
        nbr_batch_index: (E,) long, edge -> head index in batch
        """
        B = head_feat.size(0)
        composed = self._compose(nbr_entity_feat, nbr_relation_feat)  # (E, H)
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

        # Pure residual fix: preserve anchor head state, add learned neighbor delta.
        return head_feat + self.lin_out(agg)


class MLPAdapter(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + x


class SIGReg(nn.Module):
    """LeWM-style isotropic Gaussian regularizer using random projections."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = int(num_proj)
        t = torch.linspace(0, 3, int(knots), dtype=torch.float32)
        dt = 3.0 / max(int(knots) - 1, 1)
        weights = torch.full((int(knots),), 2 * dt, dtype=torch.float32)
        if weights.numel() > 1:
            weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        if proj.numel() == 0:
            return proj.new_zeros(())

        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0, keepdim=True).clamp(min=1e-12))

        t = self.t.to(dtype=proj.dtype)
        phi = self.phi.to(dtype=proj.dtype)
        weights = self.weights.to(dtype=proj.dtype)

        x_t = (proj @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()

class GWM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        legacy_hidden_dim = int(getattr(config, 'hidden_dim', getattr(config, 'text_embedding_dim', 768)))
        self.text_embedding_dim = int(getattr(config, 'text_embedding_dim', legacy_hidden_dim))

        # 1. Text Component (Entity/Relation Embeddings)
        self.text_entity_embeddings = nn.Embedding(config.num_entities, self.text_embedding_dim)
        self.text_relation_embeddings = nn.Embedding(config.num_relations, self.text_embedding_dim)

        # 2. Structural Component (Entity/Relation Embeddings)
        self.structural_dim = int(getattr(config, 'structural_dim', legacy_hidden_dim))
        self.entity_embeddings = nn.Embedding(config.num_entities, self.structural_dim)
        self.relation_embeddings = nn.Embedding(config.num_relations, self.structural_dim)

        # Keep each path in its native dimensionality unless explicitly overridden.
        self.text_compgcn_dim = int(
            getattr(config, 'text_compgcn_dim', getattr(config, 'compgcn_dim', self.text_embedding_dim))
        )
        self.struct_compgcn_dim = int(
            getattr(config, 'struct_compgcn_dim', getattr(config, 'compgcn_dim', self.structural_dim))
        )
        self.text_dynamics_dim = int(
            getattr(config, 'text_dynamics_dim', getattr(config, 'dynamics_dim', self.text_compgcn_dim))
        )
        self.struct_dynamics_dim = int(
            getattr(config, 'struct_dynamics_dim', getattr(config, 'dynamics_dim', self.struct_compgcn_dim))
        )
        self.dropout_rate = float(getattr(config, 'dropout', 0.0))
        self.recurrent_dropout = float(getattr(config, 'recurrent_dropout', 0.0))

        text_adapter_dim = int(getattr(config, 'text_adapter_dim', self.text_embedding_dim * 2))
        struct_adapter_dim = int(getattr(config, 'struct_adapter_dim', self.structural_dim * 2))
        adapter_dropout = float(getattr(config, 'adapter_dropout', 0.1))

        self.text_adapter = MLPAdapter(self.text_embedding_dim, text_adapter_dim, dropout=adapter_dropout)
        self.struct_adapter = MLPAdapter(self.structural_dim, struct_adapter_dim, dropout=adapter_dropout)

        self.input_dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        self.alpha_relation_only = bool(getattr(config, 'alpha_relation_only', False))
        alpha_in_dim = self.text_dynamics_dim + self.struct_dynamics_dim
        self.alpha_mlp = nn.Sequential(
            nn.Linear(alpha_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self._alpha_sum = 0.0
        self._alpha_count = 0

        self.per_relation_temperature = bool(getattr(config, 'per_relation_temperature', False))
        if self.per_relation_temperature:
            base_temp = float(getattr(config, 'temperature', 0.07))
            base_temp = max(base_temp, 1e-4)
            init_log_temp = math.log(base_temp)
            self.text_rel_log_temp = nn.Embedding(config.num_relations, 1)
            self.struct_rel_log_temp = nn.Embedding(config.num_relations, 1)
            nn.init.constant_(self.text_rel_log_temp.weight, init_log_temp)
            nn.init.constant_(self.struct_rel_log_temp.weight, init_log_temp)

        self.sigreg_weight = float(getattr(config, 'sigreg_weight', 0.0) or 0.0)
        if self.sigreg_weight > 0.0:
            sigreg_knots = int(getattr(config, 'sigreg_knots', 17))
            sigreg_num_proj = int(getattr(config, 'sigreg_num_proj', 1024))
            self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)
        else:
            self.sigreg = None

        def _build_projection(in_dim, out_dim):
            if in_dim == out_dim:
                return nn.Identity()
            return nn.Linear(in_dim, out_dim)

        # Text Path Projections
        self.text_spatial_projection = _build_projection(self.text_embedding_dim, self.text_compgcn_dim)
        self.text_dynamics_projection = _build_projection(self.text_embedding_dim, self.text_dynamics_dim)
        
        # Struct Path Projections
        self.struct_spatial_projection = _build_projection(self.structural_dim, self.struct_compgcn_dim)
        self.struct_dynamics_projection = _build_projection(self.structural_dim, self.struct_dynamics_dim)

        compgcn_layers = int(getattr(config, 'compgcn_layers', 1))
        
        # 2. Text Spatial Encoder
        self.text_compgcn_stack = nn.ModuleList([
            CompGCNLayer(hidden_dim=self.text_compgcn_dim, comp_op=getattr(config, 'compgcn_op', 'sub'))
            for _ in range(max(compgcn_layers, 1))
        ])

        # 3. Struct Spatial Encoder
        self.struct_compgcn_stack = nn.ModuleList([
            CompGCNLayer(hidden_dim=self.struct_compgcn_dim, comp_op=getattr(config, 'compgcn_op', 'sub'))
            for _ in range(max(compgcn_layers, 1))
        ])
        
        # 4. Text Transition Dynamics
        self.text_lstm = nn.LSTM(
            input_size=self.text_dynamics_dim, hidden_size=self.text_dynamics_dim,
            num_layers=int(getattr(config, 'dynamics_layers', 1)), batch_first=True
        )
        self.text_h0_projection = _build_projection(self.text_compgcn_dim, self.text_dynamics_dim)
        self.text_c0_projection = _build_projection(self.text_compgcn_dim, self.text_dynamics_dim)

        # 5. Struct Transition Dynamics
        self.struct_lstm = nn.LSTM(
            input_size=self.struct_dynamics_dim, hidden_size=self.struct_dynamics_dim,
            num_layers=int(getattr(config, 'dynamics_layers', 1)), batch_first=True
        )
        self.struct_h0_projection = _build_projection(self.struct_compgcn_dim, self.struct_dynamics_dim)
        self.struct_c0_projection = _build_projection(self.struct_compgcn_dim, self.struct_dynamics_dim)

        self.recurrent_dropout_layer = nn.Dropout(self.recurrent_dropout) if self.recurrent_dropout > 0 else nn.Identity()
        # FiLM (relation-conditioned) projections for modulating LSTM inputs
        self.text_film = nn.Linear(self.text_dynamics_dim, 2 * self.text_dynamics_dim)
        self.struct_film = nn.Linear(self.struct_dynamics_dim, 2 * self.struct_dynamics_dim)

    def _encode_subgraph_with_compgcn(self, h_emb, ctx_entity_emb, ctx_relation_emb, ctx_batch_index, compgcn_stack):
        h_state = h_emb
        for layer in compgcn_stack:
            h_state = layer(
                head_feat=h_state,
                nbr_entity_feat=ctx_entity_emb,
                nbr_relation_feat=ctx_relation_emb,
                nbr_batch_index=ctx_batch_index,
            )
        return h_state

    def _run_dynamics(self, world_state, step_seq, relation_emb, lstm, h0_proj, c0_proj, path='text'):
        """
        Run recurrent dynamics over a sequence of steps.

        world_state: (B, compgcn_dim) used to initialise h0/c0
        step_seq: (B, T, D) sequence of per-step inputs (already projected to dynamics dim)
        relation_emb: (B, D_rel) projected relation embedding for this path
        path: 'text' or 'struct' to select FiLM projection
        """
        h_0 = torch.tanh(h0_proj(world_state))
        c_0 = c0_proj(world_state)

        # Prepare initial LSTM states
        num_layers = lstm.num_layers
        h_0_lstm = h_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
        c_0_lstm = c_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous()

        # Relation-conditioned FiLM modulation on the step inputs
        if path == 'text':
            film_params = self.text_film(relation_emb)
        else:
            film_params = self.struct_film(relation_emb)

        shift, scale = film_params.chunk(2, dim=-1)
        # apply modulation across time steps
        step_seq = (1.0 + scale.unsqueeze(1)) * step_seq + shift.unsqueeze(1)

        # Run LSTM over the sequence
        lstm_out, (h_n, c_n) = lstm(step_seq, (h_0_lstm, c_0_lstm))
        query_vector = h_n[-1]
        query_vector = self.recurrent_dropout_layer(query_vector)
        query_norm = torch.nn.functional.normalize(query_vector, p=2, dim=1)
        return query_vector, query_norm

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

    def load_precomputed_structural_cache(self, entity_source, relation_source, freeze=False):
        """Loads precomputed structural embeddings (e.g., from RotatE or ComplEx) into the embedding tables."""
        entity_cache = self._load_embedding_tensor(
            source=entity_source,
            expected_rows=self.entity_embeddings.num_embeddings,
            name='structural_entity',
        )
        
        relation_cache = self._load_embedding_tensor(
            source=relation_source,
            expected_rows=self.relation_embeddings.num_embeddings,
            name='structural_relation',
        )

        expected_struct_dim = self.structural_dim
        if entity_cache.size(1) != expected_struct_dim:
            raise ValueError(
                f"Structural embedding dim mismatch. Config expects {expected_struct_dim}, got {entity_cache.size(1)}"
            )

        self.entity_embeddings.weight.data.copy_(entity_cache)
        self.relation_embeddings.weight.data.copy_(relation_cache)

        if freeze:
            self.entity_embeddings.weight.requires_grad = False
            self.relation_embeddings.weight.requires_grad = False

    def load_precomputed_text_cache(self, entity_source, relation_source, freeze=True):
        """Loads precomputed text embeddings into text embedding tables."""
        entity_cache = self._load_embedding_tensor(
            source=entity_source,
            expected_rows=self.text_entity_embeddings.num_embeddings,
            name='text_entity',
        )

        relation_cache = self._load_embedding_tensor(
            source=relation_source,
            expected_rows=self.text_relation_embeddings.num_embeddings,
            name='text_relation',
        )

        if entity_cache.size(1) != relation_cache.size(1):
            raise ValueError(
                "Entity and relation text embeddings must share the same embedding dimension. "
                f"Got {entity_cache.size(1)} and {relation_cache.size(1)}"
            )

        expected_text_dim = self.text_embedding_dim
        if entity_cache.size(1) != expected_text_dim:
            raise ValueError(
                "Text embedding dimension mismatch with model config. "
                f"Expected {expected_text_dim}, got {entity_cache.size(1)}"
            )

        self.text_entity_embeddings.weight.data.copy_(entity_cache)
        self.text_relation_embeddings.weight.data.copy_(relation_cache)

        if freeze:
            self.text_entity_embeddings.weight.requires_grad = False
            self.text_relation_embeddings.weight.requires_grad = False

    def reset_alpha_stats(self):
        self._alpha_sum = 0.0
        self._alpha_count = 0

    def _record_alpha(self, alpha):
        self._alpha_sum += alpha.sum().item()
        self._alpha_count += alpha.numel()

    def get_alpha_mean(self, reset=False):
        if self._alpha_count == 0:
            return None
        mean = self._alpha_sum / self._alpha_count
        if reset:
            self.reset_alpha_stats()
        return mean

    def forward(self, h_batch, r_batch, context_batch):
        h_emb_text = self.text_adapter(self.text_entity_embeddings(h_batch['id']))
        r_emb_text = self.text_adapter(self.text_relation_embeddings(r_batch['id']))
        
        # Structural Embeddings
        h_struct = self.struct_adapter(self.entity_embeddings(h_batch['id']))
        r_struct = self.struct_adapter(self.relation_embeddings(r_batch['id']))

        h_emb_text = self.input_dropout(h_emb_text)
        r_emb_text = self.input_dropout(r_emb_text)
        h_struct = self.input_dropout(h_struct)
        r_struct = self.input_dropout(r_struct)
        
        context_entity_ids = context_batch['id']
        context_relation_ids = context_batch.get('rel_id')
        context_batch_index = context_batch.get('batch_index')
        context_mask = context_batch.get('mask')

        if context_entity_ids.dim() == 2:
            B, K = context_entity_ids.shape
            if context_relation_ids is None:
                context_relation_ids = torch.zeros_like(context_entity_ids)
            if context_mask is None:
                context_mask = torch.ones_like(context_entity_ids, dtype=torch.bool)
            else:
                context_mask = context_mask.bool()

            valid_idx = context_mask.nonzero(as_tuple=False)
            if valid_idx.numel() == 0:
                flat_context_entity_ids = context_entity_ids.new_zeros((0,))
                flat_context_relation_ids = context_relation_ids.new_zeros((0,))
                context_batch_index = context_entity_ids.new_zeros((0,))
            else:
                context_batch_index = valid_idx[:, 0]
                flat_context_entity_ids = context_entity_ids[context_mask]
                flat_context_relation_ids = context_relation_ids[context_mask]
        else:
            flat_context_entity_ids = context_entity_ids
            if context_relation_ids is None:
                flat_context_relation_ids = torch.zeros_like(flat_context_entity_ids)
            else:
                flat_context_relation_ids = context_relation_ids
            if context_batch_index is None:
                raise ValueError("context_batch['batch_index'] is required for ragged context format.")

        ctx_ent_text = self.text_adapter(self.text_entity_embeddings(flat_context_entity_ids))
        ctx_ent_struct = self.struct_adapter(self.entity_embeddings(flat_context_entity_ids))
        ctx_rel_text = self.text_adapter(self.text_relation_embeddings(flat_context_relation_ids))
        ctx_rel_struct = self.struct_adapter(self.relation_embeddings(flat_context_relation_ids))

        h_spatial_text = self.text_spatial_projection(h_emb_text)
        ctx_entity_spatial_text = self.text_spatial_projection(ctx_ent_text)
        ctx_relation_spatial_text = self.text_spatial_projection(ctx_rel_text)

        h_spatial_struct = self.struct_spatial_projection(h_struct)
        ctx_entity_spatial_struct = self.struct_spatial_projection(ctx_ent_struct)
        ctx_relation_spatial_struct = self.struct_spatial_projection(ctx_rel_struct)

        # -- Text Pathway --
        world_state_text = self._encode_subgraph_with_compgcn(
            h_spatial_text, ctx_entity_spatial_text, ctx_relation_spatial_text,
            context_batch_index, self.text_compgcn_stack
        )
        world_state_text = self.input_dropout(world_state_text)
        
        # Build sequence of per-step node embeddings when context is provided as (B, K)
        if context_entity_ids.dim() == 2:
            # ctx_ent_text is flattened; reconstruct per-batch sequence directly from ids
            ctx_ent_text_seq = self.text_adapter(self.text_entity_embeddings(context_entity_ids))
            step_seq_text = self.text_dynamics_projection(ctx_ent_text_seq)
        else:
            # ragged / flattened fallback: use single-step head embedding
            step_seq_text = self.text_dynamics_projection(h_emb_text).unsqueeze(1)

        relation_emb_text = self.text_dynamics_projection(r_emb_text)

        query_text_raw, query_text = self._run_dynamics(
            world_state_text, step_seq_text, relation_emb_text,
            self.text_lstm, self.text_h0_projection, self.text_c0_projection, path='text'
        )

        # -- Struct Pathway --
        world_state_struct = self._encode_subgraph_with_compgcn(
            h_spatial_struct, ctx_entity_spatial_struct, ctx_relation_spatial_struct,
            context_batch_index, self.struct_compgcn_stack
        )
        world_state_struct = self.input_dropout(world_state_struct)
        
        if context_entity_ids.dim() == 2:
            ctx_ent_struct_seq = self.struct_adapter(self.entity_embeddings(context_entity_ids))
            step_seq_struct = self.struct_dynamics_projection(ctx_ent_struct_seq)
        else:
            step_seq_struct = self.struct_dynamics_projection(h_struct).unsqueeze(1)

        relation_emb_struct = self.struct_dynamics_projection(r_struct)

        query_struct_raw, query_struct = self._run_dynamics(
            world_state_struct, step_seq_struct, relation_emb_struct,
            self.struct_lstm, self.struct_h0_projection, self.struct_c0_projection, path='struct'
        )

        return query_text, query_struct, relation_emb_text, relation_emb_struct, query_text_raw, query_struct_raw

    def encode_target(self, t_batch):
        t_emb_text = self.text_adapter(self.text_entity_embeddings(t_batch['id']))
        t_struct = self.struct_adapter(self.entity_embeddings(t_batch['id']))
        
        t_text_proj = self.text_dynamics_projection(t_emb_text)
        t_struct_proj = self.struct_dynamics_projection(t_struct)
        
        t_text_norm = torch.nn.functional.normalize(t_text_proj, p=2, dim=1)
        t_struct_norm = torch.nn.functional.normalize(t_struct_proj, p=2, dim=1)
        return t_text_norm, t_struct_norm

    def compute_loss(self, query_vectors, target_vectors, relation_ids=None):
        loss_text, loss_struct, scores = self.compute_loss_components(
            query_vectors,
            target_vectors,
            relation_ids=relation_ids,
        )
        return loss_text + loss_struct, scores

    def _get_hard_negative_k(self, num_candidates):
        hard_negative_k = int(getattr(self.config, 'hard_negative_k', 0) or 0)
        if hard_negative_k > 0:
            return min(hard_negative_k, max(0, num_candidates - 1))

        hard_negative_fraction = float(getattr(self.config, 'hard_negative_fraction', 0.0) or 0.0)
        if hard_negative_fraction > 0.0:
            k = int(round(hard_negative_fraction * max(0, num_candidates - 1)))
            return min(k, max(0, num_candidates - 1))

        return 0

    def _weighted_infonce_loss(self, scores):
        batch_size, num_candidates = scores.shape
        labels = torch.arange(batch_size, device=scores.device)

        hard_negative_weight = float(getattr(self.config, 'hard_negative_weight', 1.0) or 1.0)
        hard_negative_k = self._get_hard_negative_k(num_candidates)

        if hard_negative_weight <= 1.0 or hard_negative_k <= 0:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(scores, labels)

        weights = torch.ones_like(scores)
        scores_neg = scores.clone()
        scores_neg.fill_diagonal_(float('-inf'))
        topk_idx = torch.topk(scores_neg, hard_negative_k, dim=1).indices
        weights.scatter_(1, topk_idx, hard_negative_weight)

        max_scores = scores.max(dim=1, keepdim=True).values
        exp_scores = torch.exp(scores - max_scores)
        denom = (weights * exp_scores).sum(dim=1)
        log_denom = torch.log(denom) + max_scores.squeeze(1)
        loss = -scores[labels, labels] + log_denom
        return loss.mean()

    def _split_query_vectors(self, query_vectors):
        if len(query_vectors) == 2:
            return query_vectors[0], query_vectors[1], None, None, None, None
        if len(query_vectors) == 4:
            return query_vectors[0], query_vectors[1], query_vectors[2], query_vectors[3], None, None
        if len(query_vectors) >= 6:
            return (
                query_vectors[0],
                query_vectors[1],
                query_vectors[2],
                query_vectors[3],
                query_vectors[4],
                query_vectors[5],
            )
        raise ValueError(f"Unexpected query vector format (len={len(query_vectors)}).")

    def compute_sigreg_loss(self, query_vectors):
        if self.sigreg is None or self.sigreg_weight <= 0.0:
            return None
        _, _, _, _, query_text_raw, query_struct_raw = self._split_query_vectors(query_vectors)
        if query_text_raw is None or query_struct_raw is None:
            return None
        sig_text = self.sigreg(query_text_raw.unsqueeze(0))
        sig_struct = self.sigreg(query_struct_raw.unsqueeze(0))
        return 0.5 * (sig_text + sig_struct)

    def compute_alpha(self, query_text, query_struct, relation_text=None, relation_struct=None):
        if self.alpha_relation_only:
            if relation_text is None or relation_struct is None:
                raise ValueError("alpha_relation_only requires relation embeddings in query_vectors.")
            alpha_input = torch.cat([relation_text, relation_struct], dim=-1)
        else:
            alpha_input = torch.cat([query_text, query_struct], dim=-1)
        alpha = self.alpha_mlp(alpha_input)
        self._record_alpha(alpha)
        return alpha

    def _get_relation_temperatures(self, relation_ids, device, dtype):
        if not self.per_relation_temperature or relation_ids is None:
            base_temp = float(getattr(self.config, 'temperature', 0.07))
            base_temp = max(base_temp, 1e-4)
            temp = torch.tensor(base_temp, device=device, dtype=dtype)
            return temp, temp

        log_temp_text = self.text_rel_log_temp(relation_ids).squeeze(-1)
        log_temp_struct = self.struct_rel_log_temp(relation_ids).squeeze(-1)
        temp_text = torch.exp(log_temp_text).clamp(min=1e-4)
        temp_struct = torch.exp(log_temp_struct).clamp(min=1e-4)
        return temp_text.unsqueeze(1), temp_struct.unsqueeze(1)

    def apply_temperature(self, scores_text, scores_struct, relation_ids=None):
        temp_text, temp_struct = self._get_relation_temperatures(
            relation_ids,
            device=scores_text.device,
            dtype=scores_text.dtype,
        )
        return scores_text / temp_text, scores_struct / temp_struct

    def compute_loss_components(self, query_vectors, target_vectors, relation_ids=None):
        query_text, query_struct, relation_text, relation_struct, _, _ = self._split_query_vectors(query_vectors)
        target_text, target_struct = target_vectors

        scores_text = torch.mm(query_text, target_text.t())
        scores_struct = torch.mm(query_struct, target_struct.t())

        scores_text, scores_struct = self.apply_temperature(
            scores_text,
            scores_struct,
            relation_ids=relation_ids,
        )

        # Score Level Fusion
        alpha = self.compute_alpha(query_text, query_struct, relation_text, relation_struct)
        scores_fused = alpha * scores_text + (1.0 - alpha) * scores_struct

        loss_text = self._weighted_infonce_loss(scores_text)
        loss_struct = self._weighted_infonce_loss(scores_struct)

        return loss_text, loss_struct, scores_fused
