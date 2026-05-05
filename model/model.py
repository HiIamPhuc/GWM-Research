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

class GWM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        legacy_hidden_dim = int(getattr(config, 'hidden_dim', getattr(config, 'text_embedding_dim', 768)))
        self.text_embedding_dim = int(getattr(config, 'text_embedding_dim', legacy_hidden_dim))

        # 1. Structural Component (Entity/Relation Embeddings)
        self.structural_dim = int(getattr(config, 'structural_dim', legacy_hidden_dim))
        self.entity_embeddings = nn.Embedding(config.num_entities, self.structural_dim)
        self.relation_embeddings = nn.Embedding(config.num_relations, self.structural_dim)

        # Shared latent spaces separated by module role.
        self.fusion_dim = int(getattr(config, 'fusion_dim', legacy_hidden_dim))
        self.compgcn_dim = int(getattr(config, 'compgcn_dim', self.fusion_dim))
        self.dynamics_dim = int(getattr(config, 'dynamics_dim', self.fusion_dim))
        self.hyper_mlp_dim = int(getattr(config, 'hyper_mlp_dim', self.dynamics_dim))
        self.dropout_rate = float(getattr(config, 'dropout', 0.0))
        self.recurrent_dropout = float(getattr(config, 'recurrent_dropout', 0.0))

        self.input_dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        self.spatial_projection = nn.Identity()
        if self.fusion_dim != self.compgcn_dim:
            self.spatial_projection = nn.Linear(self.fusion_dim, self.compgcn_dim)

        self.dynamics_projection = nn.Identity()
        if self.fusion_dim != self.dynamics_dim:
            self.dynamics_projection = nn.Linear(self.fusion_dim, self.dynamics_dim)

        self.query_projection = nn.Identity()
        if self.dynamics_dim != self.fusion_dim:
            self.query_projection = nn.Linear(self.dynamics_dim, self.fusion_dim)
        
        # 2. Spatial Encoder (Ego-centric CompGCN for subgraph context)
        # The head node is the anchor, and messages are composed from
        # (context entity, context relation) pairs.
        compgcn_layers = int(getattr(config, 'compgcn_layers', 1))
        self.compgcn_stack = nn.ModuleList(
            [
                CompGCNLayer(
                    hidden_dim=self.compgcn_dim,
                    comp_op=getattr(config, 'compgcn_op', 'sub'),
                )
                for _ in range(max(compgcn_layers, 1))
            ]
        )
        
        # 3. Transition Dynamics (Standard PyTorch LSTM)
        self.dynamics_mixer = nn.Sequential(
            nn.Linear(self.dynamics_dim * 2, self.dynamics_dim * 2),
            nn.GELU(),
            nn.Linear(self.dynamics_dim * 2, self.dynamics_dim)
        )
        
        self.lstm = nn.LSTM(
            input_size=self.dynamics_dim,
            hidden_size=self.dynamics_dim,
            num_layers=int(getattr(config, 'dynamics_layers', 1)),
            batch_first=True
        )
        self.recurrent_dropout_layer = nn.Dropout(self.recurrent_dropout) if self.recurrent_dropout > 0 else nn.Identity()
        
        self.h0_projection = nn.Linear(self.compgcn_dim, self.dynamics_dim)
        self.c0_projection = nn.Linear(self.compgcn_dim, self.dynamics_dim)
        
        # 4. Fusion Layer
        self.fusion_mode = config.fusion_mode

        # Legacy/default path: concat(text, struct) -> linear
        self.fusion = nn.Linear(self.text_embedding_dim + self.structural_dim, self.fusion_dim)

        # Dynamic gating path: learn sample-wise interpolation between raw text and structure vectors.
        if self.fusion_mode == 'gated':
            # gate reads original unprojected dimensions to compute scalar alpha
            self.gate = nn.Sequential(
                nn.Linear(self.text_embedding_dim + self.structural_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            # After weighted concatenation, project to fusion_dim
            self.fusion_projection = nn.Linear(self.text_embedding_dim + self.structural_dim, self.fusion_dim)

        # Running alpha stats for lightweight diagnostics.
        self.reset_alpha_stats()

        # Precomputed text cache loaded from preprocessing artifacts.
        self.cached_entity_text_emb = None
        self.cached_relation_text_emb = None
        self.use_text_cache = False

    def _encode_subgraph_with_compgcn(self, h_fused, ctx_entity_fused, ctx_relation_fused, ctx_batch_index):
        """
        Encode ego-centric subgraph context using CompGCN-style composition.
        
        Args:
            h_fused: (B, H) fused head embedding (anchor node)
            ctx_entity_fused: (E, H) fused context entity embeddings
            ctx_relation_fused: (E, H) fused context relation embeddings
            ctx_batch_index: (E,) long, edge -> head index in batch
        
        Returns:
            compgcn_output: (B, H) updated head representation
        """
        h_state = h_fused
        for layer in self.compgcn_stack:
            h_state = layer(
                head_feat=h_state,
                nbr_entity_feat=ctx_entity_fused,
                nbr_relation_feat=ctx_relation_fused,
                nbr_batch_index=ctx_batch_index,
            )
        return h_state
 
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

    def load_precomputed_text_embedding_cache(self, entity_source, relation_source, cache_device='cpu'):
        if self.use_text_cache and self.cached_entity_text_emb is not None and self.cached_relation_text_emb is not None:
            return

        cache_device = torch.device(cache_device)

        entity_cache = self._load_embedding_tensor(
            source=entity_source,
            expected_rows=self.entity_embeddings.num_embeddings,
            name='entity',
        ).to(cache_device)

        relation_cache = self._load_embedding_tensor(
            source=relation_source,
            expected_rows=self.relation_embeddings.num_embeddings,
            name='relation',
        ).to(cache_device)

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

        self.cached_entity_text_emb = entity_cache
        self.cached_relation_text_emb = relation_cache
        self.use_text_cache = True

        if cache_device.type == 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Backward-compatible alias.
    def build_text_embedding_cache(self, entity_source, relation_source, device='cpu', **_kwargs):
        self.load_precomputed_text_embedding_cache(entity_source, relation_source, cache_device=device)

    def _lookup_cached_text(self, ids, kind='entity'):
        cache = self.cached_entity_text_emb if kind == 'entity' else self.cached_relation_text_emb
        if cache is None:
            raise RuntimeError("Text cache is not built. Call load_precomputed_text_embedding_cache first.")

        original_shape = ids.shape
        flat_ids = ids.reshape(-1)
        if flat_ids.device != cache.device:
            flat_ids = flat_ids.to(cache.device)
        selected = cache.index_select(0, flat_ids)
        if selected.device != ids.device:
            selected = selected.to(ids.device)
        return selected.reshape(*original_shape, -1)

    def reset_alpha_stats(self):
        self._alpha_sum = 0.0
        self._alpha_count = 0

    def get_alpha_mean(self, reset=False):
        if self.fusion_mode != 'gated' or self._alpha_count == 0:
            alpha_mean = None
        else:
            alpha_mean = self._alpha_sum / self._alpha_count

        if reset:
            self.reset_alpha_stats()

        return alpha_mean

    def _fuse_modalities(self, text_emb, struct_emb):
        text_emb = self.input_dropout(text_emb)
        struct_emb = self.input_dropout(struct_emb)

        if self.fusion_mode == 'gated':
            # Compute scalar gate on concatenated original (unprojected) vectors
            gate_input = torch.cat([text_emb, struct_emb], dim=-1)
            alpha = self.gate(gate_input)
            alpha_detached = alpha.detach()
            self._alpha_sum += alpha_detached.sum().item()
            self._alpha_count += alpha_detached.numel()
            # Preserve raw text/structure by weighting and concatenating
            weighted_concat = torch.cat([alpha * text_emb, (1.0 - alpha) * struct_emb], dim=-1)
            # Project the weighted concat to fusion_dim
            return self.fusion_projection(weighted_concat)

        # Backward-compatible concat fusion
        return self.fusion(torch.cat([text_emb, struct_emb], dim=-1))
        
    def forward(self, h_batch, r_batch, context_batch):
        """
        Forward pass for a batch of triples.
        h_batch: dict {id}
        r_batch: dict {id}
        context_batch: dict {id}
          - id: (B, K)
        
        World-Model Hyper-Dynamics Paradigm:
        - Encode ego-centric subgraph context with CompGCN -> (B, H)
        - Use relation embedding as dynamic program via Hyper-LSTM
        - Run one transition step from world_state
        - Use final LSTM hidden state as query vector
        """
        if not self.use_text_cache:
            raise RuntimeError(
                "Text cache is not built. Call load_precomputed_text_embedding_cache before training/inference."
            )

        h_emb_text = self._lookup_cached_text(h_batch['id'], kind='entity')
        r_emb_text = self._lookup_cached_text(r_batch['id'], kind='relation')
        
        # Structural Embeddings
        h_struct = self.entity_embeddings(h_batch['id']) # (B, H)
        r_struct = self.relation_embeddings(r_batch['id']) # (B, H)

        # Main Fusion
        h_fused = self._fuse_modalities(h_emb_text, h_struct) # (B, H)
        r_fused = self._fuse_modalities(r_emb_text, r_struct) # (B, H)
        h_fused = self.input_dropout(h_fused)
        r_fused = self.input_dropout(r_fused)
        
        # Context (entity+relation neighbor edges around head)
        # Ragged format: flattened edges with batch index (no padding).
        context_entity_ids = context_batch['id']  # (E,) or legacy (B, K)
        context_relation_ids = context_batch.get('rel_id')
        context_batch_index = context_batch.get('batch_index')
        context_mask = context_batch.get('mask')

        if context_entity_ids.dim() == 2:
            # Legacy padded format fallback.
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

        ctx_ent_text = self._lookup_cached_text(flat_context_entity_ids, kind='entity') # (E, H)
        ctx_ent_struct = self.entity_embeddings(flat_context_entity_ids) # (E, H)
        ctx_rel_text = self._lookup_cached_text(flat_context_relation_ids, kind='relation') # (E, H)
        ctx_rel_struct = self.relation_embeddings(flat_context_relation_ids) # (E, H)

        # Fuse Context Entity/Relation modalities
        ctx_entity_fused = self._fuse_modalities(ctx_ent_text, ctx_ent_struct) # (E, H)
        ctx_relation_fused = self._fuse_modalities(ctx_rel_text, ctx_rel_struct) # (E, H)

        h_spatial = self.spatial_projection(h_fused)
        ctx_entity_spatial = self.spatial_projection(ctx_entity_fused)
        ctx_relation_spatial = self.spatial_projection(ctx_relation_fused)

        # SPATIAL ENCODER: Ego-centric CompGCN update of the head state.
        world_state = self._encode_subgraph_with_compgcn(
            h_fused=h_spatial,
            ctx_entity_fused=ctx_entity_spatial,
            ctx_relation_fused=ctx_relation_spatial,
            ctx_batch_index=context_batch_index,
        )  # (B, H)
        world_state = self.input_dropout(world_state)

        # TRANSITION DYNAMICS:
        # - world_state initializes recurrent memory
        # - head_fused is the first (and only) input step
        h_0 = torch.tanh(self.h0_projection(world_state))
        c_0 = torch.tanh(self.c0_projection(world_state))
        
        step_x = self.dynamics_projection(h_fused)
        relation_emb = self.dynamics_projection(r_fused)
        
        concat_input = torch.cat([step_x, relation_emb], dim=-1) # (B, 2H)
        mixed_input = self.dynamics_mixer(concat_input)          # (B, H)
        lstm_input = mixed_input.unsqueeze(1)                    # (B, 1, H)
        
        # LSTM expects hidden state shapes: (num_layers, B, H).
        # We broadcast the initialized h_0 and c_0 across all LSTM layers.
        num_layers = self.lstm.num_layers
        h_0_lstm = h_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous() # (num_layers, B, H)
        c_0_lstm = c_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous() # (num_layers, B, H)
        
        lstm_out, (h_n, c_n) = self.lstm(lstm_input, (h_0_lstm, c_0_lstm))
        # Extract the hidden state from the last LSTM layer
        query_vector = h_n[-1] # (B, H)
        query_vector = self.recurrent_dropout_layer(query_vector)
        
        # Project back to the fused comparison space only when the spaces differ.
        query_vector = self.query_projection(query_vector)
        
        # Ensure normalization for cosine similarity / InfoNCE
        query_vector = torch.nn.functional.normalize(query_vector, p=2, dim=1)
        
        return query_vector

    def encode_target(self, t_batch):
        """
        Encode target/tail entities symmetrically (Fusion of Text + Structure).
        t_batch: dict {id}
        Returns: (B, H) normalized fused embedding
        """
        if not self.use_text_cache:
            raise RuntimeError(
                "Text cache is not built. Call load_precomputed_text_embedding_cache before training/inference."
            )

        t_emb_text = self._lookup_cached_text(t_batch['id'], kind='entity')
        t_struct = self.entity_embeddings(t_batch['id'])
        
        t_fused = self._fuse_modalities(t_emb_text, t_struct)
        
        return torch.nn.functional.normalize(t_fused, p=2, dim=1)

    def compute_loss(self, query_vector, t_fused):
        """
        InfoNCE Loss with In-Batch Negatives.
        query_vector: (B, H) - Normalized query embeddings
        t_fused: (B, H) - Normalized target/tail embeddings (Symmetric Fusion)
        """
        # Cosine Similarity
        # (B, B)
        # score[i, j] = sim(query[i], tail[j])
        scores = torch.mm(query_vector, t_fused.t())
        
        # Temperature
        if hasattr(self.config, 'temperature'):
            scores /= self.config.temperature
        else:
            scores /= 0.07
        
        # Labels: diagonal are positives
        labels = torch.arange(scores.size(0), device=scores.device)
        
        return nn.CrossEntropyLoss()(scores, labels), scores
