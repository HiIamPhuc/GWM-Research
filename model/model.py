import torch
import torch.nn as nn
import torch.nn.functional as F

class GWM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Structural Component (Entity/Relation Embeddings)
        self.structural_dim = config.structural_dim
        self.entity_embeddings = nn.Embedding(config.num_entities, self.structural_dim)
        self.relation_embeddings = nn.Embedding(config.num_relations, self.structural_dim)

        # Project structural embeddings to hidden_dim when needed by gating fusion.
        self.structural_projection = None
        if self.structural_dim != config.hidden_dim:
            self.structural_projection = nn.Linear(self.structural_dim, config.hidden_dim)
        
        # 2. Context Processing (Transformer Core)
        self.sequence_len = 3  # [Context, Head, Relation]
        self.transformer_num_heads = int(getattr(config, 'transformer_num_heads', 8))
        if config.hidden_dim % self.transformer_num_heads != 0:
            raise ValueError(
                f"hidden_dim ({config.hidden_dim}) must be divisible by transformer_num_heads ({self.transformer_num_heads})."
            )

        transformer_num_layers = int(getattr(config, 'transformer_num_layers', getattr(config, 'num_layers', 2)))
        transformer_ffn_dim = int(getattr(config, 'transformer_ffn_dim', config.hidden_dim * 4))
        transformer_dropout = float(getattr(config, 'transformer_dropout', config.dropout))
        self.transformer_use_causal_mask = bool(getattr(config, 'transformer_use_causal_mask', False))

        self.step_position_embedding = nn.Embedding(self.sequence_len, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=self.transformer_num_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=transformer_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.sequence_core = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_num_layers,
            norm=nn.LayerNorm(config.hidden_dim),
        )
        
        # 3. Fusion Layer
        text_dim = int(getattr(config, 'text_embedding_dim', config.hidden_dim))
        self.text_projection = nn.Linear(text_dim, config.hidden_dim)
        self.fusion_mode = config.fusion_mode

        # Legacy/default path: concat(text, struct) -> linear
        self.fusion = nn.Linear(text_dim + self.structural_dim, config.hidden_dim)

        # Dynamic gating path: learn sample-wise interpolation between text and structure.
        if self.fusion_mode == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 1),
                nn.Sigmoid()
            )

        # Running alpha stats for lightweight diagnostics.
        self.reset_alpha_stats()
        
        # 4. Output Projector (Optional but good for matching embeddings)
        self.projector = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Precomputed text cache loaded from preprocessing artifacts.
        self.cached_entity_text_emb = None
        self.cached_relation_text_emb = None
        self.use_text_cache = False
 
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

        expected_text_dim = self.text_projection.in_features
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

    def _project_structural(self, struct_emb):
        if self.structural_projection is not None:
            return self.structural_projection(struct_emb)
        return struct_emb

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
        if self.fusion_mode == 'gated':
            text_proj = self.text_projection(text_emb)
            struct_proj = self._project_structural(struct_emb)
            gate_input = torch.cat([text_proj, struct_proj], dim=-1)
            alpha = self.gate(gate_input)
            alpha_detached = alpha.detach()
            self._alpha_sum += alpha_detached.sum().item()
            self._alpha_count += alpha_detached.numel()
            return alpha * text_proj + (1.0 - alpha) * struct_proj

        # Backward-compatible concat fusion
        return self.fusion(torch.cat([text_emb, struct_emb], dim=-1))
        
    def forward(self, h_batch, r_batch, context_batch):
        """
        Forward pass for a batch of triples.
        h_batch: dict {id}
        r_batch: dict {id}
        context_batch: dict {id}
          - id: (B, K)
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
        
        # Context
        context_ids = context_batch['id'] # (B, K)
        ctx_emb_text = self._lookup_cached_text(context_ids, kind='entity') # (B, K, H)
        ctx_struct = self.entity_embeddings(context_ids) # (B, K, H)

        # Fuse Context (Text + Structure)
        ctx_fused = self._fuse_modalities(ctx_emb_text, ctx_struct) # (B, K, H)
        # Aggregate Context
        ctx_summary = torch.mean(ctx_fused, dim=1) # (B, H)
        
        # Main Fusion
        h_fused = self._fuse_modalities(h_emb_text, h_struct) # (B, H)
        r_fused = self._fuse_modalities(r_emb_text, r_struct) # (B, H)

        # Transformer sequence modeling
        # Sequence: [Context, Head, Relation] -> Predict Tail
        seq_input = torch.stack([ctx_summary, h_fused, r_fused], dim=1) # (B, 3, H)
        pos_ids = torch.arange(self.sequence_len, device=seq_input.device).unsqueeze(0).expand(seq_input.size(0), -1)
        seq_input = seq_input + self.step_position_embedding(pos_ids)

        attn_mask = None
        if self.transformer_use_causal_mask:
            # Upper-triangular mask blocks attending to future positions.
            attn_mask = torch.triu(
                torch.ones(self.sequence_len, self.sequence_len, device=seq_input.device, dtype=torch.bool),
                diagonal=1,
            )

        seq_out = self.sequence_core(seq_input, mask=attn_mask)
        query_vector = seq_out[:, -1, :] # Relation-step representation (B, H)
        
        # Project Query
        query_vector = self.projector(query_vector)
        
        # Ensure normalization for cosine similarity / InfoNCE
        query_vector = F.normalize(query_vector, p=2, dim=1)
        
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
        
        return F.normalize(t_fused, p=2, dim=1)

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
