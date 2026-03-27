import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class GWM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Text Encoder (Finetunable)
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
        self.text_encoder = AutoModel.from_pretrained(config.pretrained_model)
        
        # Freezing Logic
        if not config.finetune_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
                
        # 2. Structural Component (Entity/Relation Embeddings)
        self.structural_dim = config.structural_dim
        self.entity_embeddings = nn.Embedding(config.num_entities, self.structural_dim)
        self.relation_embeddings = nn.Embedding(config.num_relations, self.structural_dim)

        # Project structural embeddings to hidden_dim when needed by gating fusion.
        self.structural_projection = None
        if self.structural_dim != config.hidden_dim:
            self.structural_projection = nn.Linear(self.structural_dim, config.hidden_dim)
        
        # 3. Context Processing (RNN / GWM Core)
        # Note: If fusion output is hidden_dim, LSTM input is hidden_dim.
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim, 
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        # 4. Fusion Layer
        # Text Encoder Dim + Structural Dim
        text_dim = self.text_encoder.config.hidden_size
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
        
        # 5. Output Projector (Optional but good for matching embeddings)
        self.projector = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Optional frozen-text cache (used when finetune_text_encoder is False).
        self.cached_entity_text_emb = None
        self.cached_relation_text_emb = None
        self.use_text_cache = False
        
    def _encode_text(self, input_ids, attention_mask):
        """Forward pass through BERT."""
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :] # [CLS] token

    def build_text_embedding_cache(
        self,
        entity_text_map,
        relation_text_map,
        device,
        batch_size=128,
        max_entity_length=512,
        max_relation_length=128
    ):
        """Precompute text embeddings once for frozen text encoder mode."""
        if self.config.finetune_text_encoder:
            self.use_text_cache = False
            return

        self.text_encoder.eval()

        def _encode_text_list(text_list, max_length):
            all_emb = []
            with torch.no_grad():
                for start in range(0, len(text_list), batch_size):
                    chunk = text_list[start:start + batch_size]
                    enc = self.tokenizer(
                        chunk,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=max_length
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}
                    emb = self._encode_text(enc['input_ids'], enc['attention_mask'])
                    all_emb.append(emb)
            return torch.cat(all_emb, dim=0)

        num_entities = self.entity_embeddings.num_embeddings
        num_relations = self.relation_embeddings.num_embeddings

        entity_texts = [entity_text_map.get(str(i), f"Entity {i}") for i in range(num_entities)]
        relation_texts = [relation_text_map.get(str(i), f"Relation {i}") for i in range(num_relations)]

        self.cached_entity_text_emb = _encode_text_list(entity_texts, max_entity_length)
        self.cached_relation_text_emb = _encode_text_list(relation_texts, max_relation_length)
        self.use_text_cache = True

    def _lookup_cached_text(self, ids, kind='entity'):
        """Lookup cached text embeddings by integer IDs."""
        cache = self.cached_entity_text_emb if kind == 'entity' else self.cached_relation_text_emb
        if cache is None:
            raise RuntimeError("Text cache is not built. Call build_text_embedding_cache first.")

        original_shape = ids.shape
        flat_ids = ids.view(-1)
        selected = cache.index_select(0, flat_ids)
        return selected.view(*original_shape, -1)

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
        h_batch: dict {input_ids, attention_mask, id}
        r_batch: dict {input_ids, attention_mask, id}
        context_batch: dict {input_ids, attention_mask, id}
          - input_ids: (B*K, L)
          - id: (B, K)
        """
        # Encode Head & Relation Text (or use cache if text encoder is frozen)
        if self.use_text_cache and not self.config.finetune_text_encoder:
            h_emb_text = self._lookup_cached_text(h_batch['id'], kind='entity')
            r_emb_text = self._lookup_cached_text(r_batch['id'], kind='relation')
        else:
            h_emb_text = self._encode_text(h_batch['input_ids'], h_batch['attention_mask'])
            r_emb_text = self._encode_text(r_batch['input_ids'], r_batch['attention_mask'])
        
        # Structural Embeddings
        h_struct = self.entity_embeddings(h_batch['id']) # (B, H)
        r_struct = self.relation_embeddings(r_batch['id']) # (B, H)
        
        # Context
        context_ids = context_batch['id'] # (B, K)
        ctx_input_ids = context_batch['input_ids'] 
        ctx_mask = context_batch['attention_mask']

        if self.use_text_cache and not self.config.finetune_text_encoder:
            ctx_emb_text = self._lookup_cached_text(context_ids, kind='entity') # (B, K, H)
        else:
            ctx_emb_text = self._encode_text(ctx_input_ids, ctx_mask) # (B*K, H)
        ctx_struct = self.entity_embeddings(context_ids) # (B, K, H)
        batch_size, k = context_ids.shape
        if not (self.use_text_cache and not self.config.finetune_text_encoder):
            ctx_emb_text = ctx_emb_text.view(batch_size, k, -1)

        # Fuse Context (Text + Structure)
        ctx_fused = self._fuse_modalities(ctx_emb_text, ctx_struct) # (B, K, H)
        # Aggregate Context
        ctx_summary = torch.mean(ctx_fused, dim=1) # (B, H)
        
        # Main Fusion
        h_fused = self._fuse_modalities(h_emb_text, h_struct) # (B, H)
        r_fused = self._fuse_modalities(r_emb_text, r_struct) # (B, H)

        # LSTM Context Aggregation
        # Sequence: [Context, Head, Relation] -> Predict Tail
        lstm_input = torch.stack([ctx_summary, h_fused, r_fused], dim=1) # (B, 3, H)
        
        lstm_out, _ = self.lstm(lstm_input)
        query_vector = lstm_out[:, -1, :] # Last hidden state (B, H)
        
        # Project Query
        query_vector = self.projector(query_vector)
        
        # Ensure normalization for cosine similarity / InfoNCE
        query_vector = torch.nn.functional.normalize(query_vector, p=2, dim=1)
        
        return query_vector

    def encode_target(self, t_batch):
        """
        Encode target/tail entities symmetrically (Fusion of Text + Structure).
        t_batch: dict {input_ids, attention_mask, id}
        Returns: (B, H) normalized fused embedding
        """
        if self.use_text_cache and not self.config.finetune_text_encoder:
            t_emb_text = self._lookup_cached_text(t_batch['id'], kind='entity')
        else:
            t_emb_text = self._encode_text(t_batch['input_ids'], t_batch['attention_mask'])
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
