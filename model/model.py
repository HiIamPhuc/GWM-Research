import torch
import torch.nn as nn
from transformers import AutoModel

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
        raise ValueError(f"Unsupported compgcn_op: {self.comp_op}")

    def forward(self, head_feat, nbr_entity_feat, nbr_relation_feat, nbr_batch_index):
        B = head_feat.size(0)
        composed = self._compose(nbr_entity_feat, nbr_relation_feat)
        msg = self.lin_msg(composed)

        agg = torch.zeros_like(head_feat)
        if msg.numel() > 0:
            agg.index_add_(0, nbr_batch_index, msg)

        denom = torch.zeros(B, 1, device=head_feat.device, dtype=head_feat.dtype)
        if msg.numel() > 0:
            ones = torch.ones(msg.size(0), 1, device=head_feat.device, dtype=head_feat.dtype)
            denom.index_add_(0, nbr_batch_index, ones)
        denom = denom.clamp(min=1.0)
        agg = agg / denom

        return head_feat + self.lin_out(agg)

class GWM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # --- TEXT ENCODER (Strategy B: Fine-tunable Sentence Transformer) ---
        pretrained_model = getattr(config, 'pretrained_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.text_encoder = AutoModel.from_pretrained(pretrained_model)
        self.text_embedding_dim = self.text_encoder.config.hidden_size

        # 1. Structural Component
        legacy_hidden_dim = int(getattr(config, 'hidden_dim', 768))
        self.structural_dim = int(getattr(config, 'structural_dim', legacy_hidden_dim))
        self.entity_embeddings = nn.Embedding(config.num_entities, self.structural_dim)
        self.relation_embeddings = nn.Embedding(config.num_relations, self.structural_dim)

        # Latent spaces
        self.fusion_dim = int(getattr(config, 'fusion_dim', legacy_hidden_dim))
        self.compgcn_dim = int(getattr(config, 'compgcn_dim', self.fusion_dim))
        self.dynamics_dim = int(getattr(config, 'dynamics_dim', self.fusion_dim))
        self.dropout_rate = float(getattr(config, 'dropout', 0.0))
        self.recurrent_dropout = float(getattr(config, 'recurrent_dropout', 0.0))

        self.input_dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        self.spatial_projection = nn.Identity() if self.fusion_dim == self.compgcn_dim else nn.Linear(self.fusion_dim, self.compgcn_dim)
        self.dynamics_projection = nn.Identity() if self.fusion_dim == self.dynamics_dim else nn.Linear(self.fusion_dim, self.dynamics_dim)
        self.query_projection = nn.Identity() if self.dynamics_dim == self.fusion_dim else nn.Linear(self.dynamics_dim, self.fusion_dim)
        
        # 2. Spatial Encoder (CompGCN)
        compgcn_layers = int(getattr(config, 'compgcn_layers', 1))
        self.compgcn_stack = nn.ModuleList([
            CompGCNLayer(hidden_dim=self.compgcn_dim, comp_op=getattr(config, 'compgcn_op', 'sub'))
            for _ in range(max(compgcn_layers, 1))
        ])
        
        # 3. Transition Dynamics (Standard single-step LSTM)
        self.lstm = nn.LSTM(
            input_size=self.dynamics_dim * 2,
            hidden_size=self.dynamics_dim,
            num_layers=int(getattr(config, 'dynamics_layers', 1)),
            batch_first=True
        )
        self.recurrent_dropout_layer = nn.Dropout(self.recurrent_dropout) if self.recurrent_dropout > 0 else nn.Identity()
        
        self.h0_projection = nn.Linear(self.compgcn_dim, self.dynamics_dim)
        self.c0_projection = nn.Linear(self.compgcn_dim, self.dynamics_dim)
        
        # 4. Fusion Layer (Raw Preservation - Option A)
        self.fusion_mode = getattr(config, 'fusion_mode', 'concat')
        self.fusion = nn.Linear(self.text_embedding_dim + self.structural_dim, self.fusion_dim)

        if self.fusion_mode == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(self.text_embedding_dim + self.structural_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            self.fusion_projection = nn.Linear(self.text_embedding_dim + self.structural_dim, self.fusion_dim)

        self.reset_alpha_stats()

    def _encode_text(self, input_ids, attention_mask):
        """Mean Pooling (Strategy B for Sentence Transformers)"""
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def reset_alpha_stats(self):
        self._alpha_sum = 0.0
        self._alpha_count = 0

    def _fuse_modalities(self, text_emb, struct_emb):
        text_emb = self.input_dropout(text_emb)
        struct_emb = self.input_dropout(struct_emb)

        if self.fusion_mode == 'gated':
            gate_input = torch.cat([text_emb, struct_emb], dim=-1)
            alpha = self.gate(gate_input)
            self._alpha_sum += alpha.detach().sum().item()
            self._alpha_count += alpha.numel()
            weighted_concat = torch.cat([alpha * text_emb, (1.0 - alpha) * struct_emb], dim=-1)
            return self.fusion_projection(weighted_concat)

        return self.fusion(torch.cat([text_emb, struct_emb], dim=-1))

    def _encode_subgraph_with_compgcn(self, h_fused, ctx_entity_fused, ctx_relation_fused, ctx_batch_index):
        h_state = h_fused
        for layer in self.compgcn_stack:
            h_state = layer(head_feat=h_state, nbr_entity_feat=ctx_entity_fused, nbr_relation_feat=ctx_relation_fused, nbr_batch_index=ctx_batch_index)
        return h_state

    def forward(self, h_batch, r_batch, context_batch):
        # 1. Text Encoder (Live Finetuning)
        h_emb_text = self._encode_text(h_batch['input_ids'], h_batch['attention_mask'])
        r_emb_text = self._encode_text(r_batch['input_ids'], r_batch['attention_mask'])
        
        # 1b. Structural Embeddings
        h_struct = self.entity_embeddings(h_batch['id'])
        r_struct = self.relation_embeddings(r_batch['id'])

        # 2. Main Fusion
        h_fused = self.input_dropout(self._fuse_modalities(h_emb_text, h_struct))
        r_fused = self.input_dropout(self._fuse_modalities(r_emb_text, r_struct))
        
        # 3. Context Processing (Ragged format matching current architecture)
        context_entity_ids = context_batch['id']
        ctx_input_ids = context_batch['input_ids']
        ctx_attention_mask = context_batch['attention_mask']
        context_mask = context_batch.get('mask')

        if context_entity_ids.dim() == 2:
            B, K = context_entity_ids.shape
            if context_mask is None:
                context_mask = torch.ones_like(context_entity_ids, dtype=torch.bool)
            else:
                context_mask = context_mask.bool()

            valid_idx = context_mask.nonzero(as_tuple=False)
            if valid_idx.numel() == 0:
                flat_ctx_input_ids = ctx_input_ids.new_zeros((0, ctx_input_ids.size(-1)))
                flat_ctx_attn = ctx_attention_mask.new_zeros((0, ctx_attention_mask.size(-1)))
                flat_context_entity_ids = context_entity_ids.new_zeros((0,))
                flat_context_relation_ids = context_entity_ids.new_zeros((0,))
                context_batch_index = context_entity_ids.new_zeros((0,))
            else:
                context_batch_index = valid_idx[:, 0]
                flat_ctx_input_ids = ctx_input_ids[context_mask]
                flat_ctx_attn = ctx_attention_mask[context_mask]
                flat_context_entity_ids = context_entity_ids[context_mask]
                flat_context_relation_ids = context_batch.get('rel_id', torch.zeros_like(context_entity_ids))[context_mask]
        else:
            flat_ctx_input_ids = ctx_input_ids
            flat_ctx_attn = ctx_attention_mask
            flat_context_entity_ids = context_entity_ids
            flat_context_relation_ids = context_batch.get('rel_id', torch.zeros_like(context_entity_ids))
            context_batch_index = context_batch['batch_index']

        # Encode Context Text & Struct
        ctx_ent_text = self._encode_text(flat_ctx_input_ids, flat_ctx_attn)
        ctx_ent_struct = self.entity_embeddings(flat_context_entity_ids)
        
        # Note: Assuming context relations don't have text strings provided for simplicity right now.
        # If they do, they should be encoded similarly. We use zero placeholders for relations without text.
        ctx_rel_text = torch.zeros(ctx_ent_text.size(0), self.text_embedding_dim, device=ctx_ent_text.device)
        ctx_rel_struct = self.relation_embeddings(flat_context_relation_ids)

        ctx_entity_fused = self._fuse_modalities(ctx_ent_text, ctx_ent_struct)
        ctx_relation_fused = self._fuse_modalities(ctx_rel_text, ctx_rel_struct)

        h_spatial = self.spatial_projection(h_fused)
        ctx_entity_spatial = self.spatial_projection(ctx_entity_fused)
        ctx_relation_spatial = self.spatial_projection(ctx_relation_fused)

        # 4. Spatial Encoder
        world_state = self.input_dropout(self._encode_subgraph_with_compgcn(
            h_spatial, ctx_entity_spatial, ctx_relation_spatial, context_batch_index
        ))

        # 5. Dynamics Single-step LSTM
        h_0 = torch.tanh(self.h0_projection(world_state))
        c_0 = torch.tanh(self.c0_projection(world_state))
        
        step_x = self.dynamics_projection(h_fused)
        rel_proj = self.dynamics_projection(r_fused)
        
        lstm_input = torch.cat([step_x, rel_proj], dim=-1).unsqueeze(1)
        h_0_lstm = h_0.unsqueeze(0).contiguous()
        c_0_lstm = c_0.unsqueeze(0).contiguous()
        
        lstm_out, (h_n, c_n) = self.lstm(lstm_input, (h_0_lstm, c_0_lstm))
        query_vector = self.query_projection(self.recurrent_dropout_layer(h_n[-1]))
        
        return torch.nn.functional.normalize(query_vector, p=2, dim=1)

    def encode_target(self, t_batch):
        t_emb_text = self._encode_text(t_batch['input_ids'], t_batch['attention_mask'])
        t_struct = self.entity_embeddings(t_batch['id'])
        return torch.nn.functional.normalize(self._fuse_modalities(t_emb_text, t_struct), p=2, dim=1)

    def compute_loss(self, query_vector, t_fused):
        scores = torch.mm(query_vector, t_fused.t())
        if hasattr(self.config, 'temperature'):
            scores /= self.config.temperature
        else:
            scores /= 0.07
        labels = torch.arange(scores.size(0), device=scores.device)
        return nn.CrossEntropyLoss()(scores, labels), scores
