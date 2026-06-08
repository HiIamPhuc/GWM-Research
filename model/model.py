import math
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
        self.lin_out = nn.Linear(hidden_dim, hidden_dim)
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

        # Pure residual fix: preserve anchor head state, add learned neighbor delta.
        return self.dropout(self.lin_out(agg))


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


class GWM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = float(getattr(config, 'dropout'))

        # 1. Text Components (Entity/Relation Embeddings)
        self.text_emb_dim = int(getattr(config, 'text_emb_dim'))
        self.text_ent_embs = nn.Embedding(config.num_entities, self.text_emb_dim)
        self.text_rel_embs = nn.Embedding(config.num_relations, self.text_emb_dim)

        self.text_adapter = MLPAdapter(self.text_emb_dim,
            int(getattr(config, 'text_adapter_dim')),
            dropout=self.dropout
            )

        self.text_compgcn = CompGCN(hidden_dim=self.text_emb_dim, dropout=self.dropout)

        self.text_h0_projection = nn.Linear(self.text_emb_dim, self.text_emb_dim)
        self.text_c0_projection = nn.Linear(self.text_emb_dim, self.text_emb_dim)

        self.text_dynamics_mixer = DynamicsMixer(self.text_emb_dim, dropout=self.dropout)

        self.text_lstm = nn.LSTM(
            input_size=self.text_emb_dim,
            hidden_size=self.text_emb_dim,
            num_layers=int(getattr(config, 'dynamics_layers', 1)),
            batch_first=True,
            dropout=self.dropout
        )

        self.text_output_projection = nn.Linear(self.text_emb_dim, self.text_emb_dim)

        # 2. Structural Components (Entity/Relation Embeddings)
        self.struct_emb_dim = int(getattr(config, 'struct_emb_dim'))
        self.struct_ent_embs = nn.Embedding(config.num_entities, self.struct_emb_dim)
        self.struct_rel_embs = nn.Embedding(config.num_relations, self.struct_emb_dim)

        self.struct_adapter = MLPAdapter(self.struct_emb_dim, int(getattr(config, 'struct_adapter_dim')), dropout=self.dropout)

        self.struct_compgcn = CompGCN(hidden_dim=self.struct_emb_dim, dropout=self.dropout)

        self.struct_h0_projection = nn.Linear(self.struct_emb_dim, self.struct_emb_dim)
        self.struct_c0_projection = nn.Linear(self.struct_emb_dim, self.struct_emb_dim)

        self.struct_dynamics_mixer = DynamicsMixer(self.struct_emb_dim, dropout=self.dropout)
        
        self.struct_lstm = nn.LSTM(
            input_size=self.struct_emb_dim, 
            hidden_size=self.struct_emb_dim,
            num_layers=int(getattr(config, 'dynamics_layers', 1)), 
            batch_first=True,
            dropout=self.dropout
        )

        self.struct_output_projection = nn.Linear(self.struct_emb_dim, self.struct_emb_dim)

        self.alpha_mlp = nn.Sequential(
            nn.Linear(self.text_emb_dim + self.struct_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self._alpha_sum = 0.0
        self._alpha_count = 0
        
        self.balance_floor = float(getattr(config, 'balance_floor', 0.0))
        self.alpha_target = float(getattr(config, 'alpha_target', 0.5))
        self.alpha_tolerance = float(getattr(config, 'alpha_tolerance', 0.05))
        self.alpha_reg_weight = float(getattr(config, 'alpha_reg_weight', 0.0))

        self.temperature = float(getattr(config, 'temperature'))

    def _prepare_context_batch(self, context_batch):
        context_entity_ids = context_batch['id']
        context_relation_ids = context_batch.get('rel_id')
        context_batch_index = context_batch.get('batch_index')
        context_mask = context_batch.get('sequence_mask')
        if context_mask is None:
            context_mask = context_batch.get('mask')
        if context_mask is not None:
            context_mask = context_mask.bool()

        context_seq_entity_ids = context_batch.get('sequence_id')
        context_seq_relation_ids = context_batch.get('sequence_rel_id')

        if context_seq_entity_ids is not None and context_mask is not None:
            valid_idx = context_mask.nonzero(as_tuple=False)
            context_batch_index = valid_idx[:, 0]
            flat_entity_ids = context_seq_entity_ids[context_mask]
            flat_relation_ids = context_seq_relation_ids[context_mask]
        elif context_entity_ids.dim() == 2:
            if context_mask is None:
                raise ValueError("context_batch['mask'] is required for padded context format.")
            valid_idx = context_mask.nonzero(as_tuple=False)
            context_batch_index = valid_idx[:, 0]
            flat_entity_ids = context_entity_ids[context_mask]
            flat_relation_ids = context_relation_ids[context_mask]
        else:
            flat_entity_ids = context_entity_ids
            flat_relation_ids = context_relation_ids
            if context_batch_index is None:
                raise ValueError("context_batch['batch_index'] is required for ragged context format.")

        return flat_entity_ids, flat_relation_ids, context_batch_index

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
        lstm_out, (h_n, c_n) = lstm(mixed_step, (h_0_lstm, c_0_lstm))
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
            self.text_ent_embs.weight.requires_grad = False
            self.text_rel_embs.weight.requires_grad = False

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
        h_text = self.text_adapter(self.text_ent_embs(h_batch['id']))
        r_text = self.text_adapter(self.text_rel_embs(r_batch['id']))
        h_struct = self.struct_adapter(self.struct_ent_embs(h_batch['id']))
        r_struct = self.struct_adapter(self.struct_rel_embs(r_batch['id']))

        flat_context_entity_ids, flat_context_relation_ids, context_batch_index = self._prepare_context_batch(context_batch)
        ctx_ent_text = self.text_adapter(self.text_ent_embs(flat_context_entity_ids))
        ctx_rel_text = self.text_adapter(self.text_rel_embs(flat_context_relation_ids))
        ctx_ent_struct = self.struct_adapter(self.struct_ent_embs(flat_context_entity_ids))
        ctx_rel_struct = self.struct_adapter(self.struct_rel_embs(flat_context_relation_ids))

        # -- Text Pathway --
        world_state_text = self.text_compgcn(
            head_feat=h_text,
            nbr_entity_feat=ctx_ent_text,
            nbr_relation_feat=ctx_rel_text,
            nbr_batch_index=context_batch_index,
        )

        query_text = self._run_dynamics(
            world_state_text, h_text, r_text,
            self.text_dynamics_mixer, self.text_lstm, self.text_h0_projection, self.text_c0_projection
        )

        query_text = self.text_output_projection(query_text)
        query_text = torch.nn.functional.normalize(query_text, p=2, dim=1)

        # -- Struct Pathway --
        world_state_struct = self.struct_compgcn(
            head_feat=h_struct,
            nbr_entity_feat=ctx_ent_struct,
            nbr_relation_feat=ctx_rel_struct,
            nbr_batch_index=context_batch_index,
        )
    
        query_struct = self._run_dynamics(
            world_state_struct, h_struct, r_struct,
            self.struct_dynamics_mixer, self.struct_lstm, self.struct_h0_projection, self.struct_c0_projection
        )

        query_struct = self.struct_output_projection(query_struct)
        query_struct = torch.nn.functional.normalize(query_struct, p=2, dim=1)

        return query_text, query_struct, r_text, r_struct

    def encode_target(self, t_batch):
        t_text = self.text_adapter(self.text_ent_embs(t_batch['id']))
        t_struct = self.struct_adapter(self.struct_ent_embs(t_batch['id']))
        
        t_text = torch.nn.functional.normalize(self.text_output_projection(t_text), p=2, dim=1)
        t_struct = torch.nn.functional.normalize(self.struct_output_projection(t_struct), p=2, dim=1)
        return t_text, t_struct

    def compute_loss(self, query_vectors, target_vectors):
        query_text, query_struct, relation_text, relation_struct = query_vectors
        target_text, target_struct = target_vectors

        scores_text = torch.mm(query_text, target_text.t())
        scores_struct = torch.mm(query_struct, target_struct.t())
        labels = torch.arange(scores_text.size(0), device=scores_text.device)

        scores_text = scores_text / self.temperature
        scores_struct = scores_struct / self.temperature

        head_combined = torch.cat([relation_text, relation_struct], dim=-1)
        alpha = self.alpha_mlp(head_combined)
        scores_fused = alpha * scores_text + (1.0 - alpha) * scores_struct
        self._record_alpha(alpha)

        loss_text_per_sample = F.cross_entropy(scores_text, labels, reduction='none')
        loss_struct_per_sample = F.cross_entropy(scores_struct, labels, reduction='none')

        balance = alpha.squeeze(-1)
        if self.balance_floor > 0.0:
            balance = balance.clamp(min=self.balance_floor, max=1.0 - self.balance_floor)

        loss_text = loss_text_per_sample.mean()
        loss_struct = loss_struct_per_sample.mean()
        loss_main = (balance * loss_text_per_sample + (1.0 - balance) * loss_struct_per_sample).mean()

        alpha_mean = balance.mean()
        alpha_deviation = (alpha_mean - self.alpha_target).abs()
        alpha_reg = torch.relu(alpha_deviation - self.alpha_tolerance).pow(2)
        loss = loss_main + self.alpha_reg_weight * alpha_reg

        return loss_text, loss_struct, loss, scores_fused, alpha, alpha_reg

