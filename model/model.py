import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AsymmetricContrastiveLoss(nn.Module):
    """Asymmetric focal contrastive loss for one-positive-many-negative ranking.

    This is a practical adaptation for KG tail reranking: each query has one
    positive target on the diagonal of the in-batch similarity matrix, and all
    off-diagonal entries are negatives. The loss down-weights easy negatives more
    aggressively than positives.
    """

    def __init__(
        self,
        temperature=0.07,
        gamma_pos=0.0,
        gamma_neg=4.0,
        clip=0.05,
        pos_weight=1.0,
        neg_weight=1.0,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)

    def forward(self, scores):
        if scores.dim() != 2 or scores.size(0) != scores.size(1):
            raise ValueError(
                f"AsymmetricContrastiveLoss expects a square score matrix, got {tuple(scores.shape)}"
            )

        logits = scores / max(self.temperature, 1e-8)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        probs = torch.sigmoid(logits)
        bsz = probs.size(0)
        eye = torch.eye(bsz, device=probs.device, dtype=torch.bool)

        pos_logits = logits.diagonal()
        neg_logits = logits.masked_select(~eye)
        pos_prob = probs.diagonal().clamp(min=1e-6, max=1.0 - 1e-6)
        neg_prob = probs.masked_select(~eye).clamp(min=1e-6, max=1.0 - 1e-6)

        if self.clip > 0.0:
            neg_prob = torch.clamp(neg_prob + self.clip, max=1.0 - 1e-6)

        pos_bce = F.binary_cross_entropy_with_logits(
            pos_logits,
            torch.ones_like(pos_logits),
            reduction='none',
        )
        neg_bce = F.binary_cross_entropy_with_logits(
            neg_logits,
            torch.zeros_like(neg_logits),
            reduction='none',
        )

        pos_term = torch.pow(1.0 - pos_prob, self.gamma_pos) * pos_bce
        neg_term = torch.pow(neg_prob, self.gamma_neg) * neg_bce

        if neg_term.numel() > 0:
            neg_term = neg_term.view(bsz, -1).mean(dim=1)
        else:
            neg_term = torch.zeros_like(pos_term)

        loss = self.pos_weight * pos_term + self.neg_weight * neg_term
        return loss.mean()

class GWM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Text Components (Entity/Relation Embeddings)
        self.text_emb_dim = int(getattr(config, 'text_emb_dim'))
        self.text_ent_embs = nn.Embedding(config.num_entities, self.text_emb_dim)
        self.text_rel_embs = nn.Embedding(config.num_relations, self.text_emb_dim)

        self.text_adapter = MLPAdapter(self.text_emb_dim,
            int(getattr(config, 'text_adapter_dim')),
            dropout=float(getattr(config, 'adapter_dropout'))
            )

        self.text_compgcn_stack = nn.ModuleList([
            CompGCNLayer(hidden_dim=self.text_emb_dim, comp_op=getattr(config, 'compgcn_op'))
            for _ in range(max(int(getattr(config, 'compgcn_layers')), 1))
        ])

        self.text_h0_projection = nn.Linear(self.text_emb_dim, self.text_emb_dim)
        self.text_c0_projection = nn.Linear(self.text_emb_dim, self.text_emb_dim)

        self.text_film = nn.Linear(self.text_emb_dim, 2 * self.text_emb_dim)

        self.text_lstm = nn.LSTM(
            input_size=self.text_emb_dim,
            hidden_size=self.text_emb_dim,
            num_layers=int(getattr(config, 'dynamics_layers', 1)),
            batch_first=True,
            dropout=float(getattr(config, 'recurrent_dropout'))
        )

        # 2. Structural Components (Entity/Relation Embeddings)
        self.struct_emb_dim = int(getattr(config, 'struct_emb_dim'))
        self.struct_ent_embs = nn.Embedding(config.num_entities, self.struct_emb_dim)
        self.struct_rel_embs = nn.Embedding(config.num_relations, self.struct_emb_dim)

        self.struct_adapter = MLPAdapter(self.struct_emb_dim, int(getattr(config, 'struct_adapter_dim')), dropout=float(getattr(config, 'adapter_dropout')))

        self.struct_compgcn_stack = nn.ModuleList([
            CompGCNLayer(hidden_dim=self.struct_emb_dim, comp_op=getattr(config, 'compgcn_op'))
            for _ in range(max(int(getattr(config, 'compgcn_layers')), 1))
        ])

        self.struct_h0_projection = nn.Linear(self.struct_emb_dim, self.struct_emb_dim)
        self.struct_c0_projection = nn.Linear(self.struct_emb_dim, self.struct_emb_dim)

        self.struct_film = nn.Linear(self.struct_emb_dim, 2 * self.struct_emb_dim)
        
        self.struct_lstm = nn.LSTM(
            input_size=self.struct_emb_dim, 
            hidden_size=self.struct_emb_dim,
            num_layers=int(getattr(config, 'dynamics_layers', 1)), 
            batch_first=True,
            dropout=float(getattr(config, 'recurrent_dropout'))
        )

        self.alpha_mlp = nn.Sequential(
            nn.Linear(self.text_emb_dim + self.struct_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self._alpha_sum = 0.0
        self._alpha_count = 0
        
        self.input_dropout = nn.Dropout(float(getattr(config, 'dropout')))

        self.sigreg_weight = float(getattr(config, 'sigreg_weight'))
        sigreg_knots = int(getattr(config, 'sigreg_knots'))
        sigreg_num_proj = int(getattr(config, 'sigreg_num_proj'))
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

        self.temperature = float(getattr(config, 'temperature'))
        self.loss_type = str(getattr(config, 'loss_type', 'acl')).lower()
        self.acl_gamma_pos = float(getattr(config, 'acl_gamma_pos', 0.0))
        self.acl_gamma_neg = float(getattr(config, 'acl_gamma_neg', 4.0))
        self.acl_clip = float(getattr(config, 'acl_clip', 0.05))
        self.acl_pos_weight = float(getattr(config, 'acl_pos_weight', 1.0))
        self.acl_neg_weight = float(getattr(config, 'acl_neg_weight', 1.0))
        self.acl = AsymmetricContrastiveLoss(
            temperature=self.temperature,
            gamma_pos=self.acl_gamma_pos,
            gamma_neg=self.acl_gamma_neg,
            clip=self.acl_clip,
            pos_weight=self.acl_pos_weight,
            neg_weight=self.acl_neg_weight,
        )

    def _ranking_loss(self, scores):
        if self.loss_type == 'acl':
            return self.acl(scores)

        labels = torch.arange(scores.size(0), device=scores.device)
        return F.cross_entropy(scores / self.temperature, labels)

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

    def _encode_context_features(self, entity_ids, relation_ids):
        entity_text = self.input_dropout(self.text_adapter(self.text_ent_embs(entity_ids)))
        relation_text = self.input_dropout(self.text_adapter(self.text_rel_embs(relation_ids)))
        entity_struct = self.input_dropout(self.struct_adapter(self.struct_ent_embs(entity_ids)))
        relation_struct = self.input_dropout(self.struct_adapter(self.struct_rel_embs(relation_ids)))
        return entity_text, relation_text, entity_struct, relation_struct

    def _run_dynamics(self, world_state, head_emb, relation_emb, lstm, h0_proj, c0_proj, path='text'):
        """
        Run recurrent dynamics over a sequence of steps.

        world_state: (B, compgcn_dim) used to initialise h0/c0
        head_emb: (B, D) head embedding for this path
        relation_emb: (B, D_rel) relation embedding for this path
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

        head_emb = (1.0 + scale) * head_emb + shift
        head_emb = head_emb.unsqueeze(1)

        # Run LSTM over the sequence
        lstm_out, (h_n, c_n) = lstm(head_emb, (h_0_lstm, c_0_lstm))
        query_vector = h_n[-1]
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
        h_text = self.input_dropout(self.text_adapter(self.text_ent_embs(h_batch['id'])))
        r_text = self.input_dropout(self.text_adapter(self.text_rel_embs(r_batch['id'])))
        h_struct = self.input_dropout(self.struct_adapter(self.struct_ent_embs(h_batch['id'])))
        r_struct = self.input_dropout(self.struct_adapter(self.struct_rel_embs(r_batch['id'])))

        flat_context_entity_ids, flat_context_relation_ids, context_batch_index = self._prepare_context_batch(context_batch)
        ctx_ent_text = self.input_dropout(self.text_adapter(self.text_ent_embs(flat_context_entity_ids)))
        ctx_rel_text = self.input_dropout(self.text_adapter(self.text_rel_embs(flat_context_relation_ids)))
        ctx_ent_struct = self.input_dropout(self.struct_adapter(self.struct_ent_embs(flat_context_entity_ids)))
        ctx_rel_struct = self.input_dropout(self.struct_adapter(self.struct_rel_embs(flat_context_relation_ids)))

        # -- Text Pathway --
        world_state_text = self._encode_subgraph_with_compgcn(
            h_text, ctx_ent_text, ctx_rel_text,
            context_batch_index, self.text_compgcn_stack
        )

        query_text_raw, query_text = self._run_dynamics(
            world_state_text, h_text, r_text,
            self.text_lstm, self.text_h0_projection, self.text_c0_projection, path='text'
        )

        # -- Struct Pathway --
        world_state_struct = self._encode_subgraph_with_compgcn(
            h_struct, ctx_ent_struct, ctx_rel_struct,
            context_batch_index, self.struct_compgcn_stack
        )
    
        query_struct_raw, query_struct = self._run_dynamics(
            world_state_struct, h_struct, r_struct,
            self.struct_lstm, self.struct_h0_projection, self.struct_c0_projection, path='struct'
        )

        return query_text, query_struct, r_text, r_struct, query_text_raw, query_struct_raw

    def encode_target(self, t_batch):
        t_text = self.text_adapter(self.text_ent_embs(t_batch['id']))
        t_struct = self.struct_adapter(self.struct_ent_embs(t_batch['id']))
        
        t_text_norm = torch.nn.functional.normalize(t_text, p=2, dim=1)
        t_struct_norm = torch.nn.functional.normalize(t_struct, p=2, dim=1)
        return t_text_norm, t_struct_norm

    def compute_loss(self, query_vectors, target_vectors):
        query_text, query_struct, relation_text, relation_struct, _, _ = query_vectors
        target_text, target_struct = target_vectors

        scores_text = torch.mm(query_text, target_text.t())
        scores_struct = torch.mm(query_struct, target_struct.t())

        scores_text = scores_text / self.temperature
        scores_struct = scores_struct / self.temperature

        head_combined = torch.cat([relation_text, relation_struct], dim=-1)
        alpha = self.alpha_mlp(head_combined)
        scores_fused = alpha * scores_text + (1.0 - alpha) * scores_struct
        self._record_alpha(alpha)

        loss_text = self._ranking_loss(scores_text)
        loss_struct = self._ranking_loss(scores_struct)

        return loss_text, loss_struct, scores_fused

    def compute_sigreg_loss(self, query_vectors):
        if self.sigreg is None or self.sigreg_weight <= 0.0:
            return None
        _, _, _, _, query_text_raw, query_struct_raw = query_vectors
        if query_text_raw is None or query_struct_raw is None:
            return None
        sig_text = self.sigreg(query_text_raw.unsqueeze(0))
        sig_struct = self.sigreg(query_struct_raw.unsqueeze(0))
        return 0.5 * (sig_text + sig_struct)