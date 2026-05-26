import torch
import torch.nn as nn


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

        self.text_emb_dim = int(getattr(config, 'text_emb_dim'))
        self.struct_emb_dim = int(getattr(config, 'struct_emb_dim'))

        # 1. Text Component (Entity/Relation Embeddings)
        self.text_ent_embs = nn.Embedding(config.num_entities, self.text_emb_dim)
        self.text_rel_embs = nn.Embedding(config.num_relations, self.text_emb_dim)

        # 2. Structural Component (Entity/Relation Embeddings)
        self.struct_ent_embs = nn.Embedding(config.num_entities, self.struct_emb_dim)
        self.struct_rel_embs = nn.Embedding(config.num_relations, self.struct_emb_dim)

        self.text_adapter = MLPAdapter(self.text_emb_dim, int(getattr(config, 'text_adapter_dim')), dropout=float(getattr(config, 'adapter_dropout')))
        self.struct_adapter = MLPAdapter(self.struct_emb_dim, int(getattr(config, 'struct_adapter_dim')), dropout=float(getattr(config, 'adapter_dropout')))

        self.input_dropout = nn.Dropout(float(getattr(config, 'dropout')))

        self.fusion_alpha = min(max(float(getattr(config, 'fusion_alpha')), 0.0), 1.0)
        self.temperature = float(getattr(config, 'temperature'))

        self.sigreg_weight = float(getattr(config, 'sigreg_weight'))
        sigreg_knots = int(getattr(config, 'sigreg_knots'))
        sigreg_num_proj = int(getattr(config, 'sigreg_num_proj'))
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)
        self.multi_step_weight = float(getattr(config, 'multi_step_weight'))

        self.text_step_projection = nn.Sequential(
            nn.Linear(self.text_emb_dim * 2, self.text_emb_dim),
            nn.GELU(),
            nn.Linear(self.text_emb_dim, self.text_emb_dim),
        )
        self.text_h0_projection = nn.Linear(self.text_emb_dim, self.text_emb_dim)
        self.text_c0_projection = nn.Linear(self.text_emb_dim, self.text_emb_dim)
        
        self.struct_step_projection = nn.Sequential(
            nn.Linear(self.struct_emb_dim * 2, self.struct_emb_dim),
            nn.GELU(),
            nn.Linear(self.struct_emb_dim, self.struct_emb_dim),
        )
        self.struct_h0_projection = nn.Linear(self.struct_emb_dim, self.struct_emb_dim)
        self.struct_c0_projection = nn.Linear(self.struct_emb_dim, self.struct_emb_dim)
        
        self.text_lstm = nn.LSTM(
            input_size=self.text_emb_dim, hidden_size=self.text_emb_dim,
            num_layers=int(getattr(config, 'dynamics_layers', 1)), batch_first=True
        )
        self.struct_lstm = nn.LSTM(
            input_size=self.struct_emb_dim, hidden_size=self.struct_emb_dim,
            num_layers=int(getattr(config, 'dynamics_layers', 1)), batch_first=True
        )
        self.recurrent_dropout_layer = nn.Dropout(float(getattr(config, 'recurrent_dropout')))
        
        # FiLM (relation-conditioned) projections for modulating LSTM inputs
        self.text_film = nn.Linear(self.text_emb_dim, 2 * self.text_emb_dim)
        self.struct_film = nn.Linear(self.struct_emb_dim, 2 * self.struct_emb_dim)

        # Learned query tokens for the final query step
        self.text_query_token = nn.Parameter(torch.zeros(1, 1, self.text_emb_dim))
        self.struct_query_token = nn.Parameter(torch.zeros(1, 1, self.struct_emb_dim))
        nn.init.normal_(self.text_query_token, mean=0.0, std=0.02)
        nn.init.normal_(self.struct_query_token, mean=0.0, std=0.02)

    def _build_context_sequences(self, context_entity_ids, context_relation_ids, context_mask, context_batch_index, batch_size, device):
        if context_entity_ids.dim() == 2:
            if context_relation_ids is None:
                context_relation_ids = torch.zeros_like(context_entity_ids)
            if context_mask is None:
                context_mask = torch.ones_like(context_entity_ids, dtype=torch.bool)
            else:
                context_mask = context_mask.bool()

            lengths = context_mask.long().sum(dim=1)
            lengths = lengths.clamp(min=1)
            if (context_mask.long().sum(dim=1) == 0).any():
                context_entity_ids = context_entity_ids.clone()
                context_relation_ids = context_relation_ids.clone()
                context_mask = context_mask.clone()
                empty_rows = context_mask.long().sum(dim=1) == 0
                context_entity_ids[empty_rows, 0] = 0
                context_relation_ids[empty_rows, 0] = 0
                context_mask[empty_rows, 0] = True
                lengths[empty_rows] = 1
            return context_entity_ids, context_relation_ids, context_mask, lengths

        if context_batch_index is None:
            raise ValueError("context_batch['batch_index'] is required for ragged context format.")

        if context_relation_ids is None:
            context_relation_ids = torch.zeros_like(context_entity_ids)
        if context_mask is None:
            context_mask = torch.ones_like(context_entity_ids, dtype=torch.bool)
        else:
            context_mask = context_mask.bool()

        batch_index_cpu = context_batch_index.detach().cpu()
        entity_cpu = context_entity_ids.detach().cpu()
        relation_cpu = context_relation_ids.detach().cpu()
        mask_cpu = context_mask.detach().cpu()

        per_batch_entities = []
        per_batch_relations = []
        per_batch_lengths = []
        for batch_id in range(batch_size):
            selected = (batch_index_cpu == batch_id) & mask_cpu
            batch_entities = entity_cpu[selected]
            batch_relations = relation_cpu[selected]
            length = int(batch_entities.numel())
            if length <= 0:
                length = 1
                batch_entities = torch.zeros(1, dtype=entity_cpu.dtype)
                batch_relations = torch.zeros(1, dtype=relation_cpu.dtype)
            per_batch_entities.append(batch_entities)
            per_batch_relations.append(batch_relations)
            per_batch_lengths.append(length)

        max_len = max(per_batch_lengths) if per_batch_lengths else 1
        entity_ids = torch.zeros(batch_size, max_len, dtype=context_entity_ids.dtype, device=device)
        relation_ids = torch.zeros(batch_size, max_len, dtype=context_relation_ids.dtype, device=device)
        context_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

        for batch_id, (batch_entities, batch_relations, length) in enumerate(zip(per_batch_entities, per_batch_relations, per_batch_lengths)):
            entity_ids[batch_id, :length] = batch_entities.to(device)
            relation_ids[batch_id, :length] = batch_relations.to(device)
            context_mask[batch_id, :length] = True
            lengths[batch_id] = length

        return entity_ids, relation_ids, context_mask, lengths

    def _run_dynamics(
        self,
        head_state,
        step_seq,
        step_rel_emb,
        query_step,
        query_rel_emb,
        lstm,
        h0_proj,
        c0_proj,
        step_mask,
        path='text',
    ):
        h_0 = torch.tanh(h0_proj(head_state))
        c_0 = c0_proj(head_state)

        num_layers = lstm.num_layers
        h_0_lstm = h_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
        c_0_lstm = c_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous()

        if path == 'text':
            film_params = self.text_film(step_rel_emb)
        else:
            film_params = self.struct_film(step_rel_emb)

        shift, scale = film_params.chunk(2, dim=-1)
        step_seq = (1.0 + scale) * step_seq + shift

        step_states, (h_n, c_n) = lstm(step_seq, (h_0_lstm, c_0_lstm))

        prev_states = torch.cat([h_0.unsqueeze(1), step_states[:, :-1, :]], dim=1)
        deltas = step_states - prev_states

        if step_mask is not None:
            mask = step_mask.unsqueeze(-1).to(deltas.dtype)
            deltas = deltas * mask

        if step_mask is None:
            last_state = step_states[:, -1, :]
        else:
            lengths = step_mask.long().sum(dim=1).clamp(min=1)
            last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, step_states.size(-1))
            last_state = step_states.gather(1, last_idx).squeeze(1)

        if path == 'text':
            query_film = self.text_film(query_rel_emb)
        else:
            query_film = self.struct_film(query_rel_emb)
        q_shift, q_scale = query_film.chunk(2, dim=-1)
        query_step = (1.0 + q_scale.unsqueeze(1)) * query_step + q_shift.unsqueeze(1)

        query_states, _ = lstm(query_step, (h_n, c_n))
        query_state = query_states[:, -1, :]
        query_delta = query_state - last_state

        query_vector = self.recurrent_dropout_layer(query_delta)
        query_norm = torch.nn.functional.normalize(query_vector, p=2, dim=1)
        return query_vector, query_norm, deltas

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

    def forward(self, h_batch, r_batch, context_batch):
        h_text = self.input_dropout(self.text_adapter(self.text_ent_embs(h_batch['id'])))
        r_text = self.input_dropout(self.text_adapter(self.text_rel_embs(r_batch['id'])))
        h_struct = self.input_dropout(self.struct_adapter(self.struct_ent_embs(h_batch['id'])))
        r_struct = self.input_dropout(self.struct_adapter(self.struct_rel_embs(r_batch['id'])))

        context_entity_ids = context_batch['id']
        context_relation_ids = context_batch.get('rel_id')
        context_batch_index = context_batch.get('batch_index')
        context_mask = context_batch.get('mask')

        context_entity_ids, context_relation_ids, context_mask, lengths = self._build_context_sequences(
            context_entity_ids,
            context_relation_ids,
            context_mask,
            context_batch_index,
            h_batch['id'].size(0),
            h_batch['id'].device,
        )

        ctx_ent_text = self.text_adapter(self.text_ent_embs(context_entity_ids))
        ctx_ent_struct = self.struct_adapter(self.struct_ent_embs(context_entity_ids))
        ctx_rel_text = self.text_adapter(self.text_rel_embs(context_relation_ids))
        ctx_rel_struct = self.struct_adapter(self.struct_rel_embs(context_relation_ids))

        step_seq_text = self.text_step_projection(torch.cat([ctx_ent_text, ctx_rel_text], dim=-1))
        step_seq_struct = self.struct_step_projection(torch.cat([ctx_ent_struct, ctx_rel_struct], dim=-1))

        query_token_text = self.text_query_token.expand(h_text.size(0), 1, -1)
        query_token_struct = self.struct_query_token.expand(h_struct.size(0), 1, -1)
        query_step_text = self.text_step_projection(torch.cat([query_token_text, r_text.unsqueeze(1)], dim=-1))
        query_step_struct = self.struct_step_projection(torch.cat([query_token_struct, r_struct.unsqueeze(1)], dim=-1))

        query_text_raw, query_text, text_deltas = self._run_dynamics(
            h_text,
            step_seq_text,
            ctx_rel_text,
            query_step_text,
            r_text,
            self.text_lstm,
            self.text_h0_projection,
            self.text_c0_projection,
            context_mask,
            path='text',
        )

        query_struct_raw, query_struct, struct_deltas = self._run_dynamics(
            h_struct,
            step_seq_struct,
            ctx_rel_struct,
            query_step_struct,
            r_struct,
            self.struct_lstm,
            self.struct_h0_projection,
            self.struct_c0_projection,
            context_mask,
            path='struct',
        )

        self._last_step_cache = {
            'text': {
                'deltas': text_deltas,
                'mask': context_mask,
                'context_ids': context_entity_ids,
            },
            'struct': {
                'deltas': struct_deltas,
                'mask': context_mask,
                'context_ids': context_entity_ids,
            },
        }
        self._last_query_raw = (query_text_raw, query_struct_raw)

        return query_text, query_struct

    def encode_target(self, t_batch):
        t_text_norm = torch.nn.functional.normalize(
            self.text_adapter(self.text_ent_embs(t_batch['id'])), 
            p=2, dim=1)
        t_struct_norm = torch.nn.functional.normalize(
            self.struct_adapter(self.struct_ent_embs(t_batch['id'])), 
            p=2, dim=1)
        return t_text_norm, t_struct_norm

    def compute_auxiliary_losses(self):
        if self.multi_step_weight <= 0.0:
            return None
        if not hasattr(self, '_last_step_cache') or self._last_step_cache is None:
            return None

        losses = {}
        total_aux = 0.0

        for branch_name, adapter, emb_table in (
            ('text', self.text_adapter, self.text_ent_embs),
            ('struct', self.struct_adapter, self.struct_ent_embs),
        ):
            cache = self._last_step_cache.get(branch_name)
            if cache is None:
                continue
            deltas = cache['deltas']
            mask = cache['mask']
            context_ids = cache['context_ids']
            if deltas is None or context_ids is None:
                continue

            deltas = torch.nn.functional.normalize(deltas, p=2, dim=-1)
            targets = torch.nn.functional.normalize(adapter(emb_table(context_ids)), p=2, dim=-1)

            flat_deltas = deltas.reshape(-1, deltas.size(-1))
            flat_targets = targets.reshape(-1, targets.size(-1))
            flat_mask = mask.reshape(-1) if mask is not None else None

            if flat_mask is not None:
                flat_deltas = flat_deltas[flat_mask]
                flat_targets = flat_targets[flat_mask]
            if flat_deltas.numel() == 0:
                continue

            scores = torch.mm(flat_deltas, flat_targets.t())
            labels = torch.arange(scores.size(0), device=scores.device)
            branch_loss = torch.nn.functional.cross_entropy(scores, labels)
            losses[f'{branch_name}_aux'] = branch_loss
            total_aux = total_aux + branch_loss

        if not losses:
            return None
        losses['total_aux'] = total_aux
        return losses

    def compute_loss(self, query_vectors, target_vectors, relation_ids=None):
        loss_text, loss_struct, scores = self.compute_loss_components(
            query_vectors,
            target_vectors,
            relation_ids=relation_ids,
        )
        return loss_text + loss_struct, scores

    def compute_sigreg_loss(self, query_vectors):
        if self.sigreg is None or self.sigreg_weight <= 0.0:
            return None
        if not hasattr(self, '_last_query_raw') or self._last_query_raw is None:
            return None
        query_text_raw, query_struct_raw = self._last_query_raw
        sig_text = self.sigreg(query_text_raw.unsqueeze(0))
        sig_struct = self.sigreg(query_struct_raw.unsqueeze(0))
        return 0.5 * (sig_text + sig_struct)

    def compute_loss_components(self, query_vectors, target_vectors, relation_ids=None):
        query_text, query_struct = query_vectors
        target_text, target_struct = target_vectors

        scores_text = torch.mm(query_text, target_text.t()) / self.temperature
        scores_struct = torch.mm(query_struct, target_struct.t()) / self.temperature

        alpha = query_text.new_full((query_text.size(0), 1), self.fusion_alpha)
        scores_fused = alpha * scores_text + (1.0 - alpha) * scores_struct

        labels = torch.arange(scores_text.size(0), device=scores_text.device)
        loss_text = torch.nn.functional.cross_entropy(scores_text, labels)
        loss_struct = torch.nn.functional.cross_entropy(scores_struct, labels)

        return loss_text, loss_struct, scores_fused
