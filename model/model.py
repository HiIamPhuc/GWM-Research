import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None


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

        pretrained_model = getattr(config, 'pretrained_model', 'bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        self.text_encoder = AutoModel.from_pretrained(pretrained_model)
        self._maybe_enable_lora()
        self.text_embedding_dim = self.text_encoder.config.hidden_size

        self.structural_dim = int(getattr(config, 'structural_dim'))
        self.entity_embeddings = nn.Embedding(config.num_entities, self.structural_dim)
        self.relation_embeddings = nn.Embedding(config.num_relations, self.structural_dim)

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
        self.input_dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()

        self.score_lambda = nn.Parameter(torch.tensor(0.5))

        self.text_spatial_projection = self._build_projection(self.text_embedding_dim, self.text_compgcn_dim)
        self.struct_spatial_projection = self._build_projection(self.structural_dim, self.struct_compgcn_dim)
        self.text_dynamics_projection = self._build_projection(self.text_embedding_dim, self.text_dynamics_dim)
        self.struct_dynamics_projection = self._build_projection(self.structural_dim, self.struct_dynamics_dim)

        compgcn_layers = int(getattr(config, 'compgcn_layers', 1))
        self.text_compgcn_stack = nn.ModuleList([
            CompGCNLayer(hidden_dim=self.text_compgcn_dim, comp_op=getattr(config, 'compgcn_op', 'sub'))
            for _ in range(max(compgcn_layers, 1))
        ])
        self.struct_compgcn_stack = nn.ModuleList([
            CompGCNLayer(hidden_dim=self.struct_compgcn_dim, comp_op=getattr(config, 'compgcn_op', 'sub'))
            for _ in range(max(compgcn_layers, 1))
        ])

        dynamics_layers = int(getattr(config, 'dynamics_layers', getattr(config, 'num_layers', 1)))

        self.text_dynamics_mixer = nn.Sequential(
            nn.Linear(self.text_dynamics_dim * 2, self.text_dynamics_dim * 2),
            nn.GELU(),
            nn.Linear(self.text_dynamics_dim * 2, self.text_dynamics_dim)
        )
        self.text_lstm = nn.LSTM(
            input_size=self.text_dynamics_dim,
            hidden_size=self.text_dynamics_dim,
            num_layers=dynamics_layers,
            batch_first=True,
        )
        self.text_h0_projection = self._build_projection(self.text_compgcn_dim, self.text_dynamics_dim)
        self.text_c0_projection = self._build_projection(self.text_compgcn_dim, self.text_dynamics_dim)

        self.struct_dynamics_mixer = nn.Sequential(
            nn.Linear(self.struct_dynamics_dim * 2, self.struct_dynamics_dim * 2),
            nn.GELU(),
            nn.Linear(self.struct_dynamics_dim * 2, self.struct_dynamics_dim)
        )
        self.struct_lstm = nn.LSTM(
            input_size=self.struct_dynamics_dim,
            hidden_size=self.struct_dynamics_dim,
            num_layers=dynamics_layers,
            batch_first=True,
        )
        self.struct_h0_projection = self._build_projection(self.struct_compgcn_dim, self.struct_dynamics_dim)
        self.struct_c0_projection = self._build_projection(self.struct_compgcn_dim, self.struct_dynamics_dim)

        self.recurrent_dropout_layer = nn.Dropout(self.recurrent_dropout) if self.recurrent_dropout > 0 else nn.Identity()

    def _build_projection(self, in_dim, out_dim):
        if in_dim == out_dim:
            return nn.Identity()
        return nn.Linear(in_dim, out_dim)

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

        if entity_cache.size(1) != self.structural_dim:
            raise ValueError(
                f"Structural embedding dim mismatch. Config expects {self.structural_dim}, got {entity_cache.size(1)}"
            )

        self.entity_embeddings.weight.data.copy_(entity_cache)
        self.relation_embeddings.weight.data.copy_(relation_cache)

        if freeze:
            self.entity_embeddings.weight.requires_grad = False
            self.relation_embeddings.weight.requires_grad = False

    def _maybe_enable_lora(self):
        if not getattr(self.config, 'lora_enabled', False):
            return
        if LoraConfig is None or get_peft_model is None:
            raise RuntimeError("peft is required for LoRA. Install with: pip install peft")

        target_modules = getattr(self.config, 'lora_target_modules', ['query', 'value'])
        if isinstance(target_modules, str):
            target_modules = [m.strip() for m in target_modules.split(',') if m.strip()]

        lora_config = LoraConfig(
            r=int(getattr(self.config, 'lora_rank', 8)),
            lora_alpha=int(getattr(self.config, 'lora_alpha', 16)),
            lora_dropout=float(getattr(self.config, 'lora_dropout', 0.05)),
            target_modules=target_modules,
            bias='none',
            task_type='FEATURE_EXTRACTION',
        )
        self.text_encoder = get_peft_model(self.text_encoder, lora_config)

    def _encode_text(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

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

    def _run_dynamics(self, world_state, step_x, relation_emb, mixer, lstm, h0_proj, c0_proj):
        h_0 = torch.tanh(h0_proj(world_state))
        c_0 = torch.tanh(c0_proj(world_state))

        concat_input = torch.cat([step_x, relation_emb], dim=-1)
        mixed_input = mixer(concat_input)
        lstm_input = mixed_input.unsqueeze(1)

        num_layers = lstm.num_layers
        h_0_lstm = h_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
        c_0_lstm = c_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous()

        _, (h_n, _) = lstm(lstm_input, (h_0_lstm, c_0_lstm))
        query_vector = h_n[-1]
        query_vector = self.recurrent_dropout_layer(query_vector)

        return torch.nn.functional.normalize(query_vector, p=2, dim=1)

    def get_alpha_mean(self, reset=False):
        return torch.sigmoid(self.score_lambda).item()

    def forward(self, h_batch, r_batch, context_batch):
        h_emb_text = self._encode_text(h_batch['input_ids'], h_batch['attention_mask'])
        r_emb_text = self._encode_text(r_batch['input_ids'], r_batch['attention_mask'])

        h_struct = self.entity_embeddings(h_batch['id'])
        r_struct = self.relation_embeddings(r_batch['id'])

        h_emb_text = self.input_dropout(h_emb_text)
        r_emb_text = self.input_dropout(r_emb_text)
        h_struct = self.input_dropout(h_struct)
        r_struct = self.input_dropout(r_struct)

        context_entity_ids = context_batch['id']
        ctx_input_ids = context_batch['input_ids']
        ctx_attention_mask = context_batch['attention_mask']
        context_mask = context_batch.get('mask')

        if context_entity_ids.dim() == 2:
            batch_size, context_size = context_entity_ids.shape
            seq_len = ctx_input_ids.size(-1)
            ctx_input_ids = ctx_input_ids.reshape(batch_size, context_size, seq_len)
            ctx_attention_mask = ctx_attention_mask.reshape(batch_size, context_size, seq_len)
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

        ctx_ent_text = self._encode_text(flat_ctx_input_ids, flat_ctx_attn)
        ctx_ent_struct = self.entity_embeddings(flat_context_entity_ids)

        ctx_rel_text = torch.zeros(ctx_ent_text.size(0), self.text_embedding_dim, device=ctx_ent_text.device)
        ctx_rel_struct = self.relation_embeddings(flat_context_relation_ids)

        h_spatial_text = self.text_spatial_projection(h_emb_text)
        ctx_entity_spatial_text = self.text_spatial_projection(ctx_ent_text)
        ctx_relation_spatial_text = self.text_spatial_projection(ctx_rel_text)

        h_spatial_struct = self.struct_spatial_projection(h_struct)
        ctx_entity_spatial_struct = self.struct_spatial_projection(ctx_ent_struct)
        ctx_relation_spatial_struct = self.struct_spatial_projection(ctx_rel_struct)

        world_state_text = self._encode_subgraph_with_compgcn(
            h_spatial_text,
            ctx_entity_spatial_text,
            ctx_relation_spatial_text,
            context_batch_index,
            self.text_compgcn_stack,
        )
        world_state_text = self.input_dropout(world_state_text)

        world_state_struct = self._encode_subgraph_with_compgcn(
            h_spatial_struct,
            ctx_entity_spatial_struct,
            ctx_relation_spatial_struct,
            context_batch_index,
            self.struct_compgcn_stack,
        )
        world_state_struct = self.input_dropout(world_state_struct)

        step_x_text = self.text_dynamics_projection(h_emb_text)
        relation_emb_text = self.text_dynamics_projection(r_emb_text)
        query_text = self._run_dynamics(
            world_state_text,
            step_x_text,
            relation_emb_text,
            self.text_dynamics_mixer,
            self.text_lstm,
            self.text_h0_projection,
            self.text_c0_projection,
        )

        step_x_struct = self.struct_dynamics_projection(h_struct)
        relation_emb_struct = self.struct_dynamics_projection(r_struct)
        query_struct = self._run_dynamics(
            world_state_struct,
            step_x_struct,
            relation_emb_struct,
            self.struct_dynamics_mixer,
            self.struct_lstm,
            self.struct_h0_projection,
            self.struct_c0_projection,
        )

        return query_text, query_struct

    def encode_target(self, t_batch):
        t_emb_text = self._encode_text(t_batch['input_ids'], t_batch['attention_mask'])
        t_struct = self.entity_embeddings(t_batch['id'])

        t_text_proj = self.text_dynamics_projection(t_emb_text)
        t_struct_proj = self.struct_dynamics_projection(t_struct)

        t_text_norm = torch.nn.functional.normalize(t_text_proj, p=2, dim=1)
        t_struct_norm = torch.nn.functional.normalize(t_struct_proj, p=2, dim=1)
        return t_text_norm, t_struct_norm

    def compute_loss(self, query_vectors, target_vectors):
        query_text, query_struct = query_vectors
        target_text, target_struct = target_vectors

        scores_text = torch.mm(query_text, target_text.t())
        scores_struct = torch.mm(query_struct, target_struct.t())

        temp = getattr(self.config, 'temperature', 0.07)
        scores_text /= temp
        scores_struct /= temp

        alpha = torch.sigmoid(self.score_lambda)
        scores = alpha * scores_text + (1.0 - alpha) * scores_struct

        labels = torch.arange(scores.size(0), device=scores.device)
        return nn.CrossEntropyLoss()(scores, labels), scores
