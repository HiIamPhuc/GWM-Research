import torch
import torch.nn as nn
import torch.nn.functional as F


class GWM(nn.Module):
    """Head-centered world-state encoder with a transition decoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config.struct_emb_dim
        self.temperature = config.temperature
        self.temperature_min = config.temperature_min
        self.temperature_max = config.temperature_max

        self.struct_ent_embs = nn.Embedding(
            config.num_entities,
            self.embedding_dim,
        )
        self.base_rel_embs = nn.Embedding(
            config.num_base_relations,
            self.embedding_dim,
        )
        self.direction_embs = nn.Embedding(2, self.embedding_dim)
        self.inverse_adapter = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=False,
        )
        self.relation_norm = nn.LayerNorm(self.embedding_dim)
        self.temperature_head = nn.Linear(self.embedding_dim, 1)

        relation_base_ids = torch.as_tensor(
            config.relation_base_ids,
            dtype=torch.long,
        )
        relation_directions = torch.as_tensor(
            config.relation_directions,
            dtype=torch.long,
        )
        self.register_buffer('relation_base_ids', relation_base_ids)
        self.register_buffer('relation_directions', relation_directions)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=config.context_encoder_heads,
            dim_feedforward=(
                config.context_encoder_ffn_multiplier * self.embedding_dim
            ),
            dropout=config.context_encoder_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.context_encoder_layers,
            norm=nn.LayerNorm(self.embedding_dim),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embedding_dim,
            nhead=config.transition_decoder_heads,
            dim_feedforward=(
                config.transition_decoder_ffn_multiplier * self.embedding_dim
            ),
            dropout=config.transition_decoder_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transition_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.transition_decoder_layers,
            norm=nn.LayerNorm(self.embedding_dim),
        )
        self.next_state_projection = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=False,
        )
        self.context_fact_norm = nn.LayerNorm(self.embedding_dim)
        self.token_roles = nn.Embedding(3, self.embedding_dim)
        self.next_state_token = nn.Parameter(
            torch.empty(1, 1, self.embedding_dim)
        )
        self.masked_head_token = nn.Parameter(
            torch.empty(1, 1, self.embedding_dim)
        )
        self.register_buffer(
            'transition_mask',
            torch.tensor(
                [[False, True], [False, False]],
                dtype=torch.bool,
            ),
            persistent=False,
        )
        nn.init.normal_(self.token_roles.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.next_state_token, mean=0.0, std=0.02)
        nn.init.normal_(self.direction_embs.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.inverse_adapter.weight)
        nn.init.eye_(self.next_state_projection.weight)
        nn.init.normal_(self.masked_head_token, mean=0.0, std=0.02)
        initial_temperature = (
            (self.temperature - self.temperature_min)
            / (self.temperature_max - self.temperature_min)
        )
        nn.init.zeros_(self.temperature_head.weight)
        nn.init.constant_(
            self.temperature_head.bias,
            torch.logit(torch.tensor(initial_temperature)).item(),
        )

    def encode_relation(self, relation_ids):
        base_ids = self.relation_base_ids[relation_ids]
        directions = self.relation_directions[relation_ids]
        base = self.base_rel_embs(base_ids)
        relation = base + self.direction_embs(directions)
        inverse_mask = directions.unsqueeze(-1).to(dtype=base.dtype)
        relation = relation + inverse_mask * self.inverse_adapter(base)
        return self.relation_norm(relation)

    def relation_temperature(self, relation):
        return self.temperature_min + (
            self.temperature_max - self.temperature_min
        ) * torch.sigmoid(self.temperature_head(relation))

    def encode_world_state(
        self,
        h_batch,
        context_batch,
        mask_head=False,
    ):
        entity_ids = context_batch['id']
        relation_ids = context_batch['rel_id']
        context_mask = context_batch['mask'].bool()

        if mask_head:
            head_token = self.masked_head_token.expand(
                entity_ids.size(0),
                -1,
                -1,
            ).squeeze(1)
        else:
            head_token = self.struct_ent_embs(h_batch['id'])
        head_token = head_token + self.token_roles.weight[0]
        safe_entity_ids = entity_ids.masked_fill(~context_mask, 0)
        safe_relation_ids = relation_ids.masked_fill(~context_mask, 0)
        context_entities = self.struct_ent_embs(safe_entity_ids)
        context_relations = self.encode_relation(safe_relation_ids)
        fact_tokens = self.context_fact_norm(
            context_entities + context_relations
        )
        fact_tokens = fact_tokens + self.token_roles.weight[1]
        fact_tokens = fact_tokens.masked_fill(
            ~context_mask.unsqueeze(-1),
            0.0,
        )

        batch_size = entity_ids.size(0)
        memory_tokens = torch.cat(
            [head_token.unsqueeze(1), fact_tokens],
            dim=1,
        )
        memory_padding_mask = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=context_mask.device,
                ),
                ~context_mask,
            ],
            dim=1,
        )
        memory = self.context_encoder(
            memory_tokens,
            src_key_padding_mask=memory_padding_mask,
        )
        return memory, memory_padding_mask

    def encode_masked_world_state(self, h_batch, context_batch):
        memory, _ = self.encode_world_state(
            h_batch,
            context_batch,
            mask_head=True,
        )
        return F.normalize(memory[:, 0], p=2, dim=-1)

    def encode_query(self, h_batch, r_batch, context_batch):
        relation = self.encode_relation(r_batch['id'])
        temperature = self.relation_temperature(relation)
        relation_token = relation + self.token_roles.weight[2]
        memory, memory_padding_mask = self.encode_world_state(
            h_batch,
            context_batch,
        )
        next_state_token = self.next_state_token.expand(
            relation_token.size(0),
            -1,
            -1,
        )
        transition_tokens = torch.cat(
            [relation_token.unsqueeze(1), next_state_token],
            dim=1,
        )
        decoded = self.transition_decoder(
            tgt=transition_tokens,
            memory=memory,
            tgt_mask=self.transition_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        next_state = self.next_state_projection(decoded[:, 1])
        return (
            F.normalize(next_state, p=2, dim=-1),
            temperature,
        )

    def forward(self, h_batch, r_batch, context_batch):
        return self.encode_query(h_batch, r_batch, context_batch)

    def encode_target(self, t_batch):
        target = self.struct_ent_embs(t_batch['id'])
        return F.normalize(target, p=2, dim=-1)

    def score_candidates(
        self,
        query_vector,
        temperature,
        candidate_vectors,
    ):
        return torch.mm(query_vector, candidate_vectors.t()) / temperature

    def compute_loss(
        self,
        query_vector,
        temperature,
        target_ids,
    ):
        candidate_vectors = F.normalize(
            self.struct_ent_embs.weight,
            p=2,
            dim=-1,
        )
        scores = self.score_candidates(
            query_vector,
            temperature,
            candidate_vectors,
        )
        return F.cross_entropy(scores, target_ids)

    def compute_state_reconstruction_loss(
        self,
        reconstructed_states,
        entity_ids,
    ):
        candidate_vectors = F.normalize(
            self.struct_ent_embs.weight,
            p=2,
            dim=-1,
        )
        scores = torch.mm(
            reconstructed_states,
            candidate_vectors.t(),
        )
        scores = scores / self.temperature
        return F.cross_entropy(scores, entity_ids)

    def score_all_entities(
        self,
        h_batch,
        r_batch,
        context_batch,
        candidate_vectors=None,
    ):
        query_vector, temperature = self.encode_query(
            h_batch,
            r_batch,
            context_batch,
        )
        if candidate_vectors is None:
            entity_ids = torch.arange(
                self.config.num_entities,
                device=query_vector.device,
                dtype=torch.long,
            )
            candidate_vectors = self.encode_target({'id': entity_ids})

        return self.score_candidates(
            query_vector,
            temperature,
            candidate_vectors,
        )
