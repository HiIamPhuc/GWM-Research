import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GWM(nn.Module):
    """Relation-conditioned next-state prediction from local graph memory."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config.struct_emb_dim
        self.temperature = config.temperature
        self.complex_residual_weight = config.complex_residual_weight

        self.struct_ent_embs = nn.Embedding(
            config.num_entities,
            self.embedding_dim,
        )

        self.base_rel_embs = nn.Embedding(
            config.num_base_relations,
            self.embedding_dim,
        )
        self.inverse_adapter = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=False,
        )
        self.relation_norm = nn.LayerNorm(self.embedding_dim)

        self.register_buffer(
            'relation_base_ids',
            torch.as_tensor(config.relation_base_ids, dtype=torch.long),
        )
        self.register_buffer(
            'relation_directions',
            torch.as_tensor(config.relation_directions, dtype=torch.long),
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
        self.memory_roles = nn.Parameter(torch.empty(2, self.embedding_dim))
        self.next_state_token = nn.Parameter(
            torch.empty(1, 1, self.embedding_dim)
        )

        nn.init.zeros_(self.inverse_adapter.weight)
        nn.init.normal_(self.memory_roles, mean=0.0, std=0.02)
        nn.init.normal_(self.next_state_token, mean=0.0, std=0.02)

    def encode_relation(self, relation_ids):
        base = self.base_rel_embs(self.relation_base_ids[relation_ids])
        inverse_mask = self.relation_directions[relation_ids]
        inverse_mask = inverse_mask.unsqueeze(-1).to(base.dtype)
        relation = base + inverse_mask * self.inverse_adapter(base)
        return self.relation_norm(relation)

    def build_world_memory(self, h_batch, context_batch):
        context_mask = context_batch['mask'].bool()
        entity_ids = context_batch['id'].masked_fill(~context_mask, 0)
        relation_ids = context_batch['rel_id'].masked_fill(~context_mask, 0)

        head = self.struct_ent_embs(h_batch['id'])
        context_facts = (
            self.struct_ent_embs(entity_ids)
            + self.encode_relation(relation_ids)
        )
        context_facts = context_facts.masked_fill(
            ~context_mask.unsqueeze(-1),
            0.0,
        )

        memory = torch.cat(
            [
                (head + self.memory_roles[0]).unsqueeze(1),
                context_facts + self.memory_roles[1],
            ],
            dim=1,
        )
        memory_padding_mask = torch.cat(
            [
                torch.zeros(
                    context_mask.size(0),
                    1,
                    dtype=torch.bool,
                    device=context_mask.device,
                ),
                ~context_mask,
            ],
            dim=1,
        )
        return memory, memory_padding_mask

    def encode_query(self, h_batch, r_batch, context_batch):
        relation = self.encode_relation(r_batch['id'])
        memory, memory_padding_mask = self.build_world_memory(
            h_batch,
            context_batch,
        )
        transition_query = self.next_state_token + relation.unsqueeze(1)
        next_state = self.transition_decoder(
            tgt=transition_query,
            memory=memory,
            memory_key_padding_mask=memory_padding_mask,
        )[:, 0]
        return F.normalize(next_state, p=2, dim=-1)

    def forward(self, h_batch, r_batch, context_batch):
        return self.encode_query(h_batch, r_batch, context_batch)

    def encode_target(self, t_batch):
        return F.normalize(
            self.struct_ent_embs(t_batch['id']),
            p=2,
            dim=-1,
        )

    def complex_scores(self, h_ids, r_ids, candidates):
        head = F.normalize(self.struct_ent_embs(h_ids), p=2, dim=-1)
        relation = F.normalize(self.encode_relation(r_ids), p=2, dim=-1)
        head_real, head_imag = head.chunk(2, dim=-1)
        relation_real, relation_imag = relation.chunk(2, dim=-1)
        tail_real, tail_imag = candidates.chunk(2, dim=-1)

        real = head_real * relation_real - head_imag * relation_imag
        imag = head_real * relation_imag + head_imag * relation_real
        scores = torch.mm(real, tail_real.t()) + torch.mm(imag, tail_imag.t())
        return scores * math.sqrt(self.embedding_dim)

    def score_candidates(self, query, candidates, h_ids, r_ids):
        dot_scores = torch.mm(query, candidates.t())
        complex_scores = self.complex_scores(h_ids, r_ids, candidates)
        return (
            dot_scores
            + self.complex_residual_weight * complex_scores
        ) / self.temperature

    def compute_loss(self, query, h_ids, r_ids, target_ids):
        candidates = F.normalize(
            self.struct_ent_embs.weight,
            p=2,
            dim=-1,
        )
        return F.cross_entropy(
            self.score_candidates(query, candidates, h_ids, r_ids),
            target_ids,
        )

    def score_all_entities(
        self,
        h_batch,
        r_batch,
        context_batch,
        candidate_vectors=None,
    ):
        query = self.encode_query(h_batch, r_batch, context_batch)
        if candidate_vectors is None:
            entity_ids = torch.arange(
                self.config.num_entities,
                device=query.device,
                dtype=torch.long,
            )
            candidate_vectors = self.encode_target({'id': entity_ids})
        return self.score_candidates(
            query,
            candidate_vectors,
            h_batch['id'],
            r_batch['id'],
        )
