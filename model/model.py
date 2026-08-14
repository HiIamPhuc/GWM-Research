import torch
import torch.nn as nn
import torch.nn.functional as F


class GWM(nn.Module):
    """PairRE-conditioned next-state prediction from local graph memory."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config.struct_emb_dim
        self.temperature = config.temperature

        self.struct_ent_embs = nn.Embedding(
            config.num_entities,
            self.embedding_dim,
        )
        self.base_rel_head_embs = nn.Embedding(
            config.num_base_relations,
            self.embedding_dim,
        )
        self.base_rel_tail_embs = nn.Embedding(
            config.num_base_relations,
            self.embedding_dim,
        )
        self.relation_norm = nn.LayerNorm(self.embedding_dim)
        self.state_norm = nn.LayerNorm(self.embedding_dim)

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

        nn.init.normal_(self.memory_roles, mean=0.0, std=0.02)
        nn.init.normal_(self.next_state_token, mean=0.0, std=0.02)

    def encode_relation_roles(self, relation_ids):
        base_ids = self.relation_base_ids[relation_ids]
        forward_head = self.base_rel_head_embs(base_ids)
        forward_tail = self.base_rel_tail_embs(base_ids)
        inverse = self.relation_directions[relation_ids].bool().unsqueeze(-1)
        head_role = torch.where(inverse, forward_tail, forward_head)
        tail_role = torch.where(inverse, forward_head, forward_tail)
        return head_role, tail_role

    def encode_relation(self, relation_ids):
        head_role, tail_role = self.encode_relation_roles(relation_ids)
        return self.relation_norm(head_role - tail_role)

    def build_world_memory(self, h_batch, context_batch, query_head_role):
        context_mask = context_batch['mask'].bool()
        entity_ids = context_batch['id'].masked_fill(~context_mask, 0)
        relation_ids = context_batch['rel_id'].masked_fill(~context_mask, 0)

        head_entity = F.normalize(
            self.struct_ent_embs(h_batch['id']),
            p=2,
            dim=-1,
        )
        head = self.state_norm(head_entity * query_head_role)

        context_entities = F.normalize(
            self.struct_ent_embs(entity_ids),
            p=2,
            dim=-1,
        )
        _, context_tail_roles = self.encode_relation_roles(relation_ids)
        context_facts = self.state_norm(
            context_entities * context_tail_roles
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
        head_role, _ = self.encode_relation_roles(r_batch['id'])
        relation_action = self.encode_relation(r_batch['id'])
        memory, memory_padding_mask = self.build_world_memory(
            h_batch,
            context_batch,
            head_role,
        )
        transition_query = self.next_state_token + relation_action.unsqueeze(1)
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

    def score_candidates(self, query, candidates, tail_roles):
        cross_term = torch.mm(query * tail_roles, candidates.t())
        candidate_norm = torch.mm(
            tail_roles.square(),
            candidates.square().t(),
        )
        query_norm = query.square().sum(dim=-1, keepdim=True)
        return (
            2.0 * cross_term - candidate_norm - query_norm
        ) / self.temperature

    def compute_loss(self, query, relation_ids, target_ids):
        candidates = F.normalize(
            self.struct_ent_embs.weight,
            p=2,
            dim=-1,
        )
        _, tail_roles = self.encode_relation_roles(relation_ids)
        scores = self.score_candidates(query, candidates, tail_roles)
        return F.cross_entropy(scores, target_ids)

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
        _, tail_roles = self.encode_relation_roles(r_batch['id'])
        return self.score_candidates(query, candidate_vectors, tail_roles)
