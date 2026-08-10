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
        self.num_next_state_slots = config.num_next_state_slots

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
        self.register_buffer(
            'relation_slot_counts',
            torch.as_tensor(config.relation_slot_counts, dtype=torch.long),
        )
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
        self.path_query_projection = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=False,
        )
        self.path_edge_projection = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=False,
        )
        self.path_delta = nn.Linear(
            self.embedding_dim,
            self.embedding_dim,
            bias=False,
        )
        self.path_norm = nn.LayerNorm(self.embedding_dim)
        self.path_score_gate = nn.Linear(
            self.embedding_dim,
            1,
        )
        self.next_state_tokens = nn.Parameter(
            torch.empty(
                1,
                self.num_next_state_slots,
                self.embedding_dim,
            )
        )
        self.masked_head_token = nn.Parameter(
            torch.empty(1, 1, self.embedding_dim)
        )
        transition_mask = torch.ones(
            self.num_next_state_slots + 1,
            self.num_next_state_slots + 1,
            dtype=torch.bool,
        )
        transition_mask[0, 0] = False
        transition_mask[1, :2] = False
        transition_mask[2:, :] = False
        self.register_buffer('transition_mask', transition_mask, persistent=False)
        nn.init.normal_(self.token_roles.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.next_state_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.direction_embs.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.inverse_adapter.weight)
        nn.init.eye_(self.next_state_projection.weight)
        nn.init.normal_(self.masked_head_token, mean=0.0, std=0.02)
        nn.init.zeros_(self.path_score_gate.weight)
        nn.init.constant_(
            self.path_score_gate.bias,
            getattr(config, 'path_score_gate_init', -2.0),
        )

    def encode_relation(self, relation_ids):
        base_ids = self.relation_base_ids[relation_ids]
        directions = self.relation_directions[relation_ids]
        base = self.base_rel_embs(base_ids)
        relation = base + self.direction_embs(directions)
        inverse_mask = directions.unsqueeze(-1).to(dtype=base.dtype)
        relation = relation + inverse_mask * self.inverse_adapter(base)
        return self.relation_norm(relation)

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
        relation_ids = r_batch['id']
        relation = self.encode_relation(relation_ids)
        slot_counts = self.relation_slot_counts[relation_ids]
        active_slots = (
            torch.arange(
                self.num_next_state_slots,
                device=relation.device,
            ).unsqueeze(0)
            < slot_counts.unsqueeze(1)
        )
        mixture_log_weights = torch.where(
            active_slots,
            -slot_counts.to(relation.dtype).log().unsqueeze(1),
            relation.new_full((), float('-inf')),
        )
        relation_token = relation + self.token_roles.weight[2]
        memory, memory_padding_mask = self.encode_world_state(
            h_batch,
            context_batch,
        )
        next_state_tokens = self.next_state_tokens.expand(
            relation_token.size(0),
            -1,
            -1,
        )
        transition_tokens = torch.cat(
            [relation_token.unsqueeze(1), next_state_tokens],
            dim=1,
        )
        decoded = self.transition_decoder(
            tgt=transition_tokens,
            memory=memory,
            tgt_mask=self.transition_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        decoded_states = decoded[:, 1:]
        next_states = self.next_state_projection(decoded_states)
        return (
            F.normalize(next_states, p=2, dim=-1),
            mixture_log_weights,
        )

    def forward(self, h_batch, r_batch, context_batch):
        return self.encode_query(h_batch, r_batch, context_batch)

    def _path_transition(self, states, edge_relations, query_relation):
        expanded_query = query_relation
        while expanded_query.dim() < states.dim():
            expanded_query = expanded_query.unsqueeze(1)
        expanded_query = expanded_query.expand_as(states)

        query_key = F.normalize(
            self.path_query_projection(expanded_query),
            p=2,
            dim=-1,
        )
        edge_key = F.normalize(
            self.path_edge_projection(edge_relations),
            p=2,
            dim=-1,
        )
        relevance = torch.sigmoid(
            4.0 * (query_key * edge_key).sum(dim=-1)
        )
        delta = torch.tanh(self.path_delta(edge_relations))
        return F.normalize(
            self.path_norm(
                states + relevance.unsqueeze(-1) * delta
            ),
            p=2,
            dim=-1,
        )

    def encode_path_evidence(self, h_batch, r_batch, path_batch):
        relation = self.encode_relation(r_batch['id'])
        source = F.normalize(
            self.struct_ent_embs(h_batch['id']) + relation,
            p=2,
            dim=-1,
        )

        hop1_mask = path_batch['hop1_mask'].bool()
        hop1_ids = path_batch['hop1_id'].masked_fill(~hop1_mask, 0)
        hop1_rel_ids = path_batch['hop1_rel_id'].masked_fill(~hop1_mask, 0)
        hop1_relations = self.encode_relation(hop1_rel_ids)
        hop1_states = self._path_transition(
            source.unsqueeze(1).expand_as(hop1_relations),
            hop1_relations,
            relation,
        )
        hop1_targets = F.normalize(
            self.struct_ent_embs(hop1_ids),
            p=2,
            dim=-1,
        )
        hop1_scores = (hop1_states * hop1_targets).sum(dim=-1)
        hop1_scores = hop1_scores / self.temperature

        hop2_mask = path_batch['hop2_mask'].bool()
        hop2_ids = path_batch['hop2_id'].masked_fill(~hop2_mask, 0)
        hop2_rel_ids = path_batch['hop2_rel_id'].masked_fill(~hop2_mask, 0)
        hop2_relations = self.encode_relation(hop2_rel_ids)
        hop2_states = self._path_transition(
            hop1_states.unsqueeze(2).expand_as(hop2_relations),
            hop2_relations,
            relation,
        )
        hop2_targets = F.normalize(
            self.struct_ent_embs(hop2_ids),
            p=2,
            dim=-1,
        )
        hop2_scores = (hop2_states * hop2_targets).sum(dim=-1)
        hop2_scores = hop2_scores / self.temperature

        return {
            'entity_ids': torch.cat(
                [hop1_ids, hop2_ids.flatten(1)],
                dim=1,
            ),
            'scores': torch.cat(
                [hop1_scores, hop2_scores.flatten(1)],
                dim=1,
            ),
            'mask': torch.cat(
                [hop1_mask, hop2_mask.flatten(1)],
                dim=1,
            ),
            'gate': torch.sigmoid(self.path_score_gate(relation)),
        }

    def _aggregate_path_evidence(self, path_evidence):
        entity_ids = path_evidence['entity_ids']
        path_scores = path_evidence['scores']
        path_mask = path_evidence['mask'].bool()
        path_count = entity_ids.size(1)

        same_endpoint = entity_ids.unsqueeze(2).eq(entity_ids.unsqueeze(1))
        valid_pairs = (
            same_endpoint
            & path_mask.unsqueeze(2)
            & path_mask.unsqueeze(1)
        )
        endpoint_counts = valid_pairs.sum(dim=-1).clamp_min(1)
        endpoint_scores = (
            valid_pairs.to(path_scores.dtype)
            * path_scores.unsqueeze(1)
        ).sum(dim=-1) / endpoint_counts

        previous = torch.tril(
            torch.ones(
                path_count,
                path_count,
                dtype=torch.bool,
                device=entity_ids.device,
            ),
            diagonal=-1,
        )
        unique_mask = path_mask & ~(valid_pairs & previous).any(dim=-1)
        bonuses = endpoint_scores * path_evidence['gate']
        safe_entity_ids = entity_ids.masked_fill(~unique_mask, 0)
        return safe_entity_ids, bonuses, unique_mask

    def _add_path_evidence(self, scores, path_evidence):
        if path_evidence is None:
            return scores
        entity_ids, bonuses, unique_mask = self._aggregate_path_evidence(
            path_evidence
        )
        contributions = bonuses * unique_mask.to(bonuses.dtype)
        if not torch.is_grad_enabled():
            scores.scatter_add_(1, entity_ids, contributions)
            return scores
        return scores.scatter_add(1, entity_ids, contributions)

    def encode_target(self, t_batch):
        target = self.struct_ent_embs(t_batch['id'])
        return F.normalize(target, p=2, dim=-1)

    def score_candidates(
        self,
        query_slots,
        mixture_log_weights,
        candidate_vectors,
    ):
        component_logits = torch.einsum(
            'bkd,nd->bkn',
            query_slots,
            candidate_vectors,
        ) / self.temperature
        return torch.logsumexp(
            component_logits + mixture_log_weights.unsqueeze(-1),
            dim=1,
        )

    def compute_loss(
        self,
        query_slots,
        mixture_log_weights,
        target_ids,
        path_evidence=None,
    ):
        candidate_vectors = F.normalize(
            self.struct_ent_embs.weight,
            p=2,
            dim=-1,
        )
        scores = self.score_candidates(
            query_slots,
            mixture_log_weights,
            candidate_vectors,
        )
        if path_evidence is None:
            return F.cross_entropy(scores, target_ids)

        entity_ids, bonuses, unique_mask = self._aggregate_path_evidence(
            path_evidence
        )
        log_partition = torch.logsumexp(scores, dim=1)
        selected_scores = scores.gather(1, entity_ids)
        partition_ratio = 1.0 + (
            torch.exp(selected_scores - log_partition.unsqueeze(1))
            * torch.expm1(bonuses)
            * unique_mask.to(scores.dtype)
        ).sum(dim=1)
        combined_log_partition = (
            log_partition + partition_ratio.clamp_min(1e-12).log()
        )
        target_scores = scores.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        target_bonus = (
            bonuses
            * unique_mask.to(bonuses.dtype)
            * entity_ids.eq(target_ids.unsqueeze(1)).to(bonuses.dtype)
        ).sum(dim=1)
        return (combined_log_partition - target_scores - target_bonus).mean()

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
        path_batch=None,
        candidate_vectors=None,
    ):
        query_slots, mixture_log_weights = self.encode_query(
            h_batch,
            r_batch,
            context_batch,
        )
        if candidate_vectors is None:
            entity_ids = torch.arange(
                self.config.num_entities,
                device=query_slots.device,
                dtype=torch.long,
            )
            candidate_vectors = self.encode_target({'id': entity_ids})

        scores = self.score_candidates(
            query_slots,
            mixture_log_weights,
            candidate_vectors,
        )
        path_evidence = None
        if path_batch is not None:
            path_evidence = self.encode_path_evidence(
                h_batch,
                r_batch,
                path_batch,
            )
        return self._add_path_evidence(scores, path_evidence)
