"""Text-only and structure-only variants of the conference model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import (
    ContextAggregator,
    GWM,
    MLPAdapter,
    RoleAwareTransition,
    load_embedding_cache,
)


def build_model(config):
    variants = {
        'fused': GWM,
        'text_only': TextOnlyGWM,
        'structure_only': StructureOnlyGWM,
    }
    return variants[getattr(config, 'model_variant', 'fused')](config)


class SingleModalityGWM(nn.Module):
    requires_text_embeddings = False

    def __init__(self, config, embedding_dim):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        self.fusion_dim = config.fusion_dim
        self.temperature = config.temperature

        self.ent_embs = nn.Embedding(config.num_entities, embedding_dim)
        self.rel_embs = nn.Embedding(config.num_relations, embedding_dim)
        self.adapter = MLPAdapter(embedding_dim, config.adapter_dropout)
        self.input_projection = nn.Linear(embedding_dim, config.fusion_dim)
        self.context_aggregator = ContextAggregator(config.fusion_dim)
        self.transition = RoleAwareTransition(
            config.fusion_dim,
            config.dynamics_layers,
            config.num_successor_modes,
            config.dropout,
        )
        self.output_projection = nn.Linear(config.fusion_dim, config.fusion_dim)

    def pop_gate_stats(self):
        return {}

    def _encode_entities(self, ids):
        return self.input_projection(self.adapter(self.ent_embs(ids)))

    def _encode_relations(self, ids):
        return self.input_projection(self.adapter(self.rel_embs(ids)))

    def forward(self, h_batch, r_batch, context_batch):
        head = self._encode_entities(h_batch['id'])
        relation = self._encode_relations(r_batch['id'])
        context_entities = self._encode_entities(context_batch['id'])
        context_relations = self._encode_relations(context_batch['rel_id'])
        world_state = self.context_aggregator(
            head,
            context_entities,
            context_relations,
            context_batch['batch_index'],
        )
        modes, mixture_logits = self.transition(world_state, head, relation)
        return {
            'modes': F.normalize(self.output_projection(modes), dim=-1),
            'mixture_logits': mixture_logits,
        }

    def encode_target(self, t_batch):
        target = self._encode_entities(t_batch['id'])
        return F.normalize(self.output_projection(target), dim=-1)

    def compute_loss(self, query_vectors, target_vectors, truth_mask):
        scores = self.score_candidates(query_vectors, target_vectors)
        loss = GWM._filtered_in_batch_contrastive_loss(scores, truth_mask).mean()
        return loss, scores

    def score_candidates(self, query, candidates):
        mode_scores = torch.einsum('bkd,nd->bkn', query['modes'], candidates)
        log_weights = F.log_softmax(query['mixture_logits'], dim=-1).unsqueeze(-1)
        return torch.logsumexp(log_weights + mode_scores / self.temperature, dim=1)


class TextOnlyGWM(SingleModalityGWM):
    requires_text_embeddings = True

    def __init__(self, config):
        super().__init__(config, config.text_emb_dim)
        self.text_ent_embs = self.ent_embs
        self.text_rel_embs = self.rel_embs

    def load_text_embeddings(self, entity_path, relation_path, freeze=True):
        self.ent_embs.weight.data.copy_(load_embedding_cache(entity_path))
        self.rel_embs.weight.data.copy_(load_embedding_cache(relation_path))
        self.ent_embs.weight.requires_grad = not freeze
        self.rel_embs.weight.requires_grad = not freeze


class StructureOnlyGWM(SingleModalityGWM):
    def __init__(self, config):
        super().__init__(config, config.struct_emb_dim)
        self.struct_ent_embs = self.ent_embs
        self.struct_rel_embs = self.rel_embs
