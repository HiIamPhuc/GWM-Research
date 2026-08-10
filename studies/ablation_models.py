"""Text-only and structure-only variants of the conference model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import ContextAggregator, GWM, MLPAdapter, load_embedding_cache


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
        self.h0_projection = nn.Linear(config.fusion_dim, config.fusion_dim)
        self.c0_projection = nn.Linear(config.fusion_dim, config.fusion_dim)
        self.lstm = nn.LSTM(
            config.fusion_dim,
            config.fusion_dim,
            num_layers=config.dynamics_layers,
            batch_first=True,
            dropout=config.dropout if config.dynamics_layers > 1 else 0,
        )
        self.output_projection = nn.Linear(config.fusion_dim, config.fusion_dim)

    def pop_gate_stats(self):
        return {}

    def _encode_entities(self, ids):
        return self.input_projection(self.adapter(self.ent_embs(ids)))

    def _encode_relations(self, ids):
        return self.input_projection(self.adapter(self.rel_embs(ids)))

    def _run_transition(self, world_state, head, relation):
        h0 = torch.tanh(self.h0_projection(world_state))
        c0 = torch.tanh(self.c0_projection(world_state))
        h0 = h0.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
        c0 = c0.unsqueeze(0).expand(self.lstm.num_layers, -1, -1).contiguous()
        _, (hidden, _) = self.lstm(
            torch.stack([head, relation], dim=1),
            (h0, c0),
        )
        return hidden[-1]

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
        query = self._run_transition(world_state, head, relation)
        return F.normalize(self.output_projection(query), dim=-1)

    def encode_target(self, t_batch):
        target = self._encode_entities(t_batch['id'])
        return F.normalize(self.output_projection(target), dim=-1)

    def compute_loss(self, query_vectors, target_vectors, truth_mask):
        scores = query_vectors @ target_vectors.t() / self.temperature
        loss = GWM._filtered_in_batch_contrastive_loss(scores, truth_mask).mean()
        return loss, scores


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
