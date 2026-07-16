"""Model variants for ablation studies.

The production model remains `model.model.GWM`. This module provides
drop-in alternatives with the same training/evaluation API for controlled
text-only and structure-only experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import ContextAggregator, ConvTransEDecoder, GWM, MLPAdapter


def build_model(config):
    variant = str(getattr(config, 'model_variant', 'fused')).lower()
    if variant in {'fused', 'full', 'gwm'}:
        return GWM(config)
    if variant in {'text', 'text_only', 'text-only'}:
        return TextOnlyGWM(config)
    if variant in {'structure', 'struct', 'structure_only', 'structure-only', 'struct_only'}:
        return StructureOnlyGWM(config)
    raise ValueError(
        "Unsupported model_variant. Expected one of: fused, text_only, structure_only. "
        f"Got: {variant}"
    )


class SingleModalityGWM(nn.Module):
    """World-state transition model using exactly one embedding modality."""

    modality_name = None
    requires_text_embeddings = False

    def __init__(self, config, modality_name, emb_dim, adapter_dim):
        super().__init__()
        self.config = config
        self.modality_name = modality_name
        self.dropout = float(getattr(config, 'dropout'))
        self.adapter_dropout = float(getattr(config, 'adapter_dropout', self.dropout))
        self.embedding_dim = int(emb_dim)
        self.fusion_dim = int(getattr(config, 'fusion_dim'))

        self.ent_embs = nn.Embedding(config.num_entities, self.embedding_dim)
        self.rel_embs = nn.Embedding(config.num_relations, self.embedding_dim)
        self.adapter = MLPAdapter(
            self.embedding_dim,
            int(adapter_dim),
            dropout=self.adapter_dropout,
        )
        self.input_projection = nn.Linear(self.embedding_dim, self.fusion_dim)

        self.context_aggregator = ContextAggregator(hidden_dim=self.fusion_dim)
        self.h0_projection = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.c0_projection = nn.Linear(self.fusion_dim, self.fusion_dim)

        dynamics_layers = int(getattr(config, 'dynamics_layers', 1))
        self.lstm = nn.LSTM(
            input_size=self.fusion_dim,
            hidden_size=self.fusion_dim,
            num_layers=dynamics_layers,
            batch_first=True,
            dropout=self.dropout if dynamics_layers > 1 else 0.0,
        )
        self.output_projection = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.temperature = float(getattr(config, 'temperature'))
        self.decoder_name = str(getattr(config, 'decoder', 'dot')).lower()
        if self.decoder_name == 'convtranse':
            self.decoder = ConvTransEDecoder(
                embedding_dim=self.fusion_dim,
                dropout=self.dropout,
                channels=int(getattr(config, 'convtranse_channels', 50)),
                kernel_size=int(getattr(config, 'convtranse_kernel_size', 3)),
            )
        elif self.decoder_name in {'dot', 'contrastive'}:
            self.decoder = None
        else:
            raise ValueError(f"Unsupported decoder: {self.decoder_name}")

    def reset_gate_stats(self):
        pass

    def pop_gate_stats(self):
        return {}

    def _prepare_context_batch(self, context_batch):
        context_entity_ids = context_batch['id']
        context_relation_ids = context_batch.get('rel_id')
        context_batch_index = context_batch.get('batch_index')
        if context_relation_ids is None or context_batch_index is None:
            raise ValueError(
                "context_batch requires 'id', 'rel_id', and 'batch_index'."
            )
        if (
            context_entity_ids.dim() != 1
            or context_relation_ids.dim() != 1
            or context_batch_index.dim() != 1
        ):
            raise ValueError("Ragged context tensors must all be one-dimensional.")
        if not (
            context_entity_ids.numel()
            == context_relation_ids.numel()
            == context_batch_index.numel()
        ):
            raise ValueError("Ragged context tensors must have equal lengths.")
        return context_entity_ids, context_relation_ids, context_batch_index

    def _encode_entities(self, entity_ids):
        features = self.adapter(self.ent_embs(entity_ids))
        return self.input_projection(features)

    def _encode_relations(self, relation_ids):
        features = self.adapter(self.rel_embs(relation_ids))
        return self.input_projection(features)

    def _run_dynamics(self, world_state, head_emb, relation_emb):
        h_0 = torch.tanh(self.h0_projection(world_state))
        c_0 = torch.tanh(self.c0_projection(world_state))

        num_layers = self.lstm.num_layers
        h_0_lstm = h_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous()
        c_0_lstm = c_0.unsqueeze(0).expand(num_layers, -1, -1).contiguous()

        _, (h_n, _) = self.lstm(
            torch.stack([head_emb, relation_emb], dim=1),
            (h_0_lstm, c_0_lstm),
        )
        return h_n[-1]

    def encode_query(self, h_batch, r_batch, context_batch):
        h_emb = self._encode_entities(h_batch['id'])
        r_emb = self._encode_relations(r_batch['id'])

        flat_context_entity_ids, flat_context_relation_ids, context_batch_index = (
            self._prepare_context_batch(context_batch)
        )
        ctx_ent = self._encode_entities(flat_context_entity_ids)
        ctx_rel = self._encode_relations(flat_context_relation_ids)

        world_state = self.context_aggregator(
            head_feat=h_emb,
            nbr_entity_feat=ctx_ent,
            nbr_relation_feat=ctx_rel,
            nbr_batch_index=context_batch_index,
        )
        query = self._run_dynamics(world_state, h_emb, r_emb)
        query = F.normalize(self.output_projection(query), p=2, dim=1)
        return query, r_emb

    def forward(self, h_batch, r_batch, context_batch):
        query, _ = self.encode_query(h_batch, r_batch, context_batch)
        return query

    def encode_target(self, t_batch):
        target = self._encode_entities(t_batch['id'])
        return F.normalize(self.output_projection(target), p=2, dim=1)

    @staticmethod
    def compute_loss(scores, positive_tail_ids, positive_batch_index):
        return GWM.compute_loss(
            scores,
            positive_tail_ids,
            positive_batch_index,
        )

    def score_candidates(
        self,
        query_vectors,
        candidate_vectors,
        relation_vectors=None,
    ):
        if self.decoder_name == 'convtranse':
            if relation_vectors is None:
                raise ValueError(
                    "ConvTransE scoring requires relation vectors."
                )
            return self.decoder(
                query_vectors,
                relation_vectors,
                candidate_vectors,
            )
        return torch.mm(query_vectors, candidate_vectors.t()) / self.temperature

    def score_all_entities(
        self,
        h_batch,
        r_batch,
        context_batch,
        candidate_vectors=None,
    ):
        query_vectors, relation_vectors = self.encode_query(
            h_batch,
            r_batch,
            context_batch,
        )
        if candidate_vectors is None:
            entity_ids = torch.arange(
                int(self.config.num_entities),
                device=query_vectors.device,
                dtype=torch.long,
            )
            candidate_vectors = self.encode_target({'id': entity_ids})

        return self.score_candidates(
            query_vectors,
            candidate_vectors,
            relation_vectors=relation_vectors,
        )


class TextOnlyGWM(SingleModalityGWM):
    requires_text_embeddings = True

    def __init__(self, config):
        super().__init__(
            config=config,
            modality_name='text',
            emb_dim=int(getattr(config, 'text_emb_dim')),
            adapter_dim=int(getattr(config, 'text_adapter_dim')),
        )
        self.text_ent_embs = self.ent_embs
        self.text_rel_embs = self.rel_embs

    def _load_text_embedding_tensor(self, source, expected_rows, expected_dim, name):
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
        if loaded.size(1) != expected_dim:
            raise ValueError(
                f"{name} cache dimension mismatch. Expected {expected_dim}, got {loaded.size(1)}"
            )
        return loaded

    def load_text_embeddings(self, entity_source, relation_source, freeze=True):
        entity_cache = self._load_text_embedding_tensor(
            source=entity_source,
            expected_rows=self.ent_embs.num_embeddings,
            expected_dim=self.embedding_dim,
            name='text_entity',
        )
        relation_cache = self._load_text_embedding_tensor(
            source=relation_source,
            expected_rows=self.rel_embs.num_embeddings,
            expected_dim=self.embedding_dim,
            name='text_relation',
        )

        self.ent_embs.weight.data.copy_(entity_cache)
        self.rel_embs.weight.data.copy_(relation_cache)

        if freeze:
            self.ent_embs.weight.requires_grad = False
            self.rel_embs.weight.requires_grad = False


class StructureOnlyGWM(SingleModalityGWM):
    requires_text_embeddings = False

    def __init__(self, config):
        super().__init__(
            config=config,
            modality_name='structure',
            emb_dim=int(getattr(config, 'struct_emb_dim')),
            adapter_dim=int(getattr(config, 'struct_adapter_dim')),
        )
        self.struct_ent_embs = self.ent_embs
        self.struct_rel_embs = self.rel_embs
