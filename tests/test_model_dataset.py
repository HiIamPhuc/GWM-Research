import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from model.dataset import CollateFN, GWMDataset
from model.model import ContextAggregator, GWM
from studies.ablation_models import build_model
from utils.eval import (
    build_bidirectional_eval_dataset,
    build_bidirectional_hr_map_for_filtering,
    load_inverse_relation_ids,
)


def make_config(
    context_agg='mean',
    decoder=None,
):
    config = SimpleNamespace(
        num_entities=4,
        num_relations=4,
        text_emb_dim=6,
        struct_emb_dim=4,
        fusion_dim=10,
        text_adapter_dim=6,
        struct_adapter_dim=4,
        dynamics_layers=1,
        dropout=0.0,
        adapter_dropout=0.0,
        temperature=0.1,
        context_agg=context_agg,
        inverse_relation_ids=[2, 3],
    )
    if decoder is not None:
        config.decoder = decoder
        config.convtranse_channels = 2
        config.convtranse_kernel_size = 3
    return config


class ModelTests(unittest.TestCase):
    def test_text_embedding_load_and_freeze_targets_text_tables(self):
        model = GWM(make_config())
        model.load_text_embeddings(
            torch.randn(4, 6),
            torch.randn(4, 6),
            freeze=True,
        )
        self.assertFalse(model.text_ent_embs.weight.requires_grad)
        self.assertFalse(model.text_rel_embs.weight.requires_grad)
        self.assertTrue(model.struct_ent_embs.weight.requires_grad)
        self.assertTrue(model.struct_rel_embs.weight.requires_grad)

    def test_structural_embeddings_are_trainable_by_default(self):
        model = GWM(make_config())
        self.assertTrue(model.struct_ent_embs.weight.requires_grad)
        self.assertTrue(model.struct_rel_embs.weight.requires_grad)

    def test_isolated_head_preserves_self_state(self):
        layer = ContextAggregator(hidden_dim=4)
        output = layer(
            head_feat=torch.randn(2, 4),
            query_relation_feat=torch.randn(2, 4),
            nbr_entity_feat=torch.empty(0, 4),
            nbr_relation_feat=torch.empty(0, 4),
            nbr_batch_index=torch.empty(0, dtype=torch.long),
        )
        self.assertEqual(output.shape, (2, 4))
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(torch.equal(output, torch.zeros_like(output)))

    def test_context_attention_normalizes_per_query(self):
        messages = torch.randn(4, 3)
        batch_index = torch.tensor([0, 0, 1, 1])
        query_relations = torch.randn(2, 3)
        layer = ContextAggregator(3)

        weights = layer._attention_weights(
            messages,
            query_relations,
            batch_index,
        )
        weight_sums = torch.zeros(2)
        weight_sums.index_add_(0, batch_index, weights)

        self.assertTrue(torch.allclose(weight_sums, torch.ones(2)))

    def test_uniform_attention_matches_residual_mean_pooling(self):
        head = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        query_relations = torch.tensor([[0.5, 1.0], [1.5, 0.5]])
        neighbor_entities = torch.tensor(
            [[2.0, 1.0], [4.0, 2.0], [1.0, 3.0]]
        )
        neighbor_relations = torch.tensor(
            [[1.0, 2.0], [0.5, 3.0], [2.0, 1.0]]
        )
        batch_index = torch.tensor([0, 0, 1])

        layer = ContextAggregator(2)
        with torch.no_grad():
            layer.query_projection.weight.zero_()
            layer.key_projection.weight.zero_()
            layer.score_projection.weight.zero_()
        output = layer(
            head,
            query_relations,
            neighbor_entities,
            neighbor_relations,
            batch_index,
        )

        composed = neighbor_entities * neighbor_relations
        pooled = torch.stack(
            [composed[:2].mean(dim=0), composed[2]]
        )
        expected = torch.nn.functional.layer_norm(
            head + pooled,
            normalized_shape=(2,),
        )
        self.assertTrue(torch.allclose(output, expected))

    def test_context_attention_is_conditioned_on_query_relation(self):
        layer = ContextAggregator(3)
        head = torch.randn(1, 3)
        query_relation = torch.randn(1, 3, requires_grad=True)
        neighbor_entities = torch.randn(2, 3)
        neighbor_relations = torch.randn(2, 3)
        batch_index = torch.tensor([0, 0])

        output = layer(
            head,
            query_relation,
            neighbor_entities,
            neighbor_relations,
            batch_index,
        )
        output[0, 0].backward()

        self.assertIsNotNone(query_relation.grad)
        self.assertGreater(query_relation.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(layer.query_projection.weight.grad)

    def test_entity_and_relation_adapters_are_separate(self):
        model = GWM(make_config())
        self.assertIsNot(model.text_entity_adapter, model.text_relation_adapter)
        self.assertIsNot(model.struct_entity_adapter, model.struct_relation_adapter)

    def test_single_modality_variants_use_relation_aware_pooling(self):
        h_batch = {'id': torch.tensor([0, 1])}
        r_batch = {'id': torch.tensor([0, 1])}
        context_batch = {
            'id': torch.tensor([1, 2, 0]),
            'rel_id': torch.tensor([0, 1, 2]),
            'batch_index': torch.tensor([0, 0, 1]),
        }

        for variant in ('text_only', 'structure_only'):
            config = make_config(decoder='convtranse')
            config.model_variant = variant
            model = build_model(config)
            scores = model.score_all_entities(h_batch, r_batch, context_batch)

            self.assertEqual(scores.shape, (2, 4))
            self.assertTrue(torch.isfinite(scores).all())
            self.assertIsNot(model.entity_adapter, model.relation_adapter)

    def test_early_fusion_loss_backpropagates_to_gate(self):
        model = GWM(make_config())
        h_batch = {'id': torch.tensor([0, 1])}
        r_batch = {'id': torch.tensor([0, 1])}
        t_batch = {'id': torch.tensor([2, 3])}
        context_batch = {
            'id': torch.tensor([1, 2]),
            'rel_id': torch.tensor([0, 1]),
            'batch_index': torch.tensor([0, 1]),
        }
        scores = model.score_all_entities(h_batch, r_batch, context_batch)
        loss = model.compute_loss(scores, t_batch['id'])
        self.assertEqual(scores.shape, (2, 4))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.entity_fusion.gate[1].weight.grad)
        self.assertIsNotNone(model.relation_fusion.gate[1].weight.grad)

    def test_convtranse_full_entity_loss_and_scoring(self):
        model = GWM(make_config(decoder='convtranse'))
        h_batch = {'id': torch.tensor([0, 1])}
        r_batch = {'id': torch.tensor([0, 1])}
        t_batch = {'id': torch.tensor([2, 3])}
        context_batch = {
            'id': torch.tensor([1, 2]),
            'rel_id': torch.tensor([0, 1]),
            'batch_index': torch.tensor([0, 1]),
        }

        scores = model.score_all_entities(h_batch, r_batch, context_batch)
        loss = model.compute_loss(scores, t_batch['id'])

        self.assertEqual(scores.shape, (2, 4))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.decoder.conv.weight.grad)
        self.assertIsNotNone(model.decoder.fc.weight.grad)
        self.assertIsNotNone(model.entity_fusion.gate[1].weight.grad)
        self.assertIsNotNone(model.relation_fusion.gate[1].weight.grad)

    def test_full_entity_loss_matches_cross_entropy(self):
        scores = torch.tensor(
            [[0.2, -0.3, 1.1, 0.5], [-0.7, 0.4, 0.8, -0.2]],
            requires_grad=True,
        )
        target_ids = torch.tensor([3, 2])

        model = GWM(make_config())
        actual = model.compute_loss(scores, target_ids)
        expected = torch.nn.functional.cross_entropy(scores, target_ids)

        self.assertTrue(torch.allclose(actual, expected))

    def test_decoder_defaults_to_legacy_dot_scoring(self):
        model = GWM(make_config())
        self.assertEqual(model.decoder_name, 'dot')
        self.assertIsNone(model.decoder)

    def test_gate_statistics_are_recorded_and_consumed(self):
        model = GWM(make_config())
        h_batch = {'id': torch.tensor([0, 1])}
        r_batch = {'id': torch.tensor([0, 1])}
        t_batch = {'id': torch.tensor([2, 3])}
        context_batch = {
            'id': torch.tensor([1, 2]),
            'rel_id': torch.tensor([0, 1]),
            'batch_index': torch.tensor([0, 1]),
        }

        model(h_batch, r_batch, context_batch)
        model.encode_target(t_batch)
        stats = model.pop_gate_stats()

        self.assertEqual(
            set(stats),
            {
                'entity_gate',
                'relation_gate',
            },
        )
        self.assertGreaterEqual(stats['entity_gate'], 0.0)
        self.assertLessEqual(stats['entity_gate'], 1.0)
        self.assertEqual(model.pop_gate_stats(), {})


class DatasetTests(unittest.TestCase):
    def _write_data(self, root):
        root = Path(root)
        (root / 'entity2id.json').write_text(
            json.dumps({'a': 0, 'b': 1, 'c': 2}), encoding='utf-8'
        )
        (root / 'relation2id.json').write_text(
            json.dumps({'r': 0, 'r_inv': 1}), encoding='utf-8'
        )
        torch.save(torch.tensor([[0, 0, 1]]), root / 'train_triples.pt')
        torch.save(
            {
                'entity_ids': torch.tensor([[1, 2], [0, -1], [-1, -1]]),
                'relation_ids': torch.tensor([[0, 0], [1, -1], [-1, -1]]),
                'mask': torch.tensor(
                    [[True, True], [True, False], [False, False]]
                ),
                'pad_value': -1,
            },
            root / 'context_neighbors.pt',
        )

    def test_answer_edge_removed_and_collated_as_ragged_context(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_data(root)
            item = GWMDataset(root, split='train')[0]
            self.assertEqual(item['context_mask'].tolist(), [False, True])
            batch = CollateFN()([item])
            self.assertEqual(batch['context_batch']['id'].tolist(), [2])
            self.assertEqual(
                set(batch['context_batch']),
                {'id', 'rel_id', 'batch_index'},
            )

    def test_training_preserves_triples_and_masks_each_target_edge(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_data(root)
            torch.save(
                torch.tensor([[0, 0, 1], [0, 0, 2]]),
                Path(root) / 'train_triples.pt',
            )

            dataset = GWMDataset(root, split='train')
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]['context_mask'].tolist(), [False, True])
            self.assertEqual(dataset[1]['context_mask'].tolist(), [True, False])

            batch = CollateFN()([dataset[0], dataset[1]])
            self.assertEqual(batch['t_batch']['id'].tolist(), [1, 2])
            self.assertNotIn('positive_batch', batch)

    def test_bidirectional_eval_dataset_builds_inverse_queries_on_the_fly(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            self._write_data(root_path)
            torch.save(torch.tensor([[0, 0, 1]]), root_path / 'test_triples.pt')

            base_dataset = GWMDataset(root, split='test')
            forward_dataset, backward_dataset = build_bidirectional_eval_dataset(
                base_dataset,
                root,
            )

            self.assertEqual(forward_dataset.triples.tolist(), [[0, 0, 1]])
            self.assertEqual(backward_dataset.triples.tolist(), [[1, 1, 0]])

            forward_batch = CollateFN()([forward_dataset[0]])
            backward_batch = CollateFN()([backward_dataset[0]])
            self.assertNotIn('positive_batch', forward_batch)
            self.assertNotIn('positive_batch', backward_batch)
            self.assertEqual(forward_batch['t_batch']['id'].tolist(), [1])
            self.assertEqual(backward_batch['t_batch']['id'].tolist(), [0])

    def test_bidirectional_filter_map_adds_inverse_truths(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            (root_path / 'relation2id.json').write_text(
                json.dumps({'r': 0, 'r_inv': 1}), encoding='utf-8'
            )
            torch.save(torch.tensor([[0, 0, 1], [1, 1, 0]]), root_path / 'train_triples.pt')
            torch.save(torch.tensor([[0, 0, 2]]), root_path / 'valid_triples.pt')
            torch.save(torch.tensor([[0, 0, 1]]), root_path / 'test_triples.pt')

            hr_map = build_bidirectional_hr_map_for_filtering(
                root,
                splits=['train', 'valid', 'test'],
            )

            self.assertEqual(hr_map[(0, 0)], {1, 2})
            self.assertEqual(hr_map[(1, 1)], {0})
            self.assertEqual(hr_map[(2, 1)], {0})
            self.assertEqual(load_inverse_relation_ids(root), [1])

if __name__ == '__main__':
    unittest.main()
