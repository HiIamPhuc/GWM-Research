import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from model.dataset import CollateFN, GWMDataset, TrainTruthIndex
from model.model import ContextAggregator, GWM
from utils.eval import (
    build_bidirectional_eval_dataset,
    build_bidirectional_hr_map_for_filtering,
)


def make_config(context_agg='mean', decoder=None):
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
            nbr_entity_feat=torch.empty(0, 4),
            nbr_relation_feat=torch.empty(0, 4),
            nbr_batch_index=torch.empty(0, dtype=torch.long),
        )
        self.assertEqual(output.shape, (2, 4))
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(torch.equal(output, torch.zeros_like(output)))

    def test_context_aggregator_mean_reduction(self):
        messages = torch.tensor(
            [[1.0, 3.0], [5.0, 2.0], [7.0, 9.0]]
        )
        batch_index = torch.tensor([0, 0, 1])
        reference = torch.zeros(2, 2)

        mean_layer = ContextAggregator(2)

        mean_result = mean_layer._aggregate(
            messages, batch_index, batch_size=2, reference=reference
        )

        self.assertTrue(
            torch.equal(mean_result, torch.tensor([[3.0, 2.5], [7.0, 9.0]]))
        )

    def test_context_aggregator_forward_is_residual_pooling_then_norm(self):
        head = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        neighbor_entities = torch.tensor(
            [[2.0, 1.0], [4.0, 2.0], [1.0, 3.0]]
        )
        neighbor_relations = torch.tensor(
            [[1.0, 2.0], [0.5, 3.0], [2.0, 1.0]]
        )
        batch_index = torch.tensor([0, 0, 1])

        layer = ContextAggregator(2)
        output = layer(
            head,
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

    def test_query_aware_truth_mask_finds_distinct_valid_tails(self):
        truth_index = TrainTruthIndex(
            torch.tensor(
                [
                    [0, 0, 2],
                    [0, 0, 3],
                    [1, 0, 4],
                ]
            )
        )
        mask = truth_index.build_in_batch_truth_mask(
            head_ids=torch.tensor([0, 0, 1]),
            relation_ids=torch.tensor([0, 0, 0]),
            candidate_tail_ids=torch.tensor([2, 3, 4]),
        )
        expected = torch.tensor(
            [
                [True, True, False],
                [True, True, False],
                [False, False, True],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_query_aware_truth_mask_does_not_use_unseen_answers(self):
        truth_index = TrainTruthIndex(
            torch.tensor(
                [
                    [0, 0, 2],
                    [1, 0, 5],
                ]
            )
        )
        mask = truth_index.build_in_batch_truth_mask(
            head_ids=torch.tensor([0, 1]),
            relation_ids=torch.tensor([0, 0]),
            candidate_tail_ids=torch.tensor([2, 5]),
        )
        self.assertFalse(mask[0, 1].item())
        self.assertFalse(mask[1, 0].item())

    def test_filtered_loss_ignores_other_training_truths(self):
        scores = torch.tensor(
            [
                [2.0, 20.0, 0.0],
                [20.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
        truth_mask = torch.tensor(
            [
                [True, True, False],
                [True, True, False],
                [False, False, True],
            ]
        )
        losses = GWM._filtered_in_batch_contrastive_loss(
            scores,
            truth_mask=truth_mask,
        )
        expected = torch.tensor(
            [
                torch.log1p(torch.exp(torch.tensor(-2.0))),
                torch.log1p(torch.exp(torch.tensor(-2.0))),
                torch.log1p(2 * torch.exp(torch.tensor(-2.0))),
            ]
        )
        self.assertTrue(torch.allclose(losses, expected))

    def test_filtered_loss_penalizes_unrelated_high_score(self):
        scores = torch.tensor([[2.0, 20.0], [0.0, 2.0]])
        truth_mask = torch.eye(2, dtype=torch.bool)
        losses = GWM._filtered_in_batch_contrastive_loss(
            scores,
            truth_mask=truth_mask,
        )
        self.assertGreater(losses[0].item(), 17.0)

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
        query = model(h_batch, r_batch, context_batch)
        targets = model.encode_target(t_batch)
        loss, scores = model.compute_loss(
            query,
            targets,
            truth_mask=torch.eye(2, dtype=torch.bool),
        )
        self.assertEqual(scores.shape, (2, 2))
        self.assertEqual(query.shape, (2, 10))
        self.assertEqual(targets.shape, (2, 10))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.entity_fusion.gate[1].weight.grad)
        self.assertIsNotNone(model.relation_fusion.gate[1].weight.grad)

    def test_convtranse_scores_all_entities_and_backpropagates(self):
        model = GWM(make_config(decoder='convtranse'))
        h_batch = {'id': torch.tensor([0, 1])}
        r_batch = {'id': torch.tensor([0, 1])}
        context_batch = {
            'id': torch.tensor([1, 2]),
            'rel_id': torch.tensor([0, 1]),
            'batch_index': torch.tensor([0, 1]),
        }

        scores = model.score_all_entities(h_batch, r_batch, context_batch)
        loss = model.compute_full_softmax_loss(
            scores,
            torch.tensor([2, 3]),
        )

        self.assertEqual(scores.shape, (2, 4))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.decoder.conv.weight.grad)
        self.assertIsNotNone(model.decoder.fc.weight.grad)
        self.assertIsNotNone(model.entity_fusion.gate[1].weight.grad)
        self.assertIsNotNone(model.relation_fusion.gate[1].weight.grad)

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

if __name__ == '__main__':
    unittest.main()
