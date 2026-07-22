import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from model.dataset import CollateFN, GWMDataset, TrainTruthIndex
from model.model import GWM, filtered_in_batch_contrastive_loss
from studies.ablation_models import build_model
from utils.eval import (
    build_bidirectional_eval_dataset,
    build_bidirectional_hr_map_for_filtering,
    load_inverse_relation_ids,
)


def make_config():
    return SimpleNamespace(
        num_entities=4,
        num_relations=4,
        struct_emb_dim=4,
        dynamics_layers=1,
        temperature=0.07,
    )


def make_context_batch():
    return {
        'id': torch.tensor([1, 2, 0]),
        'rel_id': torch.tensor([0, 1, 2]),
        'batch_index': torch.tensor([0, 0, 1]),
    }


class ModelTests(unittest.TestCase):
    def test_relation_gate_adds_parameters_but_no_heavy_module(self):
        model = GWM(make_config())
        child_modules = set(dict(model.named_children()))

        self.assertEqual(
            child_modules,
            {'struct_ent_embs', 'struct_rel_embs', 'lstm'},
        )
        self.assertFalse(hasattr(model, 'text_ent_embs'))
        self.assertFalse(hasattr(model, 'adapter'))
        self.assertFalse(hasattr(model, 'fusion'))
        self.assertFalse(hasattr(model, 'world_state_encoder'))
        self.assertFalse(hasattr(model, 'output_projection'))
        self.assertEqual(model.context_gate_scale.numel(), 4)
        self.assertEqual(model.context_gate_bias.numel(), 4)
        self.assertEqual(model.context_strength.numel(), 1)

    def test_zero_strength_starts_from_basic_head_relation_sequence(self):
        model = GWM(make_config())
        captured = {}

        def capture_input(module, inputs):
            captured['sequence'] = inputs[0].detach().clone()
            captured['input_count'] = len(inputs)

        handle = model.lstm.register_forward_pre_hook(capture_input)
        h_ids = torch.tensor([0, 1])
        r_ids = torch.tensor([2, 3])
        model({'id': h_ids}, {'id': r_ids}, make_context_batch())
        handle.remove()

        expected = torch.stack(
            [model.struct_ent_embs(h_ids), model.struct_rel_embs(r_ids)],
            dim=1,
        )
        self.assertTrue(torch.equal(captured['sequence'], expected))
        self.assertEqual(captured['input_count'], 1)

    def test_relation_gate_modifies_contextual_head_not_lstm_state(self):
        model = GWM(make_config())
        with torch.no_grad():
            model.struct_ent_embs.weight.zero_()
            model.struct_rel_embs.weight.fill_(1.0)
            model.struct_ent_embs.weight[1] = torch.tensor([1.0, 2.0, 3.0, 4.0])
            model.struct_ent_embs.weight[2] = torch.tensor([3.0, 4.0, 5.0, 6.0])
            model.context_strength.fill_(torch.atanh(torch.tensor(0.5)))

        context_batch = {
            'id': torch.tensor([1, 2]),
            'rel_id': torch.tensor([0, 0]),
            'batch_index': torch.tensor([0, 0]),
        }
        captured = {}

        def capture_sequence(module, inputs):
            captured['sequence'] = inputs[0].detach().clone()
            captured['input_count'] = len(inputs)

        handle = model.lstm.register_forward_pre_hook(capture_sequence)
        model(
            {'id': torch.tensor([0, 3])},
            {'id': torch.tensor([2, 3])},
            context_batch,
        )
        handle.remove()

        expected_contextual_heads = torch.tensor(
            [[0.5, 0.75, 1.0, 1.25], [0.0, 0.0, 0.0, 0.0]]
        )
        self.assertTrue(
            torch.allclose(captured['sequence'][:, 0], expected_contextual_heads)
        )
        self.assertTrue(
            torch.equal(captured['sequence'][:, 1], torch.ones(2, 4))
        )
        self.assertEqual(captured['input_count'], 1)

        stats = model.context_stats()
        self.assertAlmostEqual(stats['context_strength'], 0.5, places=6)
        self.assertAlmostEqual(stats['context_gate_mean'], 0.5, places=6)
        self.assertAlmostEqual(stats['context_gate_std'], 0.0, places=6)

    def test_query_relation_changes_the_contextual_head(self):
        model = GWM(make_config())
        with torch.no_grad():
            model.struct_ent_embs.weight.zero_()
            model.struct_ent_embs.weight[1].fill_(1.0)
            model.struct_rel_embs.weight.zero_()
            model.struct_rel_embs.weight[0].fill_(1.0)
            model.struct_rel_embs.weight[3].fill_(2.0)
            model.context_gate_scale.fill_(1.0)
            model.context_strength.fill_(torch.atanh(torch.tensor(0.5)))

        context_batch = {
            'id': torch.tensor([1, 1]),
            'rel_id': torch.tensor([0, 0]),
            'batch_index': torch.tensor([0, 1]),
        }
        captured = {}

        def capture_sequence(module, inputs):
            captured['sequence'] = inputs[0].detach().clone()

        handle = model.lstm.register_forward_pre_hook(capture_sequence)
        model(
            {'id': torch.tensor([0, 0])},
            {'id': torch.tensor([2, 3])},
            context_batch,
        )
        handle.remove()

        self.assertFalse(
            torch.allclose(
                captured['sequence'][0, 0],
                captured['sequence'][1, 0],
            )
        )

    def test_query_target_loss_backpropagates(self):
        model = GWM(make_config())
        query = model(
            {'id': torch.tensor([0, 1])},
            {'id': torch.tensor([0, 1])},
            make_context_batch(),
        )
        targets = model.encode_target({'id': torch.tensor([2, 3])})
        loss, scores = model.compute_loss(
            query,
            targets,
            truth_mask=torch.eye(2, dtype=torch.bool),
        )
        loss.backward()

        self.assertEqual(scores.shape, (2, 2))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.struct_ent_embs.weight.grad)
        self.assertIsNotNone(model.struct_rel_embs.weight.grad)
        self.assertIsNotNone(model.lstm.weight_ih_l0.grad)
        self.assertIsNotNone(model.context_strength.grad)
        self.assertIsNotNone(model.context_gate_scale.grad)
        self.assertIsNotNone(model.context_gate_bias.grad)

    def test_targets_are_normalized_entity_embeddings(self):
        model = GWM(make_config())
        ids = torch.tensor([1, 3])
        actual = model.encode_target({'id': ids})
        expected = F.normalize(model.struct_ent_embs(ids), p=2, dim=-1)
        self.assertTrue(torch.allclose(actual, expected))

    def test_dot_product_scores_all_entities(self):
        model = GWM(make_config())
        scores = model.score_all_entities(
            {'id': torch.tensor([0, 1])},
            {'id': torch.tensor([0, 1])},
            make_context_batch(),
        )
        self.assertEqual(scores.shape, (2, 4))
        self.assertTrue(torch.isfinite(scores).all())

    def test_study_factory_returns_the_same_basic_model(self):
        self.assertIsInstance(build_model(make_config()), GWM)

    def test_filtered_in_batch_loss_ignores_other_true_tails(self):
        scores = torch.tensor(
            [[2.0, 3.0, 1.0], [0.5, 2.0, 1.0], [0.0, 1.0, 2.0]],
            requires_grad=True,
        )
        truth_mask = torch.eye(3, dtype=torch.bool)
        truth_mask[0, 1] = True

        losses = filtered_in_batch_contrastive_loss(scores, truth_mask)
        losses.sum().backward()

        self.assertEqual(scores.grad[0, 1].item(), 0.0)
        self.assertLess(scores.grad[0, 0].item(), 0.0)
        self.assertGreater(scores.grad[0, 2].item(), 0.0)


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
                'k_requested': 2,
                'k_effective': 2,
            },
            root / 'context_neighbors.pt',
        )

    def test_answer_edge_is_removed_from_training_context(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_data(root)
            dataset = GWMDataset(root, split='train')
            item = dataset[0]
            batch = CollateFN()([item])

            self.assertEqual(
                set(item),
                {
                    'h_id',
                    'r_id',
                    't_id',
                    'context_entity_ids',
                    'context_relation_ids',
                    'context_mask',
                },
            )
            self.assertEqual(
                set(batch),
                {'h_batch', 'r_batch', 't_batch', 'context_batch'},
            )
            self.assertEqual(batch['h_batch']['id'].tolist(), [0])
            self.assertEqual(batch['r_batch']['id'].tolist(), [0])
            self.assertEqual(batch['t_batch']['id'].tolist(), [1])
            self.assertEqual(batch['context_batch']['id'].tolist(), [2])
            self.assertEqual(batch['context_batch']['rel_id'].tolist(), [0])
            self.assertEqual(batch['context_batch']['batch_index'].tolist(), [0])

    def test_dataset_requires_precomputed_context(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_data(root)
            (Path(root) / 'context_neighbors.pt').unlink()
            with self.assertRaises(FileNotFoundError):
                GWMDataset(root, split='train')

    def test_train_truth_index_marks_all_known_in_batch_tails(self):
        index = TrainTruthIndex(
            torch.tensor([[0, 0, 1], [0, 0, 2], [1, 0, 2]])
        )
        mask = index.build_in_batch_truth_mask(
            head_ids=torch.tensor([0, 0, 1]),
            relation_ids=torch.tensor([0, 0, 0]),
            candidate_tail_ids=torch.tensor([1, 2, 2]),
        )
        self.assertEqual(
            mask.tolist(),
            [[True, True, True], [True, True, True], [False, True, True]],
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
            expected_keys = {
                'h_id',
                'r_id',
                't_id',
                'context_entity_ids',
                'context_relation_ids',
                'context_mask',
            }
            self.assertEqual(set(forward_dataset[0]), expected_keys)
            self.assertEqual(set(backward_dataset[0]), expected_keys)

    def test_bidirectional_filter_map_adds_inverse_truths(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            self._write_data(root_path)
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
