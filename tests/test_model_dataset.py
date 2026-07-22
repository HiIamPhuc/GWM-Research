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
        transformer_layers=1,
        transformer_heads=2,
        transformer_ffn_multiplier=2,
        transformer_dropout=0.0,
        temperature=0.07,
    )


def make_context_batch():
    return {
        'id': torch.tensor([1, 2, 0]),
        'rel_id': torch.tensor([0, 1, 2]),
        'batch_index': torch.tensor([0, 0, 1]),
    }


class ModelTests(unittest.TestCase):
    def test_context_free_transformer_has_expected_modules(self):
        model = GWM(make_config())
        child_modules = set(dict(model.named_children()))

        self.assertEqual(
            child_modules,
            {
                'struct_ent_embs',
                'struct_rel_embs',
                'transformer',
                'transition_projection',
                'output_norm',
            },
        )
        self.assertFalse(hasattr(model, 'text_ent_embs'))
        self.assertFalse(hasattr(model, 'adapter'))
        self.assertFalse(hasattr(model, 'fusion'))
        self.assertFalse(hasattr(model, 'lstm'))
        self.assertEqual(tuple(model.token_roles.shape), (2, 4))

    def test_transformer_receives_role_encoded_tokens_and_causal_mask(self):
        model = GWM(make_config())
        model.eval()
        captured = {}

        def capture_input(module, inputs):
            captured['tokens'] = inputs[0].detach().clone()

        handle = model.transformer.register_forward_pre_hook(capture_input)
        h_ids = torch.tensor([0, 1])
        r_ids = torch.tensor([2, 3])
        model({'id': h_ids}, {'id': r_ids}, make_context_batch())
        handle.remove()

        expected = torch.stack(
            [model.struct_ent_embs(h_ids), model.struct_rel_embs(r_ids)],
            dim=1,
        ) + model.token_roles.unsqueeze(0)
        self.assertTrue(torch.equal(captured['tokens'], expected))
        self.assertEqual(
            model.transition_mask.tolist(),
            [[False, True], [False, False]],
        )

    def test_query_is_independent_of_context(self):
        model = GWM(make_config())
        model.eval()
        h_batch = {'id': torch.tensor([0, 1])}
        r_batch = {'id': torch.tensor([2, 3])}
        first = model(h_batch, r_batch, make_context_batch())
        second = model(
            h_batch,
            r_batch,
            {
                'id': torch.tensor([], dtype=torch.long),
                'rel_id': torch.tensor([], dtype=torch.long),
                'batch_index': torch.tensor([], dtype=torch.long),
            },
        )
        self.assertTrue(torch.equal(first, second))

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
        self.assertIsNotNone(
            model.transformer.layers[0].self_attn.in_proj_weight.grad
        )
        self.assertIsNotNone(model.transition_projection.weight.grad)
        self.assertIsNotNone(model.token_roles.grad)

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
