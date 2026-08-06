import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from model.dataset import CollateFN, GWMDataset
from model.model import GWM
from studies.ablation_models import build_model
from utils.compute_context import ContextProcessor
from utils.eval import (
    build_bidirectional_eval_dataset,
    build_bidirectional_hr_map_for_filtering,
    load_inverse_relation_ids,
)
from utils.relation_mapping import build_relation_direction_mapping


def make_config():
    return SimpleNamespace(
        num_entities=4,
        num_relations=4,
        num_base_relations=2,
        relation_base_ids=[0, 0, 1, 1],
        relation_directions=[0, 1, 0, 1],
        struct_emb_dim=4,
        context_encoder_layers=1,
        transition_decoder_layers=2,
        context_encoder_heads=1,
        context_encoder_ffn_multiplier=2,
        context_encoder_dropout=0.0,
        transition_decoder_heads=2,
        transition_decoder_ffn_multiplier=3,
        transition_decoder_dropout=0.0,
        temperature=0.07,
        temperature_min=0.03,
        temperature_max=0.20,
    )


def make_context_batch():
    return {
        'id': torch.tensor([[1, 2], [0, -1]]),
        'rel_id': torch.tensor([[0, 1], [2, -1]]),
        'mask': torch.tensor([[True, True], [True, False]]),
    }


class ModelTests(unittest.TestCase):
    def test_graph_memory_transformer_has_expected_modules(self):
        model = GWM(make_config())
        child_modules = set(dict(model.named_children()))

        self.assertEqual(
            child_modules,
            {
                'struct_ent_embs',
                'base_rel_embs',
                'direction_embs',
                'inverse_adapter',
                'relation_norm',
                'temperature_head',
                'context_encoder',
                'transition_decoder',
                'next_state_projection',
                'context_fact_norm',
                'token_roles',
            },
        )
        self.assertFalse(hasattr(model, 'text_ent_embs'))
        self.assertFalse(hasattr(model, 'adapter'))
        self.assertFalse(hasattr(model, 'fusion'))
        self.assertFalse(hasattr(model, 'lstm'))
        self.assertFalse(hasattr(model, 'context_entity_projection'))
        self.assertFalse(hasattr(model, 'context_relation_projection'))
        self.assertFalse(hasattr(model, 'transition_projection'))
        self.assertFalse(hasattr(model, 'output_norm'))
        self.assertFalse(hasattr(model, 'context_state_token'))
        self.assertEqual(tuple(model.token_roles.weight.shape), (3, 4))
        self.assertEqual(tuple(model.next_state_token.shape), (1, 1, 4))
        self.assertEqual(tuple(model.masked_head_token.shape), (1, 1, 4))
        self.assertEqual(
            model.context_encoder.layers[0].self_attn.num_heads,
            1,
        )
        self.assertEqual(
            model.transition_decoder.layers[0].self_attn.num_heads,
            2,
        )
        self.assertEqual(
            model.context_encoder.layers[0].linear1.out_features,
            8,
        )
        self.assertEqual(
            model.transition_decoder.layers[0].linear1.out_features,
            12,
        )
        self.assertTrue(torch.equal(
            model.next_state_projection.weight,
            torch.eye(4),
        ))
        self.assertTrue(torch.equal(
            model.temperature_head.weight,
            torch.zeros(1, 4),
        ))

    def test_context_and_transition_use_separate_sequences(self):
        model = GWM(make_config())
        model.eval()
        captured = {}

        def capture_context(module, inputs):
            captured['context_tokens'] = inputs[0].detach().clone()

        def capture_transition(module, inputs):
            captured['transition_tokens'] = inputs[0].detach().clone()
            captured['memory'] = inputs[1].detach().clone()

        context_handle = model.context_encoder.register_forward_pre_hook(
            capture_context
        )
        transition_handle = model.transition_decoder.layers[
            0
        ].register_forward_pre_hook(capture_transition)
        h_ids = torch.tensor([0, 1])
        r_ids = torch.tensor([2, 3])
        model({'id': h_ids}, {'id': r_ids}, make_context_batch())
        context_handle.remove()
        transition_handle.remove()

        expected_context_head = (
            model.struct_ent_embs(h_ids)
            + model.token_roles.weight[0]
        )
        expected_relation = (
            model.encode_relation(r_ids)
            + model.token_roles.weight[2]
        )
        expected_transition = torch.cat(
            [
                expected_relation.unsqueeze(1),
                model.next_state_token.expand(2, -1, -1),
            ],
            dim=1,
        )
        self.assertTrue(torch.equal(
            captured['transition_tokens'],
            expected_transition,
        ))
        self.assertTrue(torch.equal(
            captured['context_tokens'][:, 0],
            expected_context_head,
        ))
        self.assertEqual(captured['context_tokens'].shape, (2, 3, 4))
        self.assertEqual(captured['memory'].shape, (2, 3, 4))
        self.assertEqual(
            model.transition_mask.tolist(),
            [
                [False, True],
                [False, False],
            ],
        )

    def test_query_depends_on_context(self):
        model = GWM(make_config())
        model.eval()
        h_batch = {'id': torch.tensor([0, 1])}
        r_batch = {'id': torch.tensor([2, 3])}
        first_query, first_temperature = model(
            h_batch,
            r_batch,
            make_context_batch(),
        )
        second_query, second_temperature = model(
            h_batch,
            r_batch,
            {
                'id': torch.full((2, 2), -1, dtype=torch.long),
                'rel_id': torch.full((2, 2), -1, dtype=torch.long),
                'mask': torch.zeros((2, 2), dtype=torch.bool),
            },
        )
        self.assertFalse(torch.allclose(first_query, second_query))
        self.assertTrue(torch.equal(first_temperature, second_temperature))
        self.assertEqual(second_query.shape, (2, 4))
        self.assertTrue(torch.isfinite(second_query).all())
        self.assertTrue(torch.allclose(
            second_temperature,
            torch.full((2, 1), 0.07),
        ))

    def test_query_target_loss_backpropagates(self):
        model = GWM(make_config())
        query_vector, relation_temperature = model(
            {'id': torch.tensor([0, 1])},
            {'id': torch.tensor([0, 1])},
            make_context_batch(),
        )
        target_ids = torch.tensor([2, 3])
        kg_loss = model.compute_loss(
            query_vector,
            relation_temperature,
            target_ids,
        )
        reconstructed_heads = model.encode_masked_world_state(
            {'id': torch.tensor([0, 1])},
            make_context_batch(),
        )
        state_loss = model.compute_state_reconstruction_loss(
            reconstructed_heads,
            torch.tensor([0, 1]),
        )
        loss = kg_loss + 0.1 * state_loss
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.struct_ent_embs.weight.grad)
        self.assertIsNotNone(model.base_rel_embs.weight.grad)
        self.assertIsNotNone(model.direction_embs.weight.grad)
        self.assertIsNotNone(model.inverse_adapter.weight.grad)
        self.assertIsNotNone(
            model.context_encoder.layers[0].self_attn.in_proj_weight.grad
        )
        self.assertIsNotNone(
            model.transition_decoder.layers[0].multihead_attn.in_proj_weight.grad
        )
        self.assertIsNotNone(model.next_state_projection.weight.grad)
        self.assertIsNotNone(model.temperature_head.weight.grad)
        self.assertIsNotNone(model.context_fact_norm.weight.grad)
        self.assertIsNotNone(model.token_roles.weight.grad)
        self.assertIsNotNone(model.next_state_token.grad)
        self.assertIsNotNone(model.masked_head_token.grad)

    def test_forward_and_inverse_relations_share_the_same_base_row(self):
        model = GWM(make_config())
        self.assertEqual(model.relation_base_ids[[0, 1]].tolist(), [0, 0])
        self.assertEqual(model.relation_directions[[0, 1]].tolist(), [0, 1])

        relation_vectors = model.encode_relation(torch.tensor([0, 1]))
        self.assertFalse(torch.allclose(relation_vectors[0], relation_vectors[1]))

        weights = torch.arange(1, 5, dtype=relation_vectors.dtype)
        (relation_vectors * weights).sum().backward()
        self.assertGreater(model.base_rel_embs.weight.grad[0].abs().sum().item(), 0.0)
        self.assertEqual(model.base_rel_embs.weight.grad[1].abs().sum().item(), 0.0)

    def test_relation_mapping_does_not_assume_contiguous_relation_pairs(self):
        mapping = build_relation_direction_mapping({
            'r_a_inv': 0,
            'r_b_inv': 1,
            'r_a': 2,
            'r_b': 3,
        })

        self.assertEqual(mapping['num_base_relations'], 2)
        self.assertEqual(mapping['full_to_base'], [0, 1, 0, 1])
        self.assertEqual(mapping['directions'], [1, 1, 0, 0])

    def test_targets_are_normalized_entity_embeddings(self):
        model = GWM(make_config())
        ids = torch.tensor([1, 3])
        actual = model.encode_target({'id': ids})
        expected = F.normalize(model.struct_ent_embs(ids), p=2, dim=-1)
        self.assertTrue(torch.allclose(actual, expected))

    def test_single_state_scorer_scores_all_entities(self):
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

    def test_relation_diverse_context_selection_covers_relations_first(self):
        selected = ContextProcessor._select_relation_diverse_neighbors(
            [(0, 1), (0, 2), (1, 3), (2, 0)],
            limit=3,
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual({relation for relation, _ in selected}, {0, 1, 2})

    def test_loss_matches_relation_temperature_cross_entropy(self):
        model = GWM(make_config())
        query_vector = F.normalize(
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            p=2,
            dim=-1,
        )
        temperature = torch.tensor([[0.08]])
        target_ids = torch.tensor([2])

        actual = model.compute_loss(
            query_vector,
            temperature,
            target_ids,
        )
        candidates = F.normalize(
            model.struct_ent_embs.weight,
            p=2,
            dim=-1,
        )
        scores = torch.mm(query_vector, candidates.t()) / temperature
        expected = F.cross_entropy(scores, target_ids)

        self.assertTrue(torch.allclose(actual, expected))

    def test_relation_temperature_is_initialized_and_bounded(self):
        model = GWM(make_config())
        relation_features = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        initial = model.relation_temperature(relation_features)
        self.assertTrue(torch.allclose(
            initial,
            torch.full((2, 1), 0.07),
        ))
        with torch.no_grad():
            model.temperature_head.weight.copy_(
                torch.tensor([[1.0, -1.0, 0.0, 0.0]])
            )
        learned = model.relation_temperature(relation_features)
        self.assertFalse(torch.allclose(learned[0], learned[1]))
        self.assertTrue(torch.all(learned > model.temperature_min))
        self.assertTrue(torch.all(learned < model.temperature_max))

    def test_state_reconstruction_uses_shared_entity_table(self):
        model = GWM(make_config())
        query = F.normalize(
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            p=2,
            dim=-1,
        )
        target_ids = torch.tensor([2])

        actual = model.compute_state_reconstruction_loss(
            query,
            target_ids,
        )
        candidates = F.normalize(
            model.struct_ent_embs.weight,
            p=2,
            dim=-1,
        )
        scores = torch.mm(query, candidates.t()) / model.temperature
        expected = F.cross_entropy(scores, target_ids)

        self.assertTrue(torch.allclose(actual, expected))

    def test_masked_world_state_ignores_head_identity(self):
        model = GWM(make_config())
        model.eval()
        context = make_context_batch()

        first = model.encode_masked_world_state(
            {'id': torch.tensor([0, 1])},
            context,
        )
        second = model.encode_masked_world_state(
            {'id': torch.tensor([2, 3])},
            context,
        )

        self.assertTrue(torch.allclose(first, second))

    def test_masked_world_state_depends_on_context(self):
        model = GWM(make_config())
        model.eval()
        h_batch = {'id': torch.tensor([0, 1])}

        contextualized = model.encode_masked_world_state(
            h_batch,
            make_context_batch(),
        )
        empty_context = model.encode_masked_world_state(
            h_batch,
            {
                'id': torch.full((2, 2), -1, dtype=torch.long),
                'rel_id': torch.full((2, 2), -1, dtype=torch.long),
                'mask': torch.zeros((2, 2), dtype=torch.bool),
            },
        )

        self.assertFalse(torch.allclose(contextualized, empty_context))


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
            self.assertEqual(batch['context_batch']['id'].tolist(), [[1, 2]])
            self.assertEqual(
                batch['context_batch']['rel_id'].tolist(),
                [[0, 0]],
            )
            self.assertEqual(
                batch['context_batch']['mask'].tolist(),
                [[False, True]],
            )

    def test_dataset_requires_precomputed_context(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_data(root)
            (Path(root) / 'context_neighbors.pt').unlink()
            with self.assertRaises(FileNotFoundError):
                GWMDataset(root, split='train')

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
