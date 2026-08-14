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
from utils.relation_mapping import (
    attach_relation_direction_mapping,
    build_relation_direction_mapping,
)


def make_config():
    return SimpleNamespace(
        num_entities=4,
        num_relations=4,
        num_base_relations=2,
        relation_base_ids=[0, 0, 1, 1],
        relation_directions=[0, 1, 0, 1],
        struct_emb_dim=4,
        transition_decoder_layers=2,
        transition_decoder_heads=2,
        transition_decoder_ffn_multiplier=3,
        transition_decoder_dropout=0.0,
        temperature=0.07,
    )


def make_context_batch():
    return {
        'id': torch.tensor([[1, 2], [0, -1]]),
        'rel_id': torch.tensor([[0, 1], [2, -1]]),
        'mask': torch.tensor([[True, True], [True, False]]),
    }


class ModelTests(unittest.TestCase):
    def test_model_contains_only_minimal_components(self):
        model = GWM(make_config())
        self.assertEqual(
            set(dict(model.named_children())),
            {
                'struct_ent_embs',
                'base_rel_head_embs',
                'base_rel_tail_embs',
                'relation_norm',
                'state_norm',
                'transition_decoder',
            },
        )
        self.assertEqual(tuple(model.memory_roles.shape), (2, 4))
        self.assertEqual(tuple(model.next_state_token.shape), (1, 1, 4))
        self.assertFalse(hasattr(model, 'text_ent_embs'))
        self.assertFalse(hasattr(model, 'text_rel_embs'))
        self.assertFalse(hasattr(model, 'context_encoder'))
        self.assertFalse(hasattr(model, 'next_state_projection'))
        self.assertFalse(hasattr(model, 'masked_head_token'))
        self.assertFalse(hasattr(model, 'transition_mask'))

    def test_decoder_uses_one_relation_conditioned_query_and_raw_memory(self):
        model = GWM(make_config())
        model.eval()
        captured = {}

        def capture_transition(module, inputs):
            captured['query'] = inputs[0].detach().clone()
            captured['memory'] = inputs[1].detach().clone()

        handle = model.transition_decoder.layers[0].register_forward_pre_hook(
            capture_transition
        )
        h_ids = torch.tensor([0, 1])
        r_ids = torch.tensor([2, 3])
        model({'id': h_ids}, {'id': r_ids}, make_context_batch())
        handle.remove()

        head_role, _ = model.encode_relation_roles(r_ids)
        relation = model.encode_relation(r_ids)
        memory, _ = model.build_world_memory(
            {'id': h_ids},
            make_context_batch(),
            head_role,
        )
        expected_query = model.next_state_token + relation.unsqueeze(1)
        self.assertTrue(torch.allclose(captured['query'], expected_query))
        self.assertTrue(torch.allclose(captured['memory'], memory))
        self.assertEqual(captured['query'].shape, (2, 1, 4))
        self.assertEqual(captured['memory'].shape, (2, 3, 4))

    def test_query_depends_on_context(self):
        model = GWM(make_config())
        model.eval()
        h_batch = {'id': torch.tensor([0, 1])}
        r_batch = {'id': torch.tensor([2, 3])}
        contextualized = model(h_batch, r_batch, make_context_batch())
        empty_context = model(
            h_batch,
            r_batch,
            {
                'id': torch.full((2, 2), -1, dtype=torch.long),
                'rel_id': torch.full((2, 2), -1, dtype=torch.long),
                'mask': torch.zeros((2, 2), dtype=torch.bool),
            },
        )
        self.assertEqual(contextualized.shape, (2, 4))
        self.assertTrue(torch.isfinite(contextualized).all())
        self.assertFalse(torch.allclose(contextualized, empty_context))

    def test_full_entity_loss_backpropagates_through_minimal_model(self):
        model = GWM(make_config())
        query = model(
            {'id': torch.tensor([0, 1])},
            {'id': torch.tensor([0, 1])},
            make_context_batch(),
        )
        relation_ids = torch.tensor([0, 1])
        loss = model.compute_loss(
            query,
            relation_ids,
            torch.tensor([2, 3]),
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.struct_ent_embs.weight.grad)
        self.assertIsNotNone(model.base_rel_head_embs.weight.grad)
        self.assertIsNotNone(model.base_rel_tail_embs.weight.grad)
        self.assertIsNotNone(model.memory_roles.grad)
        self.assertIsNotNone(model.next_state_token.grad)
        self.assertIsNotNone(
            model.transition_decoder.layers[0].multihead_attn.in_proj_weight.grad
        )

    def test_inverse_relation_exactly_swaps_pairre_roles(self):
        model = GWM(make_config())
        self.assertEqual(model.relation_base_ids[[0, 1]].tolist(), [0, 0])
        head_roles, tail_roles = model.encode_relation_roles(
            torch.tensor([0, 1])
        )
        self.assertTrue(torch.equal(head_roles[0], tail_roles[1]))
        self.assertTrue(torch.equal(tail_roles[0], head_roles[1]))

    def test_relation_mapping_does_not_assume_contiguous_pairs(self):
        mapping = build_relation_direction_mapping({
            'r_a_inv': 0,
            'r_b_inv': 1,
            'r_a': 2,
            'r_b': 3,
        })
        self.assertEqual(mapping['num_base_relations'], 2)
        self.assertEqual(mapping['full_to_base'], [0, 1, 0, 1])
        self.assertEqual(mapping['directions'], [1, 1, 0, 0])

    def test_targets_are_normalized_structural_entities(self):
        model = GWM(make_config())
        ids = torch.tensor([1, 3])
        actual = model.encode_target({'id': ids})
        expected = F.normalize(model.struct_ent_embs(ids), p=2, dim=-1)
        self.assertTrue(torch.allclose(actual, expected))

    def test_loss_is_full_entity_cross_entropy(self):
        model = GWM(make_config())
        query = F.normalize(
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            p=2,
            dim=-1,
        )
        relation_ids = torch.tensor([0])
        target_ids = torch.tensor([2])
        actual = model.compute_loss(query, relation_ids, target_ids)
        candidates = F.normalize(model.struct_ent_embs.weight, p=2, dim=-1)
        _, tail_roles = model.encode_relation_roles(relation_ids)
        expected = F.cross_entropy(
            model.score_candidates(query, candidates, tail_roles),
            target_ids,
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_pairre_scores_match_explicit_squared_distance(self):
        model = GWM(make_config())
        query = torch.tensor([[0.5, -0.5, 0.5, -0.5]])
        candidates = F.normalize(
            torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
            ),
            p=2,
            dim=-1,
        )
        tail_roles = torch.tensor([[1.0, 0.5, -1.0, 2.0]])
        actual = model.score_candidates(query, candidates, tail_roles)
        expected = -(
            query.unsqueeze(1)
            - candidates.unsqueeze(0) * tail_roles.unsqueeze(1)
        ).square().sum(dim=-1) / model.temperature
        self.assertTrue(torch.allclose(actual, expected))

    def test_scorer_scores_every_entity(self):
        model = GWM(make_config())
        scores = model.score_all_entities(
            {'id': torch.tensor([0, 1])},
            {'id': torch.tensor([0, 1])},
            make_context_batch(),
        )
        self.assertEqual(scores.shape, (2, 4))
        self.assertTrue(torch.isfinite(scores).all())

    def test_study_factory_returns_current_model(self):
        self.assertIsInstance(build_model(make_config()), GWM)

    def test_relation_diverse_context_selection_covers_relations_first(self):
        selected = ContextProcessor._select_relation_diverse_neighbors(
            [(0, 1), (0, 2), (1, 3), (2, 0)],
            limit=3,
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual({relation for relation, _ in selected}, {0, 1, 2})


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
            self.assertEqual(batch['context_batch']['rel_id'].tolist(), [[0, 0]])
            self.assertEqual(batch['context_batch']['mask'].tolist(), [[False, True]])

    def test_dataset_requires_precomputed_context(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_data(root)
            (Path(root) / 'context_neighbors.pt').unlink()
            with self.assertRaises(FileNotFoundError):
                GWMDataset(root, split='train')

    def test_relation_mapping_attaches_only_base_and_direction(self):
        with tempfile.TemporaryDirectory() as root:
            relation2id = {
                'r_a': 0,
                'r_a_inv': 1,
                'r_b': 2,
                'r_b_inv': 3,
            }
            (Path(root) / 'relation2id.json').write_text(
                json.dumps(relation2id),
                encoding='utf-8',
            )
            config = SimpleNamespace()
            mapping = attach_relation_direction_mapping(config, root)

            self.assertEqual(mapping['full_to_base'], [0, 0, 1, 1])
            self.assertEqual(mapping['directions'], [0, 1, 0, 1])
            self.assertEqual(config.num_base_relations, 2)
            self.assertFalse(hasattr(config, 'relation_slot_counts'))

    def test_bidirectional_eval_dataset_builds_inverse_queries_on_the_fly(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_data(root)
            torch.save(torch.tensor([[0, 0, 1]]), Path(root) / 'test_triples.pt')
            base_dataset = GWMDataset(root, split='test')
            forward, backward = build_bidirectional_eval_dataset(
                base_dataset,
                root,
            )
            self.assertEqual(forward.triples.tolist(), [[0, 0, 1]])
            self.assertEqual(backward.triples.tolist(), [[1, 1, 0]])

    def test_bidirectional_filter_map_adds_inverse_truths(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_data(root)
            torch.save(
                torch.tensor([[0, 0, 2]]),
                Path(root) / 'valid_triples.pt',
            )
            torch.save(
                torch.tensor([[0, 0, 1]]),
                Path(root) / 'test_triples.pt',
            )
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
