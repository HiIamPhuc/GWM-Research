import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from model.dataset import CollateFN, GWMDataset, TrainTruthIndex
from model.model import (
    CompGCNWorldStateEncoder,
    GWM,
    filtered_in_batch_contrastive_loss,
)
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
        text_emb_dim=6,
        struct_emb_dim=4,
        fusion_dim=10,
        text_adapter_dim=6,
        struct_adapter_dim=4,
        dynamics_layers=1,
        dropout=0.0,
        adapter_dropout=0.0,
        temperature=0.07,
    )


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
        layer = CompGCNWorldStateEncoder(hidden_dim=4)
        output = layer(
            head_feat=torch.randn(2, 4),
            nbr_entity_feat=torch.empty(0, 4),
            nbr_relation_feat=torch.empty(0, 4),
            nbr_direction=torch.empty(0, dtype=torch.long),
            nbr_batch_index=torch.empty(0, dtype=torch.long),
        )
        self.assertEqual(output.shape, (2, 4))
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(torch.equal(output, torch.zeros_like(output)))

    def test_compgcn_sum_by_batch(self):
        messages = torch.tensor(
            [[1.0, 3.0], [5.0, 2.0], [7.0, 9.0]]
        )
        batch_index = torch.tensor([0, 0, 1])
        reference = torch.zeros(2, 2)
        layer = CompGCNWorldStateEncoder(2)

        result = layer._sum_by_batch(
            messages,
            batch_index,
            batch_size=2,
            reference=reference,
        )

        self.assertTrue(
            torch.equal(result, torch.tensor([[6.0, 5.0], [7.0, 9.0]]))
        )

    def test_compgcn_uses_distinct_direction_projections(self):
        layer = CompGCNWorldStateEncoder(2, dropout=0.0)
        with torch.no_grad():
            layer.outgoing_projection.weight.copy_(torch.eye(2))
            layer.incoming_projection.weight.copy_(2.0 * torch.eye(2))
            layer.self_projection.weight.zero_()

        head = torch.zeros(1, 2)
        entities = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
        relations = torch.ones(2, 2)
        batch_index = torch.zeros(2, dtype=torch.long)
        outgoing = layer(
            head,
            entities,
            relations,
            torch.tensor([0, 0]),
            batch_index,
        )
        incoming = layer(
            head,
            entities,
            relations,
            torch.tensor([1, 1]),
            batch_index,
        )

        self.assertFalse(torch.allclose(outgoing, incoming))

    def test_baseline_uses_shared_modality_adapters(self):
        model = GWM(make_config())
        self.assertIsInstance(model.text_adapter, torch.nn.Module)
        self.assertIsInstance(model.struct_adapter, torch.nn.Module)
        self.assertFalse(hasattr(model, 'text_entity_adapter'))
        self.assertFalse(hasattr(model, 'struct_entity_adapter'))

    def test_single_modality_variants_use_compgcn_world_state(self):
        h_batch = {'id': torch.tensor([0, 1])}
        r_batch = {'id': torch.tensor([0, 1])}
        context_batch = {
            'id': torch.tensor([1, 2, 0]),
            'rel_id': torch.tensor([0, 1, 2]),
            'direction_id': torch.tensor([0, 0, 1]),
            'batch_index': torch.tensor([0, 0, 1]),
        }

        for variant in ('text_only', 'structure_only'):
            config = make_config()
            config.model_variant = variant
            model = build_model(config)
            scores = model.score_all_entities(h_batch, r_batch, context_batch)

            self.assertEqual(scores.shape, (2, 4))
            self.assertTrue(torch.isfinite(scores).all())
            self.assertIsInstance(model.adapter, torch.nn.Module)
            self.assertIsInstance(
                model.world_state_encoder,
                CompGCNWorldStateEncoder,
            )

    def test_early_fusion_loss_backpropagates_to_gate(self):
        model = GWM(make_config())
        h_batch = {'id': torch.tensor([0, 1])}
        r_batch = {'id': torch.tensor([0, 1])}
        t_batch = {'id': torch.tensor([2, 3])}
        context_batch = {
            'id': torch.tensor([1, 2]),
            'rel_id': torch.tensor([0, 1]),
            'direction_id': torch.tensor([0, 0]),
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
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.entity_fusion.gate[1].weight.grad)
        self.assertIsNotNone(model.relation_fusion.gate[1].weight.grad)

    def test_dot_product_scores_all_entities(self):
        model = GWM(make_config())
        h_batch = {'id': torch.tensor([0, 1])}
        r_batch = {'id': torch.tensor([0, 1])}
        t_batch = {'id': torch.tensor([2, 3])}
        context_batch = {
            'id': torch.tensor([1, 2]),
            'rel_id': torch.tensor([0, 1]),
            'direction_id': torch.tensor([0, 0]),
            'batch_index': torch.tensor([0, 1]),
        }

        scores = model.score_all_entities(h_batch, r_batch, context_batch)
        self.assertEqual(scores.shape, (2, 4))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertFalse(hasattr(model, 'decoder'))

    def test_output_projection_is_shared_by_query_and_target(self):
        model = GWM(make_config())
        h_batch = {'id': torch.tensor([0, 1])}
        r_batch = {'id': torch.tensor([0, 1])}
        context_batch = {
            'id': torch.tensor([1, 2]),
            'rel_id': torch.tensor([0, 1]),
            'direction_id': torch.tensor([0, 0]),
            'batch_index': torch.tensor([0, 1]),
        }

        query, _ = model.encode_query(h_batch, r_batch, context_batch)
        query.sum().backward()
        self.assertIsNotNone(model.fused_output_projection.weight.grad)

        model.zero_grad(set_to_none=True)
        target = model.encode_target({'id': torch.tensor([2, 3])})
        target[:, 0].sum().backward()
        self.assertIsNotNone(model.fused_output_projection.weight.grad)

    def test_relation_update_receives_gradient(self):
        model = GWM(make_config())
        context_batch = {
            'id': torch.tensor([1, 2]),
            'rel_id': torch.tensor([0, 1]),
            'direction_id': torch.tensor([0, 1]),
            'batch_index': torch.tensor([0, 1]),
        }
        query = model(
            {'id': torch.tensor([0, 1])},
            {'id': torch.tensor([0, 1])},
            context_batch,
        )
        query.sum().backward()
        self.assertIsNotNone(
            model.world_state_encoder.relation_projection.weight.grad
        )

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

    def test_gate_statistics_are_recorded_and_consumed(self):
        model = GWM(make_config())
        h_batch = {'id': torch.tensor([0, 1])}
        r_batch = {'id': torch.tensor([0, 1])}
        t_batch = {'id': torch.tensor([2, 3])}
        context_batch = {
            'id': torch.tensor([1, 2]),
            'rel_id': torch.tensor([0, 1]),
            'direction_id': torch.tensor([0, 0]),
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
                {'id', 'rel_id', 'direction_id', 'batch_index'},
            )

    def test_inverse_context_relations_are_marked_incoming(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_data(root)
            dataset = GWMDataset(root, split='train')
            self.assertEqual(dataset.context_direction_ids[0].tolist(), [0, 0])
            self.assertEqual(dataset.context_direction_ids[1].tolist(), [1, 0])

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
