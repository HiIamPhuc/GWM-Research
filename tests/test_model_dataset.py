import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from model.dataset import CollateFN, GWMDataset
from model.model import CompGCN, GWM


def make_config():
    return SimpleNamespace(
        num_entities=4,
        num_relations=4,
        text_emb_dim=6,
        struct_emb_dim=4,
        fusion_dim=5,
        text_adapter_dim=6,
        struct_adapter_dim=4,
        dynamics_layers=1,
        dropout=0.0,
        adapter_dropout=0.0,
        temperature=0.1,
    )


class ModelTests(unittest.TestCase):
    def test_structural_freeze_targets_structural_tables(self):
        model = GWM(make_config())
        model.load_embeddings(
            torch.randn(4, 4),
            torch.randn(4, 4),
            kind='structural',
            freeze=True,
        )
        self.assertFalse(model.struct_ent_embs.weight.requires_grad)
        self.assertFalse(model.struct_rel_embs.weight.requires_grad)
        self.assertTrue(model.text_ent_embs.weight.requires_grad)

    def test_isolated_head_preserves_self_state(self):
        layer = CompGCN(hidden_dim=4, dropout=0.0)
        output = layer(
            head_feat=torch.randn(2, 4),
            nbr_entity_feat=torch.empty(0, 4),
            nbr_relation_feat=torch.empty(0, 4),
            nbr_batch_index=torch.empty(0, dtype=torch.long),
        )
        self.assertEqual(output.shape, (2, 4))
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(torch.equal(output, torch.zeros_like(output)))

    def test_repeated_targets_are_multiple_positives(self):
        scores = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        losses = GWM._multi_positive_contrastive_loss(
            scores, target_ids=torch.tensor([2, 2])
        )
        self.assertTrue(torch.equal(losses, torch.zeros_like(losses)))

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
            query, targets, target_ids=t_batch['id']
        )
        self.assertEqual(scores.shape, (2, 2))
        self.assertEqual(query.shape, (2, 5))
        self.assertEqual(targets.shape, (2, 5))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.entity_fusion.gate[1].weight.grad)
        self.assertIsNotNone(model.relation_fusion.gate[1].weight.grad)


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


if __name__ == '__main__':
    unittest.main()
