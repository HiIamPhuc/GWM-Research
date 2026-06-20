import tempfile
import unittest
from pathlib import Path

from utils.preprocess_data import (
    load_nell995_dataset,
    process_text_nell995,
)


class NELLPreprocessTests(unittest.TestCase):
    def test_nell995_counted_id_format_and_htr_order(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / 'entity2id.txt').write_text(
                '\n'.join([
                    '3',
                    'concept_city_boston\t0',
                    'concept_state_massachusetts\t1',
                    'concept_country_united_states\t2',
                ]),
                encoding='utf-8',
            )
            (root / 'relation2id.txt').write_text(
                '\n'.join([
                    '1',
                    'concept:citylocatedinstate\t0',
                ]),
                encoding='utf-8',
            )
            for split in ('train', 'valid', 'test'):
                (root / f'{split}.txt').write_text(
                    '1\n0 1 0\n',
                    encoding='utf-8',
                )

            train, valid, test, entity2id, relation2id = load_nell995_dataset(
                root,
                add_inverse=True,
            )

            self.assertEqual(
                train[0],
                (
                    'concept_city_boston',
                    'concept:citylocatedinstate',
                    'concept_state_massachusetts',
                ),
            )
            self.assertEqual(valid, train)
            self.assertEqual(test, train)
            self.assertEqual(entity2id['concept_city_boston'], 0)
            self.assertEqual(relation2id['concept:citylocatedinstate'], 0)
            self.assertEqual(relation2id['concept:citylocatedinstate_inv'], 1)

            entity_text, relation_text = process_text_nell995(
                root,
                entity2id,
                relation2id,
            )
            self.assertEqual(entity_text['0'], 'city boston')
            self.assertEqual(
                relation_text['1'],
                'inverse of citylocatedinstate',
            )


if __name__ == '__main__':
    unittest.main()
