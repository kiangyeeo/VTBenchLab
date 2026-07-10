import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "linear_probe_tokenizers.py"
SPEC = importlib.util.spec_from_file_location("linear_probe_tokenizers", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TokenizerLinearProbeTest(unittest.TestCase):
    def test_support_is_balanced_unique_and_nested(self):
        targets = np.repeat(np.arange(MODULE.NUM_CLASSES), 20)
        support = MODULE.make_nested_support_indices(targets, max_shot=16, seed=7)

        self.assertEqual(support.shape, (MODULE.NUM_CLASSES, 16))
        expected_labels = np.repeat(np.arange(MODULE.NUM_CLASSES)[:, None], 16, axis=1)
        np.testing.assert_array_equal(targets[support], expected_labels)
        self.assertEqual(len(np.unique(support)), support.size)
        np.testing.assert_array_equal(support[:, :1], support[:, :2][:, :1])
        np.testing.assert_array_equal(support[:, :8], support[:, :16][:, :8])

    def test_topk_correct(self):
        logits = np.asarray(
            [
                [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ]
        )
        targets = np.asarray([0, 1])
        self.assertEqual(MODULE._topk_correct(logits, targets, 1), 1)
        self.assertEqual(MODULE._topk_correct(logits, targets, 5), 2)


if __name__ == "__main__":
    unittest.main()
