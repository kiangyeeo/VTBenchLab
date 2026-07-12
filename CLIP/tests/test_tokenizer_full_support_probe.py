import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


CLIP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLIP_DIR))
MODULE_PATH = CLIP_DIR / "linear_probe_tokenizers_full_support.py"
SPEC = importlib.util.spec_from_file_location("linear_probe_tokenizers_full_support", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TokenizerFullSupportProbeTest(unittest.TestCase):
    def test_class_count_summary(self):
        labels = np.repeat(np.arange(MODULE.kshot.NUM_CLASSES), 3)
        summary = MODULE.class_count_summary(labels)
        self.assertEqual(summary["num_classes"], MODULE.kshot.NUM_CLASSES)
        self.assertEqual(summary["mean"], 3.0)
        self.assertEqual(summary["min"], 3)
        self.assertEqual(summary["max"], 3)

    def test_resumable_search_reuses_every_completed_candidate(self):
        calls = []

        def evaluate(exponent):
            calls.append(exponent)
            return -((exponent - 0.3) ** 2), {"converged": True, "n_iter_max": 2}

        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "search.json"
            selected_first, history_first = MODULE.resumable_parametric_c_search(
                evaluate, state_path, "protocol-hash"
            )
            first_call_count = len(calls)
            selected_second, history_second = MODULE.resumable_parametric_c_search(
                evaluate, state_path, "protocol-hash"
            )

        self.assertGreater(first_call_count, 0)
        self.assertEqual(len(calls), first_call_count)
        self.assertEqual(selected_first, selected_second)
        self.assertEqual(history_first, history_second)
        self.assertLessEqual(
            abs(math.log10(selected_first) - 0.3), MODULE.kshot.DEFAULT_C_RESOLUTION / 2
        )

    def test_cli_defaults_match_kshot_protocol(self):
        args = MODULE.parse_args(["--model", "unitok"])
        self.assertEqual(args.protocol, "clip-paper-v1")
        self.assertEqual(args.selection_seed, 0)
        self.assertEqual(args.selection_fraction, 0.1)
        self.assertEqual(args.batch_size, 100)
        self.assertEqual(args.max_iter, 1_000)
        self.assertIsNone(args.fixed_c)


if __name__ == "__main__":
    unittest.main()
