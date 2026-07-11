import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "linear_probe_tokenizers.py"
SPEC = importlib.util.spec_from_file_location("linear_probe_tokenizers", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TokenizerLinearProbeTest(unittest.TestCase):
    def test_selection_partition_is_exact_deterministic_and_disjoint(self):
        selection_a, pool_a = MODULE.make_train_selection_indices(40_000, fraction=0.1, seed=0)
        selection_b, pool_b = MODULE.make_train_selection_indices(40_000, fraction=0.1, seed=0)

        self.assertEqual(len(selection_a), 4_000)
        self.assertEqual(len(pool_a), 36_000)
        np.testing.assert_array_equal(selection_a, selection_b)
        np.testing.assert_array_equal(pool_a, pool_b)
        self.assertEqual(len(np.intersect1d(selection_a, pool_a)), 0)
        self.assertEqual(len(np.unique(np.concatenate((selection_a, pool_a)))), 40_000)

    def test_support_is_balanced_unique_nested_and_outside_selection(self):
        targets = np.repeat(np.arange(MODULE.NUM_CLASSES), 40)
        selection, pool = MODULE.make_train_selection_indices(len(targets), fraction=0.1, seed=0)
        support = MODULE.make_nested_support_indices(targets, max_shot=16, seed=7, pool_indices=pool)

        self.assertEqual(support.shape, (MODULE.NUM_CLASSES, 16))
        expected_labels = np.repeat(np.arange(MODULE.NUM_CLASSES)[:, None], 16, axis=1)
        np.testing.assert_array_equal(targets[support], expected_labels)
        self.assertEqual(len(np.unique(support)), support.size)
        self.assertEqual(len(np.intersect1d(selection, support)), 0)
        np.testing.assert_array_equal(support[:, :1], support[:, :2][:, :1])
        np.testing.assert_array_equal(support[:, :8], support[:, :16][:, :8])

    def test_support_seed_is_deterministic_and_model_independent(self):
        targets = np.repeat(np.arange(MODULE.NUM_CLASSES), 24)
        pool = np.arange(len(targets))
        first = MODULE.make_nested_support_indices(targets, max_shot=16, seed=2, pool_indices=pool)
        second = MODULE.make_nested_support_indices(targets, max_shot=16, seed=2, pool_indices=pool)
        other_seed = MODULE.make_nested_support_indices(targets, max_shot=16, seed=3, pool_indices=pool)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other_seed))

    def test_persisted_splits_record_and_validate_hashes(self):
        class DummyDataset:
            targets = np.repeat(np.arange(MODULE.NUM_CLASSES), 40).tolist()

            def __len__(self):
                return len(self.targets)

        dataset = DummyDataset()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection_path = root / "selection.npz"
            support_path = root / "support.npz"
            selection, pool, selection_metadata = MODULE.load_or_create_train_selection(
                dataset, selection_path, 0.1, 0, "dataset-order-hash"
            )
            support, support_metadata = MODULE.load_or_create_support(
                dataset,
                support_path,
                pool,
                16,
                0,
                "dataset-order-hash",
                selection_metadata["support_pool_indices_sha256"],
            )
            reloaded_selection, reloaded_pool, reloaded_metadata = MODULE.load_or_create_train_selection(
                dataset, selection_path, 0.1, 0, "dataset-order-hash"
            )

            np.testing.assert_array_equal(selection, reloaded_selection)
            np.testing.assert_array_equal(pool, reloaded_pool)
            self.assertEqual(len(np.intersect1d(selection, support)), 0)
            self.assertEqual(
                selection_metadata["selection_indices_sha256"],
                reloaded_metadata["selection_indices_sha256"],
            )
            self.assertEqual(support_metadata["support_indices_sha256"], MODULE._sha256_array(support))
            with self.assertRaises(ValueError):
                MODULE.load_or_create_train_selection(dataset, selection_path, 0.1, 0, "different-hash")

    def test_parametric_c_search_finds_internal_peak(self):
        selected_c, history = MODULE.parametric_c_search(lambda exponent: -((exponent - 0.3) ** 2))
        selected_exponent = math.log10(selected_c)

        self.assertLessEqual(abs(selected_exponent - 0.3), MODULE.DEFAULT_C_RESOLUTION / 2)
        self.assertGreater(len(history), len(MODULE.DEFAULT_C_EXPONENTS))
        evaluated = {item["log10_C"] for item in history}
        self.assertTrue(set(MODULE.DEFAULT_C_EXPONENTS).issubset(evaluated))

    def test_parametric_c_search_handles_boundary_peak(self):
        selected_c, history = MODULE.parametric_c_search(lambda exponent: exponent)
        self.assertEqual(math.log10(selected_c), 6.0)
        self.assertTrue(all(-6.0 <= item["log10_C"] <= 6.0 for item in history))

    def test_parametric_c_search_tie_chooses_smallest_c(self):
        selected_c, _ = MODULE.parametric_c_search(lambda exponent: 1.0)
        self.assertEqual(math.log10(selected_c), -6.0)

    def test_feature_cache_requires_exact_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            np.save(cache_dir / "features.npy", np.zeros((3, 2), dtype=np.float32))
            np.save(cache_dir / "labels.npy", np.arange(3, dtype=np.int64))
            fingerprint = {"protocol": "clip-paper-v1", "batch_size": 100}
            (cache_dir / "metadata.json").write_text(json.dumps({"count": 3, "fingerprint": fingerprint}))

            self.assertTrue(MODULE._feature_cache_complete(cache_dir, 3, fingerprint))
            with self.assertRaises(RuntimeError):
                MODULE._feature_cache_complete(cache_dir, 3, {"protocol": "other"})
            self.assertFalse(
                MODULE._feature_cache_complete(cache_dir, 3, {"protocol": "other"}, overwrite=True)
            )

    def test_result_resume_requires_protocol_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "results.json"
            output_path.write_text(
                json.dumps(
                    {
                        "protocol_config_sha256": "expected",
                        "results": [{"shot": 1, "top1": 10.0}],
                    }
                )
            )
            completed = MODULE._load_completed_results(output_path, "expected", overwrite_probe=False)
            self.assertIn(1, completed)
            with self.assertRaises(RuntimeError):
                MODULE._load_completed_results(output_path, "different", overwrite_probe=False)
            self.assertEqual(MODULE._load_completed_results(output_path, "different", overwrite_probe=True), {})

    def test_cli_defaults_to_paper_protocol(self):
        args = MODULE.parse_args(["--model", "unitok"])
        self.assertEqual(args.protocol, "clip-paper-v1")
        self.assertEqual(args.shots, [1, 2, 4, 8, 16])
        self.assertEqual(args.selection_seed, 0)
        self.assertEqual(args.selection_fraction, 0.1)
        self.assertEqual(args.batch_size, 100)
        self.assertEqual(args.max_iter, 1_000)

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
