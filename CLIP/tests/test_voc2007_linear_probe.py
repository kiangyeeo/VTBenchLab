import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "linear_probe_voc2007.py"
SPEC = importlib.util.spec_from_file_location("linear_probe_voc2007", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class VOC2007LinearProbeTest(unittest.TestCase):
    def test_voc2007_ap_ignores_difficult_and_uses_eleven_recall_points(self):
        labels = np.asarray([0, 1, -1, 1, -1], dtype=np.int8)
        scores = np.asarray([100.0, 0.9, 0.8, 0.7, 0.1])

        # Six recall thresholds use precision 1.0 and five use precision 2/3.
        expected = 28.0 / 33.0
        actual = MODULE.voc2007_11_point_ap(labels, scores)

        self.assertAlmostEqual(actual, expected)

    def test_voc2007_ap_rejects_a_class_without_positive_examples(self):
        with self.assertRaisesRegex(ValueError, "without positive"):
            MODULE.voc2007_11_point_ap(
                np.asarray([-1, 0, -1], dtype=np.int8),
                np.asarray([0.9, 0.8, 0.7]),
            )

    def test_multilabel_evaluation_is_the_mean_of_twenty_class_aps(self):
        labels_one_class = np.asarray([1, -1, 1, -1], dtype=np.int8)
        scores_one_class = np.asarray([0.9, 0.8, 0.7, 0.1])
        labels = np.repeat(labels_one_class[:, None], len(MODULE.VOC_CLASSES), axis=1)
        scores = np.repeat(scores_one_class[:, None], len(MODULE.VOC_CLASSES), axis=1)

        mean_ap, per_class = MODULE.evaluate_multilabel(labels, scores)

        self.assertEqual(per_class.shape, (20,))
        self.assertTrue(np.allclose(per_class, per_class[0]))
        self.assertAlmostEqual(mean_ap, per_class[0])

    def test_regularization_grid_matches_paper_and_ties_choose_stronger_l2(self):
        lambdas = MODULE.regularization_grid()

        self.assertEqual(len(lambdas), 45)
        self.assertAlmostEqual(lambdas[0], 1e5)
        self.assertAlmostEqual(lambdas[-1], 1e-6)
        np.testing.assert_allclose(np.diff(np.log10(lambdas)), -0.25)
        self.assertAlmostEqual(MODULE.lambda_to_c(1e5), 1e-5)
        self.assertAlmostEqual(MODULE.lambda_to_c(1e-6), 1e6)

        selected = MODULE.select_best_lambda([1e5, 1.0, 1e-6], [50.0, 60.0, 60.0])
        self.assertEqual(selected, 1)

    def test_split_loader_stacks_classes_and_rejects_order_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_dir = root / "ImageSets/Main"
            split_dir.mkdir(parents=True)
            rows = "000001  1\n000002 -1\n000003  0\n"
            for class_name in MODULE.VOC_CLASSES:
                (split_dir / f"{class_name}_train.txt").write_text(rows)

            image_ids, labels = MODULE.load_voc_split(root, "train", validate_size=False)

            self.assertEqual(image_ids, ["000001", "000002", "000003"])
            self.assertEqual(labels.shape, (3, 20))
            np.testing.assert_array_equal(labels[:, 0], np.asarray([1, -1, 0], dtype=np.int8))
            with self.assertRaisesRegex(ValueError, "Unexpected VOC2007 train size"):
                MODULE.load_voc_split(root, "train")
            with self.assertRaisesRegex(FileNotFoundError, "Missing 3 VOC JPEG"):
                MODULE.VOC2007ClassificationDataset(
                    root,
                    "train",
                    validate_size=False,
                )

            (split_dir / "tvmonitor_train.txt").write_text(
                "000002 -1\n000001  1\n000003  0\n"
            )
            with self.assertRaisesRegex(ValueError, "order differs"):
                MODULE.load_voc_split(root, "train", validate_size=False)

    def test_split_loader_rejects_non_voc_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_dir = root / "ImageSets/Main"
            split_dir.mkdir(parents=True)
            for class_name in MODULE.VOC_CLASSES:
                label = 2 if class_name == "aeroplane" else -1
                (split_dir / f"{class_name}_val.txt").write_text(f"000001 {label}\n")

            with self.assertRaisesRegex(ValueError, "must be -1, 0, or 1"):
                MODULE.load_voc_split(root, "val", validate_size=False)

    def test_feature_cache_requires_exact_fingerprint_and_image_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            fingerprint = {"protocol": MODULE.PROTOCOL, "split": "train"}
            image_ids = ["000001", "000002"]
            np.save(cache_dir / "features.npy", np.zeros((2, 4), dtype=np.float32))
            np.save(cache_dir / "labels.npy", np.zeros((2, 20), dtype=np.int8))
            (cache_dir / "image_ids.txt").write_text("000001\n000002\n")
            (cache_dir / "metadata.json").write_text(
                json.dumps({"count": 2, "fingerprint": fingerprint})
            )

            self.assertTrue(
                MODULE._feature_cache_complete(cache_dir, 2, fingerprint, image_ids)
            )
            self.assertFalse(
                MODULE._feature_cache_complete(
                    cache_dir,
                    2,
                    fingerprint,
                    image_ids,
                    overwrite=True,
                )
            )
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                MODULE._feature_cache_complete(
                    cache_dir,
                    2,
                    {"protocol": "different"},
                    image_ids,
                )
            self.assertFalse(
                MODULE._feature_cache_complete(
                    cache_dir,
                    2,
                    {"protocol": "different"},
                    image_ids,
                    overwrite=True,
                )
            )

    def test_transform_specs_are_whole_image_and_model_native_size(self):
        expected_sizes = {
            "unitok": (256, 256),
            "vilau": (256, 256),
            "metaclip": (224, 224),
            "toklips": (256, 256),
            "toklipl": (384, 384),
        }
        for model, expected_size in expected_sizes.items():
            with self.subTest(model=model):
                spec = MODULE.MODEL_TRANSFORM_SPECS[model]
                self.assertEqual(spec.size, expected_size)
                self.assertEqual(spec.as_dict()["mode"], "whole-image-resize-no-crop")

    def test_class_search_cache_is_resumable_only_for_exact_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bird.npz"
            fingerprint = "abc123"
            np.savez_compressed(
                path,
                fingerprint_sha256=np.asarray(fingerprint),
                class_name=np.asarray("bird"),
                lambdas=MODULE.regularization_grid(),
                val_scores=np.zeros((45, 3), dtype=np.float64),
                n_iter=np.ones(45, dtype=np.int32),
                converged=np.ones(45, dtype=np.bool_),
                elapsed_seconds=np.zeros(45, dtype=np.float64),
            )

            self.assertTrue(
                MODULE._class_search_cache_complete(
                    path,
                    fingerprint,
                    3,
                    overwrite=False,
                )
            )
            self.assertFalse(
                MODULE._class_search_cache_complete(
                    path,
                    fingerprint,
                    3,
                    overwrite=True,
                )
            )
            with self.assertRaisesRegex(RuntimeError, "protocol does not match"):
                MODULE._class_search_cache_complete(
                    path,
                    "different",
                    3,
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()
