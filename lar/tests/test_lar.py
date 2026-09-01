from __future__ import annotations

import unittest

import numpy as np

from lar.compute_lar import compute_metrics
from lar.data import build_text_dataset
from lar.eval_e2 import anova_f
from lar.eval_e3 import mean_regret


class DatasetProtocolTest(unittest.TestCase):
    def test_expected_dataset_sizes_and_alignment(self):
        captions = build_text_dataset("caption", "coco4618")
        answers = build_text_dataset("answer", "coco4618")
        all_captions = build_text_dataset("caption", "coco5000")
        self.assertEqual(len(captions.ids), 4618)
        self.assertEqual(captions.ids, answers.ids)
        self.assertEqual(len(all_captions.ids), 5000)
        self.assertEqual(len(answers.answer_counts or []), 4618)


class MetricTest(unittest.TestCase):
    def test_metrics_are_finite_and_bounded(self):
        rng = np.random.default_rng(7)
        n, d = 320, 24
        latent = rng.normal(size=(n, d)) * np.linspace(3.0, 0.2, d)
        text = np.column_stack((latent[:, :4] + 0.05 * rng.normal(size=(n, 4)), rng.normal(size=(n, 12))))
        metrics, spectra = compute_metrics(latent.astype(np.float32), text.astype(np.float32))
        self.assertEqual(len(spectra["lam"]), d)
        self.assertEqual(int(metrics["K"]), d)
        for key in ("LAR_8", "LAR_16", "Waste"):
            self.assertTrue(np.isfinite(metrics[key]))
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0 + 1e-12)
        self.assertGreaterEqual(metrics["LAR_16"], metrics["LAR_8"])

    def test_protocol_helpers(self):
        self.assertGreater(anova_f({"a": [0.0, 0.1], "b": [2.0, 2.1]}), 100.0)
        values = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        target = values.copy()
        regret = mean_regret(values, target, 5, 20, np.random.default_rng(0))
        self.assertEqual(regret, 0.0)


if __name__ == "__main__":
    unittest.main()
