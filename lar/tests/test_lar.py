from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from lar.compute_lar import CSV_FIELDS, compute_metrics, compute_spectral_metrics
from lar.compute_lar_pool import main as pool_main
from lar.data import build_text_dataset
from lar.eval_e2 import anova_f
from lar.eval_e3 import main as e3_main, mean_regret


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
        for key in ("LAR_8", "LAR_16"):
            self.assertTrue(np.isfinite(metrics[key]))
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0 + 1e-12)
        for key in ("Lift_8", "Lift_16", "VSA", "log2_m50", "log2_m90"):
            self.assertTrue(np.isfinite(metrics[key]))
        self.assertGreaterEqual(metrics["LAR_16"], metrics["LAR_8"])
        self.assertLessEqual(metrics["m50"], metrics["m90"])

    def test_lift_removes_variance_spectrum(self):
        lam = np.geomspace(100.0, 1.0, 128)
        metrics = compute_spectral_metrics(lam, np.ones(128))
        for m in (8, 16, 32, 64, 128):
            self.assertAlmostEqual(metrics[f"Lift_{m}"], 1.0)
        self.assertAlmostEqual(metrics["VSA"], 0.0)

    def test_vsa_sign_and_quantile_component_count(self):
        lam = np.ones(128)
        r = np.arange(128, 0, -1, dtype=np.float64)
        metrics = compute_spectral_metrics(lam, r)
        self.assertAlmostEqual(metrics["VSA"], 1.0)
        self.assertEqual(metrics["m50"], 38.0)

    def test_protocol_helpers(self):
        self.assertGreater(anova_f({"a": [0.0, 0.1], "b": [2.0, 2.1]}), 100.0)
        values = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        target = values.copy()
        regret = mean_regret(values, target, 5, 20, np.random.default_rng(0))
        self.assertEqual(regret, 0.0)


class E3PipelineTest(unittest.TestCase):
    def test_complete_synthetic_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.csv"
            targets_path = root / "targets.csv"
            output_path = root / "e3.json"
            report_path = root / "e3_report.md"
            figure_path = root / "lift.png"
            metric_rows = []
            target_rows = []
            for index in range(18):
                name = f"model_{index:02d}"
                family = f"family_{index // 3}"
                quality = float(index) + 0.1 * (index % 3)
                target_rows.append(
                    {
                        "name": name, "family": family,
                        "MLLM_Avg": 40.0 + quality,
                        "qwen3": 38.0 + 0.9 * quality,
                        "qwen2_5": 40.0 + quality,
                        "smollm2": 36.0 + 1.1 * quality + 0.03 * (index % 2),
                        "probe_epoch1": 65.0 + 0.7 * quality,
                        "retrieval-ImageNet": 55.0 + 0.5 * quality,
                        "CKA": 0.2 + 0.01 * quality,
                        "pretrain_loss": 4.0 - 0.05 * quality,
                        "A_score": 0.3 + 0.015 * quality,
                    }
                )
                for domain_index, domain in enumerate(("caption", "answer")):
                    dimension = 256 + 8 * index
                    row = {field: "" for field in CSV_FIELDS}
                    row.update(
                        name=name, text_domain=domain, image_set="coco4618",
                        d=dimension, n_tokens=196 + index, N=4618,
                        K=min(dimension, 512),
                        eff_rank=20 + quality, RankMe=30 + quality,
                        m50=64 - quality, m90=110 - quality, VSA=0.01 * quality,
                        LAR_64=0.7 + 0.005 * quality,
                    )
                    for m in (8, 16, 32, 64, 128):
                        row[f"Lift_{m}"] = 0.8 + 0.01 * quality + 0.001 * domain_index
                        row[f"LAR_{m}"] = min(1.0, m / 128 + 0.001 * quality)
                    metric_rows.append(row)
            with metrics_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(metric_rows)
            target_fields = list(target_rows[0])
            with targets_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=target_fields)
                writer.writeheader()
                writer.writerows(target_rows)
            arguments = [
                "eval_e3", "--metrics", str(metrics_path), "--targets", str(targets_path),
                "--family-repeats", "20", "--regret-repeats", "30",
                "--combo-repeats", "20", "--bootstrap-repeats", "20",
                "--strict-pool", "--output", str(output_path),
                "--report", str(report_path), "--figure", str(figure_path),
            ]
            with patch.object(sys, "argv", arguments):
                e3_main()
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["coverage"]["metrics_by_domain"]["caption"], 18)
            self.assertIn(
                "probe_epoch1+Lift_64",
                payload["evaluations"]["answer"]["PC1"]["combinations"],
            )
            self.assertTrue(report_path.is_file())
            self.assertTrue(figure_path.is_file())

    def test_dual_domain_pool_computation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_root = root / "features"
            text_root = root / "text"
            spectrum_root = root / "spectra"
            feature_root.mkdir()
            text_root.mkdir()
            ids = [str(index) for index in range(40)]
            rng = np.random.default_rng(123)
            visual_path = feature_root / "synthetic__coco4618.npy"
            np.save(visual_path, rng.normal(size=(40, 12)).astype(np.float32))
            visual_path.with_suffix(".ids.txt").write_text("\n".join(ids) + "\n")
            visual_path.with_suffix(".meta.json").write_text(
                json.dumps({"n_tokens": 16}), encoding="utf-8"
            )
            for domain in ("caption", "answer"):
                text_path = text_root / f"{domain}__coco4618.npy"
                np.save(text_path, rng.normal(size=(40, 10)).astype(np.float32))
                text_path.with_suffix(".ids.txt").write_text("\n".join(ids) + "\n")
            config_path = root / "models.yaml"
            config_path.write_text(
                yaml.safe_dump({"models": [{"name": "synthetic", "enabled": True}]}),
                encoding="utf-8",
            )
            output_path = root / "metrics.csv"
            arguments = [
                "compute_lar_pool", "--models-config", str(config_path),
                "--feature-root", str(feature_root), "--text-root", str(text_root),
                "--spectrum-root", str(spectrum_root), "--output", str(output_path),
            ]
            with patch.object(sys, "argv", arguments):
                pool_main()
            with output_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["text_domain"] for row in rows}, {"caption", "answer"})
            self.assertNotIn("Waste", rows[0])
            self.assertTrue((spectrum_root / "synthetic__answer__coco4618.npz").is_file())


if __name__ == "__main__":
    unittest.main()
