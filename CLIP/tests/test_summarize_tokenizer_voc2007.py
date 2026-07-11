import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "summarize_tokenizer_voc2007.py"
SPEC = importlib.util.spec_from_file_location("summarize_tokenizer_voc2007", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _result(model: str, test_map: float, *, converged: bool = True) -> dict:
    validation_ap = {class_name: 70.0 for class_name in MODULE.VOC_CLASSES}
    test_ap = {class_name: test_map for class_name in MODULE.VOC_CLASSES}
    return {
        "protocol": MODULE.PROTOCOL,
        "model": model,
        "feature_dim": 768,
        "selection": {
            "selected_lambda": 0.1,
            "selected_C": 10.0,
            "validation_mAP_11point": 70.0,
            "validation_AP_11point": validation_ap,
            "nonconverged_classes": [] if converged else ["bird"],
        },
        "final_evaluation": {
            "mAP_11point": test_map,
            "AP_11point": test_ap,
            "converged": converged,
        },
    }


class SummarizeTokenizerVOC2007Test(unittest.TestCase):
    def test_collect_rows_preserves_requested_order_and_marks_missing_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            model_root = output_root / "unitok"
            model_root.mkdir()
            (model_root / "results.json").write_text(json.dumps(_result("unitok", 81.25)))

            summary, per_class = MODULE.collect_rows(output_root, ["unitok", "vilau"])

            self.assertEqual([row["model"] for row in summary], ["unitok", "vilau"])
            self.assertEqual(summary[0]["status"], "complete")
            self.assertEqual(summary[0]["test_mAP_11point"], 81.25)
            self.assertEqual(summary[1]["status"], "missing")
            self.assertEqual(len(per_class), 20)

    def test_nonconvergence_is_visible_in_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            model_root = output_root / "toklipl"
            model_root.mkdir()
            (model_root / "results.json").write_text(
                json.dumps(_result("toklipl", 88.5, converged=False))
            )

            summary, _ = MODULE.collect_rows(output_root, ["toklipl"])

            self.assertEqual(summary[0]["status"], "nonconverged")
            self.assertFalse(summary[0]["selection_converged"])
            self.assertFalse(summary[0]["final_converged"])

    def test_main_writes_csv_and_markdown_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            model_root = output_root / "metaclip"
            model_root.mkdir()
            (model_root / "results.json").write_text(json.dumps(_result("metaclip", 79.0)))

            with contextlib.redirect_stdout(io.StringIO()):
                MODULE.main(
                    [
                        "--output-root",
                        str(output_root),
                        "--models",
                        "metaclip",
                        "unitok",
                    ]
                )

            for filename in (
                "summary.csv",
                "summary.md",
                "per_class_ap.csv",
                "per_class_ap.md",
            ):
                self.assertTrue((output_root / filename).is_file())
            markdown = (output_root / "summary.md").read_text()
            self.assertIn("| metaclip | 768 | 0.1 | 70.00 | 79.00 | ok |", markdown)
            self.assertIn("| unitok | — | — | — | — | missing |", markdown)

    def test_mixed_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            model_root = output_root / "unitok"
            model_root.mkdir()
            payload = _result("unitok", 80.0)
            payload["protocol"] = "different"
            (model_root / "results.json").write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "Unexpected protocol"):
                MODULE.collect_rows(output_root, ["unitok"])


if __name__ == "__main__":
    unittest.main()
