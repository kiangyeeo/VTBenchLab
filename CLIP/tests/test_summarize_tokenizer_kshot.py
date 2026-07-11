import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "summarize_tokenizer_kshot.py"
SPEC = importlib.util.spec_from_file_location("summarize_tokenizer_kshot", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SummarizeTokenizerKshotTest(unittest.TestCase):
    def test_loads_new_and_legacy_c_fields_and_aggregates_population_std(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            seed0 = output_root / "unitok/seed0"
            seed1 = output_root / "unitok/seed1"
            seed0.mkdir(parents=True)
            seed1.mkdir(parents=True)
            (seed0 / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "shot": 1,
                                "top1": 10.0,
                                "top5": 20.0,
                                "selected_C": 0.1,
                                "selection_top1": 11.0,
                                "converged": True,
                            }
                        ]
                    }
                )
            )
            (seed1 / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "shot": 1,
                                "top1": 14.0,
                                "top5": 24.0,
                                "C": 1.0,
                                "converged": True,
                            }
                        ]
                    }
                )
            )

            rows = MODULE._load_rows(output_root, [0, 1])
            aggregates = MODULE._aggregate_rows(rows)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["selected_C"], 0.1)
            self.assertEqual(rows[1]["selected_C"], 1.0)
            self.assertEqual(aggregates[0]["top1_mean"], 12.0)
            self.assertEqual(aggregates[0]["top1_std"], 2.0)
            self.assertEqual(aggregates[0]["n_seeds"], 2)

    def test_main_writes_per_seed_and_aggregate_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()):
                MODULE.main(["--output-root", str(output_root), "--seeds", "0", "1", "2"])

            for seed in (0, 1, 2):
                self.assertTrue((output_root / f"summary_seed{seed}.csv").is_file())
                self.assertTrue((output_root / f"summary_seed{seed}.md").is_file())
            self.assertTrue((output_root / "summary_mean_std.csv").is_file())
            self.assertTrue((output_root / "summary_mean_std.md").is_file())


if __name__ == "__main__":
    unittest.main()
