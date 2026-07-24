import unittest
from pathlib import Path

from vtm_lcg.adapters import create_adapter
from vtm_lcg.config import load_config


class AdapterConfigTest(unittest.TestCase):
    def test_four_tokenizers_share_sequence_shape_with_native_inputs(self):
        project_root = Path(__file__).resolve().parents[1]
        config, _ = load_config(project_root / "configs" / "phase0_smoke.yaml")
        adapters = {
            item["id"]: create_adapter(item, config["preprocess"])
            for item in config["tokenizers"]
        }
        self.assertEqual(
            set(adapters),
            {
                "clip_openai__l14",
                "mc1_l14_224_2.5b",
                "siglip2_l16_256",
                "mc2_l14_224",
            },
        )
        for adapter in adapters.values():
            self.assertEqual(adapter.grid_shape, (16, 16))
            self.assertEqual(adapter.expected_hidden_dim, 1024)
            self.assertTrue(adapter.checkpoint_file.is_file())
        self.assertEqual(adapters["siglip2_l16_256"].input_size, (256, 256))
        self.assertEqual(
            adapters["siglip2_l16_256"].preprocess_config["resize_size"],
            284,
        )
        self.assertEqual(
            adapters["siglip2_l16_256"].preprocess_config["mean"],
            [0.5, 0.5, 0.5],
        )
        for tokenizer_id in (
            "clip_openai__l14",
            "mc1_l14_224_2.5b",
            "mc2_l14_224",
        ):
            self.assertEqual(adapters[tokenizer_id].input_size, (224, 224))


if __name__ == "__main__":
    unittest.main()
