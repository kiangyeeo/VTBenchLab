import gc
import os
import unittest
from pathlib import Path

import torch

from vtm_lcg.adapters import create_adapter
from vtm_lcg.cache.dataset import CocoImageTensorDataset, load_coco_karpathy_records
from vtm_lcg.config import load_config, torch_dtype_from_name


@unittest.skipUnless(
    os.environ.get("VTM_LCG_RUN_INTEGRATION") == "1",
    "set VTM_LCG_RUN_INTEGRATION=1 to load the real local checkpoints",
)
class AdapterIntegrationTest(unittest.TestCase):
    def test_real_checkpoints_return_matching_patch_sequences(self):
        project_root = Path(__file__).resolve().parents[1]
        config, _ = load_config(project_root / "configs" / "phase0_smoke.yaml")
        records, _ = load_coco_karpathy_records(
            Path(config["dataset"]["annotations"]),
            Path(config["dataset"]["image_root"]),
            limit=1,
        )
        device_name = os.environ.get("VTM_LCG_INTEGRATION_DEVICE", "cpu")
        device = torch.device(device_name)
        dtype_name = (
            config["runtime"]["backbone_dtype"]
            if device.type == "cuda"
            else "float32"
        )
        dtype = torch_dtype_from_name(dtype_name)

        for tokenizer_config in config["tokenizers"]:
            with self.subTest(tokenizer=tokenizer_config["id"]):
                adapter = create_adapter(tokenizer_config, config["preprocess"])
                dataset = CocoImageTensorDataset(records, [0], adapter.preprocess)
                image, image_id, record_index = dataset[0]
                self.assertEqual(image_id, records[0].image_id)
                self.assertEqual(record_index, 0)
                adapter.load(device, dtype)
                with torch.inference_mode():
                    batch = adapter.encode(image.unsqueeze(0))
                self.assertEqual(tuple(batch.values.shape), (1, 256, 1024))
                self.assertTrue(bool(torch.isfinite(batch.values).all()))
                self.assertFalse(bool(batch.special_mask.any()))
                del adapter, batch
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()

