import json
import tempfile
import unittest
from pathlib import Path

import torch

from vtm_lcg.cvrvtm.cache import write_cross_view_shard
from vtm_lcg.cvrvtm.train import (
    _load_caches_and_indices,
    train_run,
)


class CrossViewTrainingIntegrationTest(unittest.TestCase):
    def test_tiny_streaming_training_and_sanity_evaluation(self):
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_dir = root / "cache" / "dummy" / "key"
            shard_dir = cache_dir / "shards"
            records = [
                {
                    "image_id": 100 + index,
                    "file_name": f"{index}.jpg",
                    "image_path": f"/unused/{index}.jpg",
                    "caption_ids": [index],
                    "captions": ["unused"],
                }
                for index in range(8)
            ]
            descriptors = []
            generator = torch.Generator().manual_seed(7)
            positions = torch.randn(1, 16, 8, generator=generator)
            all_values = []
            for shard_index, start in enumerate((0, 4)):
                global_features = torch.randn(4, 1, 8, generator=generator)
                view_a = global_features + positions
                view_b = 1.01 * global_features + positions
                view_b = view_b + 0.01 * torch.randn(
                    view_b.shape,
                    generator=generator,
                )
                all_values.extend((view_a, view_b))
                descriptors.append(
                    write_cross_view_shard(
                        shard_dir / f"{shard_index:05d}.safetensors",
                        shard_index=shard_index,
                        view_a=view_a.to(torch.float16),
                        view_b=view_b.to(torch.float16),
                        image_ids=torch.tensor(
                            [100 + index for index in range(start, start + 4)],
                            dtype=torch.int64,
                        ),
                    )
                )
            combined = torch.cat(all_values, dim=0).float()
            mean = combined.mean(dim=(0, 1))
            std = combined.std(dim=(0, 1), unbiased=False).clamp_min(1.0e-3)
            manifest = {
                "schema_version": 1,
                "protocol": "cross_view_aligned_cache_v1",
                "cache_key": "key",
                "complete": True,
                "record_count": 8,
                "shard_size": 4,
                "expected_shards": 2,
                "token_count": 16,
                "hidden_dim": 8,
                "feature_dtype": "float16",
                "shards": [item.to_dict() for item in descriptors],
            }
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "manifest.json").write_text(json.dumps(manifest))
            (cache_dir / "records.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": {"dataset_fingerprint": "dummy"},
                        "records": records,
                    }
                )
            )
            (cache_dir / "stats.json").write_text(
                json.dumps(
                    {
                        "cache_key": "key",
                        "channel_mean": mean.tolist(),
                        "channel_std": std.tolist(),
                    }
                )
            )
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol": "cross_view_aligned_cache_v1",
                        "all_acceptance_checks_passed": True,
                        "tokenizers": [
                            {
                                "tokenizer_id": "dummy",
                                "cache_key": "key",
                                "cache_dir": str(cache_dir),
                            }
                        ],
                    }
                )
            )
            config = {
                "phase0_summary": str(summary_path),
                "artifact_root": str(root / "artifacts"),
                "split": {
                    "seed": 0,
                    "train": 4,
                    "validation": 2,
                    "test": 2,
                },
                "predictor": {
                    "visual_input_dim": 8,
                    "model_dim": 16,
                    "depth": 1,
                    "num_heads": 4,
                    "mlp_ratio": 2,
                    "dropout": 0.0,
                    "grid_shape": [4, 4],
                    "block_shape": [2, 2],
                    "mask_ratio": 0.75,
                    "symmetric": True,
                },
                "training": {
                    "epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1.0e-3,
                    "weight_decay": 0.0,
                    "warmup_ratio": 0.0,
                    "precision": "float32",
                    "stats_epsilon": 1.0e-6,
                    "seeds": [0],
                },
                "evaluation": {
                    "batch_size": 4,
                    "mask_seed": 41,
                    "sanity_seed": 42,
                    "noise_std": 0.5,
                },
            }
            caches, indices = _load_caches_and_indices(config, "dummy")
            result = train_run(
                tokenizer_id="dummy",
                caches=caches,
                indices=indices,
                config=config,
                project_root=project_root,
                device=torch.device("cpu"),
                seed=0,
            )
            self.assertTrue(torch.isfinite(torch.tensor(result["test"]["scores"]["CVRVTM"])))
            self.assertEqual(
                set(result["test"]["sanity_pass"]),
                {
                    "collapsed_below_main",
                    "noise_below_main",
                    "spatial_shuffle_below_main",
                },
            )


if __name__ == "__main__":
    unittest.main()
