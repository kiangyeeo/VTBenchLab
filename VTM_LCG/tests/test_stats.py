import tempfile
import unittest
from pathlib import Path

import torch

from vtm_lcg.cache.io import write_shard_atomic
from vtm_lcg.cache.stats import compute_cache_stats, validate_stats_acceptance


class CacheStatsTest(unittest.TestCase):
    def test_streaming_stats_and_normalized_readback(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            shards_dir = cache_dir / "shards"
            first = torch.tensor(
                [
                    [[0.0, 2.0], [2.0, 4.0]],
                    [[4.0, 6.0], [6.0, 8.0]],
                ],
                dtype=torch.float16,
            )
            second = torch.tensor(
                [
                    [[8.0, 10.0], [10.0, 12.0]],
                    [[12.0, 14.0], [14.0, 16.0]],
                ],
                dtype=torch.float16,
            )
            descriptors = [
                write_shard_atomic(
                    shards_dir / "00000.safetensors",
                    shard_index=0,
                    values=first,
                    image_ids=torch.tensor([1, 2], dtype=torch.int64),
                ),
                write_shard_atomic(
                    shards_dir / "00001.safetensors",
                    shard_index=1,
                    values=second,
                    image_ids=torch.tensor([3, 4], dtype=torch.int64),
                ),
            ]
            stats = compute_cache_stats(cache_dir, descriptors, epsilon=1.0e-6)
            self.assertEqual(stats["image_count"], 4)
            self.assertEqual(stats["token_observation_count"], 8)
            self.assertEqual(stats["nan_count"], 0)
            self.assertEqual(stats["inf_count"], 0)
            self.assertGreater(stats["mean_token_variance"], 0.0)
            self.assertLess(stats["normalized_max_abs_channel_mean"], 1.0e-10)
            self.assertLess(stats["normalized_max_abs_channel_std_error"], 5.0e-3)
            validate_stats_acceptance(stats)


if __name__ == "__main__":
    unittest.main()

