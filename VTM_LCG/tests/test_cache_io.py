import tempfile
import unittest
from pathlib import Path

import torch

from vtm_lcg.cache.io import validate_shard, write_shard_atomic
from vtm_lcg.utils import sha256_json


class CacheIoTest(unittest.TestCase):
    def test_round_trip_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "00000.safetensors"
            values = torch.arange(24, dtype=torch.float16).reshape(2, 4, 3)
            image_ids = torch.tensor([9, 4], dtype=torch.int64)
            descriptor = write_shard_atomic(
                path,
                shard_index=0,
                values=values,
                image_ids=image_ids,
            )
            loaded_values, loaded_ids = validate_shard(path, descriptor)
            torch.testing.assert_close(loaded_values, values)
            torch.testing.assert_close(loaded_ids, image_ids)

            payload = bytearray(path.read_bytes())
            payload[-1] ^= 0x01
            path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_shard(path, descriptor)

    def test_cache_identity_is_stable_and_sensitive(self):
        identity = {
            "checkpoint_sha256": "abc",
            "surface": "final normalized patches",
            "preprocess": {"size": 224},
        }
        first = sha256_json(identity)
        second = sha256_json(
            {
                "preprocess": {"size": 224},
                "surface": "final normalized patches",
                "checkpoint_sha256": "abc",
            }
        )
        changed = sha256_json({**identity, "surface": "penultimate patches"})
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()

