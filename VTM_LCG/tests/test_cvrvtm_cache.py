import tempfile
import unittest
from pathlib import Path

import torch

from vtm_lcg.cvrvtm.cache import (
    validate_cross_view_shard,
    write_cross_view_shard,
)


class CrossViewCacheTest(unittest.TestCase):
    def test_cross_view_shard_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "00000.safetensors"
            view_a = torch.randn(2, 16, 8).to(torch.float16)
            view_b = torch.randn(2, 16, 8).to(torch.float16)
            image_ids = torch.tensor([11, 12], dtype=torch.int64)
            descriptor = write_cross_view_shard(
                path,
                shard_index=0,
                view_a=view_a,
                view_b=view_b,
                image_ids=image_ids,
            )
            loaded_a, loaded_b, loaded_ids = validate_cross_view_shard(
                path,
                descriptor,
                verify_checksum=True,
            )
            self.assertTrue(torch.equal(view_a, loaded_a))
            self.assertTrue(torch.equal(view_b, loaded_b))
            self.assertTrue(torch.equal(image_ids, loaded_ids))


if __name__ == "__main__":
    unittest.main()
