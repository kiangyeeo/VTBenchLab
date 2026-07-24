import unittest

import torch

from vtm_lcg.schemas import TokenBatch, row_major_coords


class TokenBatchTest(unittest.TestCase):
    def make_batch(self) -> TokenBatch:
        values = torch.randn(2, 4, 3)
        return TokenBatch(
            values=values,
            mask=torch.ones(2, 4, dtype=torch.bool),
            coords=row_major_coords((2, 2)).unsqueeze(0).expand(2, -1, -1),
            special_mask=torch.zeros(2, 4, dtype=torch.bool),
            grid_shape=(2, 2),
            surface="unit-test surface",
            input_size=(32, 32),
            metadata={"tokenizer_id": "dummy"},
        )

    def test_valid_dense_batch(self):
        batch = self.make_batch()
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.token_count, 4)
        self.assertEqual(batch.hidden_dim, 3)
        self.assertEqual(batch.coords[0].tolist(), [[0, 0], [0, 1], [1, 0], [1, 1]])
        with self.assertRaises(TypeError):
            batch.metadata["new"] = "forbidden"

    def test_rejects_non_row_major_coords(self):
        with self.assertRaisesRegex(ValueError, "row-major"):
            TokenBatch(
                values=torch.randn(1, 4, 3),
                mask=torch.ones(1, 4, dtype=torch.bool),
                coords=torch.tensor([[[0, 1], [0, 0], [1, 0], [1, 1]]]),
                special_mask=torch.zeros(1, 4, dtype=torch.bool),
                grid_shape=(2, 2),
                surface="bad coords",
                input_size=(32, 32),
            )

    def test_rejects_special_tokens(self):
        special_mask = torch.zeros(1, 4, dtype=torch.bool)
        special_mask[0, 0] = True
        with self.assertRaisesRegex(ValueError, "remove all special"):
            TokenBatch(
                values=torch.randn(1, 4, 3),
                mask=torch.ones(1, 4, dtype=torch.bool),
                coords=row_major_coords((2, 2)).unsqueeze(0),
                special_mask=special_mask,
                grid_shape=(2, 2),
                surface="contains cls",
                input_size=(32, 32),
            )


if __name__ == "__main__":
    unittest.main()

