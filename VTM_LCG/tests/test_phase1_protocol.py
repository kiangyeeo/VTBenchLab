import unittest

import torch

from vtm_lcg.eval.compute_scores import compute_vtm_lcg_scores
from vtm_lcg.eval.sanity_checks import (
    caption_keep_mask,
    make_deterministic_mask,
    spatially_shuffle_visible_tokens,
)
from vtm_lcg.train.data import fit_channel_standardization, make_split_indices


class Phase1ProtocolTest(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        first = make_split_indices(
            10,
            train_count=6,
            validation_count=2,
            test_count=2,
            seed=7,
        )
        second = make_split_indices(
            10,
            train_count=6,
            validation_count=2,
            test_count=2,
            seed=7,
        )
        self.assertEqual(first, second)
        all_indices = first["train"] + first["validation"] + first["test"]
        self.assertEqual(sorted(all_indices), list(range(10)))
        self.assertEqual(len(set(all_indices)), 10)

    def test_train_only_standardization(self):
        values = torch.tensor(
            [
                [[0.0, 2.0], [2.0, 4.0]],
                [[4.0, 6.0], [6.0, 8.0]],
                [[100.0, 200.0], [100.0, 200.0]],
            ],
            dtype=torch.float16,
        )
        mean, std = fit_channel_standardization(
            values,
            [0, 1],
            epsilon=1.0e-6,
        )
        torch.testing.assert_close(mean, torch.tensor([3.0, 5.0]))
        self.assertTrue(bool((std > 0).all()))

    def test_masks_and_caption_dropout_are_deterministic(self):
        first = make_deterministic_mask(
            [3, 8],
            token_count=8,
            mask_ratio=0.5,
            seed=11,
            epoch=2,
            device="cpu",
        )
        second = make_deterministic_mask(
            [3, 8],
            token_count=8,
            mask_ratio=0.5,
            seed=11,
            epoch=2,
            device="cpu",
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.sum(dim=1).tolist(), [4, 4])
        keep_first = caption_keep_mask(
            [3, 8],
            dropout=0.5,
            seed=19,
            epoch=2,
            device="cpu",
        )
        keep_second = caption_keep_mask(
            [3, 8],
            dropout=0.5,
            seed=19,
            epoch=2,
            device="cpu",
        )
        self.assertTrue(torch.equal(keep_first, keep_second))

    def test_spatial_shuffle_preserves_visible_multiset(self):
        visual = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
        masked = torch.tensor([[False, True, False, True]])
        shuffled = spatially_shuffle_visible_tokens(
            visual,
            masked,
            [5],
            seed=3,
        )
        visible = ~masked[0]
        original_rows = sorted(map(tuple, visual[0, visible].tolist()))
        shuffled_rows = sorted(map(tuple, shuffled[0, visible].tolist()))
        self.assertEqual(original_rows, shuffled_rows)
        torch.testing.assert_close(shuffled[0, masked[0]], visual[0, masked[0]])

    def test_score_formulas(self):
        scores = compute_vtm_lcg_scores(
            {
                "L_mean": 1.0,
                "L_visual": 0.8,
                "L_visual_text": 0.72,
                "L_visual_shuffled_text": 0.79,
                "L_visual_spatial_shuffle": 0.9,
                "L_no_visible": 1.01,
            }
        )
        self.assertAlmostEqual(scores["VTM"], 0.2)
        self.assertAlmostEqual(scores["LCG"], 0.1)
        self.assertAlmostEqual(scores["LCG_specific"], 0.0875)
        self.assertTrue(scores["caption_specificity_pass"])
        self.assertTrue(scores["spatial_structure_pass"])


if __name__ == "__main__":
    unittest.main()

