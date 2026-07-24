import unittest

import torch

from vtm_lcg.cvrvtm import (
    CrossViewResidualPredictor,
    compute_cvrvtm_scores,
    make_deterministic_block_mask,
    residualize_cross_view,
)
from vtm_lcg.cvrvtm.views import align_flipped_patch_tokens


class CrossViewResidualProtocolTest(unittest.TestCase):
    def test_block_mask_is_exact_and_deterministic(self):
        first = make_deterministic_block_mask(
            [3, 7],
            grid_shape=(16, 16),
            block_shape=(4, 4),
            mask_ratio=0.75,
            seed=41,
            epoch=2,
            device="cpu",
        )
        second = make_deterministic_block_mask(
            [3, 7],
            grid_shape=(16, 16),
            block_shape=(4, 4),
            mask_ratio=0.75,
            seed=41,
            epoch=2,
            device="cpu",
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.sum(dim=1).tolist(), [192, 192])
        grid = first[0].reshape(4, 4, 4, 4).permute(0, 2, 1, 3)
        block_sums = grid.reshape(4, 4, 16).sum(dim=-1)
        self.assertTrue(bool(((block_sums == 0) | (block_sums == 16)).all()))

    def test_residual_center_uses_visible_source_only(self):
        source = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        target = source + 100.0
        mask = torch.tensor(
            [[False, True, False, True], [True, False, True, False]]
        )
        source_residual, target_residual, center = residualize_cross_view(
            source,
            target,
            mask,
        )
        visible = (~mask).unsqueeze(-1)
        visible_means = (
            (source_residual * visible).sum(dim=1)
            / visible.sum(dim=1)
        )
        self.assertTrue(torch.allclose(visible_means, torch.zeros_like(visible_means)))
        self.assertTrue(torch.allclose(target_residual, target - center))

    def test_score_penalizes_collapse_and_noise(self):
        useful = compute_cvrvtm_scores(
            {
                "L_total": 1.0,
                "L_residual_null": 0.6,
                "L_residual_prediction": 0.2,
            }
        )
        collapsed = compute_cvrvtm_scores(
            {
                "L_total": 1.0,
                "L_residual_null": 0.0,
                "L_residual_prediction": 0.0,
            }
        )
        noise = compute_cvrvtm_scores(
            {
                "L_total": 1.0,
                "L_residual_null": 0.6,
                "L_residual_prediction": 0.6,
            }
        )
        self.assertAlmostEqual(useful["CVRVTM"], 0.4)
        self.assertEqual(collapsed["CVRVTM"], 0.0)
        self.assertEqual(noise["CVRVTM"], 0.0)

    def test_model_returns_patch_residual_predictions(self):
        model = CrossViewResidualPredictor(
            visual_input_dim=16,
            model_dim=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2,
            dropout=0.0,
            grid_shape=(4, 4),
        )
        values = torch.randn(2, 16, 16)
        mask = torch.zeros(2, 16, dtype=torch.bool)
        mask[:, ::2] = True
        prediction = model(values, mask)
        self.assertEqual(tuple(prediction.shape), (2, 16, 16))
        prediction[mask].square().mean().backward()
        self.assertIsNotNone(model.input_projection.weight.grad)

    def test_horizontal_flip_alignment_reverses_columns(self):
        values = torch.arange(6).reshape(1, 6, 1)
        aligned = align_flipped_patch_tokens(
            values,
            grid_shape=(2, 3),
            horizontal_flip=True,
        )
        expected = torch.tensor([[[2], [1], [0], [5], [4], [3]]])
        self.assertTrue(torch.equal(aligned, expected))


if __name__ == "__main__":
    unittest.main()
