import unittest

import torch

from vtm_lcg.models.masked_predictor import (
    MaskedVisualPredictor,
    build_2d_sincos_position_embedding,
)


class MaskedPredictorTest(unittest.TestCase):
    def test_position_embedding_shape(self):
        embedding = build_2d_sincos_position_embedding((2, 3), 16)
        self.assertEqual(tuple(embedding.shape), (6, 16))
        self.assertTrue(bool(torch.isfinite(embedding).all()))

    def test_unconditional_and_conditional_forward(self):
        model = MaskedVisualPredictor(
            visual_input_dim=8,
            text_input_dim=6,
            model_dim=16,
            depth=2,
            num_heads=4,
            mlp_ratio=2,
            dropout=0.0,
            grid_shape=(2, 2),
        )
        visual = torch.randn(3, 4, 8)
        masked = torch.tensor(
            [
                [True, False, True, False],
                [False, True, False, True],
                [True, True, False, False],
            ]
        )
        unconditional = model(visual, masked)
        conditional = model(
            visual,
            masked,
            text_embeddings=torch.randn(3, 5, 6),
            text_attention_mask=torch.tensor(
                [
                    [True, True, True, False, False],
                    [True, True, False, False, False],
                    [True, True, True, True, False],
                ]
            ),
        )
        no_visible = model(visual, masked, hide_all_visual=True)
        self.assertEqual(tuple(unconditional.shape), (3, 4, 8))
        self.assertEqual(tuple(conditional.shape), (3, 4, 8))
        self.assertEqual(tuple(no_visible.shape), (3, 4, 8))
        self.assertTrue(bool(torch.isfinite(conditional).all()))

    def test_phase1_parameter_budget(self):
        model = MaskedVisualPredictor(
            visual_input_dim=1024,
            text_input_dim=768,
            model_dim=256,
            depth=4,
            num_heads=8,
            mlp_ratio=4,
            dropout=0.0,
            grid_shape=(16, 16),
        )
        self.assertGreater(model.parameter_count(), 3_500_000)
        self.assertLess(model.parameter_count(), 10_000_000)


if __name__ == "__main__":
    unittest.main()
