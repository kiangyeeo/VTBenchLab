from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn

from gradient_compatibility.data import _majority_answer, _sample_indices
from gradient_compatibility.modeling import LoRABProbeLinear, MLP3xGELU, text_only_loss
from gradient_compatibility.summarize import _alignment, _bootstrap_ci, _spearman
from gradient_compatibility.summarize_loss_proxy import _average_ranks_lower_is_better
from gradient_compatibility.token_cache import FinalNormPatchEncoder, TokenCache
from gradient_compatibility.utils import load_config
from lar.model_adapters import SpatialTokenEncoder


class _FakePatchModel(nn.Module):
    def __init__(self, return_tuple: bool) -> None:
        super().__init__()
        self.return_tuple = return_tuple

    def get_intermediate_layers(self, images, **_kwargs):
        patches = torch.ones(images.shape[0], 4, 3)
        if self.return_tuple:
            return [(patches, torch.ones(images.shape[0], 1, 3))]
        return [patches]


class _TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 6)
        self.head = nn.Linear(6, 32)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, labels, **_kwargs):
        logits = self.head(inputs_embeds)
        valid = labels.ne(-100)
        loss = nn.functional.cross_entropy(logits[valid], labels[valid])
        return type("Output", (), {"loss": loss})()


class _TinyTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [2 + (ord(char) % 30) for char in text]


class ProtocolTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_unique(self):
        first = _sample_indices(100, 20, 7)
        second = _sample_indices(100, 20, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_majority_answer_preserves_original_spelling(self):
        self.assertEqual(_majority_answer(["Cat", "cat", "dog"]), "Cat")

    def test_projector_shape(self):
        projector = MLP3xGELU(5, 7)
        self.assertEqual(tuple(projector(torch.randn(2, 3, 5)).shape), (2, 3, 7))

    def test_fresh_lora_b_preserves_output_and_has_gradient(self):
        base = nn.Linear(5, 4)
        generator = torch.Generator().manual_seed(9)
        probe = LoRABProbeLinear(base, rank=2, generator=generator)
        values = torch.randn(3, 5)
        expected = base(values).detach()
        actual = probe(values)
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        self.assertIsNotNone(probe.lora_b.grad)
        self.assertGreater(float(probe.lora_b.grad.norm()), 0.0)

    def test_alignment_distinguishes_same_and_opposite_direction(self):
        target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        source = torch.tensor([[2.0, 0.0], [-3.0, 0.0]])
        values = _alignment(source, target)
        np.testing.assert_allclose(values, [1.0, -1.0])
        low, high = _bootstrap_ci(values, seed=1, samples=100)
        self.assertLessEqual(low, 0.0)
        self.assertGreaterEqual(high, 0.0)

    def test_spearman_handles_order_and_ties(self):
        self.assertAlmostEqual(_spearman([3, 2, 1], [30, 20, 10]), 1.0)
        self.assertAlmostEqual(_spearman([1, 2, 3], [30, 20, 10]), -1.0)
        self.assertTrue(np.isfinite(_spearman([1, 1, 2], [1, 2, 3])))

    def test_loss_ranks_are_one_based_and_average_ties(self):
        ranks = _average_ranks_lower_is_better({"a": 1.0, "b": 2.0, "c": 2.0})
        self.assertEqual(ranks, {"a": 1.0, "b": 2.5, "c": 2.5})

    def test_token_cache_reads_indexed_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "shards").mkdir()
            save_file({"tokens": torch.arange(12).reshape(2, 2, 3)}, root / "shards" / "a.safetensors")
            (root / "index.json").write_text(
                '{"x":{"shard":"a.safetensors","offset":1}}', encoding="utf-8"
            )
            (root / "cache.json").write_text('{"fingerprint":"f"}', encoding="utf-8")
            cache = TokenCache(root)
            torch.testing.assert_close(cache.get("x"), torch.arange(6, 12).reshape(2, 3))

    def test_patch_encoder_accepts_prefix_and_prefix_free_timm_outputs(self):
        images = torch.randn(2, 3, 8, 8)
        for return_tuple in (False, True):
            encoder = FinalNormPatchEncoder(
                _FakePatchModel(return_tuple), expected_tokens=4, expected_dim=3
            )
            self.assertEqual(tuple(encoder(images).shape), (2, 4, 3))

    def test_spatial_adapter_validates_shape_and_finiteness(self):
        valid = torch.ones(2, 4, 3)
        self.assertIs(SpatialTokenEncoder._finish(valid), valid)
        with self.assertRaises(RuntimeError):
            SpatialTokenEncoder._finish(torch.ones(2, 3))
        invalid = valid.clone()
        invalid[0, 0, 0] = float("nan")
        with self.assertRaises(RuntimeError):
            SpatialTokenEncoder._finish(invalid)

    def test_full_sweep_registry_is_frozen_to_requested_79(self):
        config_path = Path(__file__).parents[1] / "configs" / "full_sweep.json"
        config, _ = load_config(config_path)
        self.assertEqual(len(config["tokenizers"]), 79)
        self.assertEqual(config["tokenizers"]["RAE v2"]["loader_name"], "raev2")
        self.assertEqual(config["tokenizers"]["I-JEPA"]["loader_name"], "ijepa")

    def test_text_only_loss_backpropagates(self):
        model = _TinyCausalLM()
        loss = text_only_loss(
            model, _TinyTokenizer(), ["question"], ["answer"], torch.device("cpu")
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.head.weight.grad)


if __name__ == "__main__":
    unittest.main()
