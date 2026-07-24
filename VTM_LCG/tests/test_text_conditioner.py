import unittest
from types import SimpleNamespace

import torch
from torch import nn

from vtm_lcg.models.text_conditioner import FrozenClipTextConditioner


class _FakeTokenizer:
    def __call__(self, captions, **_kwargs):
        batch_size = len(captions)
        return {
            "input_ids": torch.ones(batch_size, 3, dtype=torch.int64),
            "attention_mask": torch.ones(batch_size, 3, dtype=torch.int64),
        }


class _FakeTextModel(nn.Module):
    def forward(self, input_ids, attention_mask, return_dict):
        del attention_mask, return_dict
        embeddings = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 6)
        return SimpleNamespace(last_hidden_state=embeddings)


class FrozenClipTextConditionerTest(unittest.TestCase):
    def test_frozen_embeddings_support_downstream_backward(self):
        conditioner = FrozenClipTextConditioner.__new__(FrozenClipTextConditioner)
        conditioner.max_length = 3
        conditioner.device = torch.device("cpu")
        conditioner.tokenizer = _FakeTokenizer()
        conditioner.model = _FakeTextModel()

        embeddings, attention_mask = conditioner.encode(["one", "two"])

        self.assertFalse(embeddings.requires_grad)
        self.assertFalse(embeddings.is_inference())
        self.assertEqual(attention_mask.dtype, torch.bool)

        projection = nn.Linear(6, 4)
        projection(embeddings).square().mean().backward()
        self.assertIsNotNone(projection.weight.grad)
        self.assertTrue(bool(torch.isfinite(projection.weight.grad).all()))


if __name__ == "__main__":
    unittest.main()
