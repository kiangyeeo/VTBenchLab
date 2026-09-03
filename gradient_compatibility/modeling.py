from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


class MLP3xGELU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.layers(tokens)


def load_llm(path: str, device: torch.device, gradient_checkpointing: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.requires_grad_(False).eval()
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model, tokenizer


def _text_ids(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    values = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor(values, dtype=torch.long, device=device)


def multimodal_loss(
    llm: nn.Module,
    tokenizer,
    projector: nn.Module,
    visual_tokens: list[torch.Tensor],
    prompts: list[str],
    answers: list[str],
    device: torch.device,
) -> torch.Tensor:
    if not (len(visual_tokens) == len(prompts) == len(answers)):
        raise ValueError("visual_tokens, prompts, and answers must have equal lengths")
    embedding = llm.get_input_embeddings()
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("The language tokenizer must define eos_token_id")
    sequences = []
    labels = []
    for tokens, prompt, answer in zip(visual_tokens, prompts, answers):
        before_ids = _text_ids(tokenizer, "User: ", device)
        after_ids = _text_ids(tokenizer, f"\n{prompt}\nAssistant:", device)
        answer_ids = _text_ids(tokenizer, f" {answer}", device)
        answer_ids = torch.cat(
            [answer_ids, torch.tensor([eos_id], dtype=torch.long, device=device)]
        )
        before = embedding(before_ids)
        after = embedding(after_ids)
        answer_embeddings = embedding(answer_ids)
        projected = projector(tokens.to(device=device, non_blocking=True).unsqueeze(0))[0]
        sequence = torch.cat([before, projected, after, answer_embeddings], dim=0)
        label = torch.full((sequence.shape[0],), -100, dtype=torch.long, device=device)
        label[-answer_ids.numel() :] = answer_ids
        sequences.append(sequence)
        labels.append(label)

    max_length = max(sequence.shape[0] for sequence in sequences)
    hidden_dim = sequences[0].shape[-1]
    batch_embeddings = torch.zeros(
        (len(sequences), max_length, hidden_dim),
        dtype=sequences[0].dtype,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(sequences), max_length), dtype=torch.long, device=device
    )
    batch_labels = torch.full(
        (len(sequences), max_length), -100, dtype=torch.long, device=device
    )
    for index, (sequence, label) in enumerate(zip(sequences, labels)):
        length = sequence.shape[0]
        batch_embeddings[index, :length] = sequence
        attention_mask[index, :length] = 1
        batch_labels[index, :length] = label
    outputs = llm(
        inputs_embeds=batch_embeddings,
        attention_mask=attention_mask,
        labels=batch_labels,
        use_cache=False,
        return_dict=True,
    )
    return outputs.loss


def text_only_loss(
    llm: nn.Module,
    tokenizer,
    prompts: list[str],
    answers: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Instruction loss without tokenizer-specific visual embeddings."""
    if len(prompts) != len(answers):
        raise ValueError("prompts and answers must have equal lengths")
    embedding = llm.get_input_embeddings()
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("The language tokenizer must define eos_token_id")
    sequences = []
    labels = []
    for prompt, answer in zip(prompts, answers):
        prompt_ids = _text_ids(tokenizer, f"User:\n{prompt}\nAssistant:", device)
        answer_ids = _text_ids(tokenizer, f" {answer}", device)
        answer_ids = torch.cat(
            [answer_ids, torch.tensor([eos_id], dtype=torch.long, device=device)]
        )
        sequence_ids = torch.cat([prompt_ids, answer_ids])
        label = torch.full_like(sequence_ids, -100)
        label[-answer_ids.numel() :] = answer_ids
        sequences.append(embedding(sequence_ids))
        labels.append(label)

    max_length = max(sequence.shape[0] for sequence in sequences)
    hidden_dim = sequences[0].shape[-1]
    batch_embeddings = torch.zeros(
        (len(sequences), max_length, hidden_dim),
        dtype=sequences[0].dtype,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(sequences), max_length), dtype=torch.long, device=device
    )
    batch_labels = torch.full(
        (len(sequences), max_length), -100, dtype=torch.long, device=device
    )
    for index, (sequence, label) in enumerate(zip(sequences, labels)):
        length = sequence.shape[0]
        batch_embeddings[index, :length] = sequence
        attention_mask[index, :length] = 1
        batch_labels[index, :length] = label
    return llm(
        inputs_embeds=batch_embeddings,
        attention_mask=attention_mask,
        labels=batch_labels,
        use_cache=False,
        return_dict=True,
    ).loss


class LoRABProbeLinear(nn.Module):
    """Frozen Linear plus fixed random A and zero, trainable B probe coordinates."""

    def __init__(self, base: nn.Linear, rank: int, generator: torch.Generator) -> None:
        super().__init__()
        self.base = base.requires_grad_(False)
        std = 1.0 / math.sqrt(base.in_features)
        a = torch.randn(
            rank,
            base.in_features,
            generator=generator,
            dtype=torch.float32,
        ) * std
        self.register_buffer("lora_a", a.to(device=base.weight.device, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base_output = self.base(values)
        low_rank = F.linear(F.linear(values, self.lora_a), self.lora_b)
        return base_output + low_rank


@dataclass
class ProbeParameter:
    name: str
    parameter: nn.Parameter


def install_lora_b_probes(
    llm: nn.Module,
    last_n_layers: int,
    target_modules: Iterable[str],
    rank: int,
    seed: int,
) -> list[ProbeParameter]:
    layers = llm.model.layers
    if not 0 < last_n_layers <= len(layers):
        raise ValueError(f"Invalid last_n_layers={last_n_layers} for {len(layers)} layers")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    probes = []
    start = len(layers) - last_n_layers
    for layer_index in range(start, len(layers)):
        attention = layers[layer_index].self_attn
        for module_name in target_modules:
            base = getattr(attention, module_name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"Expected Linear at layer {layer_index}.{module_name}")
            wrapped = LoRABProbeLinear(base, rank=rank, generator=generator)
            setattr(attention, module_name, wrapped)
            probes.append(
                ProbeParameter(
                    name=f"model.layers.{layer_index}.self_attn.{module_name}.lora_b",
                    parameter=wrapped.lora_b,
                )
            )
    return probes


def clear_probe_gradients(probes: list[ProbeParameter]) -> None:
    for probe in probes:
        probe.parameter.grad = None


def flatten_probe_gradients(probes: list[ProbeParameter]) -> torch.Tensor:
    gradients = []
    for probe in probes:
        if probe.parameter.grad is None:
            raise RuntimeError(f"Missing gradient for {probe.name}")
        gradients.append(probe.parameter.grad.detach().float().reshape(-1).cpu())
    return torch.cat(gradients, dim=0)
