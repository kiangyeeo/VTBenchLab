from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn

from .data import Example, load_examples
from .modeling import MLP3xGELU, load_llm, multimodal_loss
from .token_cache import TokenCache
from .utils import atomic_write_json, choose_names, device_from_config, load_config, resolve_path, stable_seed


class ZeroVisualProjector(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (*tokens.shape[:-1], self.hidden_dim),
            dtype=tokens.dtype,
            device=tokens.device,
        )


def _deranged_donors(examples: list[Example], seed: int) -> list[Example]:
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    donor_by_index = {
        index: order[(position + 1) % len(order)] for position, index in enumerate(order)
    }
    return [examples[donor_by_index[index]] for index in range(len(examples))]


def evaluate_tokenizer_rows(
    config: dict,
    llm,
    language_tokenizer,
    name: str,
    seed: int,
    max_examples: int | None,
    batch_size: int,
    selected_domains: list[str],
    selected_seen: int,
) -> list[dict]:
    device = device_from_config(config)
    root = resolve_path(config, config["runtime"]["artifact_root"])
    cache = TokenCache(root / "tokens" / name)
    input_dim = int(cache.metadata["shape"][-1])
    checkpoint_path = (
        root / "projectors" / name / f"seed_{seed}" / f"projector_seen_{selected_seen}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    projector = MLP3xGELU(input_dim, int(config["llm"]["hidden_dim"])).to(device)
    projector.load_state_dict(checkpoint["projector"])
    projector.eval().requires_grad_(False)
    zero_projector = ZeroVisualProjector(int(config["llm"]["hidden_dim"])).to(device)
    examples = load_examples(config)
    results = []
    for domain in selected_domains:
        selected = [example for example in examples if example.domain == domain]
        if max_examples is not None:
            selected = selected[:max_examples]
        if len(selected) < 2:
            raise RuntimeError(f"Loss domain {domain!r} needs at least two examples")
        donors = _deranged_donors(
            selected, stable_seed(int(config["probe"]["shuffle_seed"]), domain)
        )
        kind_losses = {"real": [], "shuffled": [], "zero": []}
        for start in range(0, len(selected), batch_size):
            batch = selected[start : start + batch_size]
            donor_batch = donors[start : start + batch_size]
            prompts = [example.prompt for example in batch]
            answers = [example.answer for example in batch]
            real_tokens = [cache.get(example.record_id) for example in batch]
            shuffled_tokens = [cache.get(example.record_id) for example in donor_batch]
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                kind_losses["real"].append(
                    float(multimodal_loss(
                        llm, language_tokenizer, projector, real_tokens,
                        prompts, answers, device
                    ))
                )
                kind_losses["shuffled"].append(
                    float(multimodal_loss(
                        llm, language_tokenizer, projector, shuffled_tokens,
                        prompts, answers, device
                    ))
                )
                kind_losses["zero"].append(
                    float(multimodal_loss(
                        llm, language_tokenizer, zero_projector, real_tokens,
                        prompts, answers, device
                    ))
                )
        means = {kind: sum(values) / len(values) for kind, values in kind_losses.items()}
        row = {
            "tokenizer": name,
            "seed": seed,
            "domain": domain,
            "count": len(selected),
            **means,
            "real_minus_shuffled": means["real"] - means["shuffled"],
            "real_minus_zero": means["real"] - means["zero"],
        }
        results.append(row)
        print(name, domain, json.dumps(row, sort_keys=True), flush=True)
    del projector, cache
    torch.cuda.empty_cache()
    return results


def evaluate(
    config: dict,
    tokenizer_names: list[str],
    seed: int,
    max_examples: int | None,
    batch_size: int,
    domains: list[str] | None = None,
    projector_seen: int | None = None,
) -> Path:
    device = device_from_config(config)
    root = resolve_path(config, config["runtime"]["artifact_root"])
    suffix = "full" if max_examples is None else str(max_examples)
    selected_domains = domains or list(config["probe"]["domains"])
    domain_suffix = "all" if domains is None else "-".join(selected_domains)
    final_seen = int(config["data"]["counts"]["warmup"])
    selected_seen = final_seen if projector_seen is None else int(projector_seen)
    output = (
        root
        / "analysis"
        / f"loss_probe_seed{seed}_seen{selected_seen}_{suffix}_{domain_suffix}.json"
    )
    llm, language_tokenizer = load_llm(
        str(resolve_path(config, config["llm"]["path"])),
        device,
        gradient_checkpointing=False,
    )
    results = []
    for name in tokenizer_names:
        results.extend(
            evaluate_tokenizer_rows(
                config, llm, language_tokenizer, name, seed, max_examples,
                batch_size, selected_domains, selected_seen
            )
        )
    atomic_write_json(
        output,
        {
            "schema_version": 1,
            "max_examples_per_domain": max_examples,
            "batch_size": batch_size,
            "domains": selected_domains,
            "projector_seen": selected_seen,
            "rows": results,
        },
    )
    print(f"Wrote {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate real/shuffled/zero visual loss")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizers", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--projector-seen", type=int)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    names = choose_names(args.tokenizers, config["tokenizers"])
    unknown_domains = sorted(set(args.domains or []) - set(config["probe"]["domains"]))
    if unknown_domains:
        raise ValueError(f"Unknown domains: {unknown_domains}")
    evaluate(
        config,
        names,
        args.seed,
        args.max_examples,
        args.batch_size,
        domains=args.domains,
        projector_seen=args.projector_seen,
    )


if __name__ == "__main__":
    main()
