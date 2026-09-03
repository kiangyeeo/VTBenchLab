from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

from .data import load_examples
from .modeling import MLP3xGELU, load_llm, multimodal_loss
from .token_cache import TokenCache
from .utils import (
    atomic_write_json,
    canonical_hash,
    choose_names,
    device_from_config,
    load_config,
    resolve_path,
    set_seed,
    stable_seed,
)


def _save_checkpoint(
    path: Path,
    projector: MLP3xGELU,
    optimizer: torch.optim.Optimizer,
    examples_seen: int,
    optimizer_steps: int,
    identity: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "projector": projector.state_dict(),
            "optimizer": optimizer.state_dict(),
            "examples_seen": examples_seen,
            "optimizer_steps": optimizer_steps,
            "identity": identity,
        },
        temporary,
    )
    temporary.replace(path)


def train_one(
    config: dict[str, Any],
    llm,
    language_tokenizer,
    tokenizer_name: str,
    seed: int,
    force: bool = False,
) -> Path:
    device = device_from_config(config)
    artifact_root = resolve_path(config, config["runtime"]["artifact_root"])
    output_dir = artifact_root / "projectors" / tokenizer_name / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    training = config["projector_training"]
    cache = TokenCache(artifact_root / "tokens" / tokenizer_name)
    input_dim = int(cache.metadata["shape"][-1])
    final_seen = int(config["data"]["counts"]["warmup"])
    identity = {
        "schema_version": 1,
        "tokenizer": tokenizer_name,
        "seed": seed,
        "input_dim": input_dim,
        "hidden_dim": int(config["llm"]["hidden_dim"]),
        "training": training,
        "warmup_examples": final_seen,
        "token_cache_fingerprint": json.loads(
            (artifact_root / "tokens" / tokenizer_name / "cache.json").read_text(
                encoding="utf-8"
            )
        )["fingerprint"],
        "llm_path": str(resolve_path(config, config["llm"]["path"])),
    }
    identity["fingerprint"] = canonical_hash(identity)
    final_path = output_dir / f"projector_seen_{final_seen}.pt"
    protocol_path = output_dir / "protocol.json"
    if final_path.is_file() and protocol_path.is_file() and not force:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol.get("fingerprint") == identity["fingerprint"] and protocol.get("complete"):
            print(f"Reusing complete projector: {final_path}")
            return final_path
        raise RuntimeError(f"Existing projector run {output_dir} does not match; use --force")

    set_seed(seed)
    projector = MLP3xGELU(
        input_dim=input_dim,
        hidden_dim=int(config["llm"]["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    examples = [example for example in load_examples(config) if example.domain == "warmup"]
    order = list(range(len(examples)))
    random.Random(stable_seed(seed, "projector_order")).shuffle(order)
    accumulation = int(training["gradient_accumulation_steps"])
    microbatch = int(training["microbatch_size"])
    checkpoint_counts = sorted(
        set(int(value) for value in training["checkpoint_examples"] if int(value) <= final_seen)
        | {final_seen}
    )
    log_every = int(training["log_every_optimizer_steps"])
    optimizer.zero_grad(set_to_none=True)
    examples_seen = 0
    optimizer_steps = 0
    interval_loss = 0.0
    interval_microbatches = 0
    pending = 0

    for batch_start in range(0, len(order), microbatch):
        indices = order[batch_start : batch_start + microbatch]
        batch = [examples[index] for index in indices]
        tokens = [cache.get(example.record_id) for example in batch]
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            loss = multimodal_loss(
                llm,
                language_tokenizer,
                projector,
                tokens,
                [example.prompt for example in batch],
                [example.answer for example in batch],
                device,
            )
        (loss / accumulation).backward()
        interval_loss += float(loss.detach().cpu())
        interval_microbatches += 1
        examples_seen += len(batch)
        pending += 1
        should_step = pending == accumulation or examples_seen == final_seen
        if should_step:
            torch.nn.utils.clip_grad_norm_(
                projector.parameters(), float(training["max_gradient_norm"])
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            pending = 0
            if optimizer_steps % log_every == 0 or examples_seen == final_seen:
                print(
                    f"{tokenizer_name}/seed{seed}: seen={examples_seen}/{final_seen} "
                    f"steps={optimizer_steps} loss={interval_loss / interval_microbatches:.4f}"
                )
                interval_loss = 0.0
                interval_microbatches = 0
        if examples_seen in checkpoint_counts:
            _save_checkpoint(
                output_dir / f"projector_seen_{examples_seen}.pt",
                projector,
                optimizer,
                examples_seen,
                optimizer_steps,
                identity,
            )

    atomic_write_json(protocol_path, {**identity, "complete": True})
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm up tokenizer-specific projectors")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizers", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    names = choose_names(args.tokenizers, config["tokenizers"])
    seeds = args.seeds if args.seeds is not None else list(config["projector_training"]["seeds"])
    device = device_from_config(config)
    llm_path = str(resolve_path(config, config["llm"]["path"]))
    llm, language_tokenizer = load_llm(
        llm_path,
        device,
        gradient_checkpointing=bool(config["llm"]["gradient_checkpointing"]),
    )
    for name in names:
        for seed in seeds:
            train_one(
                config,
                llm,
                language_tokenizer,
                name,
                seed,
                force=args.force,
            )


if __name__ == "__main__":
    main()
