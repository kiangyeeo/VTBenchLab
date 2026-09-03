from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from .evaluate_losses import evaluate_tokenizer_rows
from .modeling import load_llm
from .token_cache import TokenCache, extract_one
from .train_projector import train_one
from .utils import (
    atomic_write_json,
    canonical_hash,
    choose_names,
    load_config,
    resolve_path,
)


def _loss_one(
    config: dict,
    llm,
    language_tokenizer,
    name: str,
    seed: int,
    batch_size: int,
    force: bool,
) -> Path:
    root = resolve_path(config, config["runtime"]["artifact_root"])
    output = root / "analysis" / "by_tokenizer" / f"{name}.json"
    cache = TokenCache(root / "tokens" / name)
    seen = int(config["data"]["counts"]["warmup"])
    projector_protocol = json.loads(
        (root / "projectors" / name / f"seed_{seed}" / "protocol.json").read_text(
            encoding="utf-8"
        )
    )
    domains = list(config["protocol"]["reliable_domains"])
    identity = {
        "schema_version": 1,
        "tokenizer": name,
        "seed": seed,
        "projector_seen": seen,
        "batch_size": batch_size,
        "domains": domains,
        "token_cache_fingerprint": cache.metadata["fingerprint"],
        "projector_fingerprint": projector_protocol["fingerprint"],
        "frozen_protocol": config["protocol"],
    }
    identity["fingerprint"] = canonical_hash(identity)
    if output.is_file() and not force:
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("fingerprint") == identity["fingerprint"] and previous.get("complete"):
            print(f"Reusing complete loss probe: {output}", flush=True)
            return output
        raise RuntimeError(f"Existing loss result {output} does not match; use --force-loss")

    rows = evaluate_tokenizer_rows(
        config,
        llm,
        language_tokenizer,
        name,
        seed,
        None,
        batch_size,
        domains,
        seen,
    )
    atomic_write_json(output, {**identity, "complete": True, "rows": rows})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="One resumable worker for the frozen sweep")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", required=True, help="cuda:N")
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--tokenizers", nargs="+", default=["all"])
    parser.add_argument("--stages", default="tokens,warmup,loss")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss-batch-size", type=int, default=4)
    parser.add_argument("--force-tokens", action="store_true")
    parser.add_argument("--force-warmup", action="store_true")
    parser.add_argument("--force-loss", action="store_true")
    parser.add_argument("--clean-token-cache", action="store_true")
    args = parser.parse_args()
    if args.num_workers < 1 or not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker-index must be in [0, num-workers)")

    config, _ = load_config(args.config)
    config["runtime"]["device"] = args.device
    names = choose_names(args.tokenizers, config["tokenizers"])
    names = [name for index, name in enumerate(names) if index % args.num_workers == args.worker_index]
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    unknown = stages - {"tokens", "warmup", "loss"}
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    print(
        f"worker={args.worker_index}/{args.num_workers} device={args.device} "
        f"models={len(names)} stages={sorted(stages)}",
        flush=True,
    )

    llm = language_tokenizer = None
    if stages.intersection({"warmup", "loss"}):
        device = torch.device(args.device)
        llm, language_tokenizer = load_llm(
            str(resolve_path(config, config["llm"]["path"])),
            device,
            gradient_checkpointing=bool(config["llm"]["gradient_checkpointing"]),
        )
    root = resolve_path(config, config["runtime"]["artifact_root"])
    for name in names:
        completed_loss = root / "analysis" / "by_tokenizer" / f"{name}.json"
        if (
            {"tokens", "warmup", "loss"}.issubset(stages)
            and args.clean_token_cache
            and not (args.force_tokens or args.force_warmup or args.force_loss)
            and completed_loss.is_file()
        ):
            previous = json.loads(completed_loss.read_text(encoding="utf-8"))
            if (
                previous.get("complete")
                and previous.get("seed") == args.seed
                and previous.get("frozen_protocol") == config["protocol"]
            ):
                print(f"[resume] {name} is already complete; skipping all stages", flush=True)
                continue
        if "tokens" in stages:
            print(f"[tokens] {name}", flush=True)
            extract_one(config, name, force=args.force_tokens)
        if "warmup" in stages:
            print(f"[warmup] {name}", flush=True)
            train_one(
                config,
                llm,
                language_tokenizer,
                name,
                args.seed,
                force=args.force_warmup,
            )
        if "loss" in stages:
            print(f"[loss] {name}", flush=True)
            _loss_one(
                config,
                llm,
                language_tokenizer,
                name,
                args.seed,
                args.loss_batch_size,
                args.force_loss,
            )
        if args.clean_token_cache and "loss" in stages:
            cache_path = (root / "tokens" / name).resolve()
            expected_parent = (root / "tokens").resolve()
            if cache_path.parent != expected_parent or not (cache_path / "cache.json").is_file():
                raise RuntimeError(f"Refusing to clean unexpected token cache path: {cache_path}")
            shutil.rmtree(cache_path)
            print(f"[cleanup] removed reconstructible cache {cache_path}", flush=True)


if __name__ == "__main__":
    main()
