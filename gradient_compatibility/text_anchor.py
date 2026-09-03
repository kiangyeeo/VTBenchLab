from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from .data import load_examples
from .modeling import (
    clear_probe_gradients,
    flatten_probe_gradients,
    install_lora_b_probes,
    load_llm,
    text_only_loss,
)
from .utils import atomic_write_json, canonical_hash, device_from_config, load_config, resolve_path


def extract_text_anchor(config: dict, max_examples: int | None, force: bool = False) -> Path:
    device = device_from_config(config)
    artifact_root = resolve_path(config, config["runtime"]["artifact_root"])
    suffix = "full" if max_examples is None else str(max_examples)
    output_dir = artifact_root / "analysis" / f"text_anchor_{suffix}"
    metadata_path = output_dir / "protocol.json"
    identity = {
        "schema_version": 1,
        "kind": "tokenizer-independent text-only task gradients",
        "max_examples_per_domain": max_examples,
        "llm": config["llm"],
        "probe": config["probe"],
    }
    identity["fingerprint"] = canonical_hash(identity)
    domains = list(config["probe"]["domains"])
    expected = [output_dir / f"{domain}.safetensors" for domain in domains]
    if metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("fingerprint") == identity["fingerprint"]
            and metadata.get("complete")
            and all(path.is_file() for path in expected)
        ):
            print(f"Reusing complete text anchor: {output_dir}")
            return output_dir
        raise RuntimeError(f"Existing text anchor {output_dir} does not match; use --force")

    output_dir.mkdir(parents=True, exist_ok=True)
    llm, tokenizer = load_llm(
        str(resolve_path(config, config["llm"]["path"])),
        device,
        gradient_checkpointing=bool(config["llm"]["gradient_checkpointing"]),
    )
    probe_config = config["probe"]
    probes = install_lora_b_probes(
        llm,
        last_n_layers=int(probe_config["last_n_layers"]),
        target_modules=probe_config["target_modules"],
        rank=int(probe_config["lora_rank"]),
        seed=int(probe_config["lora_seed"]),
    )
    examples = load_examples(config)
    counts = {}
    for domain in domains:
        selected = [example for example in examples if example.domain == domain]
        if max_examples is not None:
            selected = selected[:max_examples]
        rows = []
        norms = []
        for index, example in enumerate(selected, start=1):
            clear_probe_gradients(probes)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                loss = text_only_loss(
                    llm, tokenizer, [example.prompt], [example.answer], device
                )
            loss.backward()
            gradient = flatten_probe_gradients(probes)
            if not bool(torch.isfinite(gradient).all().item()):
                raise RuntimeError(f"Non-finite text gradient for {example.record_id}")
            rows.append(gradient.to(torch.float16))
            norms.append(gradient.norm())
            if index % int(probe_config["log_every_examples"]) == 0 or index == len(selected):
                print(f"text-anchor/{domain}: {index}/{len(selected)}")
        destination = output_dir / f"{domain}.safetensors"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        save_file(
            {"gradient": torch.stack(rows), "norm": torch.stack(norms).float()},
            temporary,
        )
        os.replace(temporary, destination)
        counts[domain] = len(selected)
    atomic_write_json(metadata_path, {**identity, "complete": True, "domain_counts": counts})
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract tokenizer-independent text task gradients")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    extract_text_anchor(config, args.max_examples, force=args.force)


if __name__ == "__main__":
    main()
