from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .data import Example, load_examples
from .modeling import (
    MLP3xGELU,
    clear_probe_gradients,
    flatten_probe_gradients,
    install_lora_b_probes,
    load_llm,
    multimodal_loss,
)
from .token_cache import TokenCache
from .utils import (
    atomic_write_json,
    canonical_hash,
    choose_names,
    device_from_config,
    load_config,
    resolve_path,
    stable_seed,
)


def _gradient(
    llm,
    language_tokenizer,
    projector,
    probes,
    tokens: torch.Tensor,
    example: Example,
    device: torch.device,
) -> torch.Tensor:
    clear_probe_gradients(probes)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        loss = multimodal_loss(
            llm,
            language_tokenizer,
            projector,
            [tokens],
            [example.prompt],
            [example.answer],
            device,
        )
    loss.backward()
    gradient = flatten_probe_gradients(probes)
    if not bool(torch.isfinite(gradient).all().item()):
        raise RuntimeError(f"Non-finite gradient for {example.record_id}")
    return gradient


def _deranged_donors(examples: list[Example], seed: int) -> list[Example]:
    if len(examples) < 2:
        raise ValueError("Each probe domain needs at least two examples")
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    donor_by_index = {}
    for position, index in enumerate(order):
        donor_by_index[index] = order[(position + 1) % len(order)]
    return [examples[donor_by_index[index]] for index in range(len(examples))]


def extract_one(
    config: dict[str, Any],
    llm,
    language_tokenizer,
    probes,
    probe_names: list[str],
    tokenizer_name: str,
    seed: int,
    force: bool = False,
) -> Path:
    device = device_from_config(config)
    artifact_root = resolve_path(config, config["runtime"]["artifact_root"])
    final_seen = int(config["data"]["counts"]["warmup"])
    projector_path = (
        artifact_root
        / "projectors"
        / tokenizer_name
        / f"seed_{seed}"
        / f"projector_seen_{final_seen}.pt"
    )
    if not projector_path.is_file():
        raise FileNotFoundError(projector_path)
    output_dir = artifact_root / "gradients" / tokenizer_name / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "protocol.json"
    checkpoint = torch.load(projector_path, map_location="cpu", weights_only=False)
    identity = {
        "schema_version": 1,
        "tokenizer": tokenizer_name,
        "projector_seed": seed,
        "projector_identity": checkpoint["identity"],
        "probe": config["probe"],
        "probe_parameter_names": probe_names,
    }
    identity["fingerprint"] = canonical_hash(identity)
    if metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = [output_dir / f"{domain}.safetensors" for domain in config["probe"]["domains"]]
        if (
            metadata.get("fingerprint") == identity["fingerprint"]
            and metadata.get("complete")
            and all(path.is_file() for path in expected)
        ):
            print(f"Reusing complete gradients: {output_dir}")
            return output_dir
        raise RuntimeError(f"Existing gradient run {output_dir} does not match; use --force")

    tokenizer_spec = config["tokenizers"][tokenizer_name]
    projector = MLP3xGELU(
        input_dim=int(tokenizer_spec["hidden_dim"]),
        hidden_dim=int(config["llm"]["hidden_dim"]),
    ).to(device)
    projector.load_state_dict(checkpoint["projector"])
    projector.eval().requires_grad_(False)
    cache = TokenCache(artifact_root / "tokens" / tokenizer_name)
    all_examples = load_examples(config)
    domain_counts = {}
    gradient_dim = sum(probe.parameter.numel() for probe in probes)

    for domain in config["probe"]["domains"]:
        examples = [example for example in all_examples if example.domain == domain]
        donors = _deranged_donors(
            examples,
            stable_seed(int(config["probe"]["shuffle_seed"]), domain),
        )
        real_rows = []
        delta_rows = []
        real_norms = []
        shuffled_norms = []
        delta_norms = []
        for index, (example, donor) in enumerate(zip(examples, donors), start=1):
            real = _gradient(
                llm,
                language_tokenizer,
                projector,
                probes,
                cache.get(example.record_id),
                example,
                device,
            )
            shuffled = _gradient(
                llm,
                language_tokenizer,
                projector,
                probes,
                cache.get(donor.record_id),
                example,
                device,
            )
            delta = real - shuffled
            real_rows.append(real.to(torch.float16))
            delta_rows.append(delta)
            real_norms.append(real.norm())
            shuffled_norms.append(shuffled.norm())
            delta_norms.append(delta.norm())
            if index % int(config["probe"]["log_every_examples"]) == 0 or index == len(examples):
                print(f"{tokenizer_name}/seed{seed}/{domain}: {index}/{len(examples)}")
        tensors = {
            "real": torch.stack(real_rows).contiguous(),
            "delta": torch.stack(delta_rows).contiguous(),
            "real_norm": torch.stack(real_norms).float(),
            "shuffled_norm": torch.stack(shuffled_norms).float(),
            "delta_norm": torch.stack(delta_norms).float(),
        }
        destination = output_dir / f"{domain}.safetensors"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        save_file(tensors, temporary)
        os.replace(temporary, destination)
        domain_counts[domain] = len(examples)

    atomic_write_json(
        metadata_path,
        {
            **identity,
            "complete": True,
            "gradient_dim": gradient_dim,
            "domain_counts": domain_counts,
            "feature_files": {domain: f"{domain}.safetensors" for domain in domain_counts},
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract fresh LoRA-B compatibility gradients")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizers", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    names = choose_names(args.tokenizers, config["tokenizers"])
    seeds = args.seeds if args.seeds is not None else list(config["projector_training"]["seeds"])
    device = device_from_config(config)
    llm, language_tokenizer = load_llm(
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
    probe_names = [probe.name for probe in probes]
    print(f"Installed {len(probes)} LoRA-B probes with {sum(p.parameter.numel() for p in probes)} coordinates")
    for name in names:
        for seed in seeds:
            extract_one(
                config,
                llm,
                language_tokenizer,
                probes,
                probe_names,
                name,
                seed,
                force=args.force,
            )


if __name__ == "__main__":
    main()
