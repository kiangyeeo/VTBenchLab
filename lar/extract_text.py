#!/usr/bin/env python
"""Prepare LAR manifests and extract mean-pooled Qwen2.5 text states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from .data import WORKSPACE, build_text_dataset, write_lines, write_manifests
except ImportError:  # Direct execution: python lar/extract_text.py
    from data import WORKSPACE, build_text_dataset, write_lines, write_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("caption", "answer", "imagenet"), required=True)
    parser.add_argument("--image-set", choices=("coco4618", "coco5000", "in1k10k"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output-root", type=Path, default=WORKSPACE / "lar" / "text")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def encode_unique_texts(texts: list[str], args: argparse.Namespace) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; pass --device cpu --dtype float32")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
        torch_dtype=_torch_dtype(args.dtype),
    ).to(args.device).eval().requires_grad_(False)

    unique_texts = list(dict.fromkeys(texts))
    unique_index = {text: index for index, text in enumerate(unique_texts)}
    chunks: list[np.ndarray] = []
    for start in range(0, len(unique_texts), args.batch_size):
        batch = unique_texts[start : start + args.batch_size]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(args.device) for key, value in tokens.items()}
        with torch.inference_mode():
            outputs = model(**tokens, return_dict=True)
        hidden = outputs.last_hidden_state.float()
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        chunks.append(pooled.cpu().numpy().astype(np.float32, copy=False))
        print(f"encoded {min(start + args.batch_size, len(unique_texts))}/{len(unique_texts)} unique texts", flush=True)
    unique_features = np.concatenate(chunks, axis=0)
    return unique_features[[unique_index[text] for text in texts]]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    write_manifests(seed=args.seed)
    dataset = build_text_dataset(args.domain, args.image_set, seed=args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = f"{args.domain}__{args.image_set}"
    text_json = args.output_root / f"{stem}.texts.json"
    ids_path = args.output_root / f"{stem}.ids.txt"
    output_path = args.output_root / f"{stem}.npy"
    metadata_path = args.output_root / f"{stem}.meta.json"

    text_json.write_text(json.dumps(dataset.texts, ensure_ascii=False, indent=2), encoding="utf-8")
    write_lines(ids_path, dataset.ids)
    if args.prepare_only:
        print(f"prepared {len(dataset.ids)} rows: {text_json}")
        return
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite")

    features = encode_unique_texts(dataset.texts, args)
    if features.shape[0] != len(dataset.ids) or not np.isfinite(features).all():
        raise RuntimeError(f"Invalid text feature array: {features.shape}")
    np.save(output_path, features)
    metadata = {
        "domain": dataset.domain,
        "image_set": dataset.image_set,
        "N": len(dataset.ids),
        "d": int(features.shape[1]),
        "model": args.model,
        "pooling": "last_hidden_state attention-mask mean",
        "max_length": args.max_length,
        "seed": args.seed,
        "answer_count_mean": (
            None if dataset.answer_counts is None else float(np.mean(dataset.answer_counts))
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
