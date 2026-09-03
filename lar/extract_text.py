#!/usr/bin/env python
"""Prepare LAR manifests and extract mean-pooled Qwen2.5 text states."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

try:
    from .data import (
        EVAL_ANSWER_SOURCES, WORKSPACE, build_text_dataset, write_lines, write_manifests,
    )
except ImportError:  # Direct execution: python lar/extract_text.py
    from data import EVAL_ANSWER_SOURCES, WORKSPACE, build_text_dataset, write_lines, write_manifests


COCO_DOMAINS = (
    "caption", "answer", "answer_other", "question_other", "qa_concat",
    "answer_all_types", "eval_answer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=(*COCO_DOMAINS, "imagenet"), required=True)
    parser.add_argument("--image-set", choices=("coco4618", "coco5000", "in1k10k"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help=(
            "Hugging Face Hub endpoint. Defaults to HF_ENDPOINT when set, otherwise "
            "https://hf-mirror.com for mainland-China compute nodes."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=WORKSPACE / "lar" / "text")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-answer-sources", type=Path, default=EVAL_ANSWER_SOURCES,
        help="YAML describing VQAv2/GQA/TextVQA train-answer sources for eval_answer.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--tokenize-only", action="store_true",
        help="Only write token-length diagnostics; keep an existing embedding array unchanged.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _ensure_transformers_torch_compatibility() -> None:
    """Bridge optional Transformers FP8 symbols absent from older PyTorch.

    Transformers imports the fine-grained FP8 integration while defining
    Qwen2Model even though this BF16 extraction never uses FP8.  PyTorch
    releases before float8_e8m0fnu therefore fail during the lazy import.  The
    existing UniAR loader uses the same inert compatibility alias.
    """
    if not hasattr(torch, "float8_e8m0fnu"):
        fallback = getattr(torch, "float8_e4m3fn", None)
        if fallback is None:
            raise RuntimeError(
                "This Transformers version expects an FP8 dtype symbol unavailable "
                f"in PyTorch {torch.__version__}; install a compatible PyTorch/Transformers pair"
            )
        torch.float8_e8m0fnu = fallback


def tokenize_lengths(texts: list[str], args: argparse.Namespace) -> np.ndarray:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only, trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    chunks = []
    for start in range(0, len(texts), args.batch_size):
        tokens = tokenizer(
            texts[start : start + args.batch_size], padding=True, truncation=True,
            max_length=args.max_length, return_tensors="np",
        )
        chunks.append(np.asarray(tokens["attention_mask"].sum(axis=1), dtype=np.int32))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)


def encode_unique_texts(
    texts: list[str], args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    _ensure_transformers_torch_compatibility()
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
        dtype=_torch_dtype(args.dtype),
    ).to(args.device).eval().requires_grad_(False)

    unique_texts = list(dict.fromkeys(texts))
    unique_index = {text: index for index, text in enumerate(unique_texts)}
    chunks: list[np.ndarray] = []
    length_chunks: list[np.ndarray] = []
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
        length_chunks.append(
            tokens["attention_mask"].sum(dim=1).cpu().numpy().astype(np.int32, copy=False)
        )
        print(f"encoded {min(start + args.batch_size, len(unique_texts))}/{len(unique_texts)} unique texts", flush=True)
    unique_features = np.concatenate(chunks, axis=0)
    unique_lengths = np.concatenate(length_chunks, axis=0)
    indices = [unique_index[text] for text in texts]
    return unique_features[indices], unique_lengths[indices]


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        # Set this before transformers/huggingface_hub is imported by
        # encode_unique_texts; huggingface_hub reads the endpoint at import time.
        os.environ["HF_ENDPOINT"] = args.hf_endpoint.rstrip("/")
        print(f"Hugging Face endpoint: {os.environ['HF_ENDPOINT']}", flush=True)
    torch.manual_seed(args.seed)
    write_manifests(seed=args.seed)
    dataset = build_text_dataset(
        args.domain, args.image_set, seed=args.seed,
        eval_answer_sources=args.eval_answer_sources,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = f"{args.domain}__{args.image_set}"
    text_json = args.output_root / f"{stem}.texts.json"
    ids_path = args.output_root / f"{stem}.ids.txt"
    output_path = args.output_root / f"{stem}.npy"
    token_lengths_path = args.output_root / f"{stem}.token_lengths.npy"
    metadata_path = args.output_root / f"{stem}.meta.json"

    text_json.write_text(json.dumps(dataset.texts, ensure_ascii=False, indent=2), encoding="utf-8")
    write_lines(ids_path, dataset.ids)
    previous_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file() else {}
    )
    base_metadata = {
        **previous_metadata,
        "domain": dataset.domain,
        "image_set": dataset.image_set,
        "N": len(dataset.ids),
        "model": args.model,
        "max_length": args.max_length,
        "seed": args.seed,
        "unique_texts": len(set(dataset.texts)),
        "dataset": dataset.metadata or {},
        "answer_count_mean": (
            None if dataset.answer_counts is None else float(np.mean(dataset.answer_counts))
        ),
    }
    metadata_path.write_text(json.dumps(base_metadata, indent=2) + "\n", encoding="utf-8")
    if args.prepare_only:
        print(f"prepared {len(dataset.ids)} rows: {text_json}")
        return
    if not dataset.ids:
        raise RuntimeError(
            "No eval_answer rows matched the COCO-4618 pool. Check --eval-answer-sources; "
            f"coverage audit is in {metadata_path} (run --prepare-only to write it)."
        )
    if args.tokenize_only:
        token_lengths = tokenize_lengths(dataset.texts, args)
        np.save(token_lengths_path, token_lengths)
        base_metadata.update(
            mean_token_length=float(token_lengths.mean()),
            token_lengths=str(token_lengths_path),
        )
        metadata_path.write_text(json.dumps(base_metadata, indent=2) + "\n", encoding="utf-8")
        print(token_lengths_path)
        return
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite")

    features, token_lengths = encode_unique_texts(dataset.texts, args)
    if features.shape[0] != len(dataset.ids) or not np.isfinite(features).all():
        raise RuntimeError(f"Invalid text feature array: {features.shape}")
    np.save(output_path, features)
    np.save(token_lengths_path, token_lengths)
    metadata = {
        **base_metadata,
        "d": int(features.shape[1]),
        "pooling": "last_hidden_state attention-mask mean",
        "mean_token_length": float(token_lengths.mean()),
        "token_lengths": str(token_lengths_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
