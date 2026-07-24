import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.optim import AdamW

from vtm_lcg.cache.io import write_shard_atomic
from vtm_lcg.models import MaskedVisualPredictor
from vtm_lcg.train.streaming_data import (
    Phase0ShardCache,
    validate_karpathy_split_caches,
)
from vtm_lcg.train.train_full_coco import (
    evaluate_streaming,
    train_streaming_epoch,
)
from vtm_lcg.train.train_predictor import make_scheduler


class FakeTextConditioner:
    def encode(self, captions):
        embeddings = torch.zeros(len(captions), 3, 6)
        for row, caption in enumerate(captions):
            embeddings[row, :, 0] = float(len(caption))
        return embeddings, torch.ones(len(captions), 3, dtype=torch.bool)


def build_cache(root: Path, split_name: str, image_ids: list[int]) -> Phase0ShardCache:
    cache_dir = root / split_name
    shards_dir = cache_dir / "shards"
    records = [
        {
            "image_id": image_id,
            "caption_ids": [image_id * 10, image_id * 10 + 1],
            "captions": [f"image {image_id}", f"alternate {image_id}"],
        }
        for image_id in image_ids
    ]
    descriptors = []
    for shard_index, start in enumerate(range(0, len(records), 2)):
        stop = min(start + 2, len(records))
        values = torch.arange(
            (stop - start) * 4 * 8,
            dtype=torch.float16,
        ).reshape(stop - start, 4, 8)
        values = values / 32.0 + shard_index
        ids = torch.tensor(image_ids[start:stop], dtype=torch.int64)
        descriptors.append(
            write_shard_atomic(
                shards_dir / f"{shard_index:05d}.safetensors",
                shard_index=shard_index,
                values=values,
                image_ids=ids,
            )
        )
    manifest = {
        "complete": True,
        "record_count": len(records),
        "shard_size": 2,
        "token_count": 4,
        "hidden_dim": 8,
        "shards": [descriptor.to_dict() for descriptor in descriptors],
    }
    stats = {
        "channel_mean": [0.0] * 8,
        "channel_std": [1.0] * 8,
    }
    dataset_metadata = {"dataset_fingerprint": f"{split_name}-fingerprint"}
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (cache_dir / "records.json").write_text(
        json.dumps({"records": records, "dataset": dataset_metadata}),
        encoding="utf-8",
    )
    (cache_dir / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    return Phase0ShardCache(
        split_name=split_name,
        tokenizer_id="dummy",
        cache_key=f"{split_name}-key",
        cache_dir=cache_dir,
        manifest=manifest,
        records=records,
        dataset_metadata=dataset_metadata,
        stats=stats,
        descriptors=descriptors,
    )


class FullCocoStreamingTest(unittest.TestCase):
    def test_streaming_batches_and_split_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = build_cache(root, "train", [1, 2, 3, 4])
            validation = build_cache(root, "validation", [11, 12, 13, 14])
            test = build_cache(root, "test", [21, 22, 23, 24])
            validate_karpathy_split_caches(train, validation, test)
            batches = list(
                train.iter_batches(
                    batch_size=2,
                    shuffle=False,
                    seed=0,
                    verify_checksum=True,
                )
            )
            self.assertEqual(len(batches), 2)
            self.assertEqual(
                torch.cat([indices for _values, indices in batches]).tolist(),
                [0, 1, 2, 3],
            )

    def test_online_text_training_and_full_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = build_cache(Path(temporary), "train", [1, 2, 3, 4])
            config = {
                "predictor": {
                    "mask_ratio": 0.5,
                    "caption_dropout": 0.5,
                },
                "training": {
                    "batch_size": 2,
                    "precision": "float32",
                },
                "evaluation": {
                    "batch_size": 2,
                    "caption_mode": "first",
                    "mask_seed": 10,
                    "caption_seed": 20,
                    "shuffle_seed": 30,
                },
            }
            model = MaskedVisualPredictor(
                visual_input_dim=8,
                text_input_dim=6,
                model_dim=16,
                depth=1,
                num_heads=4,
                mlp_ratio=2,
                dropout=0.0,
                grid_shape=(2, 2),
            )
            optimizer = AdamW(model.parameters(), lr=1.0e-3)
            scheduler = make_scheduler(
                optimizer,
                total_steps=cache.batch_count(2),
                warmup_ratio=0.0,
            )
            mean = torch.zeros(1, 1, 8)
            std = torch.ones(1, 1, 8)
            train_metrics = train_streaming_epoch(
                model=model,
                text_conditioner=FakeTextConditioner(),
                cache=cache,
                mean=mean,
                std=std,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                device=torch.device("cpu"),
                seed=0,
                epoch=0,
                verify_checksum=True,
            )
            self.assertGreater(train_metrics["loss"], 0.0)
            evaluation = evaluate_streaming(
                model=model,
                text_conditioner=FakeTextConditioner(),
                cache=cache,
                mean=mean,
                std=std,
                config=config,
                device=torch.device("cpu"),
                full_sanity_checks=True,
                verify_checksum=True,
            )
            self.assertEqual(
                set(evaluation["losses"]),
                {
                    "L_mean",
                    "L_visual",
                    "L_visual_text",
                    "L_visual_shuffled_text",
                    "L_visual_spatial_shuffle",
                    "L_no_visible",
                },
            )
            self.assertTrue(
                all(torch.isfinite(torch.tensor(value)) for value in evaluation["losses"].values())
            )


if __name__ == "__main__":
    unittest.main()

