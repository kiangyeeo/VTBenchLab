import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vtm_lcg.cache.dataset import load_coco_karpathy_records


class CocoDatasetTest(unittest.TestCase):
    def test_stable_order_and_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            image_root.mkdir()
            for filename, color in (
                ("second.jpg", (255, 0, 0)),
                ("first.jpg", (0, 255, 0)),
            ):
                Image.new("RGB", (8, 8), color).save(image_root / filename)
            annotations = {
                "images": [
                    {"id": 20, "file_name": "second.jpg"},
                    {"id": 10, "file_name": "first.jpg"},
                ],
                "annotations": [
                    {"id": 2, "image_id": 10, "caption": "first caption"},
                    {"id": 1, "image_id": 20, "caption": "second caption"},
                    {"id": 3, "image_id": 20, "caption": "another caption"},
                ],
            }
            annotations_path = root / "captions.json"
            annotations_path.write_text(json.dumps(annotations), encoding="utf-8")

            records, metadata = load_coco_karpathy_records(
                annotations_path,
                image_root,
                limit=1,
            )
            self.assertEqual([record.image_id for record in records], [20])
            self.assertEqual(records[0].caption_ids, (1, 3))
            self.assertEqual(metadata["record_count"], 1)
            self.assertEqual(len(metadata["dataset_fingerprint"]), 64)

            records_again, metadata_again = load_coco_karpathy_records(
                annotations_path,
                image_root,
                limit=1,
            )
            self.assertEqual(records, records_again)
            self.assertEqual(metadata, metadata_again)

    def test_missing_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            image_root.mkdir()
            annotations_path = root / "captions.json"
            annotations_path.write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "missing.jpg"}],
                        "annotations": [
                            {"id": 1, "image_id": 1, "caption": "missing image"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                load_coco_karpathy_records(
                    annotations_path,
                    image_root,
                    limit=1,
                )


if __name__ == "__main__":
    unittest.main()

