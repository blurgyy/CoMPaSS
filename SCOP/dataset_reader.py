from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import final

from PIL import Image


@final
class DatasetReader:
    """Abstraction for reading COCO dataset from either directory or zip files."""

    def __init__(
        self,
        coco_root: Path | None = None,
        annotations_zip: Path | None = None,
        images_zip: Path | None = None,
    ):
        """
        Initialize dataset reader with either directory or zip files.

        Args:
            coco_root: Path to COCO dataset root directory
            annotations_zip: Path to annotations zip file
            images_zip: Path to images zip file
        """
        if coco_root and (annotations_zip or images_zip):
            raise ValueError("Provide either coco_root or zip files, not both")

        if (annotations_zip and not images_zip) or (not annotations_zip and images_zip):
            raise ValueError("Must provide both zip files or neither")

        self.coco_root = coco_root
        self._annotations_zip = None
        self._images_zip = None

        if annotations_zip and images_zip:
            self._annotations_zip = zipfile.ZipFile(annotations_zip)
            self._images_zip = zipfile.ZipFile(images_zip)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._annotations_zip:
            self._annotations_zip.close()
        if self._images_zip:
            self._images_zip.close()

    def get_instances_annotations(self) -> dict:
        """Read and parse instances_train2017.json."""
        if self.coco_root:
            path = self.coco_root / "annotations" / "instances_train2017.json"
            return json.loads(path.read_text())
        else:
            with self._annotations_zip.open(
                "annotations/instances_train2017.json"
            ) as f:
                return json.load(f)

    def get_image_bytes(self, image_id: int) -> bytes:
        """Read raw image bytes by its ID."""
        image_path = f"train2017/{image_id:012d}.jpg"

        if self.coco_root:
            return (self.coco_root / "train2017" / f"{image_id:012d}.jpg").read_bytes()
        else:
            return self._images_zip.read(image_path)

    def get_image(self, image_id: int) -> Image.Image:
        """Read an image by its ID into PIL Image object."""
        if self.coco_root:
            return Image.open(self.coco_root / "train2017" / f"{image_id:012d}.jpg")
        else:
            image_data = self.get_image_bytes(image_id)
            return Image.open(io.BytesIO(image_data))

    @staticmethod
    def validate_input(
        coco_root: Path | None = None,
        annotations_zip: Path | None = None,
        images_zip: Path | None = None,
    ) -> bool:
        """Validate input paths exist and have correct content."""
        if coco_root:
            return (coco_root / "train2017").exists() and (
                coco_root / "annotations" / "instances_train2017.json"
            ).exists()

        if not (annotations_zip and images_zip):
            return False

        try:
            with zipfile.ZipFile(annotations_zip) as zf:
                if "annotations/instances_train2017.json" not in zf.namelist():
                    return False

            with zipfile.ZipFile(images_zip) as zf:
                # Check if it contains train2017 directory
                if not any(name.startswith("train2017/") for name in zf.namelist()):
                    return False

            return True
        except zipfile.BadZipFile:
            return False
