from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

import inflect
from dacite import from_dict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

BoundingBox = tuple[float, float, float, float]
CocoCategoryName = Literal[
    "person",
    "chair",
    "car",
    "dining table",
    "cup",
    "bottle",
    "bowl",
    "handbag",
    "truck",
    "bench",
    "backpack",
    "book",
    "cell phone",
    "sink",
    "clock",
    "tv",
    "potted plant",
    "couch",
    "dog",
    "knife",
    "sports ball",
    "traffic light",
    "cat",
    "umbrella",
    "bus",
    "tie",
    "bed",
    "vase",
    "train",
    "fork",
    "spoon",
    "laptop",
    "motorcycle",
    "surfboard",
    "skateboard",
    "tennis racket",
    "toilet",
    "bicycle",
    "bird",
    "pizza",
    "skis",
    "remote",
    "boat",
    "airplane",
    "horse",
    "cake",
    "oven",
    "baseball glove",
    "giraffe",
    "wine glass",
    "baseball bat",
    "suitcase",
    "sandwich",
    "refrigerator",
    "kite",
    "banana",
    "frisbee",
    "elephant",
    "teddy bear",
    "keyboard",
    "cow",
    "broccoli",
    "zebra",
    "mouse",
    "stop sign",
    "fire hydrant",
    "orange",
    "carrot",
    "snowboard",
    "apple",
    "microwave",
    "sheep",
    "donut",
    "hot dog",
    "toothbrush",
    "bear",
    "scissors",
    "parking meter",
    "toaster",
    "hair drier",
]
ARB = tuple[CocoCategoryName, str, CocoCategoryName]


def deserialize_jsonl(s: str, exe: ThreadPoolExecutor | None) -> list[dict]:
    if exe is not None:
        map = exe.map
    return list(map(json.loads, s.strip().splitlines()))


@dataclass(kw_only=True, frozen=True)
class CocoInstanceAnnotation:
    @dataclass(kw_only=True, frozen=True)
    class SegmentationDict:
        counts: list[int]
        size: list[int]

    segmentation: list[list[float]] | SegmentationDict
    area: float
    iscrowd: int  # actually value can only be 0 or 1
    image_id: int
    bbox: BoundingBox
    category_id: int
    id: int

    # from_dict = classmethod(from_dict)
    @classmethod
    def from_dict(cls, d: dict) -> CocoInstanceAnnotation:
        d["bbox"] = tuple(d["bbox"])
        return from_dict(data_class=cls, data=d)

    def asdict(self) -> dict:
        return asdict(self)


@dataclass(kw_only=True, frozen=True)
class SCOPDataPoint:
    seq: int
    file_path: Path
    oros: list[ARB]
    annots: tuple[CocoInstanceAnnotation, CocoInstanceAnnotation]

    root: Path

    @classmethod
    def from_dict(cls, d: dict, root: Path) -> SCOPDataPoint:
        root = Path(root)
        d["oros"] = list(map(tuple, d["oros"]))
        d["annots"] = tuple(map(lambda x: x | {"bbox": tuple(x["bbox"])}, d["annots"]))
        d["root"] = root
        d["file_path"] = root / d["file_name"]
        d.pop("file_name")
        return from_dict(data_class=cls, data=d)

    def asdict(self) -> dict:
        d = asdict(self)
        d["file_name"] = self.file_path.relative_to(self.root)
        d.pop("root")
        d.pop("file_path")
        return d


class SCOPDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        crop_scale_range: tuple[float, float] = (1.0, 1.1),
        image_size: int = 512,
        prompt_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.root = root = Path(root)
        self.ie = inflect.engine()
        self.crop_scale_range = crop_scale_range
        self.image_size = image_size
        self.prompt_dropout = prompt_dropout

        self.metadata = list(
            map(
                lambda m: SCOPDataPoint.from_dict(m, root=root),
                tqdm(
                    deserialize_jsonl(
                        root.joinpath("metadata.jsonl").read_text(),
                        exe=ThreadPoolExecutor(),
                    ),
                    desc="loading metadata",
                ),
            )
        )

        # compute some metrics of the loaded dataset, including the top-5 most frequent relations
        # and their occurrences, top-10 most frequent object names and their occurrences, etc.
        self.display_metrics()

    def display_metrics(self):
        relation_counts = {}
        object_counts = {}

        for data_point in self.metadata:
            for s, p, o in data_point.oros:
                relation = (s, p, o)
                relation_counts[relation] = relation_counts.get(relation, 0) + 1
                object_counts[s] = object_counts.get(s, 0) + 1
                object_counts[o] = object_counts.get(o, 0) + 1

        sorted_relations = sorted(
            relation_counts.items(), key=lambda item: item[1], reverse=True
        )
        sorted_objects = sorted(
            object_counts.items(), key=lambda item: item[1], reverse=True
        )

        total_relations = sum(relation_counts.values())
        total_objects = sum(object_counts.values())

        print("Top 10 most frequent relations:")
        for relation, count in sorted_relations[:10]:
            print(f"- {relation}: {count} ({count / total_relations:.2%})")

        print("\nTop 10 most frequent objects:")
        for obj, count in sorted_objects[:10]:
            print(f"- {obj}: {count} ({count / total_objects:.2%})")

        relation_descriptor_counts = {}
        for data_point in self.metadata:
            for s, p, o in data_point.oros:
                relation_descriptor_counts[p] = relation_descriptor_counts.get(p, 0) + 1

        total_relations = sum(relation_counts.values())
        total_relation_descriptors = sum(
            relation_descriptor_counts.values()
        )  # added this line for total

        print("\nRelation Descriptor Statistics:")
        for descriptor, count in relation_descriptor_counts.items():
            percentage = (
                (count / total_relation_descriptors) * 100
                if total_relation_descriptors > 0
                else 0
            )  # check division by zero
            print(f"- {descriptor}: {count} ({percentage:.2f}%)")

    @cached_property
    def transforms(self) -> T.Compose:
        return T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5),
                T.Resize(
                    (self.image_size, self.image_size),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
            ]
        )

    @cached_property
    def attn_transforms(self) -> T.Compose:
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (self.image_size, self.image_size),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
            ]
        )

    def __getitem__(self, index: int) -> dict:
        m = self.metadata[index]

        image = Image.open(m.file_path).convert("RGB")  # Ensure RGB format

        annot1, annot2 = m.annots

        # Calculate combined bounding box
        min_x = min(annot1.bbox[0], annot2.bbox[0])
        min_y = min(annot1.bbox[1], annot2.bbox[1])
        max_x = max(annot1.bbox[0] + annot1.bbox[2], annot2.bbox[0] + annot2.bbox[2])
        max_y = max(annot1.bbox[1] + annot1.bbox[3], annot2.bbox[1] + annot2.bbox[3])

        # Calculate center of the combined bounding box
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Calculate the side length of the square that encloses both bounding boxes
        side = max(max_x - min_x, max_y - min_y)

        # Apply random scale factor
        scale = random.uniform(*self.crop_scale_range)
        side *= scale

        # Ensure the cropping area is within the image bounds
        img_width, img_height = image.size
        max_side = min(img_width, img_height)

        # If the side is larger than the image, shrink it
        if side > max_side:
            side = max_side

        # Calculate crop coordinates
        crop_min_x = max(0, center_x - side / 2)
        crop_min_y = max(0, center_y - side / 2)
        crop_max_x = min(img_width, center_x + side / 2)
        crop_max_y = min(img_height, center_y + side / 2)

        # Adjust crop coordinates to maintain square shape
        if crop_max_x - crop_min_x > crop_max_y - crop_min_y:
            diff = (crop_max_x - crop_min_x) - (crop_max_y - crop_min_y)
            crop_max_y += diff / 2
            crop_min_y -= diff / 2
        elif crop_max_y - crop_min_y > crop_max_x - crop_min_x:
            diff = (crop_max_y - crop_min_y) - (crop_max_x - crop_min_x)
            crop_max_x += diff / 2
            crop_min_x -= diff / 2

        # Ensure crop coordinates are within image bounds
        crop_min_x = max(0, crop_min_x)
        crop_min_y = max(0, crop_min_y)
        crop_max_x = min(img_width, crop_max_x)
        crop_max_y = min(img_height, crop_max_y)

        # Crop and resize
        cropped_image = image.crop((crop_min_x, crop_min_y, crop_max_x, crop_max_y))

        # Resize to ensure square shape
        target_size = (self.image_size, self.image_size)
        cropped_image = cropped_image.resize(target_size, Image.Resampling.LANCZOS)

        pixel_values = self.transforms(cropped_image)

        o1, rel, o2 = random.choice(m.oros)

        if random.uniform(0.0, 1.0) < self.prompt_dropout:
            prompt = ""
        else:
            prompt = f"{self.ie.a(o1)} {rel} {self.ie.a(o2)}"
            # NOTE: use geneval's format
            prompt = prompt.replace(" to the", "").replace("", f"a photo of ", 1)

        return {
            "text": prompt,
            "pixel_values": pixel_values,
        }

    def __len__(self) -> int:
        return len(self.metadata)
