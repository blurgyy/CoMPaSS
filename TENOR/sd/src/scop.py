"""
Folloing code partly adapted from
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, final, override

import inflect
import numpy as np
from dacite import from_dict
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision import transforms as T
from tqdm import tqdm
from transformers import CLIPTokenizer

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
RelativePositionDescriptor = Literal[
    "to the left of", "to the right of", "above", "below"
]
RelativePositionLogic = Literal["visor", "improved"]
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


@final
class SCOPDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        tokenizer: CLIPTokenizer,
        crop_scale_range: tuple[float, float] = (1.0, 1.1),
        image_size: int = 512,
        prompt_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.root = root = Path(root)
        self.ie = inflect.engine()
        self.tokenizer = tokenizer
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

    def __iter__(self):
        while True:
            index = random.choice(range(len(self.metadata)))
            yield self[index]

    @override
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

        attn_map_obj1 = Image.open(
            self.root
            / "masks"
            / str(m.file_path.stem)
            / f"mask_{m.file_path.stem}_{o1}.{m.seq}.png"
        )
        attn_map_obj2 = Image.open(
            self.root
            / "masks"
            / str(m.file_path.stem)
            / f"mask_{m.file_path.stem}_{o2}.{m.seq}.png"
        )
        cropped_attn_map_obj1 = attn_map_obj1.crop(
            (crop_min_x, crop_min_y, crop_max_x, crop_max_y)
        )
        cropped_attn_map_obj2 = attn_map_obj2.crop(
            (crop_min_x, crop_min_y, crop_max_x, crop_max_y)
        )
        cropped_attn_map_obj1 = cropped_attn_map_obj1.resize(
            target_size, Image.Resampling.LANCZOS
        )
        cropped_attn_map_obj2 = cropped_attn_map_obj2.resize(
            target_size, Image.Resampling.LANCZOS
        )

        attn_value_obj1 = self.attn_transforms(cropped_attn_map_obj1)
        attn_value_obj2 = self.attn_transforms(cropped_attn_map_obj2)

        if random.uniform(0.0, 1.0) < self.prompt_dropout:
            prompt = ""
        else:
            prompt = f"{self.ie.a(o1)} {rel} {self.ie.a(o2)}"
            # NOTE: use geneval's format
            prompt = prompt.replace(" to the", "").replace("", f"a photo of ", 1)

        input_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {
            "text": prompt,
            "pixel_values": pixel_values,
            # "attn_values": attn_values,
            "postprocess_seg_ls": [
                [o1, attn_value_obj1],
                [o2, attn_value_obj2],
            ],
            "input_ids": input_ids,
        }

    def __len__(self) -> int:
        return len(self.metadata)


@final
class ProbabilitySampledMultiDataset(IterableDataset):
    def __init__(
        self,
        sub_datasets: list[Dataset | IterableDataset],
        relative_sampling_importance: list[float],
    ) -> None:
        super().__init__()
        assert len(sub_datasets) == len(relative_sampling_importance), (
            f"got {len(sub_datasets)} datasets while got {len(relative_sampling_importance)} relative sampling importance"
        )
        self.sub_datasets = sub_datasets
        self.cdf = np.cumsum(
            [
                imp / sum(relative_sampling_importance)
                for imp in relative_sampling_importance
            ]
        )
        self.iterators = list(map(iter, self.sub_datasets))

        print("sampling CDF:", self.cdf.tolist())

    @override
    def __iter__(self):
        return self  # if returning self for __iter__, self must implement __next__

    def __next__(self):
        subset_index = np.searchsorted(
            self.cdf, np.random.uniform(low=0.0, high=1.0), side="left"
        )
        try:
            return next(self.iterators[subset_index])
        except StopIteration:  # subdataset must throw StopIteration on item drain
            self.iterators[subset_index] = iter(self.sub_datasets[subset_index])
            return next(self.iterators[subset_index])
