#!/usr/bin/env python3

from __future__ import annotations

import argparse
import functools
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, final, get_args, override

import numpy as np
import torch
from dacite import from_dict
from PIL import Image, ImageOps, PngImagePlugin
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch.utils.data import DataLoader, Dataset
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
COCO_CATID_TO_NAME: dict[int, CocoCategoryName] = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}
ARB = tuple[CocoCategoryName, str, CocoCategoryName]


if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


PngImagePlugin.MAX_TEXT_CHUNK = (
    10 * 1024 * 1024
)  # 10MiB, avoids "ValueError: Decompressed Data Too Large" error while loading `ffhq/in-the-wild-images/{35680,53177}.png`
Image.MAX_IMAGE_PIXELS = None  # disable MAX_IMAGE_PIXELS check


SAM2ModelID = Literal[
    "facebook/sam2-hiera-tiny",
    "facebook/sam2-hiera-small",
    "facebook/sam2-hiera-base-plus",
    "facebook/sam2-hiera-large",
]


def coco_bbox_to_regular_bbox(bbox: BoundingBox) -> BoundingBox:
    x1, y1, w, h = bbox
    return x1, y1, x1 + w, y1 + h


def deserialize_jsonl(s: str, exe: ThreadPoolExecutor | None) -> list[dict]:
    if exe is None:
        return list(map(json.loads, s.strip().splitlines()))
    else:
        return list(exe.map(json.loads, s.strip().splitlines()))


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
class JustLoadStuff(Dataset):
    def __init__(self, root: str | Path) -> None:
        super().__init__()
        self.root = root = Path(root)

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

    def __len__(self) -> int:
        return len(self.metadata)

    @override
    def __getitem__(self, index: int) -> list[dict]:
        m = self.metadata[index]
        image = np.asarray(
            ImageOps.exif_transpose(Image.open(m.file_path).convert("RGB")),
            copy=True,  # resolves UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors.
        )
        return list(
            map(
                lambda annot: {
                    "image_path": m.file_path,
                    "seq": m.seq,
                    "image": image,
                    "label": COCO_CATID_TO_NAME[annot.category_id],
                    "bbox": coco_bbox_to_regular_bbox(annot.bbox),
                },
                m.annots,
            )
        )


def collate_fn(instances: list[list[dict]]) -> list[dict]:
    return functools.reduce(
        lambda lhs, rhs: lhs + rhs,
        instances,
    )


def save_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    image_path: Path,
    label: str,
    seq: int,
    output_dir: Path | None,
):
    best_index = np.argmax(scores)
    best_mask = masks[best_index]
    output_path = (
        image_path.parent.parent
        / "masks"
        / image_path.stem
        / f"mask_{image_path.stem}_{label}.{seq}.png"
    )
    if output_dir is not None:
        output_path = output_dir / output_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_image = Image.fromarray((best_mask * 255).astype(np.uint8))
    mask_image.save(output_path)


@torch.inference_mode()
@torch.autocast("cuda")
def do_predict(
    predictor: SAM2ImagePredictor, batch: dict
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    predictor.set_image_batch([b["image"] for b in batch])
    masks, scores, _ = predictor.predict_batch(box_batch=[b["bbox"] for b in batch])
    return masks, scores


def main(
    scop_root: Path,
    sam2_model_id: SAM2ModelID,
    batch_size: int,
    num_workers: int,
    output_dir: Path | None,
) -> int:
    dataloader = DataLoader(
        JustLoadStuff(scop_root),
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = build_sam2_hf(sam2_model_id)
    predictor = SAM2ImagePredictor(model)

    exe = ThreadPoolExecutor()
    for batch in tqdm(dataloader):
        masks, scores = do_predict(predictor, batch)

        collected = exe.map(
            save_mask,
            masks,
            scores,
            [b["image_path"] for b in batch],
            [b["label"] for b in batch],
            [b["seq"] for b in batch],
            [output_dir] * len(masks),
        )
        for _ in collected:
            ...

    return 0


def make_parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser()

    cli.add_argument(
        "scop_root",
        type=Path,
    )
    cli.add_argument(
        "-m",
        "--model",
        choices=get_args(SAM2ModelID),
        default="facebook/sam2-hiera-large",
    )
    cli.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
    )
    cli.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=0,
    )
    cli.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
    )

    return cli


if __name__ == "__main__":
    args = make_parser().parse_args()
    exit(
        main(
            scop_root=args.scop_root,
            sam2_model_id=args.model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_dir=args.output,
        )
    )
