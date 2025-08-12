from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

BoundingBox = tuple[float, float, float, float]
RelativePositionDescriptor = Literal[
    "to the left of", "to the right of", "above", "below"
]
ARB = tuple[str, str, str]  # (Object1, RelativePosition, Object2)


@dataclass(frozen=True)
class CocoInstanceAnnotation:
    """COCO instance annotation with bounding box and category information."""

    @dataclass(frozen=True)
    class Segmentationdict:
        counts: list[int]
        size: list[int]

    segmentation: list[list[float]] | Segmentationdict
    area: float
    iscrowd: int  # actually value can only be 0 or 1
    image_id: int
    bbox: BoundingBox
    category_id: int
    id: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CocoInstanceAnnotation:
        """Create a CocoInstanceAnnotation from a dictionary."""
        d_copy = d.copy()
        d_copy["bbox"] = tuple(d_copy["bbox"])

        # Handle segmentation - could be list of lists or dict
        if isinstance(d_copy["segmentation"], dict):
            d_copy["segmentation"] = cls.Segmentationdict(**d_copy["segmentation"])

        return cls(**d_copy)

    def asdict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def serialize_jsonl(data_list: list[dict[str, Any]]) -> str:
    """Serialize list of dictionaries to JSONL format."""
    return "\n".join(json.dumps(d, separators=(",", ":")) for d in data_list) + "\n"
