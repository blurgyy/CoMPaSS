from __future__ import annotations

import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from SCOP.dataset_reader import DatasetReader

from .filters import (
    get_relative_position,
    has_minimal_overlap,
    has_semantic_distinction,
    has_size_balance,
    has_spatial_clarity,
    is_visually_significant,
    remove_small_bboxes,
)
from .models import CocoInstanceAnnotation, serialize_jsonl


@dataclass(frozen=True, kw_only=True)
class SCOPParameters:
    min_bbox_area: int = 32 * 32

    # \tau_v
    visual_significance: float = 0.2

    # semantic_distinction

    # \tau_u
    spatial_clarity: float = 2.0

    # \tau_o
    minimal_overlap: float = 0.3

    # \tau_s
    size_balance: float = 0.5

    @staticmethod
    def best_geneval() -> SCOPParameters:
        return SCOPParameters(
            visual_significance=0.40087064191445176,
            spatial_clarity=1.077654397636866,
            minimal_overlap=0.23206626338746755,
            size_balance=0.4823213426319574,
        )

    def as_json_str(self) -> str:
        return json.dumps(asdict(self), indent=2)


def load_coco_instance_annotations(coco_root: Path) -> dict[str, Any]:
    """Load COCO instance annotations from disk."""
    coco_inst_json_path = coco_root / "annotations" / "instances_train2017.json"
    return json.loads(coco_inst_json_path.read_text())


def create_per_image_annotations(
    coco_inst: dict[str, Any], output_path: Path
) -> dict[int, list[CocoInstanceAnnotation]]:
    """
    Organize annotations by image_id and save to disk.
    Returns the organized annotations dictionary.
    """
    per_image_annots: dict[int, list[CocoInstanceAnnotation]] = {}

    for inst_annot in tqdm(
        coco_inst["annotations"], desc="Creating per-image annotations"
    ):
        image_id: int = inst_annot["image_id"]
        annot = CocoInstanceAnnotation.from_dict(inst_annot)

        if image_id in per_image_annots:
            per_image_annots[image_id].append(annot)
        else:
            per_image_annots[image_id] = [annot]

    # Save to disk
    if output_path:
        serialized = {
            k: [asdict(ann) for ann in v] for k, v in per_image_annots.items()
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(serialized, separators=(",", ":")))

    return per_image_annots


def load_per_image_annotations(
    annotation_path: Path,
) -> dict[int, list[CocoInstanceAnnotation]]:
    """Load per-image annotations from a previously saved JSON file."""
    per_image_dict = json.loads(annotation_path.read_text())

    with ThreadPoolExecutor() as executor:

        def process_item(key_val):
            key, val = key_val
            return int(key), [CocoInstanceAnnotation.from_dict(a) for a in val]

        result_dict = {}
        for image_id, annots in tqdm(
            executor.map(process_item, per_image_dict.items()),
            total=len(per_image_dict),
            desc="Loading per-image annotations",
        ):
            result_dict[image_id] = annots

    return result_dict


def generate_object_relationships(
    coco_inst: dict[str, Any],
    per_image_annots: dict[int, list[CocoInstanceAnnotation]],
    params: SCOPParameters,
) -> tuple[dict[int, list[dict[str, Any]]], int]:
    """
    Apply spatial constraints and generate clear spatial relationships.
    Returns a dictionary mapping image_ids to lists of relationship dictionaries.
    """
    image_metadata = {m["id"]: m for m in coco_inst["images"]}
    category_metadata = {cat["id"]: cat["name"] for cat in coco_inst["categories"]}

    # Create a dictionary to store results
    image_id_to_relationships: dict[int, list[dict[str, Any]]] = {}

    # Process each image's annotations
    total_image_count = len(per_image_annots)
    total_filtered_pairs = 0

    for image_id, annots in tqdm(
        per_image_annots.items(), total=total_image_count, desc="Processing images"
    ):
        # 1. Remove small bounding boxes
        filtered_annots = remove_small_bboxes(annots, min_area=params.min_bbox_area)

        # Skip if there are fewer than 2 objects after filtering
        if len(filtered_annots) < 2:
            continue

        # 2. Create pairs of objects
        object_pairs = list(itertools.combinations(filtered_annots, 2))
        valid_relationships = []

        # 3. Apply spatial constraints
        for a1, a2 in object_pairs:
            # Apply all spatial constraints
            if (
                is_visually_significant(
                    a1, a2, image_metadata, params.visual_significance
                )
                and has_semantic_distinction(a1, a2)
                and has_spatial_clarity(a1, a2, params.spatial_clarity)
                and has_minimal_overlap(a1, a2, params.minimal_overlap)
                and has_size_balance(a1, a2, params.size_balance)
            ):
                # Create relationship descriptor
                rel_positions = get_relative_position(a1.bbox, a2.bbox)
                object_name1 = category_metadata[a1.category_id]
                object_name2 = category_metadata[a2.category_id]

                # Split rel_positions into two parts - first half for o1->o2, second half for o2->o1
                mid_idx = len(rel_positions) // 2
                rel12 = rel_positions[:mid_idx] if mid_idx > 0 else rel_positions
                rel21 = rel_positions[mid_idx:] if mid_idx < len(rel_positions) else []

                relative_positions = []
                for r12 in rel12:
                    relative_positions.append([object_name1, r12, object_name2])
                for r21 in rel21:
                    relative_positions.append([object_name2, r21, object_name1])

                # Add to valid relationships
                valid_relationships.append(
                    {"annotations": [a1, a2], "relative_positions": relative_positions}
                )

        if valid_relationships:
            image_id_to_relationships[image_id] = valid_relationships
            total_filtered_pairs += len(valid_relationships)

    print(
        f"Found {len(image_id_to_relationships)} images with {total_filtered_pairs} valid object pairs"
    )
    return image_id_to_relationships, total_filtered_pairs


def export_dataset(
    reader: DatasetReader,
    image_id_to_relationships: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    shared_images_dir: Path | None = None,
) -> None:
    """
    Export the SCOP dataset to the specified directory.
    Creates:
        - A metadata.jsonl file with relationship information
        - Either:
          - Symbolic links to shared_images_dir if provided
          - Symbolic link to original dataset if using directory input
          - Copy needed images if using zip input without shared_images_dir

    Args:
        reader: DatasetReader instance to access data
        image_id_to_relationships: Dictionary mapping image IDs to relationships
        output_dir: Directory to save the dataset
        shared_images_dir: Optional directory to store actual images
    """
    output_dir = Path(output_dir)
    images_outdir = output_dir / "images"
    metadata_jsonl_path = output_dir / "metadata.jsonl"

    # Function to process each image
    def process_image(image_data):
        image_id, relationships = image_data
        metadata_entries = []

        # Prepare filename
        image_name = f"{image_id:012d}.jpg"

        # Create metadata entries for each relationship
        for i, rel_data in enumerate(relationships):
            annotations = rel_data["annotations"]
            rel_positions = rel_data["relative_positions"]

            metadata_entries.append(
                {
                    "seq": i,
                    "file_name": f"images/{image_name}",  # Relative path
                    "oros": rel_positions,
                    "annots": [ann.asdict() for ann in annotations],
                }
            )

        return metadata_entries

    # Process images in parallel
    with ThreadPoolExecutor() as executor:
        all_metadata = list(
            tqdm(
                itertools.chain.from_iterable(
                    executor.map(process_image, image_id_to_relationships.items())
                ),
                desc="Exporting dataset",
                total=sum(len(rels) for rels in image_id_to_relationships.values()),
            )
        )

    output_dir.mkdir(exist_ok=True, parents=True)
    # Write metadata file
    metadata_jsonl_path.write_text(serialize_jsonl(all_metadata))

    # Handle images based on configuration
    needed_image_ids = set(image_id_to_relationships.keys())

    if shared_images_dir:
        # Using shared images directory
        shared_images_dir.mkdir(exist_ok=True, parents=True)

        # Create or ensure all needed images exist in shared directory
        for image_id in tqdm(
            needed_image_ids, desc="Ensuring images in shared directory"
        ):
            target_path = shared_images_dir / f"{image_id:012d}.jpg"
            if not target_path.exists():
                if reader.coco_root:
                    # For directory input, create symlink to original
                    target_path.symlink_to(
                        reader.coco_root / "train2017" / f"{image_id:012d}.jpg"
                    )
                else:
                    # For zip input, extract image
                    image_bytes = reader.get_image_bytes(image_id)
                    target_path.write_bytes(image_bytes)

        # Create symlinks in output directory
        images_outdir.mkdir(exist_ok=True, parents=True)
        for image_id in tqdm(needed_image_ids, desc="Creating image symlinks"):
            symlink_path = images_outdir / f"{image_id:012d}.jpg"
            if not symlink_path.exists():
                # Use relative path for symlink to make the output directory relocatable
                rel_path = os.path.relpath(
                    shared_images_dir / f"{image_id:012d}.jpg", images_outdir
                )
                symlink_path.symlink_to(rel_path)

    else:
        # No shared directory - use original behavior
        images_outdir.mkdir(exist_ok=True, parents=True)
        if reader.coco_root:
            # If using directory input, create symlink to original dataset
            if not images_outdir.exists():
                images_outdir.symlink_to(
                    reader.coco_root / "train2017", target_is_directory=True
                )
        else:
            # If using zip input, extract only needed images
            for image_id in tqdm(needed_image_ids, desc="Extracting images"):
                image_bytes = reader.get_image_bytes(image_id)
                (images_outdir / f"{image_id:012d}.jpg").write_bytes(image_bytes)

    print(f"Exported {len(all_metadata)} relationship entries to {metadata_jsonl_path}")
    print(f"Exported images to {images_outdir}")
