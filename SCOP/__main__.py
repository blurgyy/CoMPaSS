#!/usr/bin/env python3

"""
SCOP: Spatial Constraints-Oriented Pairing

A data engine that identifies and validates spatial relationships
between object pairs through carefully designed spatial constraints.
"""

import argparse
import sys
import time
from pathlib import Path

from .data_processing import (
    SCOPParameters,
    create_per_image_annotations,
    export_dataset,
    generate_object_relationships,
    load_per_image_annotations,
)
from .dataset_reader import DatasetReader
from .visualize import create_sample_visualization


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SCOP: Spatial Constraints-Oriented Pairing data engine"
    )

    parser.add_argument(
        "--shared-images-dir",
        type=Path,
        help="Directory to store actual images, other outputs will use symlinks to this directory",
    )

    # Create mutually exclusive group for input method
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--coco-root",
        type=Path,
        help="Path to COCO 2017 dataset root directory",
    )
    input_group.add_argument(
        "--zip-input",
        nargs=2,
        type=Path,
        metavar=("ANNOTATIONS_ZIP", "IMAGES_ZIP"),
        help="Paths to annotation and image zip files",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for SCOP dataset",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory to store intermediate processing files",
    )

    parser.add_argument(
        "--visual-significance",
        type=float,
        default=0.2,
        help="Visual significance (\\tau_v) threshold (default: 0.2)",
    )

    parser.add_argument(
        "--spatial-clarity",
        type=float,
        default=2.0,
        help="Spatial clarity (\\tau_u) threshold (default: 2.0)",
    )

    parser.add_argument(
        "--minimal-overlap",
        type=float,
        default=0.3,
        help="Minimal overlap (\\tau_o) threshold (default: 0.3)",
    )

    parser.add_argument(
        "--size-balance",
        type=float,
        default=0.5,
        help="Size balance (\\tau_s) threshold (default: 0.5)",
    )

    parser.add_argument(
        "--min-bbox-area",
        type=int,
        default=32 * 32,
        help="Minimum bounding box area in pixels (default: 1024)",
    )

    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="Create sample visualizations",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sample visualizations to create (default: 5)",
    )

    args = parser.parse_args()

    # Validate input paths
    if args.coco_root:
        if not DatasetReader.validate_input(coco_root=args.coco_root):
            print(
                f"Error: Invalid COCO root directory {args.coco_root}",
                file=sys.stderr,
            )
            return 1
    else:
        annotations_zip, images_zip = args.zip_input
        if not DatasetReader.validate_input(
            annotations_zip=annotations_zip, images_zip=images_zip
        ):
            print(
                "Error: Invalid zip files provided",
                file=sys.stderr,
            )
            return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set cache directory
    cache_dir = args.cache_dir if args.cache_dir else args.output_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Process start time
    start_time = time.time()

    # Initialize dataset reader
    with DatasetReader(
        coco_root=args.coco_root,
        annotations_zip=args.zip_input[0] if args.zip_input else None,
        images_zip=args.zip_input[1] if args.zip_input else None,
    ) as reader:
        # Step 1: Load COCO annotations
        print("Loading COCO instance annotations...")
        coco_inst = reader.get_instances_annotations()

        # Step 2: Organize annotations by image
        per_image_annotations_path = cache_dir / "per_image_annotations.json"

        if per_image_annotations_path.exists():
            print(
                f"Loading existing per-image annotations from {per_image_annotations_path}..."
            )
            per_image_annots = load_per_image_annotations(per_image_annotations_path)
        else:
            print("Creating per-image annotations...")
            per_image_annots = create_per_image_annotations(
                coco_inst, per_image_annotations_path
            )

        # Step 3: Apply spatial constraints and generate relationships
        print("Applying spatial constraints and generating object relationships...")
        params = SCOPParameters(
            visual_significance=args.visual_significance,
            spatial_clarity=args.spatial_clarity,
            minimal_overlap=args.minimal_overlap,
            size_balance=args.size_balance,
            min_bbox_area=args.min_bbox_area,
        )
        image_id_to_relationships, _ = generate_object_relationships(
            coco_inst,
            per_image_annots,
            params=params,
        )

        # Step 4: Export dataset
        print(f"Exporting SCOP dataset to {args.output_dir}...")
        export_dataset(
            reader,
            image_id_to_relationships,
            args.output_dir,
            shared_images_dir=args.shared_images_dir,
        )

        # Step 5: Create sample visualizations if requested
        if args.create_samples:
            print(f"Creating {args.num_samples} sample visualizations...")
            samples_dir = args.output_dir / "samples"
            create_sample_visualization(
                reader, image_id_to_relationships, samples_dir, args.num_samples
            )

        elapsed_time = time.time() - start_time
        print(f"SCOP processing completed in {elapsed_time:.2f} seconds")

        # Print dataset statistics
        total_images = len(image_id_to_relationships)
        total_relationships = sum(
            len(rels) for rels in image_id_to_relationships.values()
        )

        print("\nSCOP Dataset Statistics:")
        print(f"  - Images: {total_images}")
        print(f"  - Object pairs: {total_relationships}")
        print(f"  - Output directory: {args.output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
