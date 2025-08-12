import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .dataset_reader import DatasetReader
from .models import CocoInstanceAnnotation


def visualize_object_pair(
    annot_pair: tuple[CocoInstanceAnnotation, CocoInstanceAnnotation],
    reader: DatasetReader,
    category_dict: dict[int, str],
    font_path: str | None = None,
) -> tuple[Image.Image, Image.Image]:
    """
    Visualize a pair of objects with bounding boxes.

    Args:
        annot_pair: Tuple of two annotations to visualize
        reader: DatasetReader instance to access images
        category_dict: Dictionary mapping category IDs to names
        font_path: Optional path to font file

    Returns:
        tuple: (original_image, image_with_annotations)
    """
    a1, a2 = annot_pair
    assert a1.image_id == a2.image_id

    # Get the image using the reader
    image = reader.get_image(a1.image_id)
    image_with_annotations = image.copy()

    # Create draw object
    draw = ImageDraw.Draw(image_with_annotations)

    # Try to load font, fallback to default if specified font isn't available
    try:
        font = ImageFont.truetype(
            font_path if font_path else "DejaVuSerif.ttf", size=24
        )
    except IOError:
        # Fallback to default font
        font = ImageFont.load_default()

    # Draw bounding boxes and labels
    colors = ["cyan", "magenta"]

    for i, annot in enumerate([a1, a2]):
        x, y, w, h = annot.bbox
        category = category_dict[annot.category_id]
        color = colors[i % len(colors)]

        # Draw rectangle
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        # Draw label background
        text_bbox = draw.textbbox((x, y), category, font=font)
        draw.rectangle(text_bbox, fill="black")

        # Draw label text
        draw.text((x, y), category, font=font, fill=color)

    return image, image_with_annotations


def create_sample_visualization(
    reader: DatasetReader,
    image_id_to_relationships: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    num_samples: int = 5,
) -> None:
    """
    Create sample visualizations of spatial relationships.

    Args:
        reader: DatasetReader instance to access data
        image_id_to_relationships: Dictionary mapping image IDs to relationships
        output_dir: Directory to save visualizations
        num_samples: Number of samples to create
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load category information from reader
    coco_inst = reader.get_instances_annotations()
    category_dict = {cat["id"]: cat["name"] for cat in coco_inst["categories"]}

    # Get sample image IDs
    sample_image_ids = random.sample(
        list(image_id_to_relationships.keys()),
        min(num_samples, len(image_id_to_relationships)),
    )

    for i, image_id in enumerate(sample_image_ids):
        # Get one relationship from the image
        relationship = image_id_to_relationships[image_id][0]
        annot_pair = relationship["annotations"]
        relative_positions = relationship["relative_positions"]

        # Create visualization
        _, img_with_annotations = visualize_object_pair(
            (annot_pair[0], annot_pair[1]), reader, category_dict
        )

        # Add text to describe the relationship
        draw = ImageDraw.Draw(img_with_annotations)
        rel_text = f"{relative_positions[0][0]} {relative_positions[0][1]} {relative_positions[0][2]}"
        draw.text((10, 10), rel_text, fill="white", stroke_width=2, stroke_fill="black")

        # Save the image
        output_path = output_dir / f"sample_{i}_{image_id}.jpg"
        img_with_annotations.save(output_path)

    print(f"Created {len(sample_image_ids)} sample visualizations in {output_dir}")
