from .models import BoundingBox, CocoInstanceAnnotation


def remove_small_bboxes(
    annots: list[CocoInstanceAnnotation], min_area: int = 32 * 32
) -> list[CocoInstanceAnnotation]:
    """Remove annotations with very small bounding boxes."""
    return [ann for ann in annots if ann.bbox[2] * ann.bbox[3] >= min_area]


def is_visually_significant(
    a1: CocoInstanceAnnotation,
    a2: CocoInstanceAnnotation,
    image_metadata: dict[int, dict],
    threshold: float = 0.2,
) -> bool:
    """
    Check if the sum of the areas of boxes are larger than `threshold * image_size`.
    """
    assert a1.image_id == a2.image_id
    *_, w1, h1 = a1.bbox
    *_, w2, h2 = a2.bbox
    area1 = w1 * h1
    area2 = w2 * h2
    total_area = (
        image_metadata[a1.image_id]["height"] * image_metadata[a1.image_id]["width"]
    )
    return area1 + area2 >= threshold * total_area


def has_semantic_distinction(
    a1: CocoInstanceAnnotation, a2: CocoInstanceAnnotation
) -> bool:
    """Check if the two annotations have different categories."""
    return a1.category_id != a2.category_id


def has_spatial_clarity(
    a1: CocoInstanceAnnotation, a2: CocoInstanceAnnotation, threshold: float = 2.0
) -> bool:
    """Check if objects are within a reasonable proximity."""
    x1, y1, w1, h1 = a1.bbox
    x2, y2, w2, h2 = a2.bbox

    d1 = (w1 * w1 + h1 * h1) ** 0.5
    d2 = (w2 * w2 + h2 * h2) ** 0.5

    # Make d1 the longer diagonal if not already so
    if d1 < d2:
        d1, d2 = d2, d1

    center_x1 = x1 + w1 / 2
    center_y1 = y1 + h1 / 2
    center_x2 = x2 + w2 / 2
    center_y2 = y2 + h2 / 2

    distance = ((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2) ** 0.5
    return distance <= d2 * threshold


def has_minimal_overlap(
    a1: CocoInstanceAnnotation, a2: CocoInstanceAnnotation, threshold: float = 0.3
) -> bool:
    """Check if the overlapping area is below a threshold percentage of the smaller box."""
    x1, y1, w1, h1 = a1.bbox
    x2, y2, w2, h2 = a2.bbox

    # Calculate intersection coordinates
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    # Check for no intersection
    if x_right < x_left or y_bottom < y_top:
        return True  # No intersection, so overlap is definitely small

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate areas of the bounding boxes
    area1 = w1 * h1
    area2 = w2 * h2

    # Find the smaller area
    smaller_area = min(area1, area2)

    # Check if overlap is small enough
    return intersection_area < threshold * smaller_area


def has_size_balance(
    a1: CocoInstanceAnnotation, a2: CocoInstanceAnnotation, threshold: float = 0.5
) -> bool:
    """Check if objects have comparable visual prominence."""
    *_, w1, h1 = a1.bbox
    *_, w2, h2 = a2.bbox
    area1 = w1 * h1
    area2 = w2 * h2

    # Make area1 the larger rect if not already so
    if area1 < area2:
        area1, area2 = area2, area1

    return area2 >= area1 * threshold


def get_relative_position(bbox1: BoundingBox, bbox2: BoundingBox) -> list[str]:
    """
    Determine the relative positions of two bounding boxes, considering overlap.
    Returns a list of applicable position descriptors.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    center_x1 = x1 + w1 / 2
    center_y1 = y1 + h1 / 2
    center_x2 = x2 + w2 / 2
    center_y2 = y2 + h2 / 2

    # Determine horizontal position
    if center_x1 < center_x2:
        horizontal_rel1 = "to the left of"
        horizontal_rel2 = "to the right of"
    else:
        horizontal_rel1 = "to the right of"
        horizontal_rel2 = "to the left of"

    # Determine vertical position
    if center_y1 < center_y2:
        vertical_rel1 = "above"
        vertical_rel2 = "below"
    else:
        vertical_rel1 = "below"
        vertical_rel2 = "above"

    # Calculate overlap in x and y directions
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) / min(w1, w2)
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2)) / min(h1, h2)

    if x_overlap < 1 / 3 and y_overlap < 1 / 3:
        # Return combined positions when there's minimal overlap
        combined_positions = [
            f"{horizontal_rel1} and {vertical_rel1}",
            f"{vertical_rel1} and {horizontal_rel1}",
            f"{horizontal_rel2} and {vertical_rel2}",
            f"{vertical_rel2} and {horizontal_rel2}",
        ]
        return combined_positions

    # Prioritize relationship with less overlap
    if x_overlap < y_overlap:
        return [horizontal_rel1, horizontal_rel2]  # Prioritize left/right
    else:
        return [vertical_rel1, vertical_rel2]  # Prioritize above/below
