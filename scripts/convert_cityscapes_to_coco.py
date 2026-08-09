from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "human"},
    {"id": 2, "name": "rider", "supercategory": "human"},
    {"id": 3, "name": "car", "supercategory": "vehicle"},
    {"id": 4, "name": "truck", "supercategory": "vehicle"},
    {"id": 5, "name": "bus", "supercategory": "vehicle"},
    {"id": 6, "name": "train", "supercategory": "vehicle"},
    {"id": 7, "name": "motorcycle", "supercategory": "vehicle"},
    {"id": 8, "name": "bicycle", "supercategory": "vehicle"},
]

LABEL_TO_CATEGORY_ID = {category["name"]: category["id"] for category in CATEGORIES}


def polygon_area(points: list[list[float]]) -> float:
    """Compute polygon area using the shoelace formula."""
    if len(points) < 3:
        return 0.0

    area = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


def convert_split(
    root: Path,
    split: str,
    output_dir: Path,
    include_groups: bool,
) -> Path:
    annotations_dir = root / "gtFine" / split
    images_dir = root / "leftImg8bit" / split

    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory was not found: {annotations_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory was not found: {images_dir}")

    json_paths = sorted(annotations_dir.glob("*/*_gtFine_polygons.json"))
    if not json_paths:
        raise FileNotFoundError(f"No polygon JSON files were found in {annotations_dir}")

    coco: dict[str, Any] = {
        "info": {
            "description": f"Cityscapes {split} converted to COCO object detection",
        },
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }

    annotation_id = 1

    for image_id, annotation_path in enumerate(json_paths, start=1):
        with annotation_path.open("r", encoding="utf-8") as file:
            source = json.load(file)

        width = int(source["imgWidth"])
        height = int(source["imgHeight"])
        city = annotation_path.parent.name
        base_name = annotation_path.name.removesuffix("_gtFine_polygons.json")
        image_name = f"{base_name}_leftImg8bit.png"
        image_path = images_dir / city / image_name

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image for annotation {annotation_path} was not found: {image_path}"
            )

        coco["images"].append(
            {
                "id": image_id,
                "file_name": f"{city}/{image_name}",
                "width": width,
                "height": height,
            }
        )

        for obj in source.get("objects", []):
            if obj.get("deleted", False):
                continue

            raw_label = str(obj.get("label", ""))
            is_group = raw_label.endswith("group")
            label = raw_label.removesuffix("group") if is_group else raw_label

            if label not in LABEL_TO_CATEGORY_ID:
                continue
            if is_group and not include_groups:
                continue

            polygon = obj.get("polygon", [])
            if len(polygon) < 3:
                continue

            points = [[float(x), float(y)] for x, y in polygon]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]

            x_min = max(0.0, min(xs))
            y_min = max(0.0, min(ys))
            x_max = min(float(width), max(xs))
            y_max = min(float(height), max(ys))

            box_width = x_max - x_min
            box_height = y_max - y_min
            area = polygon_area(points)

            if box_width <= 1.0 or box_height <= 1.0 or area <= 1.0:
                continue

            segmentation = [[coordinate for point in points for coordinate in point]]

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": LABEL_TO_CATEGORY_ID[label],
                    "bbox": [x_min, y_min, box_width, box_height],
                    "area": area,
                    "iscrowd": int(is_group),
                    "segmentation": segmentation,
                }
            )
            annotation_id += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(coco, file, ensure_ascii=False)

    print(
        f"{split}: images={len(coco['images'])}, "
        f"annotations={len(coco['annotations'])}, output={output_path}"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Cityscapes gtFine polygons to COCO object detection."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Cityscapes root containing leftImg8bit/ and gtFine/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/annotations",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
    )
    parser.add_argument(
        "--include-groups",
        action="store_true",
        help="Include labels such as persongroup/cargroup as iscrowd=1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "annotations"
    )

    for split in args.splits:
        convert_split(
            root=root,
            split=split,
            output_dir=output_dir,
            include_groups=args.include_groups,
        )


if __name__ == "__main__":
    main()
