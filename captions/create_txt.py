#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

PLANE_TO_FOLDER = {
    "trans_thalamic": "1_thalamic",
    "trans_cerebellum": "2_cerebellum",
    "trans_ventricular": "3_ventricular",
}


def normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[\s-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def normalize_phrase(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[_-]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ,.;")


def parse_cap_pairs(cap: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for part in cap.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        pairs.append((key.strip(), value.strip()))
    return pairs


def parse_cap_fields(cap: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key, value in parse_cap_pairs(cap):
        fields[key.strip().lower()] = value.strip()
    return fields


def detailed_caption_from_cap(cap: str) -> str | None:
    pairs = parse_cap_pairs(cap)
    if not pairs:
        return None

    fields = {k.strip().lower(): v.strip() for k, v in pairs}
    region = fields.get("region")
    plane = fields.get("plane")
    if not region or not plane:
        return None

    region_phrase = normalize_phrase(region)
    plane_phrase = normalize_phrase(plane)

    parts = [
        f"{plane_phrase} plane",
        "fetal ultrasound scan",
        region_phrase,
    ]

    for raw_key, raw_value in pairs:
        key_phrase = normalize_phrase(raw_key)
        value_phrase = normalize_phrase(raw_value)
        if not key_phrase or not value_phrase:
            continue
        if key_phrase in {"plane", "region"}:
            continue
        parts.append(f"{key_phrase} {value_phrase}")

    return ", ".join(parts)


def caption_and_folder_from_record(record: dict) -> tuple[str, str] | None:
    cap = record.get("cap", "")
    if not cap:
        return None

    fields = parse_cap_fields(cap)
    region = fields.get("region")
    plane = fields.get("plane")
    if not region or not plane:
        return None

    norm_region = normalize_text(region)
    norm_plane = normalize_text(plane)
    folder_name = PLANE_TO_FOLDER.get(norm_plane)
    if not folder_name:
        return None

    caption = detailed_caption_from_cap(cap)
    if not caption:
        return None

    return caption, folder_name


def build_caption_map(jsonl_path: Path) -> dict[str, tuple[str, str]]:
    caption_map: dict[str, tuple[str, str]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_no}")
                continue

            image_path = record.get("image_path")
            if not image_path:
                continue

            item = caption_and_folder_from_record(record)
            if not item:
                continue

            image_name = Path(image_path).name
            caption_map[image_name] = item
    return caption_map


def create_txt_files(img_dir: Path, caption_map: dict[str, tuple[str, str]]) -> tuple[int, int]:
    written = 0
    deleted = 0
    moved = 0

    for folder_name in PLANE_TO_FOLDER.values():
        (img_dir / folder_name).mkdir(exist_ok=True)

    for image_path in sorted(img_dir.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        caption_info = caption_map.get(image_path.name)
        if caption_info is None:
            image_path.unlink()
            deleted += 1
            print(f"Deleted image without description: {image_path.name}")
            continue

        caption, folder_name = caption_info
        target_dir = img_dir / folder_name
        target_image_path = target_dir / image_path.name
        if image_path != target_image_path:
            image_path.replace(target_image_path)
            moved += 1

        root_txt_path = img_dir / f"{image_path.stem}.txt"
        if root_txt_path.exists():
            root_txt_path.unlink()

        txt_path = target_image_path.with_suffix(".txt")
        txt_path.write_text(caption + "\n", encoding="utf-8")
        written += 1

    print(f"Moved {moved} images into plane folders.")
    return written, deleted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create .txt caption files for images using region+plane from JSONL cap field."
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("./brain_descriptions_full.jsonl"),
        help="Path to brain_descriptions_full.jsonl",
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=Path("./fetal_ultrasound/img"),
        help="Folder containing images to caption",
    )
    args = parser.parse_args()

    if not args.img_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {args.img_dir}")

    if not args.jsonl.is_file():
        raise FileNotFoundError(f"JSONL file not found: {args.jsonl}")

    caption_map = build_caption_map(args.jsonl)
    written, deleted = create_txt_files(args.img_dir, caption_map)
    print(f"Done. Wrote {written} txt files. Deleted {deleted} images without descriptions.")


if __name__ == "__main__":
    main()
