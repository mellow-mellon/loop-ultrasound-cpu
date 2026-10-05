"""Case-grouped metadata, pixel auditing, and full-frame ultrasound inputs.

Grouping by released Case prevents paired-view leakage. It does NOT verify the
unreleased Case-to-patient mapping, so these are demonstration research splits.
Ground-truth masks/BBOX/BI-RADS are never used as image inputs or ROI selectors.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

MANIFEST_FIELDS = ["image_id", "case_id", "label", "image_path", "mask_path", "split", "device"]
SPLITS = {"train", "tune", "cal", "test"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def validate_case_groups(rows: list[dict]) -> None:
    """Reject label conflicts, duplicate image IDs, and Case crossing splits."""
    seen = set()
    cases = defaultdict(list)
    for row in rows:
        if not all(field in row for field in MANIFEST_FIELDS):
            raise ValueError(f"Manifest must contain {MANIFEST_FIELDS}")
        if row["image_id"] in seen:
            raise ValueError(f"Duplicate image_id: {row['image_id']}")
        seen.add(row["image_id"])
        if str(row["split"]) not in SPLITS or float(row["label"]) not in (0.0, 1.0):
            raise ValueError("Invalid split or binary pathology label")
        if not str(row["case_id"]).strip():
            raise ValueError("Empty case_id")
        cases[str(row["case_id"])].append(row)
    for case, group in cases.items():
        if len({r["split"] for r in group}) != 1:
            raise ValueError(f"Case {case} crosses splits")
        if len({float(r["label"]) for r in group}) != 1:
            raise ValueError(f"Case {case} has inconsistent pathology labels")


def load_manifest(path: str | Path) -> list[dict]:
    """Read CSV; label becomes int and identifiers remain strings.

    Relative image/mask paths resolve relative to the manifest's directory.
    """
    path = Path(path).resolve()
    rows = _read_csv(path)
    validate_case_groups(rows)
    for row in rows:
        row["label"] = int(float(row["label"]))
        row["case_id"] = str(row["case_id"])
        for key in ("image_path", "mask_path"):
            p = Path(row[key])
            row[key] = str(p if p.is_absolute() else (path.parent / p).resolve())
    return rows


def build_manifest(bus_dir: str | Path, output_path: str | Path,
                   seed: int = 20261004, outer_fold: int = 1) -> list[dict]:
    """Retain official outer fold; remake internal Case-grouped split.

    The remaining ~80% is stratified by pathology into 55:15:10 portions of the
    whole cohort. Official valid_k flags are intentionally ignored, because
    paired Cases cross their train/validation assignments. This is a fixed
    reproducible CPU-demonstration split, not an approved clinical protocol.
    """
    bus_dir, output_path = Path(bus_dir).resolve(), Path(output_path).resolve()
    original = _read_csv(bus_dir / "bus_data.csv")
    official = _read_csv(bus_dir / "5-fold-cv.csv")
    if outer_fold not in range(1, 6):
        raise ValueError("outer_fold must be 1..5")
    by_id = {r["ID"]: r for r in official}
    if len(by_id) != len(official) or len({r["ID"] for r in original}) != len(original):
        raise ValueError("Duplicate original image IDs")
    if set(by_id) != {r["ID"] for r in original}:
        raise ValueError("Metadata and official fold IDs differ")
    cases = defaultdict(list)
    for row in original:
        cases[str(row["Case"])].append(row)
        if row["Pathology"] not in ("benign", "malignant"):
            raise ValueError("Unexpected original pathology label")
        if by_id[row["ID"]]["Pathology"] != row["Pathology"]:
            raise ValueError("Metadata/fold pathology mismatch")
    assignments = {}
    remaining = defaultdict(list)
    for case, group in cases.items():
        folds = {int(by_id[r["ID"]]["kFold"]) for r in group}
        labels = {r["Pathology"] for r in group}
        if len(folds) != 1 or len(labels) != 1:
            raise ValueError(f"Case {case} has inconsistent fold/label")
        if next(iter(folds)) == outer_fold:
            assignments[case] = "test"
        else:
            remaining[next(iter(labels))].append(case)
    rng = random.Random(seed)
    for label in sorted(remaining):
        group = sorted(remaining[label], key=lambda x: (int(x) if x.isdigit() else 0, x))
        rng.shuffle(group)
        n_train = round(len(group) * 55 / 80)
        n_tune = round(len(group) * 15 / 80)
        for i, case in enumerate(group):
            assignments[case] = "train" if i < n_train else "tune" if i < n_train + n_tune else "cal"
    rows = []
    for original_row in original:
        image_id, case = original_row["ID"], str(original_row["Case"])
        rows.append({"image_id": image_id, "case_id": case,
                     "label": int(original_row["Pathology"] == "malignant"),
                     "image_path": str(bus_dir / "Images" / f"{image_id}.png"),
                     "mask_path": str(bus_dir / "Masks" / f"mask{image_id[3:]}.png"),
                     "split": assignments[case], "device": original_row["Device"]})
    validate_case_groups(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "seed": seed, "official_outer_fold": outer_fold,
        "group_key": "Case", "case_to_patient_mapping_verified": False,
        "clinical_protocol_eligibility": "demonstration only; patient mapping unresolved",
        "internal_valid_flags_used": False,
        "target_case_proportions": {"train": 0.55, "tune": 0.15, "cal": 0.10, "test": 0.20},
        "cases_per_split": dict(Counter(assignments.values())),
        "images_per_split": dict(Counter(r["split"] for r in rows)),
        "pathology_by_case_split": {s: dict(Counter(group[0]["Pathology"] for case, group in cases.items()
                                                    if assignments[case] == s)) for s in sorted(SPLITS)},
        "source_metadata_sha256": hashlib.sha256((bus_dir / "bus_data.csv").read_bytes()).hexdigest(),
        "source_folds_sha256": hashlib.sha256((bus_dir / "5-fold-cv.csv").read_bytes()).hexdigest(),
        "warning": "No inference input may include ground-truth mask/BBOX/BI-RADS/pathology.",
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return rows


def binary_mask_array(mask: Image.Image) -> np.ndarray:
    """Accept published 0/1 or 0/255 encodings; reject gray/color interpolation."""
    array = np.asarray(mask)
    if array.ndim == 3:
        rgb = array[:, :, :3]
        if not np.all(rgb == rgb[:, :, :1]):
            raise ValueError("Color-coded mask is not a binary grayscale tumor mask")
        array = rgb[:, :, 0]
    if array.ndim != 2:
        raise ValueError("Tumor mask must be two dimensional")
    values = set(np.unique(array).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError(f"Nonbinary tumor mask values: {sorted(values)}")
    return (array != 0).astype(np.uint8)


def _letterbox_resized_shape(size: tuple[int, int], image_size: int) -> tuple[int, int]:
    if image_size < 1:
        raise ValueError("image_size must be positive")
    width, height = size
    scale = image_size / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def audit_images(rows: list[dict], output_path: str | Path | None = None,
                 image_size: int = 224) -> dict:
    """Read each image/mask, validate metadata/alignment, and hash exact pixels.

    Reports loss of a positive lesion mask after the SAME nearest-neighbor
    downsampling as letterbox_pair. Such geometry remains a valid source pair;
    the resize warning does not filter rows based on pathology/model outcomes.
    Reports exact pixel duplicates across Cases/splits; masks alone being equal
    do not establish image leakage. Near-duplicate/perceptual and clinical
    contour QA remain separate unresolved checks.
    """
    validate_case_groups(rows)
    if image_size < 1:
        raise ValueError("image_size must be positive")
    errors, warnings = [], []
    resized_empty_masks = 0
    image_hashes, mask_hashes = defaultdict(list), defaultdict(list)
    dimensions, mask_values = Counter(), Counter()
    metadata_cache = {}
    valid_count = 0
    for row in rows:
        image_id = row["image_id"]
        try:
            image_path, mask_path = Path(row["image_path"]), Path(row["mask_path"])
            metadata_path = image_path.parent.parent / "bus_data.csv"
            if metadata_path.exists() and metadata_path not in metadata_cache:
                metadata_cache[metadata_path] = {r["ID"]: r for r in _read_csv(metadata_path)}
            with Image.open(image_path) as source:
                source.load()
                original_size = source.size
                pixels = np.asarray(source.convert("RGB"))
            with Image.open(mask_path) as source_mask:
                source_mask.load()
                if source_mask.size != original_size:
                    raise ValueError(f"Image/mask size mismatch: {original_size} vs {source_mask.size}")
                mask_values[str(sorted(np.unique(np.asarray(source_mask)).tolist()))] += 1
                mask = binary_mask_array(source_mask)
            original = metadata_cache.get(metadata_path, {}).get(image_id)
            if original:
                metadata_size = (int(original["Width"]), int(original["Height"]))
                if original_size != metadata_size:
                    # Pixel geometry governs image/mask preprocessing. A CSV
                    # dimension discrepancy does not invalidate an aligned pair
                    # or remove it from the complete duplicate-pixel audit.
                    warnings.append({"image_id": image_id,
                                     "code": "metadata_dimensions_mismatch",
                                     "actual_size": list(original_size),
                                     "metadata_size": list(metadata_size)})
            if not mask.any():
                raise ValueError("Empty tumor mask in a lesion image")
            resized_shape = _letterbox_resized_shape(original_size, image_size)
            resized_mask = np.asarray(Image.fromarray(mask).resize(resized_shape, Image.Resampling.NEAREST))
            if not resized_mask.any():
                resized_empty_masks += 1
                warnings.append({"image_id": image_id,
                                 "code": "lesion_vanished_after_resize",
                                 "image_size": image_size,
                                 "resized_size": list(resized_shape),
                                 "original_positive_pixels": int(mask.sum())})
            if mask.all():
                warnings.append({"image_id": image_id, "warning": "mask covers entire image"})
            dimensions[str(original_size)] += 1
            size_bytes = repr(original_size).encode()
            image_hashes[hashlib.sha256(size_bytes + pixels.tobytes()).hexdigest()].append(row)
            mask_hashes[hashlib.sha256(size_bytes + mask.tobytes()).hexdigest()].append(row)
            valid_count += 1
        except (OSError, ValueError, KeyError) as error:
            errors.append({"image_id": image_id, "error": str(error)})

    def duplicates(groups):
        return [{"sha256": digest, "image_ids": [r["image_id"] for r in group],
                 "case_ids": sorted({str(r["case_id"]) for r in group}),
                 "splits": sorted({r["split"] for r in group}),
                 "cross_case": len({r["case_id"] for r in group}) > 1,
                 "cross_split": len({r["split"] for r in group}) > 1}
                for digest, group in groups.items() if len(group) > 1]
    report = {
        "images_checked": len(rows), "valid_pairs": valid_count,
        "resized_mask_image_size": image_size, "resized_empty_masks": resized_empty_masks,
        "errors": errors, "warnings": warnings,
        "original_dimensions_counts": dict(dimensions), "original_mask_values_counts": dict(mask_values),
        "exact_duplicate_images": duplicates(image_hashes), "exact_duplicate_masks": duplicates(mask_hashes),
        "case_to_patient_mapping_verified": False,
        "near_duplicate_review_completed": False, "expert_contour_review_completed": False,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def letterbox_pair(image: Image.Image, mask: Image.Image, image_size: int = 224) -> dict[str, torch.Tensor]:
    """Keep the full frame and aspect ratio; nearest-neighbor binary mask.

    Padding is invalid for segmentation loss. Outputs are normalized 3xHxW
    image and binary float 1xHxW mask/valid_pixels, all float32.
    """
    if image_size < 1 or image.size != mask.size:
        raise ValueError("Invalid image_size or image/mask size mismatch")
    array = binary_mask_array(mask)
    image = image.convert("RGB")
    resized = _letterbox_resized_shape(image.size, image_size)
    offset = ((image_size - resized[0]) // 2, (image_size - resized[1]) // 2)
    canvas = Image.new("RGB", (image_size, image_size), tuple(round(x * 255) for x in IMAGENET_MEAN))
    canvas.paste(image.resize(resized, Image.Resampling.BILINEAR), offset)
    mask_canvas = Image.new("L", (image_size, image_size), 0)
    mask_canvas.paste(Image.fromarray(array).resize(resized, Image.Resampling.NEAREST), offset)
    valid = np.zeros((image_size, image_size), dtype=np.float32)
    x, y = offset
    valid[y:y + resized[1], x:x + resized[0]] = 1
    image_array = np.array(canvas, dtype=np.float32) / 255
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    return {"image": (image_tensor - mean) / std,
            "mask": torch.from_numpy(np.array(mask_canvas, dtype=np.float32)[None, ...]),
            "valid_pixels": torch.from_numpy(valid[None, ...])}


class UltrasoundDataset(Dataset):
    """Full-image inputs and paired masks; augmentation allowed only on train."""
    def __init__(self, rows: list[dict], image_size: int = 224, augment: bool = False):
        validate_case_groups(rows)
        if augment and any(row["split"] != "train" for row in rows):
            raise ValueError("Augmentation is restricted to training rows")
        self.rows = list(rows)
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        with Image.open(row["mask_path"]) as source:
            # Decode before any transform, preserving 0/1 and 0/255 semantics.
            mask = Image.fromarray(binary_mask_array(source))
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        output = letterbox_pair(image, mask, self.image_size)
        output.update(label=torch.tensor(float(row["label"]), dtype=torch.float32),
                      case_id=str(row["case_id"]), image_id=str(row["image_id"]))
        return output


def balanced_train_rows(rows: list[dict], max_cases_per_class: int | None = None,
                        seed: int = 20261004) -> list[dict]:
    """A small balanced train-only Case subset, retaining every selected view."""
    validate_case_groups(rows)
    groups = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            groups[str(row["case_id"])].append(row)
    by_label = defaultdict(list)
    for case, group in groups.items():
        by_label[int(float(group[0]["label"]))].append(case)
    if not by_label[0] or not by_label[1]:
        raise ValueError("Balanced train subset needs both pathology classes")
    n = min(len(by_label[0]), len(by_label[1]))
    if max_cases_per_class is not None:
        if max_cases_per_class < 1:
            raise ValueError("max_cases_per_class must be positive")
        n = min(n, max_cases_per_class)
    rng, selected = random.Random(seed), set()
    for label in (0, 1):
        candidates = sorted(by_label[label])
        rng.shuffle(candidates)
        selected.update(candidates[:n])
    return [row for row in rows if row["split"] == "train" and str(row["case_id"]) in selected]


def paired_view_cases(rows: list[dict], split: str | None = None) -> dict[str, list[dict]]:
    """Group Cases with >1 view; do not assume l/r indicate breast laterality."""
    validate_case_groups(rows)
    groups = defaultdict(list)
    for row in rows:
        if split is None or row["split"] == split:
            groups[str(row["case_id"])].append(row)
    return {case: group for case, group in groups.items() if len(group) > 1}


def aggregate_case_predictions(image_ids: Iterable[str], probabilities: Iterable[float],
                               rows: list[dict], reduction: str = "mean") -> list[dict]:
    """Aggregate paired views with a declared rule; one output row per Case.

    A Case is accepted only when all its manifest views have predictions.
    Mean is the default; max is available as a separately declared sensitivity
    analysis. Neither rule makes independent-patient claims.
    """
    validate_case_groups(rows)
    if reduction not in ("mean", "max"):
        raise ValueError("reduction must be mean or max")
    ids, probabilities = list(image_ids), list(probabilities)
    if len(ids) != len(probabilities) or len(set(ids)) != len(ids):
        raise ValueError("Prediction IDs must be unique and match probability count")
    lookup = {r["image_id"]: r for r in rows}
    expected = Counter(str(r["case_id"]) for r in rows)
    groups = defaultdict(list)
    for image_id, probability in zip(ids, probabilities):
        if image_id not in lookup or not 0 <= float(probability) <= 1:
            raise ValueError("Unknown prediction image or invalid probability")
        row = lookup[image_id]
        groups[str(row["case_id"])].append((row, float(probability)))
    output = []
    for case, group in sorted(groups.items()):
        if len(group) != expected[case]:
            raise ValueError(f"Missing paired-view prediction for Case {case}")
        values = [p for _, p in group]
        value = sum(values) / len(values) if reduction == "mean" else max(values)
        output.append({"case_id": case, "label": int(float(group[0][0]["label"])),
                       "probability": value, "n_images": len(values), "reduction": reduction})
    return output
