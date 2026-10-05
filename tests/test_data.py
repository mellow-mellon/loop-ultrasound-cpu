import csv
import hashlib
import json
import stat
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

from loop_ultrasound.data import (
    IMAGENET_MEAN, IMAGENET_STD, UltrasoundDataset, aggregate_case_predictions,
    audit_images, balanced_train_rows, binary_mask_array, build_manifest,
    letterbox_pair, load_manifest, paired_view_cases, validate_case_groups,
)
from loop_ultrasound import download


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_row(tmp_path, image_id="toy", case="1", label=1, split="train", mask_scale=255):
    (tmp_path / "Images").mkdir(exist_ok=True)
    (tmp_path / "Masks").mkdir(exist_ok=True)
    mask = np.zeros((2, 4), dtype=np.uint8)
    mask[:, :2] = 1
    image_path = tmp_path / "Images" / f"{image_id}.png"
    mask_path = tmp_path / "Masks" / f"{image_id}.png"
    Image.fromarray(mask * 255).save(image_path)
    Image.fromarray(mask * mask_scale).save(mask_path)
    return dict(image_id=image_id, case_id=case, label=label, image_path=str(image_path),
                mask_path=str(mask_path), split=split, device="toy-scanner")


@pytest.mark.parametrize("scale", [1, 255])
def test_letterbox_binary_encodings_and_alignment(tmp_path, scale):
    row = sample_row(tmp_path, mask_scale=scale)
    sample = UltrasoundDataset([row], image_size=8)[0]
    assert sample["image"].shape == (3, 8, 8)
    assert sample["mask"].shape == sample["valid_pixels"].shape == (1, 8, 8)
    assert sample["label"].shape == () and sample["label"].dtype == torch.float32
    assert sample["case_id"] == "1" and sample["image_id"] == "toy"
    assert set(sample["mask"].unique().tolist()) == {0.0, 1.0}
    assert sample["valid_pixels"].sum().item() == 32
    assert sample["mask"].sum().item() == 16
    assert not sample["valid_pixels"][0, :2].any()
    assert not sample["valid_pixels"][0, 6:].any()
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    recovered = sample["image"] * std + mean
    image_foreground = (recovered[0] > 0.5) & sample["valid_pixels"][0].bool()
    assert torch.equal(image_foreground, sample["mask"][0].bool())
    assert not (sample["mask"] * (1 - sample["valid_pixels"])).any()


def test_training_flip_transforms_mask_and_image_together(tmp_path, monkeypatch):
    row = sample_row(tmp_path)
    monkeypatch.setattr("loop_ultrasound.data.random.random", lambda: 0.0)
    monkeypatch.setattr("loop_ultrasound.data.random.uniform", lambda a, b: 1.0)
    dataset = UltrasoundDataset([row], image_size=8, augment=True)
    flipped = dataset[0]
    plain = UltrasoundDataset([row], image_size=8)[0]
    assert torch.equal(flipped["mask"], plain["mask"].flip(-1))
    assert torch.equal(flipped["image"], plain["image"].flip(-1))
    test_row = {**row, "split": "test"}
    with pytest.raises(ValueError, match="training"):
        UltrasoundDataset([test_row], augment=True)


def test_reject_interpolated_mask_and_size_mismatch():
    with pytest.raises(ValueError, match="Nonbinary"):
        binary_mask_array(Image.fromarray(np.array([[0, 128, 255]], dtype=np.uint8)))
    with pytest.raises(ValueError, match="mismatch"):
        letterbox_pair(Image.new("L", (3, 2)), Image.new("L", (2, 2)))


def test_case_cross_split_and_label_conflict_rejected(tmp_path):
    a = sample_row(tmp_path, "a")
    b = sample_row(tmp_path, "b", split="test")
    with pytest.raises(ValueError, match="crosses splits"):
        validate_case_groups([a, b])
    with pytest.raises(ValueError, match="inconsistent pathology"):
        validate_case_groups([a, {**b, "split": "train", "label": 0}])


def test_fixed_outer_fold_and_reproducible_group_manifest(tmp_path):
    bus = tmp_path / "BUSBRA"
    bus.mkdir()
    originals, folds = [], []
    for case in range(1, 41):
        label = "malignant" if case % 2 else "benign"
        for suffix in ("l", "r"):
            image_id = f"bus_{case:04d}-{suffix}"
            originals.append(dict(ID=image_id, Case=case, Pathology=label, Device="toy",
                                  Width=4, Height=2))
            # Deliberately leaking official internal flags must be ignored.
            folds.append(dict(ID=image_id, Pathology=label, kFold=(case % 5) + 1,
                              valid_1=int(suffix == "l")))
    write_csv(bus / "bus_data.csv", originals)
    write_csv(bus / "5-fold-cv.csv", folds)
    output = tmp_path / "manifest.csv"
    rows = build_manifest(bus, output, seed=19, outer_fold=1)
    assert len(rows) == 80
    validate_case_groups(rows)
    expected_test = {str(case) for case in range(1, 41) if case % 5 == 0}
    assert {r["case_id"] for r in rows if r["split"] == "test"} == expected_test
    first_bytes = output.read_bytes()
    build_manifest(bus, output, seed=19, outer_fold=1)
    assert output.read_bytes() == first_bytes
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert metadata["case_to_patient_mapping_verified"] is False
    assert metadata["internal_valid_flags_used"] is False
    assert sum(metadata["cases_per_split"].values()) == 40
    assert load_manifest(output) == rows
    subset = balanced_train_rows(rows, max_cases_per_class=2, seed=20)
    assert {r["split"] for r in subset} == {"train"}
    assert len(subset) == 8  # two labels x two Cases x both views
    assert len(paired_view_cases(subset)) == 4


def test_pixel_audit_detects_cross_split_duplicates_and_bad_mask(tmp_path):
    a = sample_row(tmp_path, "a", case="1", split="train")
    b = sample_row(tmp_path, "b", case="2", split="test", mask_scale=1)
    report = audit_images([a, b], tmp_path / "audit.json")
    assert report["valid_pairs"] == 2 and report["errors"] == []
    assert len(report["exact_duplicate_images"]) == 1
    assert report["exact_duplicate_images"][0]["cross_split"] is True
    assert report["case_to_patient_mapping_verified"] is False
    Image.fromarray(np.full((2, 4), 128, dtype=np.uint8)).save(b["mask_path"])
    report = audit_images([a, b])
    assert report["valid_pairs"] == 1
    assert report["errors"][0]["image_id"] == "b"


def test_metadata_dimension_warning_keeps_pair_and_duplicate_audit(tmp_path):
    a = sample_row(tmp_path, "a", case="1", split="train")
    b = sample_row(tmp_path, "b", case="2", split="test")
    write_csv(tmp_path / "bus_data.csv", [dict(ID="a", Width=9, Height=2),
                                          dict(ID="b", Width=4, Height=2)])
    report = audit_images([a, b])
    assert report["valid_pairs"] == 2 and report["errors"] == []
    assert report["warnings"] == [{"image_id": "a", "code": "metadata_dimensions_mismatch",
                                   "actual_size": [4, 2], "metadata_size": [9, 2]}]
    assert report["exact_duplicate_images"][0]["image_ids"] == ["a", "b"]
    assert report["exact_duplicate_images"][0]["cross_split"] is True
    assert report["exact_duplicate_masks"][0]["image_ids"] == ["a", "b"]
    assert sum(report["original_dimensions_counts"].values()) == 2


def test_audit_image_mask_geometry_mismatch_remains_error(tmp_path):
    row = sample_row(tmp_path)
    Image.fromarray(np.ones((3, 4), dtype=np.uint8)).save(row["mask_path"])
    report = audit_images([row])
    assert report["valid_pairs"] == 0 and not report["warnings"]
    assert "Image/mask size mismatch" in report["errors"][0]["error"]


def test_tiny_lesion_vanishing_at_224_warned_without_exclusion(tmp_path):
    row = sample_row(tmp_path)
    tiny = np.zeros((512, 512), dtype=np.uint8)
    tiny[0, 0] = 255
    Image.fromarray(tiny).save(row["image_path"])
    Image.fromarray(tiny).save(row["mask_path"])
    report = audit_images([row])
    assert report["images_checked"] == report["valid_pairs"] == 1
    assert report["errors"] == []
    assert report["resized_mask_image_size"] == 224
    assert report["resized_empty_masks"] == 1
    assert report["warnings"] == [{"image_id": "toy", "code": "lesion_vanished_after_resize",
                                   "image_size": 224, "resized_size": [224, 224],
                                   "original_positive_pixels": 1}]
    # Verify the audit matches the actual input transform rather than an
    # unrelated approximate area heuristic. Original pair remains available.
    dataset = UltrasoundDataset([row], image_size=224)
    assert len(dataset) == 1
    assert dataset[0]["mask"].sum().item() == 0
    assert dataset[0]["valid_pixels"].sum().item() == 224 * 224


def test_pair_prediction_aggregation_requires_all_views(tmp_path):
    a = sample_row(tmp_path, "a")
    b = sample_row(tmp_path, "b")
    rows = [a, b]
    summary = aggregate_case_predictions(["a", "b"], [0.2, 0.8], rows)
    assert summary[0]["probability"] == pytest.approx(0.5)
    assert summary[0]["n_images"] == 2
    assert aggregate_case_predictions(["a", "b"], [0.2, 0.8], rows, "max")[0]["probability"] == 0.8
    with pytest.raises(ValueError, match="Missing paired-view"):
        aggregate_case_predictions(["a"], [0.2], rows)


@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt", "dir\\escape.txt", "C:evil.txt"])
def test_zip_traversal_rejected_before_writes(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("safe.txt", "safe")
        z.writestr(name, "unsafe")
    target = tmp_path / "extract"
    with pytest.raises(ValueError, match="Unsafe"):
        download.safe_extract_zip(archive, target)
    assert not (target / "safe.txt").exists()


def test_zip_symlink_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    member = zipfile.ZipInfo("link")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(member, "../outside")
    with pytest.raises(ValueError, match="symlink"):
        download.safe_extract_zip(archive, tmp_path / "extract")


def test_existing_archive_must_pass_pinned_checksum(tmp_path, monkeypatch):
    archive = tmp_path / "BUSBRA.zip"
    archive.write_bytes(b"not the archive")
    monkeypatch.setattr(download, "BUS_BRA_ARCHIVE_BYTES", archive.stat().st_size)
    with pytest.raises(ValueError, match="size/MD5"):
        download.download_bus_bra(tmp_path)
    assert not (tmp_path / "BUSBRA").exists()


def test_verified_existing_zip_is_reused_and_atomically_extracted(tmp_path, monkeypatch):
    archive = tmp_path / "BUSBRA.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name in ("bus_data.csv", "5-fold-cv.csv", "10-fold-cv.csv"):
            z.writestr("BUSBRA/" + name, "toy")
    monkeypatch.setattr(download, "BUS_BRA_ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(download, "BUS_BRA_MD5", hashlib.md5(archive.read_bytes()).hexdigest())
    monkeypatch.setattr(download, "_complete_bus_tree", lambda p: (p / "bus_data.csv").exists())
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected redownload"))
    target = download.download_bus_bra(tmp_path)
    assert target == tmp_path / "BUSBRA"
    assert (target / "bus_data.csv").read_text() == "toy"
    assert download.download_bus_bra(tmp_path) == target
    assert not list(tmp_path.glob("bus-extract-*"))
