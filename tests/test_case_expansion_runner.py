import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from loop_ultrasound import case_expansion as runner
from loop_ultrasound.cohort import select_training_rows
from loop_ultrasound.train import initialization_fingerprint


def test_workers_partition_six_paired_fits():
    expected = {(seed, depth) for seed in (17, 29, 43) for depth in (2, 4)}
    first, second = set(runner.assigned_runs(2, 0)), set(runner.assigned_runs(2, 1))
    assert not first & second and first | second == expected
    assert len(first) == len(second) == 3
    assert set(runner.assigned_runs()) == expected


@pytest.mark.parametrize("workers,index", [(0, 0), (3, 0), (2, 2), (1, 1), (2, -1)])
def test_invalid_worker_refused(workers, index):
    with pytest.raises(ValueError):
        runner.assigned_runs(workers, index)


def test_nested_check_requires_cases_views_and_labels():
    reference = [{"case_id": "a", "image_id": "view1", "label": 1},
                 {"case_id": "a", "image_id": "view2", "label": 1}]
    larger = reference + [{"case_id": "b", "image_id": "view3", "label": 0}]
    assert all(runner.nested_check(reference, larger).values())
    with pytest.raises(ValueError):
        runner.nested_check(reference, [larger[0], larger[2]])
    bad = copy.deepcopy(larger)
    bad[0]["label"] = 0
    with pytest.raises(ValueError):
        runner.nested_check(reference, bad)


@pytest.fixture
def reference(tmp_path):
    rows = [{"case_id": f"{split}-{i}", "image_id": f"{split}-view-{i}",
             "label": int(i % 3 == 0), "split": split, "image_path": f"{split}-image-{i}.png",
             "mask_path": f"{split}-mask-{i}.png", "device": "synthetic"}
            for split, size in (("train", 256), ("tune", 159)) for i in range(size)]
    old = json.loads(Path("configs/cpu_depth_supervision128.json").read_text())
    old["train"] = runner.counts(select_training_rows(rows, 128))
    old["tune"] = runner.counts([r for r in rows if r["split"] == "tune"])
    for seed in runner.SEEDS:
        torch.manual_seed(seed)
        old["initialization_sha256_by_seed"][str(seed)] = initialization_fingerprint(torch.nn.Linear(2, 2))
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(old))
    args = SimpleNamespace(threads=1, reference_protocol=str(path), manifest="manifest", encoder_weights="encoder")
    def digest(value):
        return old["manifest_sha256"] if str(value) == "manifest" else old["encoder_sha256"] if str(value) == "encoder" else "f" * 64
    with patch.object(runner, "load_manifest", return_value=rows), \
         patch.object(runner, "file_digest", side_effect=digest), \
         patch.object(runner, "training_source_fingerprint", return_value=old["training_source_sha256"]), \
         patch.object(runner, "make_model", side_effect=lambda arm, steps: torch.nn.Linear(2, 2)):
        yield args, old, rows, path


def test_locked_budget_matches_updates_not_epochs(reference):
    args, old, _, _ = reference
    protocol, train, tune = runner.build_protocol(args)
    assert protocol["reference_train"] == old["train"]
    assert protocol["train"]["cases"] == len(train) == 256 and len(tune) == 159
    assert protocol["epochs"] == 25 and protocol["reference_epochs"] == 50
    assert protocol["updates_per_epoch"] * protocol["epochs"] == 3200
    assert protocol["reference_updates_per_epoch"] * protocol["reference_epochs"] == 3200
    assert protocol["matched_update_points"] == [640, 1280, 1920, 2560, 3200]
    assert protocol["nested_cohort_verified"] is True
    assert protocol["initialization_sha256_by_seed"] == old["initialization_sha256_by_seed"]
    assert protocol["supervisions"] == ["terminal"]
    assert protocol["calibration_and_test_predictions"] is False


def test_changed_source_refused(reference):
    args, _, _, _ = reference
    with patch.object(runner, "training_source_fingerprint", return_value="a" * 64):
        with pytest.raises(ValueError, match="Training source"):
            runner.build_protocol(args)


def test_unpaired_reference_initialization_refused(reference):
    args, old, _, path = reference
    old["initialization_sha256_by_seed"]["17"] = "a" * 64
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="Initial parameters"):
        runner.build_protocol(args)


def test_completed_fit_cannot_silently_reuse_old_50_epoch_recipe(reference):
    args, _, _, _ = reference
    p, _, _ = runner.build_protocol(args)
    s = {"status": "executed_cpu_engineering_pilot", "clinical_validation": False, "device": "cpu",
         "evaluation_split": "tune_development", "model_selection": "fixed_final_epoch_no_best_epoch_selection",
         "config": runner.expected_config(17, 2), "initialization_sha256": p["initialization_sha256_by_seed"]["17"],
         "timing": {"updates": 3200}, "n_training_cases": 256, "n_training_images": p["train"]["images"],
         **{key: p[key] for key in ("training_source_sha256", "encoder_sha256", "manifest_sha256")},
         "cohort_selection": {"train_fingerprint": p["train"]["fingerprint"], "tune_fingerprint": p["tune"]["fingerprint"]}}
    assert runner.validate_completed(s, p, 17, 2) == s
    s["config"]["epochs"] = 50
    with pytest.raises(ValueError):
        runner.validate_completed(s, p, 17, 2)
