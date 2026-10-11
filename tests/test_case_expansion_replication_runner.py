import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from loop_ultrasound import case_expansion_replication as runner
from loop_ultrasound.cohort import select_training_rows
from loop_ultrasound.depth_supervision import expected_config
from loop_ultrasound.train import initialization_fingerprint


def test_workers_partition_ten_fresh_fits():
    expected = {(seed, depth) for seed in (53, 71, 89, 107, 131) for depth in (2, 4)}
    first, second = set(runner.assigned_runs(2, 0)), set(runner.assigned_runs(2, 1))
    assert not first & second and first | second == expected
    assert len(first) == len(second) == 5
    assert set(runner.assigned_runs()) == expected


@pytest.mark.parametrize("workers,index", [(0, 0), (3, 0), (2, 2), (1, 1), (2, -1)])
def test_invalid_worker_refused(workers, index):
    with pytest.raises(ValueError):
        runner.assigned_runs(workers, index)


@pytest.fixture
def locked(tmp_path):
    rows = [{"case_id": f"{split}-{i}", "image_id": f"{split}-view-{i}",
             "label": int(i % 3 == 0), "split": split, "image_path": f"{split}-image-{i}.png",
             "mask_path": f"{split}-mask-{i}.png", "device": "synthetic"}
            for split, size in (("train", 256), ("tune", 159)) for i in range(size)]
    old = json.loads(Path("configs/cpu_seed_replication128.json").read_text())
    original = json.loads(Path("configs/cpu_case_expansion256.json").read_text())
    old["train"] = runner.counts(select_training_rows(rows, 128))
    old["tune"] = runner.counts([r for r in rows if r["split"] == "tune"])
    original["reference_train"] = copy.deepcopy(old["train"])
    original["train"] = runner.counts(select_training_rows(rows, 256))
    original["tune"] = copy.deepcopy(old["tune"])
    reference_dir = tmp_path / "reference"
    for seed in runner.SEEDS:
        torch.manual_seed(seed)
        digest = initialization_fingerprint(torch.nn.Linear(2, 2))
        old["initialization_sha256_by_seed"][str(seed)] = digest
        for depth in runner.DEPTHS:
            directory = reference_dir / f"SJ-depth{depth}-terminal-seed{seed}"
            directory.mkdir(parents=True)
            summary = {"status": "executed_cpu_engineering_pilot", "clinical_validation": False,
                       "device": "cpu", "evaluation_split": "tune_development",
                       "model_selection": "fixed_final_epoch_no_best_epoch_selection",
                       "config": expected_config(seed, depth, "terminal"), "initialization_sha256": digest,
                       "timing": {"updates": 3200}, "n_training_cases": 128,
                       "n_training_images": old["train"]["images"],
                       **{key: old[key] for key in ("training_source_sha256", "encoder_sha256", "manifest_sha256")},
                       "cohort_selection": {"train_fingerprint": old["train"]["fingerprint"],
                                            "tune_fingerprint": old["tune"]["fingerprint"]}}
            (directory / "summary.json").write_text(json.dumps(summary))
    old_path, original_path = tmp_path / "old.json", tmp_path / "original.json"
    old_path.write_text(json.dumps(old))
    original_path.write_text(json.dumps(original))
    baseline_dirs = []
    for name, record in (("base128", old), ("base256", original)):
        directory = tmp_path / name
        directory.mkdir()
        summary = {"sources": {key: record[key] for key in ("encoder_sha256", "manifest_sha256")},
                   "cohort_fingerprints": {split: record[split]["fingerprint"] for split in ("train", "tune")}}
        (directory / "baselines_summary.json").write_text(json.dumps(summary))
        (directory / "baseline_case_predictions.csv").write_text("synthetic fixture\n")
        baseline_dirs.append(directory)
    args = runner.parser().parse_args(["--reference-protocol", str(old_path),
        "--original-expansion-protocol", str(original_path), "--reference-run-dir", str(reference_dir),
        "--baseline-dir", str(baseline_dirs[1]), "--reference-baseline-dir", str(baseline_dirs[0]),
        "--manifest", "manifest", "--encoder-weights", "encoder", "--threads", "1"])
    def digest(value):
        if str(value) == "manifest": return old["manifest_sha256"]
        if str(value) == "encoder": return old["encoder_sha256"]
        return hashlib.sha256(Path(value).read_bytes()).hexdigest()
    with patch.object(runner, "load_manifest", return_value=rows), \
         patch.object(runner, "file_digest", side_effect=digest), \
         patch.object(runner, "training_source_fingerprint", return_value=old["training_source_sha256"]), \
         patch.object(runner, "make_model", side_effect=lambda arm, steps: torch.nn.Linear(2, 2)):
        yield args, old, original, old_path, original_path


def test_new_five_preserve_nested_budget_and_separate_history(locked):
    args, old, original, _, _ = locked
    protocol, train, tune = runner.build_protocol(args)
    assert protocol["seeds"] == protocol["reference_seeds"] == list(runner.SEEDS)
    assert protocol["initialization_sha256_by_seed"] == old["initialization_sha256_by_seed"]
    assert protocol["train"] == original["train"] and len(train) == 256 and len(tune) == 159
    assert protocol["epochs"] * protocol["updates_per_epoch"] == 3200
    assert protocol["reference_epochs"] * protocol["reference_updates_per_epoch"] == 3200
    assert protocol["historical_seed_scope"] == "original_three_secondary_only"
    assert protocol["calibration_and_test_predictions"] is False
    assert "screening_gate" not in protocol
    assert protocol["launcher_source_sha256"] == runner.launcher_digest()
    assert protocol["shared_runner_source_sha256"] == runner.case_expansion.launcher_digest()
    assert len(protocol["baseline_artifact_bindings"]["new256"]["predictions_sha256"]) == 64


@pytest.mark.parametrize("record,key,value", [
    ("old", "epochs", 25), ("original", "epochs", 50),
    ("old", "seeds", [17, 29, 43]), ("original", "seeds", [53, 71, 89, 107, 131]),
    ("old", "calibration_and_test_predictions", True), ("original", "learning_rate", .001),
    ("original", "warm_starts", True), ("old", "training_updates_per_run", 3200.0),
    ("original", "training_source_sha256", "a" * 64),
])
def test_changed_locked_reference_refused(locked, record, key, value):
    args, old, original, old_path, original_path = locked
    data, path = (old, old_path) if record == "old" else (original, original_path)
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): runner.build_protocol(args)


@pytest.mark.parametrize("key,value", [("n_training_cases", 256), ("initialization_sha256", "a" * 64),
                                       ("training_source_sha256", "b" * 64)])
def test_changed_actual_reference_fit_refused(locked, key, value):
    args, *_ = locked
    path = Path(args.reference_run_dir) / "SJ-depth2-terminal-seed53/summary.json"
    summary = json.loads(path.read_text())
    summary[key] = value
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError): runner.build_protocol(args)


def test_changed_baseline_cohort_refused(locked):
    args, *_ = locked
    path = Path(args.baseline_dir) / "baselines_summary.json"
    summary = json.loads(path.read_text())
    summary["cohort_fingerprints"]["train"] = "a" * 64
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="baseline cohort/source"): runner.build_protocol(args)


def test_changed_initialization_refused(locked):
    args, old, _, path, _ = locked
    old["initialization_sha256_by_seed"]["53"] = "a" * 64
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError): runner.build_protocol(args)
