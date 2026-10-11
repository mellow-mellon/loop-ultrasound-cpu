import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from loop_ultrasound import seed_replication as runner


def test_workers_partition_new_seeds_without_reusing_old_seeds():
    expected = {(seed, depth) for seed in (53, 71, 89, 107, 131) for depth in (2, 4)}
    first, second = set(runner.assigned_runs(2, 0)), set(runner.assigned_runs(2, 1))
    assert not first & second
    assert first | second == expected
    assert len(first) == len(second) == 5
    assert set(runner.assigned_runs()) == expected
    assert not {seed for seed, _ in expected} & set(runner.REFERENCE_SEEDS)


@pytest.mark.parametrize("workers,index", [(0, 0), (3, 0), (2, 2), (1, 1), (2, -1)])
def test_invalid_worker_cannot_claim_runs(workers, index):
    with pytest.raises(ValueError):
        runner.assigned_runs(workers, index)


@pytest.fixture
def reference(tmp_path):
    rows = [{"case_id": f"{split}-{i}", "image_id": f"{split}-view-{i}",
             "label": int(i % 3 == 0), "split": split,
             "image_path": f"{split}-image-{i}.png", "mask_path": f"{split}-mask-{i}.png",
             "device": "synthetic"}
            for split, size in (("train", 128), ("tune", 159)) for i in range(size)]
    old = json.loads(Path("configs/cpu_depth_supervision128.json").read_text())
    for split in ("train", "tune"):
        old[split] = runner.counts([row for row in rows if row["split"] == split])
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(old))
    args = SimpleNamespace(threads=1, reference_protocol=str(path), manifest="manifest",
                           encoder_weights="encoder")
    def digest(value):
        if str(value) == "manifest":
            return old["manifest_sha256"]
        if str(value) == "encoder":
            return old["encoder_sha256"]
        return "f" * 64
    with patch.object(runner, "load_manifest", return_value=rows), \
         patch.object(runner, "file_digest", side_effect=digest), \
         patch.object(runner, "training_source_fingerprint", return_value=old["training_source_sha256"]), \
         patch.object(runner, "make_model", side_effect=lambda arm, steps: torch.nn.Linear(2, 2)):
        yield args, old, rows, path


def test_protocol_pairs_initial_weights_and_keeps_primary_new_only(reference):
    args, old, _, _ = reference
    protocol, train, tune = runner.build_protocol(args)
    assert protocol["train"] == old["train"]
    assert protocol["tune"] == old["tune"]
    assert len(train) == 128 and len(tune) == 159
    assert protocol["seeds"] == [53, 71, 89, 107, 131]
    assert protocol["reference_seeds"] == [17, 29, 43]
    assert protocol["supervisions"] == ["terminal"]
    assert "primary_interaction" not in protocol
    assert set(protocol["initialization_sha256_by_seed"]) == {str(seed) for seed in runner.SEEDS}
    assert len(set(protocol["initialization_sha256_by_seed"].values())) == 5
    assert protocol["screening_gate"]["minimum_positive_seeds"] == 4
    assert protocol["screening_gate"]["require_all_leave_one_seed_out_means_positive"] is True
    assert protocol["screening_gate"]["maximum_mean_nll_worsening"] == 0.0
    assert protocol["calibration_and_test_predictions"] is False
    assert protocol["training_updates_per_run"] == 3200


def test_changed_training_source_refused_before_new_training(reference):
    args, _, _, _ = reference
    with patch.object(runner, "training_source_fingerprint", return_value="a" * 64):
        with pytest.raises(ValueError, match="Training code"):
            runner.build_protocol(args)


def test_changed_cohort_refused_before_new_training(reference):
    args, _, rows, _ = reference
    changed = [dict(row) for row in rows]
    changed[0]["image_id"] = "different-view"
    with patch.object(runner, "load_manifest", return_value=changed):
        with pytest.raises(ValueError, match="exactly"):
            runner.build_protocol(args)


@pytest.mark.parametrize("field,value", [("epochs", 25), ("warm_starts", True),
                                         ("augment", True), ("calibration_and_test_predictions", True)])
def test_reference_recipe_mutation_refused(reference, field, value):
    args, old, _, path = reference
    old[field] = value
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError):
        runner.build_protocol(args)
