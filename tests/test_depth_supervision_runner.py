"""No overlapping worker assignments or mismatched resumed experiment."""
from copy import deepcopy

import pytest

from loop_ultrasound.depth_supervision import assigned_runs, expected_config, validate_completed


def fixture():
    protocol = {"training_source_sha256": "a"*64, "encoder_sha256": "b"*64,
                "manifest_sha256": "c"*64, "initialization_sha256_by_seed": {"17": "d"*64},
                "train": {"fingerprint": "e"*64}, "tune": {"fingerprint": "f"*64}}
    summary = {"status": "executed_cpu_engineering_pilot", "clinical_validation": False,
               "device": "cpu", "model_selection": "fixed_final_epoch_no_best_epoch_selection",
               "evaluation_split": "tune_development", "config": expected_config(17, 2, "terminal"),
               "initialization_sha256": "d"*64, "timing": {"updates": 3200},
               "cohort_selection": {"train_fingerprint": "e"*64, "tune_fingerprint": "f"*64},
               **{k: protocol[k] for k in ("training_source_sha256", "encoder_sha256", "manifest_sha256")}}
    return summary, protocol


def test_worker_partitions_cover_exactly_twelve_distinct_runs():
    complete = set(assigned_runs())
    left, right = set(assigned_runs(2, 0)), set(assigned_runs(2, 1))
    assert len(left) == len(right) == 6
    assert not left & right and left | right == complete and len(complete) == 12


@pytest.mark.parametrize("workers,index", [(0, 0), (3, 0), (2, 2), (2, -1), (1, 1)])
def test_worker_rejects_invalid_partition(workers, index):
    with pytest.raises(ValueError):
        assigned_runs(workers, index)


def test_completed_matching_run_is_reusable():
    summary, protocol = fixture()
    assert validate_completed(summary, protocol, 17, 2, "terminal") is summary


@pytest.mark.parametrize("field", ["initialization", "source", "cohort", "updates", "supervision", "epoch"])
def test_reuse_rejects_changed_fit_provenance(field):
    summary, protocol = fixture()
    if field == "initialization":
        summary["initialization_sha256"] = "0"*64
    elif field == "source":
        summary["training_source_sha256"] = "0"*64
    elif field == "cohort":
        summary["cohort_selection"]["train_fingerprint"] = "0"*64
    elif field == "updates":
        summary["timing"]["updates"] = 3199
    elif field == "supervision":
        summary["config"]["supervision"] = "all"
    else:
        summary["config"]["epochs"] = 49
    with pytest.raises(ValueError):
        validate_completed(summary, protocol, 17, 2, "terminal")
