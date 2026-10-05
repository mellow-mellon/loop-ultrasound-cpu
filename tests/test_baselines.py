"""Train-only and Case-weighted controls, rather than image-weighted fitting."""
import csv
import json

import numpy as np
import pytest
import torch

import loop_ultrasound.baselines as module
from loop_ultrasound.baselines import cohort_fingerprint, run_baselines


def row(case, image, label, split="train"):
    return {"case_id": case, "image_id": image, "label": label, "split": split,
            "image_path": "/private/unpublished/image.png"}


def features(values):
    # Fixed feature dimensionality is part of this particular frozen encoder.
    return torch.tensor(values, dtype=torch.float32)[:, None, None].expand(-1, 196, 192).clone()


@pytest.fixture
def cohorts():
    train = [row("a", "a1", 0), row("a", "a2", 0), row("b", "b1", 1),
             row("c", "c1", 0), row("d", "d1", 1)]
    tune = [row("e", "e1", 0, "tune"), row("f", "f1", 1, "tune")]
    return train, tune, features([0., 2., 5., 0., 7.]), features([100., 200.])


def test_scaler_fits_only_case_averaged_train_features(cohorts, tmp_path, monkeypatch):
    train, tune, train_x, tune_x = cohorts
    original_scaler = module.StandardScaler
    calls = {}

    class SpyScaler(original_scaler):
        def fit(self, x, y=None, sample_weight=None):
            calls["fit"] = x.copy()
            return super().fit(x, y=y, sample_weight=sample_weight)

        def transform(self, x, copy=None):
            calls.setdefault("transforms", []).append(x.copy())
            return super().transform(x, copy=copy)

    monkeypatch.setattr(module, "StandardScaler", SpyScaler)
    result = run_baselines(train, tune, train_x, tune_x, tmp_path)
    np.testing.assert_array_equal(calls["fit"][:, 0], [1., 5., 0., 7.])
    assert len(calls["fit"]) == 4  # Case a has two images but one fitting row.
    np.testing.assert_array_equal(calls["transforms"][-1][:, 0], [100., 200.])
    assert result["method"]["scaler_fit_split"] == "train"
    assert result["method"]["hyperparameters_selected_using_tune"] is False


def test_constant_uses_case_prevalence_not_image_prevalence(tmp_path):
    train = [row("a", "a1", 1), row("a", "a2", 1), row("a", "a3", 1),
             row("b", "b1", 0), row("c", "c1", 0), row("d", "d1", 0)]
    tune = [row("e", "e1", 0, "tune"), row("f", "f1", 1, "tune")]
    result = run_baselines(train, tune, features([0.]*6), features([0.]*2), tmp_path)
    assert result["method"]["train_case_prevalence"] == .25
    assert result["baselines"]["constant"]["tune"]["brier"] == .3125
    assert result["baselines"]["constant"]["tune"]["auroc"] == .5
    with (tmp_path / "baseline_case_predictions.csv").open() as stream:
        records = list(csv.DictReader(stream))
    constant = [r for r in records if r["baseline"] == "constant"]
    assert len(constant) == 6
    assert {float(r["probability"]) for r in constant} == {.25}
    assert int(next(r for r in constant if r["case_id"] == "a")["n_images"]) == 3


def test_tune_labels_do_not_change_predictions(cohorts, tmp_path):
    train, tune, train_x, tune_x = cohorts
    original = tmp_path / "original"
    relabeled = tmp_path / "relabeled"
    run_baselines(train, tune, train_x, tune_x, original)
    changed_tune = [dict(r, label=1-r["label"]) for r in tune]
    run_baselines(train, changed_tune, train_x, tune_x, relabeled)
    outputs = []
    for path in (original, relabeled):
        with (path / "baseline_case_predictions.csv").open() as stream:
            outputs.append({(r["baseline"], r["split"], r["case_id"]): r["probability"]
                            for r in csv.DictReader(stream)})
    assert outputs[0] == outputs[1]


def test_public_summary_contains_aggregates_and_no_case_ids_or_paths(cohorts, tmp_path):
    train, tune, train_x, tune_x = cohorts
    result = run_baselines(train, tune, train_x, tune_x, tmp_path, selection_seed=42)
    text = (tmp_path / "baselines_summary.json").read_text()
    assert json.loads(text) == result
    assert result["selection_seed"] == 42
    assert result["clinical_validation"] is False
    assert result["device"] == "cpu"
    assert result["cohorts"]["train"] == {"n_cases": 4, "n_images": 5, "n_positive": 2, "n_negative": 2}
    assert "/private/" not in text and "\"case_id\":" not in text
    assert "\"image_id\":" not in text
    for baseline in ("constant", "logistic_regression"):
        assert result["baselines"][baseline]["tune"]["threshold"] == .5


def test_fingerprint_ignores_paths_and_order_but_tracks_labels(cohorts):
    train = cohorts[0]
    digest = cohort_fingerprint(train)
    assert len(digest) == 64
    assert digest == cohort_fingerprint([dict(r, image_path="elsewhere") for r in reversed(train)])
    changed = [dict(r, label=1-r["label"]) for r in train]
    assert digest != cohort_fingerprint(changed)


@pytest.mark.parametrize("corruption,match", [
    ("duplicate_image", "Duplicate image"),
    ("conflicting_case", "Conflicting pathology"),
    ("overlap_case", "Case IDs must be disjoint"),
    ("overlap_image", "image IDs must be disjoint"),
    ("test_split", "Only split"),
    ("cal_split", "Only split"),
    ("nonbinary_label", "finite binary"),
    ("single_class", "both pathology"),
    ("missing_field", "require"),
    ("empty_id", "cannot be empty"),
])
def test_rejects_invalid_cohorts_without_writing(cohorts, tmp_path, corruption, match):
    train, tune, train_x, tune_x = cohorts
    train, tune = [dict(r) for r in train], [dict(r) for r in tune]
    if corruption == "duplicate_image":
        train[1]["image_id"] = train[0]["image_id"]
    elif corruption == "conflicting_case":
        train[1]["label"] = 1
    elif corruption == "overlap_case":
        tune[0]["case_id"] = train[0]["case_id"]
    elif corruption == "overlap_image":
        tune[0]["image_id"] = train[0]["image_id"]
    elif corruption == "test_split":
        tune[0]["split"] = "test"
    elif corruption == "cal_split":
        train[0]["split"] = "cal"
    elif corruption == "nonbinary_label":
        train[0]["label"] = .5
    elif corruption == "single_class":
        train = [dict(r, label=0) for r in train]
    elif corruption == "missing_field":
        del train[0]["image_id"]
    elif corruption == "empty_id":
        train[0]["case_id"] = " "
    with pytest.raises(ValueError, match=match):
        run_baselines(train, tune, train_x, tune_x, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("bad_features,match", [
    (torch.zeros(5, 195, 192), "shaped"),
    (torch.zeros(5, 196, 192, dtype=torch.int64), "floating-point"),
    (torch.full((5, 196, 192), float("nan")), "finite"),
    (torch.zeros(5, 196, 192, device="meta"), "CPU torch"),
    (np.zeros((5, 196, 192)), "CPU torch"),
])
def test_rejects_misaligned_nonfinite_or_noncpu_features(cohorts, tmp_path, bad_features, match):
    train, tune, _, tune_x = cohorts
    with pytest.raises(ValueError, match=match):
        run_baselines(train, tune, bad_features, tune_x, tmp_path)


def test_completed_results_are_not_overwritten(cohorts, tmp_path):
    train, tune, train_x, tune_x = cohorts
    run_baselines(train, tune, train_x, tune_x, tmp_path)
    before = (tmp_path / "baselines_summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_baselines(train, tune, train_x, tune_x, tmp_path)
    assert (tmp_path / "baselines_summary.json").read_bytes() == before
