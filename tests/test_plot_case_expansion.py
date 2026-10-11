"""Protocol/alignment guards for aggregate expansion figures, without rendering."""
import copy
import hashlib
import json

import numpy as np
import pytest

from loop_ultrasound.plot_case_expansion import (
    load_reports, _matched_history, _seed_effects, SEEDS, REPLICATION_SEEDS,
)


def _fixture(tmp_path, seeds=REPLICATION_SEEDS):
    new, reference = tmp_path / "new", tmp_path / "reference"
    new.mkdir(); reference.mkdir()
    config = {"arm": "SJ", "supervision": "terminal", "batch_size": 2,
              "learning_rate": .0003, "segmentation_weight": 1.,
              "cached_frozen_features": True, "augment": False,
              "selection_seed": 20261004, "sampling": "proportional", "eval_every": 5}
    sources = {key: letter*64 for key, letter in (("encoder_sha256", "a"),
               ("manifest_sha256", "b"), ("training_source_sha256", "c"))}
    init = {str(seed): hashlib.sha256(str(seed).encode()).hexdigest() for seed in seeds}
    train = {"cases": 256, "images": 445, "benign": 174, "malignant": 82, "fingerprint": "d"*64}
    reference_train = {"cases": 128, "images": 227, "benign": 87, "malignant": 41, "fingerprint": "e"*64}
    tune = {"cases": 159, "images": 281, "benign": 108, "malignant": 51, "fingerprint": "f"*64}
    protocol = {**{key: value for key, value in config.items() if key not in {"arm", "supervision"}},
                **sources, "status": "fixed_before_case_expansion_seed_replication_training" if seeds == REPLICATION_SEEDS
                else "fixed_before_case_expansion_training", "seeds": list(seeds), "depths": [2, 4],
                "supervisions": ["terminal"], "arms": ["SJ"], "device": "cpu", "clinical_validation": False,
                "calibration_and_test_predictions": False, "warm_starts": False,
                "patient_mapping": "unverified_Case_grouped_only",
                "model_selection": "fixed_final_epoch_no_best_epoch_selection", "epochs": 25,
                "reference_epochs": 50, "updates_per_epoch": 128, "reference_updates_per_epoch": 64,
                "training_updates_per_run": 3200, "matched_update_points": [640, 1280, 1920, 2560, 3200],
                "nested_cohort_verified": True, "reference_protocol_sha256": "1"*64,
                "original_expansion_protocol_sha256": "2"*64,
                "nested_cohort_check": {"reference_cases_subset": True, "reference_images_subset": True,
                                        "shared_view_labels_match": True},
                "initialization_sha256_by_seed": init, "train": train, "reference_train": reference_train,
                "tune": tune}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    for size, directory, cohort, epochs in ((128, reference, reference_train, 50), (256, new, train, 25)):
        for depth in (2, 4):
            for index, seed in enumerate(seeds):
                history = []
                for epoch in range(5, epochs+1, 5):
                    by_step = {str(step): {"n_cases": 159, "n_positive": 51, "n_negative": 108,
                                          "brier": .2 + index*.01 + step*.001 + epoch*.00001 - (.015 if size == 256 else 0),
                                          "auroc": .75 + index*.005, "nll": .7, "mean_case_dice": .72}
                               for step in range(1, depth+1)}
                    history.append({"epoch": epoch, "by_step": by_step})
                run = {**sources, "status": "executed_cpu_engineering_pilot", "clinical_validation": False,
                       "device": "cpu", "patient_mapping": "unverified_Case_grouped_only",
                       "evaluation_split": "tune_development",
                       "model_selection": "fixed_final_epoch_no_best_epoch_selection",
                       "config": {**config, "steps": depth, "seed": seed, "epochs": epochs},
                       "n_training_cases": size, "n_training_images": cohort["images"],
                       "timing": {"updates": 3200}, "initialization_sha256": init[str(seed)],
                       "cohort_selection": {"sampling": "proportional", "selection_seed": 20261004,
                                            "train_fingerprint": cohort["fingerprint"], "tune_fingerprint": tune["fingerprint"]},
                       "readout_supervision": {"mode": "terminal", "directly_supervised_steps": [depth],
                                               "intermediate_predictions_scope": "diagnostic_only_no_direct_intermediate_readout_loss"},
                       "development_history": history, "development_subset": copy.deepcopy(history[-1]["by_step"])}
                (directory / f"SJ-depth{depth}-terminal-seed{seed}.json").write_text(json.dumps(run))
    return new, reference, protocol_path


@pytest.mark.parametrize("seeds", [SEEDS, REPLICATION_SEEDS])
def test_accepts_separate_original_and_replication_protocols(tmp_path, seeds):
    new, reference, protocol = _fixture(tmp_path, seeds)
    runs = load_reports(new, reference, protocol_path=protocol)
    assert len(runs) == 4*len(seeds)
    assert {key[2] for key in runs} == set(seeds)


def test_replication_does_not_read_or_pool_original_seeds(tmp_path):
    new, reference, protocol = _fixture(tmp_path)
    for directory in (new, reference):
        (directory / "SJ-depth2-terminal-seed17.json").write_text("invalid old report deliberately ignored")
    runs = load_reports(new, reference, protocol_path=protocol)
    assert len(runs) == 20
    assert {key[2] for key in runs} == set(REPLICATION_SEEDS)


def test_curves_match_updates_rather_than_equal_epoch_numbers(tmp_path):
    new, reference, protocol = _fixture(tmp_path)
    runs = load_reports(new, reference, protocol_path=protocol)
    reference_values = _matched_history(runs[128, 2, 53], 128, 2, "brier")
    new_values = _matched_history(runs[256, 2, 53], 256, 2, "brier")
    assert reference_values[0] == pytest.approx(.2021)  # Reference epoch 10 -> 640 updates.
    assert new_values[0] == pytest.approx(.18705)  # New epoch 5 -> 640 updates.
    assert reference_values[-1] == pytest.approx(.2025)  # Reference epoch 50.
    assert new_values[-1] == pytest.approx(.18725)  # New epoch 25.


@pytest.mark.parametrize("change", ["epochs", "updates", "source", "initialization", "cohort", "final_metric", "supervision"])
def test_rejects_mispaired_or_incomplete_runs(tmp_path, change):
    new, reference, protocol = _fixture(tmp_path)
    path = new / "SJ-depth4-terminal-seed53.json"
    run = json.loads(path.read_text())
    if change == "epochs": run["config"]["epochs"] = 50
    elif change == "updates": run["timing"]["updates"] = 3199
    elif change == "source": run["training_source_sha256"] = "0"*64
    elif change == "initialization": run["initialization_sha256"] = "0"*64
    elif change == "cohort": run["cohort_selection"]["train_fingerprint"] = "0"*64
    elif change == "final_metric": run["development_subset"]["4"]["brier"] = .1
    elif change == "supervision": run["readout_supervision"]["directly_supervised_steps"] = [1, 2, 3, 4]
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError): load_reports(new, reference, protocol_path=protocol)


@pytest.mark.parametrize("change", ["old_seed_set", "nested", "reference_epoch", "update_grid", "missing_original_provenance"])
def test_replication_rejects_changed_protocol(tmp_path, change):
    new, reference, protocol = _fixture(tmp_path)
    value = json.loads(protocol.read_text())
    if change == "old_seed_set": value["seeds"] = list(SEEDS)
    elif change == "nested": value["nested_cohort_verified"] = False
    elif change == "reference_epoch": value["reference_epochs"] = 25
    elif change == "update_grid": value["matched_update_points"] = [320, 640, 960, 1280, 1600]
    elif change == "missing_original_provenance": value.pop("original_expansion_protocol_sha256")
    protocol.write_text(json.dumps(value))
    with pytest.raises(ValueError): load_reports(new, reference, protocol_path=protocol)


def test_seed_effect_signs_and_pairing():
    values = {(128, 2): [.25, .30], (128, 4): [.20, .35],
              (256, 2): [.22, .25], (256, 4): [.18, .21]}
    runs = {(size, depth, seed): {"development_subset": {str(depth): {"brier": numbers[index]}}}
            for (size, depth), numbers in values.items() for index, seed in enumerate((53, 71))}
    gains, data = _seed_effects(runs, (53, 71))
    np.testing.assert_allclose(gains[128], [.05, -.05])
    np.testing.assert_allclose(gains[256], [.04, .04])
    np.testing.assert_allclose(gains[256]-gains[128], [-.01, .09])
    np.testing.assert_allclose(data[2], [.03, .05])
    np.testing.assert_allclose(data[4], [.02, .14])
