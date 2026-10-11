"""OOF leakage prevention, locked recipes/artifacts, numeric policy and privacy."""
import copy
import hashlib
import json

import numpy as np
import pytest

from loop_ultrasound.analyze_expanded import _fingerprint
from loop_ultrasound.metrics import classification_metrics
from loop_ultrasound.temperature_diagnostic import (
    LOG_T_BOUNDS,assign_folds,build_protocol,calibrate_probabilities,case_logits,
    cross_fit_temperature,execute,fit_temperature,main,prepare,
)
from test_analyze_case_expansion import experiment,replication
from test_analyze_seed_replication import csv_edit,csv_read,csv_write,json_edit,metrics


@pytest.fixture
def source(replication):
    """Ten tune Cases, with shared views and labels in every source run."""
    for directory in replication["runs"]+replication["old_runs"]:
        fields,rows = csv_read(directory/"development_predictions.csv")
        expanded = copy.deepcopy(rows)
        depth = json.loads((directory/"summary.json").read_text())["config"]["steps"]
        for step in range(1,depth+1):
            for label in (0,1):
                prototype = next(row for row in rows if int(row["step"])==step and int(row["label"])==label)
                for index,p in enumerate((.05,.95,.2,.35) if label==0 else (.95,.05,.8,.65)):
                    candidate = copy.deepcopy(prototype)
                    candidate.update(case_id=f"private_tune_extra_case{label}_{index}",
                                     image_id=f"private_tune_extra_image{label}_{index}",probability=p)
                    expanded.append(candidate)
        csv_write(directory/"development_predictions.csv",expanded,fields)
        observed = metrics(expanded,depth)
        summary = json.loads((directory/"summary.json").read_text())
        summary["development_subset"] = observed
        for row in summary["development_history"]:
            row["by_step"] = copy.deepcopy(observed)
        (directory/"summary.json").write_text(json.dumps(summary))
    _,rows = csv_read(replication["runs"][0]/"development_predictions.csv")
    index = {}
    for row in rows:
        if int(row["step"])!=1:
            continue
        case = row["case_id"]
        entry = index.setdefault(case,{"label":int(row["label"]),"image_ids":[]})
        entry["image_ids"].append(row["image_id"])
    tune = {"cases":10,"images":11,"benign":5,"malignant":5,"fingerprint":_fingerprint(index)}
    json_edit(replication["old_protocol"],lambda p:p.update(tune=tune))
    json_edit(replication["protocol"],lambda p:p.update(tune=tune,
              reference_protocol_sha256=hashlib.sha256(replication["old_protocol"].read_bytes()).hexdigest()))
    for directory in replication["runs"]+replication["old_runs"]:
        json_edit(directory/"summary.json",lambda s:s["cohort_selection"].update(tune_fingerprint=tune["fingerprint"]))
    return replication


def source_kwargs(source,*,secondary=True):
    result = {"source_protocol_path":source["protocol"],"expected_training_cases":4,
              "expected_reference_training_cases":2,"expected_development_cases":10}
    if secondary:
        result.update(reference_run_dirs=source["old_runs"],reference_protocol_path=source["old_protocol"])
    return result


def test_identity_candidate_returns_exact_original_probability_copy():
    p = np.array([0.,.1,.5,.9,1.])
    before = p.copy()
    output = calibrate_probabilities(p,1.)
    assert np.array_equal(output,before)
    assert np.array_equal(p,before)
    assert not np.shares_memory(output,p)


def test_uninformative_scores_select_exact_identity_and_preserve_ties():
    fit = fit_temperature([0,1,0,1],[.5,.5,.5,.5])
    assert fit["temperature"]==1.
    assert fit["selected_identity"] is True
    assert fit["fit_nll_stable_softplus"]==pytest.approx(np.log(2))


def test_fixed_temperature_optimizer_is_deterministic_and_cannot_worsen_fit_nll():
    y = [0,1,0,1,0,1]
    p = [.01,.99,.01,.99,.99,.01]
    before = np.array(p)
    fitted = fit_temperature(y,p)
    assert fitted==fit_temperature(y,p)
    assert fitted["fit_nll_stable_softplus"] <= fitted["identity_fit_nll_stable_softplus"]+1e-12
    assert np.array_equal(before,np.array(p))
    assert np.exp(-4)<=fitted["temperature"]<=np.exp(4)


def test_bounded_fit_reports_boundary_selection_for_all_wrong_scores():
    fit = fit_temperature([0,1,0,1],[.9,.1,.8,.2])
    assert fit["selected_exact_upper_bound"] is True
    assert fit["near_upper_bound"] is True
    assert fit["log_temperature"]==LOG_T_BOUNDS[1]


def test_input_boundary_logits_are_finite_and_raw_probabilities_unchanged():
    p = np.array([0.,1.])
    logits = case_logits(p)
    assert np.isfinite(logits).all()
    assert np.array_equal(p,[0.,1.])


def test_positive_temperature_preserves_threshold_side_even_near_roundoff():
    p = np.array([np.nextafter(.5,0.),.5,np.nextafter(.5,1.)])
    output = calibrate_probabilities(p,54.)
    assert np.array_equal(output>=.5,p>=.5)


def test_crossfit_fit_does_not_use_heldout_labels_or_probabilities():
    labels = np.array([0,1,0,1,0,1,0,1,0,1])
    p = np.array([.1,.9,.2,.8,.3,.7,.4,.6,.05,.95])
    folds = np.repeat(np.arange(5),2)
    _,original = cross_fit_temperature(labels,p,folds)
    altered_y,altered_p = labels.copy(),p.copy()
    altered_y[:2] = 1-altered_y[:2]
    altered_p[:2] = [.99,.01]
    _,altered = cross_fit_temperature(altered_y,altered_p,folds)
    assert original[0]==altered[0]
    assert original[0]["n_fit_cases"]==8
    assert original[0]["n_holdout_cases"]==2


def test_identity_fit_candidate_does_not_guarantee_heldout_improvement():
    y = np.array([0,1,0,1,0,1])
    p = np.array([.1,.9,.1,.9,.9,.1])
    oof,_ = cross_fit_temperature(y,p,np.repeat(np.arange(3),2))
    assert classification_metrics(y,oof)["nll"] > classification_metrics(y,p)["nll"]


def test_pooled_oof_auroc_can_change_across_fold_temperatures():
    y,p = np.array([0,1,0,1]),np.array([.7,.8,.2,.6])
    output = np.array([calibrate_probabilities(np.array([value]),.1 if fold==0 else 10.)[0]
                       for value,fold in zip(p,[0,1,0,1])])
    assert classification_metrics(y,p)["auroc"]==pytest.approx(.75)
    assert classification_metrics(y,output)["auroc"]==pytest.approx(.5)
    assert np.array_equal(p>=.5,output>=.5)


@pytest.mark.parametrize("y,p",[([0,1],[.2,float("nan")]),([0,2],[.2,.8]),([0,0],[.2,.8]),([0,1],[.2,1.1])])
def test_temperature_fit_rejects_invalid_or_single_class_inputs(y,p):
    with pytest.raises(ValueError):
        fit_temperature(y,p)


@pytest.mark.parametrize("temperature",[0.,-1.,float("nan"),True])
def test_calibration_requires_positive_finite_temperature(temperature):
    with pytest.raises(ValueError):
        calibrate_probabilities([.2,.8],temperature)


def test_case_folds_are_deterministic_stratified_and_one_assignment_per_case():
    index = {f"private_case{i}":{"label":i%2,"image_ids":[f"image{i}"]} for i in range(20)}
    cases,y,folds,digest,counts = assign_folds(index)
    assert cases==sorted(index)
    assert len(folds)==20
    assert np.array_equal(folds,assign_folds(dict(reversed(list(index.items()))))[2])
    assert all(row["n_holdout_cases"]==4 and row["n_holdout_positive"]==2 for row in counts)
    assert len(digest)==64


def test_build_protocol_is_fitting_free_and_public_aggregate_only(source,monkeypatch,tmp_path):
    def forbidden(*args,**kwargs):
        raise AssertionError("Temperature fitted during preparation")
    monkeypatch.setattr("loop_ultrasound.temperature_diagnostic.fit_temperature",forbidden)
    protocol,_ = build_protocol(source["runs"],**source_kwargs(source))
    assert protocol["n_splits"]==5
    assert protocol["stratified_case_fold_seed"]==20261011
    assert protocol["bootstrap"]["refit_temperatures"] is False
    encoded = json.dumps(protocol,allow_nan=False)
    for private in ("private_tune","private_train",str(tmp_path),"private_local_filename"):
        assert private not in encoded


def test_prepare_writes_only_locked_protocol_and_local_fold_file(source,tmp_path,monkeypatch):
    root = tmp_path/"diagnostic"
    def forbidden(*args,**kwargs):
        raise AssertionError("Preparation fitted a temperature")
    monkeypatch.setattr("loop_ultrasound.temperature_diagnostic.fit_temperature",forbidden)
    protocol = prepare(source["runs"],output_dir=root,**source_kwargs(source))
    assert protocol["status"]=="fixed_before_case_grouped_crossfit_temperature_diagnostic"
    assert sorted(path.name for path in root.iterdir())==["fold_assignments.csv","protocol.json"]
    assert "private_tune" in (root/"fold_assignments.csv").read_text()
    assert prepare(source["runs"],output_dir=root,**source_kwargs(source))==protocol


def test_execute_requires_preparation_and_never_overwrites_results(source,tmp_path):
    root = tmp_path/"diagnostic"
    with pytest.raises(ValueError,match="Prepare and lock"):
        execute(source["runs"],output_dir=root,**source_kwargs(source))
    prepare(source["runs"],output_dir=root,**source_kwargs(source),draws=100)
    report = execute(source["runs"],output_dir=root,**source_kwargs(source),draws=100)
    assert report["status"]=="executed_case_grouped_crossfit_temperature_diagnostic"
    assert report["primary256"]["training_seeds"]==list((53,71,89,107,131))
    assert report["secondary128"] is not None
    assert len(report["primary256"]["seed_results"])==5
    assert len(report["primary256"]["seed_results"][0]["regimes"]["depth2_terminal"]["temperature_fits"])==5
    assert report["primary256"]["raw_depth_gain_Brier"]["bootstrap"]["n_cases"]==10
    assert report["primary256"]["raw_depth_gain_Brier"]["bootstrap"]["n_fitted_seeds"]==5
    assert (root/"oof_case_predictions.csv").exists()
    before = (root/"temperature_diagnostic_analysis.json").read_bytes()
    with pytest.raises(ValueError,match="already exist"):
        execute(source["runs"],output_dir=root,**source_kwargs(source),draws=100)
    assert (root/"temperature_diagnostic_analysis.json").read_bytes()==before


def test_oof_report_privacy_and_fixed_threshold_classification_preserved(source,tmp_path):
    root = tmp_path/"diagnostic"
    prepare(source["runs"],output_dir=root,**source_kwargs(source),draws=100)
    report = execute(source["runs"],output_dir=root,**source_kwargs(source),draws=100)
    for block in (report["primary256"],report["secondary128"]):
        for row in block["seed_results"]:
            for regime in row["regimes"].values():
                assert all(regime["raw"][key]==regime["oof"][key] for key in ("tp","tn","fp","fn"))
    encoded = json.dumps(report,allow_nan=False)
    for private in ("private_tune","private_train",str(tmp_path),"private_local_filename"):
        assert private not in encoded
    assert "ranking and AUROC" in encoded


def test_primary_only_diagnostic_requires_no_secondary_artifacts(source,tmp_path):
    root = tmp_path/"diagnostic"
    kwargs = source_kwargs(source,secondary=False)
    prepare(source["runs"],output_dir=root,**kwargs,draws=100)
    report = execute(source["runs"],output_dir=root,**kwargs,draws=100)
    assert report["secondary128"] is None


@pytest.mark.parametrize("filename",["summary.json","training_predictions.csv","development_predictions.csv"])
def test_source_artifact_bytes_are_pinned_before_fitting(source,tmp_path,filename):
    root = tmp_path/"diagnostic"
    prepare(source["runs"],output_dir=root,**source_kwargs(source))
    path = source["runs"][-1]/filename
    path.write_bytes(path.read_bytes()+b"\n")
    with pytest.raises(ValueError,match="changed after preparation"):
        execute(source["runs"],output_dir=root,**source_kwargs(source))
    assert not (root/"temperature_diagnostic_analysis.json").exists()


def test_changed_local_fold_assignment_rejected_before_fitting(source,tmp_path):
    root = tmp_path/"diagnostic"
    prepare(source["runs"],output_dir=root,**source_kwargs(source))
    csv_edit(root/"fold_assignments.csv",lambda rows:rows[0].update(fold=4))
    with pytest.raises(ValueError,match="Case folds"):
        execute(source["runs"],output_dir=root,**source_kwargs(source))


def test_changed_bootstrap_recipe_rejected_after_preparation(source,tmp_path):
    root = tmp_path/"diagnostic"
    prepare(source["runs"],output_dir=root,**source_kwargs(source),draws=100)
    with pytest.raises(ValueError,match="changed after preparation"):
        execute(source["runs"],output_dir=root,**source_kwargs(source),draws=2000)


def test_non_tune_source_predictions_are_rejected(source):
    csv_edit(source["runs"][-1]/"development_predictions.csv",lambda rows:rows[0].update(split="test"))
    with pytest.raises(ValueError,match="train/tune split"):
        build_protocol(source["runs"],**source_kwargs(source))


def test_cli_prepare_and_execute_are_distinct_actions(source,tmp_path,capsys):
    root = tmp_path/"diagnostic"
    args = ["--run-dirs",*map(str,source["runs"]),"--source-protocol",str(source["protocol"]),
            "--reference-run-dirs",*map(str,source["old_runs"]),"--reference-protocol",str(source["old_protocol"]),
            "--output-dir",str(root),"--expected-training-cases","4","--expected-reference-training-cases","2",
            "--expected-development-cases","10"]
    assert main(["--prepare",*args])==0
    assert json.loads(capsys.readouterr().out)["temperatures_fitted"] is False
    assert main(["--execute",*args])==0
    assert json.loads(capsys.readouterr().out)["temperatures_fitted"] is True
