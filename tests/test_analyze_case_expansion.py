"""Nested Case/view validation and equal-update separately fitted depth contrasts."""
import copy
import hashlib
import json

import numpy as np
import pytest

from loop_ultrasound.analyze_case_expansion import NEW_SEEDS,analyze_case_expansion,main
from loop_ultrasound.analyze_expanded import _fingerprint
from loop_ultrasound.metrics import aggregate_cases,classification_metrics
from test_analyze_seed_replication import (GATE,REFERENCE_SEEDS,cohort,csv_edit,csv_read,csv_write,
                                          json_edit,make_protocol,make_runs,metrics,update_endpoint)


@pytest.fixture
def experiment(tmp_path):
    reference = make_protocol(REFERENCE_SEEDS,"fixed_before_depth_supervision_training")
    reference["supervisions"] = ["all","terminal"]
    old_path = tmp_path/"reference.json"
    old_path.write_text(json.dumps(reference))
    old_runs = make_runs(tmp_path/"old",REFERENCE_SEEDS,reference)
    nested = copy.deepcopy(cohort("train"))
    for label in (0,1):
        nested[f"private_extra_case{label}"] = {"label":label,"n_images":1,"image_ids":[f"private_extra_image{label}"]}
    protocol = copy.deepcopy(reference)
    protocol.update(status="fixed_before_case_expansion_training",
                    phase="exploratory_nested_training_case_expansion_on_observed_tune",epochs=25,reference_epochs=50,
                    reference_train=copy.deepcopy(reference["train"]),supervisions=["terminal"],
                    updates_per_epoch=2,reference_updates_per_epoch=1,matched_update_points=[10,20,30,40,50],
                    train={"cases":4,"images":4,"benign":2,"malignant":2,"fingerprint":_fingerprint(nested)},
                    reference_protocol_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
                    nested_cohort_verified=True,nested_cohort_check={"reference_cases_subset":True,"reference_images_subset":True,"shared_view_labels_match":True})
    path = tmp_path/"protocol.json"
    path.write_text(json.dumps(protocol))
    new_runs = make_runs(tmp_path/"new",REFERENCE_SEEDS,reference)
    for directory in new_runs:
        fields,rows = csv_read(directory/"training_predictions.csv")
        extra = []
        for row in rows:
            candidate = copy.deepcopy(row)
            label = row["label"]
            candidate.update(case_id=f"private_extra_case{label}",image_id=f"private_extra_image{label}")
            extra.append(candidate)
        csv_write(directory/"training_predictions.csv",rows+extra,fields)
        json_edit(directory/"summary.json",lambda s:s["config"].update(epochs=25))
        json_edit(directory/"summary.json",lambda s:s.update(n_training_cases=4,n_training_images=4))
        json_edit(directory/"summary.json",lambda s:s["cohort_selection"].update(train_fingerprint=protocol["train"]["fingerprint"]))
        depth = json.loads((directory/"summary.json").read_text())["config"]["steps"]
        if depth==4:
            seed = int(directory.name.split("seed")[-1])
            update_endpoint(directory,(.05,.15,.25)[REFERENCE_SEEDS.index(seed)])
        summary = json.loads((directory/"summary.json").read_text())
        summary["development_history"] = [{"epoch":epoch,"by_step":copy.deepcopy(summary["development_subset"])} for epoch in range(5,26,5)]
        (directory/"summary.json").write_text(json.dumps(summary))
    return {"runs":new_runs,"protocol":path,"old_runs":old_runs,"old_protocol":old_path}


def analyze(experiment,**kwargs):
    return analyze_case_expansion(experiment["runs"],protocol_path=experiment["protocol"],
                                   reference_run_dirs=experiment["old_runs"],reference_protocol_path=experiment["old_protocol"],
                                   expected_training_cases=4,expected_reference_training_cases=2,
                                   expected_development_cases=2,draws=100,**kwargs)


def make_baselines(directory,runs):
    directory.mkdir()
    summary = {"status":"executed_cpu_preexperiment_baselines","clinical_validation":False,
               "device":"cpu","patient_mapping":"unverified_Case_grouped_only","selection_seed":20261004,
               "sources":{"encoder_sha256":"a"*64,"manifest_sha256":"b"*64},
               "cohort_fingerprints":{},"baselines":{name:{} for name in ("constant","logistic_regression")},
               "method":{"feature_shape_per_image":[196,192],
                         "feature_pooling":"mean spatial tokens per image, then mean all available image features per Case",
                         "fit_split":"train","scaler_fit_split":"train","hyperparameters_selected_using_tune":False,
                         "threshold":.5,"train_case_prevalence":.5,
                         "unknown_private_field":"private_local_filename",
                         "logistic_regression":{"C":1.,"solver":"lbfgs","max_iter":2000,"random_state":17,
                                                "class_weight":None,"fit_intercept":True,"n_iter":1}}}
    records = []
    for split,filename in (("train","training_predictions.csv"),("tune","development_predictions.csv")):
        _,rows = csv_read(runs[0]/filename)
        typed = [{**row,"step":int(row["step"]),"seed":int(row["seed"]),"label":int(row["label"]),
                  "probability":float(row["probability"]),"dice":float(row["dice"])} for row in rows if row["step"]=="1"]
        cases = aggregate_cases(typed)
        index = {row["case_id"]:row for row in cases}
        summary["cohort_fingerprints"][split] = _fingerprint(index)
        for name in ("constant","logistic_regression"):
            probabilities = [.5 if name=="constant" else (.2,.8)[row["label"]] for row in cases]
            summary["baselines"][name][split] = classification_metrics([row["label"] for row in cases],probabilities)
            records.extend({"baseline":name,"split":split,"seed":17,"case_id":row["case_id"],"label":row["label"],
                            "probability":p,"n_images":row["n_images"]} for row,p in zip(cases,probabilities))
    csv_write(directory/"baseline_case_predictions.csv",records,
              ("baseline","split","seed","case_id","label","probability","n_images"))
    (directory/"baselines_summary.json").write_text(json.dumps(summary))
    return directory


@pytest.fixture
def replication(experiment,tmp_path):
    original = json.loads(experiment["protocol"].read_text())
    reference = make_protocol(NEW_SEEDS,"fixed_before_seed_replication_training")
    reference.update(phase="exploratory_training_randomness_replication_on_observed_tune",
                     supervisions=["terminal"],reference_seeds=list(REFERENCE_SEEDS),screening_gate=dict(GATE),
                     reference_protocol_sha256=hashlib.sha256(experiment["old_protocol"].read_bytes()).hexdigest())
    reference_path = tmp_path/"replication_reference.json"
    reference_path.write_text(json.dumps(reference))
    protocol = copy.deepcopy(original)
    protocol.update(status="fixed_before_case_expansion_seed_replication_training",
                    phase="exploratory_nested_case_expansion_training_randomness_replication_on_observed_tune",
                    seeds=list(NEW_SEEDS),reference_seeds=list(NEW_SEEDS),historical_seed_scope="original_three_secondary_only",
                    initialization_sha256_by_seed=copy.deepcopy(reference["initialization_sha256_by_seed"]),
                    reference_protocol_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                    original_expansion_protocol_sha256=hashlib.sha256(experiment["protocol"].read_bytes()).hexdigest())
    protocol_path = tmp_path/"replication.json"
    protocol_path.write_text(json.dumps(protocol))
    old_runs = make_runs(tmp_path/"replication_old",NEW_SEEDS,reference)
    new_runs = make_runs(tmp_path/"replication_new",NEW_SEEDS,reference)
    for directory in new_runs:
        fields,rows = csv_read(directory/"training_predictions.csv")
        extras = []
        for row in rows:
            candidate = copy.deepcopy(row)
            label = row["label"]
            candidate.update(case_id=f"private_extra_case{label}",image_id=f"private_extra_image{label}")
            extras.append(candidate)
        csv_write(directory/"training_predictions.csv",rows+extras,fields)
        json_edit(directory/"summary.json",lambda s:s["config"].update(epochs=25))
        json_edit(directory/"summary.json",lambda s:s.update(n_training_cases=4,n_training_images=4))
        json_edit(directory/"summary.json",lambda s:s["cohort_selection"].update(train_fingerprint=protocol["train"]["fingerprint"]))
        if "depth4" in directory.name:
            seed = int(directory.name.split("seed")[-1])
            update_endpoint(directory,(.1,.2,.3,.4,.5)[NEW_SEEDS.index(seed)])
        summary = json.loads((directory/"summary.json").read_text())
        summary["development_history"] = [{"epoch":epoch,"by_step":copy.deepcopy(summary["development_subset"])} for epoch in range(5,26,5)]
        (directory/"summary.json").write_text(json.dumps(summary))
    baseline_dirs = {"new256":make_baselines(tmp_path/"replication_new_baselines",new_runs),
                     "reference128":make_baselines(tmp_path/"replication_reference_baselines",old_runs)}
    protocol["baseline_artifact_bindings"] = {size:{
        "summary_sha256":hashlib.sha256((directory/"baselines_summary.json").read_bytes()).hexdigest(),
        "predictions_sha256":hashlib.sha256((directory/"baseline_case_predictions.csv").read_bytes()).hexdigest()}
        for size,directory in baseline_dirs.items()}
    protocol_path.write_text(json.dumps(protocol))
    return {"runs":new_runs,"protocol":protocol_path,"old_runs":old_runs,"old_protocol":reference_path,
            "original":experiment,"baseline_dirs":baseline_dirs}


def analyze_replication(replication,*,secondary=False,**kwargs):
    original = replication["original"]
    optional = {"secondary_run_dirs":original["runs"],"secondary_protocol_path":original["protocol"],
                "secondary_reference_run_dirs":original["old_runs"],"secondary_reference_protocol_path":original["old_protocol"]} if secondary else {}
    return analyze(replication,**optional,**kwargs)


def test_known_endpoint_gain_and_nested_data_size_interaction(experiment):
    report = analyze(experiment)
    expected = .4**2-np.array([.05,.15,.25])**2
    primary = report["primary_256"]["depth_benefit_Brier"]
    assert primary["estimate"] == pytest.approx(expected.mean())
    assert primary["std_across_training_seeds"] == pytest.approx(expected.std(ddof=1))
    assert primary["positive_seed_count"] == 3
    assert report["reference_128"]["depth_benefit_Brier"]["mean"] == pytest.approx(.15)
    assert report["interaction_data_size_depth"]["estimate"] == pytest.approx(expected.mean()-.15)
    assert report["interaction_data_size_depth"]["estimate"] < 0
    assert report["alignment_checks"]["strictly_nested_training_cases"] is True
    assert report["decision_rule_automatically_applied"] is False


def test_metrics_are_seed_fit_means_not_ensembles(experiment):
    report = analyze(experiment)
    expected = np.mean(np.array([.05,.15,.25])**2)
    assert report["primary_256"]["regimes"]["depth4_terminal"]["endpoint"]["brier"]["mean"] == pytest.approx(expected)
    assert expected != pytest.approx(.15**2)


def test_data_size_effects_by_depth_explain_interaction_without_conflation(experiment):
    report = analyze(experiment)
    changes = report["data_size_benefit_by_depth"]
    assert changes["depth2_terminal"]["Brier_benefit"]["mean"] == pytest.approx(0.)
    expected_depth4 = .01-np.mean(np.array([.05,.15,.25])**2)
    assert changes["depth4_terminal"]["Brier_benefit"]["mean"] == pytest.approx(expected_depth4)
    # Interaction equals depth4's data-size benefit minus depth2's.
    assert report["interaction_data_size_depth"]["mean"] == pytest.approx(expected_depth4)


def test_matched_update_learning_uses_different_epoch_indices(experiment):
    report = analyze(experiment)
    curve = report["matched_update_learning"]
    assert [item["updates"] for item in curve] == [10,20,30,40,50]
    assert [item["new_epoch"] for item in curve] == [5,10,15,20,25]
    assert [item["reference_epoch"] for item in curve] == [10,20,30,40,50]
    assert curve[-1]["new256"]["depth4_terminal"]["brier"] == report["primary_256"]["regimes"]["depth4_terminal"]["endpoint"]["brier"]
    assert curve[-1]["reference128"]["depth4_terminal"]["brier"] == report["reference_128"]["regimes"]["depth4_terminal"]["endpoint"]["brier"]


def test_train_development_brier_gap_recomputed_from_case_traces(experiment):
    report = analyze(experiment)
    new = report["primary_256"]
    for row in new["seed_results"]:
        regime = row["regimes"]["depth4_terminal"]
        assert regime["training_endpoint"]["brier"] == pytest.approx(.01)
        assert regime["development_minus_training_brier"] == pytest.approx(regime["endpoint"]["brier"]-.01)
    expected = np.mean(np.array([.05,.15,.25])**2)-.01
    assert new["regimes"]["depth4_terminal"]["development_minus_training_brier"]["mean"] == pytest.approx(expected)


def test_bootstrap_keeps_two_actual_cases_and_excludes_seed_population_uncertainty(experiment):
    report = analyze(experiment)
    for value in (report["primary_256"]["depth_benefit_Brier"],report["interaction_data_size_depth"]):
        assert value["bootstrap"]["n_cases"] == 2
        assert value["bootstrap"]["n_fitted_seeds"] == 3
        assert value["bootstrap"]["includes_training_seed_sampling_uncertainty"] is False
        assert value["ci95"] == pytest.approx([value["estimate"],value["estimate"]])


def test_output_has_aggregate_metrics_without_paths_ids_or_unknown_fields(experiment,tmp_path):
    report = analyze(experiment)
    text = json.dumps(report,allow_nan=False)
    for private in ("private_train","private_tune","private_extra","private_local_filename",str(tmp_path)):
        assert private not in text
    assert "unknown_field" not in report["config"]
    assert report["clinical_validation"] is False


def test_size_specific_baselines_include_recomputed_train_tune_metrics(experiment,tmp_path):
    new_baseline = make_baselines(tmp_path/"new_baseline",experiment["runs"])
    old_baseline = make_baselines(tmp_path/"old_baseline",experiment["old_runs"])
    report = analyze(experiment,baseline_dir=new_baseline,reference_baseline_dir=old_baseline)
    baselines = report["baselines"]
    assert baselines["new256"]["n_training_cases"] == 4
    assert baselines["reference128"]["n_training_cases"] == 2
    assert baselines["new256"]["baselines"]["constant"]["tune"]["brier"] == pytest.approx(.25)
    assert baselines["new256"]["baselines"]["logistic_regression"]["train"]["brier"] == pytest.approx(.04)
    assert "unknown_private_field" not in baselines["new256"]["method"]
    assert "private_local_filename" not in json.dumps(baselines)


def test_old_size_baseline_cannot_be_reported_as_new_size_fit(experiment,tmp_path):
    old = make_baselines(tmp_path/"old_baseline",experiment["old_runs"])
    with pytest.raises(ValueError,match="fingerprints"):
        analyze(experiment,baseline_dir=old)


@pytest.mark.parametrize("change",[lambda b:b["method"].update(fit_split="tune"),
                                   lambda b:b["method"].update(scaler_fit_split="tune"),
                                   lambda b:b["method"].update(hyperparameters_selected_using_tune=True),
                                   lambda b:b["method"]["logistic_regression"].update(C=.1),
                                   lambda b:b["method"]["logistic_regression"].update(random_state=43),
                                   lambda b:b["method"].update(train_case_prevalence=.4),
                                   lambda b:b["baselines"]["constant"]["tune"].update(brier=.123)])
def test_baseline_method_and_declared_metrics_locked(experiment,tmp_path,change):
    directory = make_baselines(tmp_path/"baselines",experiment["runs"])
    json_edit(directory/"baselines_summary.json",change)
    with pytest.raises(ValueError):
        analyze(experiment,baseline_dir=directory)


def test_constant_baseline_uses_only_own_training_case_prevalence(experiment,tmp_path):
    directory = make_baselines(tmp_path/"baselines",experiment["runs"])
    csv_edit(directory/"baseline_case_predictions.csv",lambda rows:rows[0].update(probability=.7))
    with pytest.raises(ValueError,match="train-only Case prevalence"):
        analyze(experiment,baseline_dir=directory)


@pytest.mark.parametrize("which",["runs","old_runs"])
@pytest.mark.parametrize("mutation",["missing","duplicate"])
def test_exactly_six_unique_fits_per_data_size(experiment,which,mutation):
    runs = experiment[which]
    experiment[which] = runs[:-1] if mutation=="missing" else runs[:-1]+runs[:1]
    with pytest.raises(ValueError,match="distinct"):
        analyze(experiment)


@pytest.mark.parametrize("change",[lambda s:s["config"].update(epochs=50),lambda s:s["config"].update(seed=53),
                                   lambda s:s["config"].update(supervision="all"),lambda s:s["config"].update(steps=3),
                                   lambda s:s["config"].update(augment=True),lambda s:s["timing"].update(updates=49),
                                   lambda s:s.update(training_source_sha256="e"*64),lambda s:s.update(initialization_sha256="f"*64),
                                   lambda s:s.update(evaluation_split="test"),lambda s:s["development_history"].pop(),
                                   lambda s:s["development_subset"]["4"].update(brier=.123)])
def test_new_fit_config_completion_and_provenance_strict(experiment,change):
    json_edit(experiment["runs"][-1]/"summary.json",change)
    with pytest.raises(ValueError):
        analyze(experiment)


@pytest.mark.parametrize("split",["train","cal","calibration","test",""])
def test_only_tune_development_predictions(experiment,split):
    csv_edit(experiment["runs"][-1]/"development_predictions.csv",lambda rows:rows[0].update(split=split))
    with pytest.raises(ValueError,match="train/tune split"):
        analyze(experiment)


@pytest.mark.parametrize("field,value",[("case_id","private_other_case"),("image_id","private_other_image"),("label","1")])
def test_equal_counts_cannot_hide_changed_cases_views_or_labels(experiment,field,value):
    def change(rows):
        for row in rows:
            if row["case_id"]=="private_tune_case0" and (field!="image_id" or row["image_id"]=="private_tune_image0_0"):
                row[field] = value
    csv_edit(experiment["runs"][-1]/"development_predictions.csv",change)
    with pytest.raises(ValueError):
        analyze(experiment)


def test_nested_case_and_view_claims_checked_against_actual_csv(experiment):
    # Relabel one shared train image consistently in all new fits and update
    # new locked fingerprints. Counts and claimed nested booleans still match.
    for directory in experiment["runs"]:
        csv_edit(directory/"training_predictions.csv",lambda rows:[row.update(image_id="private_changed_shared_view")
                 for row in rows if row["image_id"]=="private_train_image0_0"])
    new_index = copy.deepcopy(cohort("train"))
    new_index["private_train_case0"]["image_ids"] = ["private_changed_shared_view"]
    for label in (0,1):
        new_index[f"private_extra_case{label}"] = {"label":label,"n_images":1,"image_ids":[f"private_extra_image{label}"]}
    fingerprint = _fingerprint(new_index)
    json_edit(experiment["protocol"],lambda p:p["train"].update(fingerprint=fingerprint))
    for directory in experiment["runs"]:
        json_edit(directory/"summary.json",lambda s:s["cohort_selection"].update(train_fingerprint=fingerprint))
    with pytest.raises(ValueError,match="all shared views"):
        analyze(experiment)


def test_old_protocol_bytes_pinned_and_old_fits_validated(experiment):
    json_edit(experiment["old_runs"][-1]/"summary.json",lambda s:s["timing"].update(updates=49))
    with pytest.raises(ValueError,match="update"):
        analyze(experiment)


def test_reference_protocol_byte_change_rejected(experiment):
    json_edit(experiment["old_protocol"],lambda p:p.update(unknown="changed"))
    with pytest.raises(ValueError,match="protocol bytes"):
        analyze(experiment)


@pytest.mark.parametrize("change",[lambda p:p.update(epochs=50),lambda p:p.update(reference_epochs=25),
                                   lambda p:p.update(training_updates_per_run=49),lambda p:p.update(warm_starts=True),
                                   lambda p:p.update(matched_update_points=[5,10,15,20,25]),
                                   lambda p:p["train"].update(fingerprint="e"*64),
                                   lambda p:p["initialization_sha256_by_seed"].update({"43":"e"*64})])
def test_locked_protocol_and_equal_budget_required(experiment,change):
    json_edit(experiment["protocol"],change)
    with pytest.raises(ValueError):
        analyze(experiment)


def test_matched_history_rejects_nonfinite_endpoint(experiment):
    json_edit(experiment["runs"][-1]/"summary.json",lambda s:s["development_history"][0]["by_step"]["4"].update(nll=float("nan")))
    with pytest.raises(ValueError,match="finite"):
        analyze(experiment)


def test_cli_publishes_aggregate_only_and_refuses_overwrite(experiment,tmp_path,capsys):
    output = tmp_path/"aggregate.json"
    args = ["--run-dirs",*map(str,experiment["runs"]),"--protocol",str(experiment["protocol"]),
            "--reference-run-dirs",*map(str,experiment["old_runs"]),"--reference-protocol",str(experiment["old_protocol"]),
            "--expected-training-cases","4","--expected-reference-training-cases","2",
            "--expected-development-cases","2","--draws","50","--output",str(output)]
    assert main(args)==0
    before = output.read_bytes()
    assert json.loads(before)["status"]=="executed_cpu_case_expansion_analysis"
    assert "private_train" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(args)
    assert output.read_bytes()==before


def test_five_new_seeds_are_primary_with_correct_metrics_and_leave_one_out(replication):
    report = analyze_replication(replication)
    expected = .4**2-np.array([.1,.2,.3,.4,.5])**2
    assert report["training_seeds"] == list(NEW_SEEDS)
    assert report["primary_seed_role"] == "five_new_seed_replication"
    assert report["primary_256"]["n_seeds"] == 5
    assert report["primary_256"]["depth_benefit_Brier"]["mean"] == pytest.approx(expected.mean())
    assert report["primary_256"]["depth_benefit_Brier"]["std_across_training_seeds"] == pytest.approx(expected.std(ddof=1))
    assert [row["seed"] for row in report["primary_256"]["depth_benefit_Brier"]["by_seed"]] == list(NEW_SEEDS)
    assert [row["mean_gain"] for row in report["primary_256"]["leave_one_seed_out"]] == pytest.approx(
        [np.delete(expected,index).mean() for index in range(5)])
    assert report["primary_256"]["depth_benefit_Brier"]["bootstrap"]["n_cases"] == 2
    assert report["primary_256"]["depth_benefit_Brier"]["bootstrap"]["n_fitted_seeds"] == 5
    assert report["secondary_combined_eight_seeds"] is None
    assert report["decision_rule_automatically_applied"] is False


def test_original_three_and_combined_eight_are_explicit_secondary_without_replacing_primary(replication):
    report = analyze_replication(replication,secondary=True)
    primary = report["primary_256"]["depth_benefit_Brier"]["mean"]
    original = report["secondary_original_three_seeds"]
    combined = report["secondary_combined_eight_seeds"]
    assert original["training_seeds"] == list(REFERENCE_SEEDS)
    assert combined["training_seeds"] == list(REFERENCE_SEEDS+NEW_SEEDS)
    assert combined["primary_256"]["n_seeds"] == 8
    assert combined["primary_256"]["depth_benefit_Brier"]["mean"] == pytest.approx(
        (primary*5+original["primary_256"]["depth_benefit_Brier"]["mean"]*3)/8)
    assert combined["primary_256"]["depth_benefit_Brier"]["mean"] > primary
    assert original["can_override_primary"] is False
    assert combined["can_override_primary"] is False
    assert report["primary_256"]["n_seeds"] == 5
    assert len(report["primary_256"]["seed_results"]) == 5


def test_secondary_uncertainty_still_resamples_cases_not_eight_seed_case_pairs(replication):
    combined = analyze_replication(replication,secondary=True)["secondary_combined_eight_seeds"]
    for effect in (combined["primary_256"]["depth_benefit_Brier"],combined["interaction_data_size_depth"]):
        assert effect["bootstrap"]["n_cases"] == 2
        assert effect["bootstrap"]["n_fitted_seeds"] == 8
        assert effect["bootstrap"]["includes_training_seed_sampling_uncertainty"] is False
        assert effect["ci95"] == pytest.approx([effect["estimate"],effect["estimate"]])


@pytest.mark.parametrize("which",["runs","old_runs"])
def test_five_seed_mode_requires_ten_fits_at_each_size(replication,which):
    replication[which] = replication[which][:-1]
    with pytest.raises(ValueError,match="distinct"):
        analyze_replication(replication)


def test_original_seed_cannot_be_mixed_into_new_primary_runset(replication):
    json_edit(replication["runs"][-1]/"summary.json",lambda s:s["config"].update(seed=43))
    with pytest.raises(ValueError,match="expected paired seed"):
        analyze_replication(replication)


def test_replication_reference_validation_requires_matching_new_five_seed128fits(replication):
    json_edit(replication["old_runs"][-1]/"summary.json",lambda s:s.update(initialization_sha256="f"*64))
    with pytest.raises(ValueError,match="paired seed digest"):
        analyze_replication(replication)


@pytest.mark.parametrize("change",[lambda p:p.update(seeds=list(REFERENCE_SEEDS)+list(NEW_SEEDS)),
                                   lambda p:p.update(reference_seeds=list(REFERENCE_SEEDS)),
                                   lambda p:p.update(historical_seed_scope="all_primary"),
                                   lambda p:p.update(original_expansion_protocol_sha256="invalid")])
def test_replication_protocol_cannot_pool_old_seeds_or_relax_primary_scope(replication,change):
    json_edit(replication["protocol"],change)
    with pytest.raises(ValueError):
        analyze_replication(replication)


def test_secondary_requires_all_four_explicit_artifacts(replication):
    with pytest.raises(ValueError,match="all four artifacts"):
        analyze_replication(replication,secondary_run_dirs=replication["original"]["runs"])


def test_secondary_original_protocol_bytes_pinned_before_new_training(replication):
    json_edit(replication["original"]["protocol"],lambda p:p.update(extra="changed"))
    with pytest.raises(ValueError,match="Original expansion protocol bytes"):
        analyze_replication(replication,secondary=True)


def test_secondary_original_fits_are_fully_validated_before_combining(replication):
    json_edit(replication["original"]["runs"][-1]/"summary.json",lambda s:s["timing"].update(updates=49))
    with pytest.raises(ValueError,match="optimizer-update"):
        analyze_replication(replication,secondary=True)


def test_five_and_eight_seed_public_reports_exclude_case_paths_and_private_fields(replication,tmp_path):
    report = analyze_replication(replication,secondary=True)
    text = json.dumps(report,allow_nan=False)
    for private in ("private_train","private_tune","private_extra",str(tmp_path),"private_local_filename"):
        assert private not in text


def test_new_five_baseline_artifact_bytes_are_bound_and_validated(replication):
    directories = replication["baseline_dirs"]
    report = analyze_replication(replication,baseline_dir=directories["new256"],reference_baseline_dir=directories["reference128"])
    assert report["alignment_checks"]["locked_baseline_artifact_bytes_verified"] is True
    assert report["baselines"]["new256"]["n_training_cases"] == 4


@pytest.mark.parametrize("artifact",["baselines_summary.json","baseline_case_predictions.csv"])
def test_changed_baseline_artifact_bytes_cannot_replace_locked_fits(replication,artifact):
    directory = replication["baseline_dirs"]["new256"]
    path = directory/artifact
    path.write_bytes(path.read_bytes()+b"\n")
    with pytest.raises(ValueError,match="baseline artifact bytes"):
        analyze_replication(replication,baseline_dir=directory)
