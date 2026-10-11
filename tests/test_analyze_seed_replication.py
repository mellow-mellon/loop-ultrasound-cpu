"""Five-seed gate robustness, exact paired provenance, secondary-fit separation."""
import copy
import csv
import hashlib
import json

import numpy as np
import pytest

from loop_ultrasound.analyze_depth_supervision import FIXED_CONFIG
from loop_ultrasound.analyze_expanded import _fingerprint
from loop_ultrasound.analyze_seed_replication import (
    GATE, NEW_SEEDS, REFERENCE_SEEDS, _concentration, _effect,
    _high_confidence_errors, _transitions, analyze_seed_replication, main,
)
from loop_ultrasound.metrics import aggregate_cases, classification_metrics


FIELDS = ("case_id","image_id","label","probability","step","arm","seed","dice","split")


def csv_write(path,rows,fields=FIELDS):
    with path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def csv_read(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames,list(reader)


def csv_edit(path,change):
    fields,rows = csv_read(path)
    change(rows)
    csv_write(path,rows,fields)


def json_edit(path,change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def cohort(split):
    counts = (1,1) if split=="train" else (2,1)
    return {f"private_{split}_case{label}": {"label":label,"n_images":count,
            "image_ids":[f"private_{split}_image{label}_{view}" for view in range(count)]}
            for label,count in enumerate(counts)}


def init(seed):
    return f"{seed:064x}"


def metrics(rows,depth):
    numeric = [{**row,"step":int(row["step"]),"seed":int(row["seed"]),"label":float(row["label"]),
                "probability":float(row["probability"]),"dice":float(row["dice"])} for row in rows]
    cases = aggregate_cases(numeric)
    result = {}
    for step in range(1,depth+1):
        selected = [row for row in cases if row["step"]==step]
        value = classification_metrics([row["label"] for row in selected],[row["probability"] for row in selected])
        value.update(mean_case_dice=float(np.mean([row["dice"] for row in selected])),
                     dice_space="224x224_letterbox_valid_pixels")
        result[str(step)] = value
    return result


def update_endpoint(directory,error):
    summary = json.loads((directory/"summary.json").read_text())
    depth = summary["config"]["steps"]
    fields,rows = csv_read(directory/"development_predictions.csv")
    for row in rows:
        if int(row["step"])==depth:
            # Retain exact multi-view Case means while changing endpoint fit.
            probability = error if row["label"]=="0" else 1-error
            row["probability"] = probability
    csv_write(directory/"development_predictions.csv",rows,fields)
    value = metrics(rows,depth)
    summary["development_subset"] = value
    summary["development_history"][-1]["by_step"] = value
    (directory/"summary.json").write_text(json.dumps(summary))


def make_protocol(seeds,status):
    protocol = {**{key:value for key,value in FIXED_CONFIG.items() if key!="arm"},
                "status":status,"clinical_validation":False,"device":"cpu",
                "patient_mapping":"unverified_Case_grouped_only","calibration_and_test_predictions":False,
                "model_selection":"fixed_final_epoch_no_best_epoch_selection","warm_starts":False,
                "training_updates_per_run":50,"arms":["SJ"],"depths":[2,4],"seeds":list(seeds),
                "encoder_sha256":"a"*64,"manifest_sha256":"b"*64,"training_source_sha256":"c"*64,
                "launcher_source_sha256":"d"*64,
                "initialization_sha256_by_seed":{str(seed):init(seed) for seed in seeds}}
    for split in ("train","tune"):
        index = cohort(split)
        protocol[split] = {"cases":2,"images":sum(row["n_images"] for row in index.values()),
                           "benign":1,"malignant":1,"fingerprint":_fingerprint(index)}
    return protocol


def make_runs(root,seeds,protocol):
    runs = []
    for seed in seeds:
        for depth in (2,4):
            directory = root/f"SJ-depth{depth}-terminal-seed{seed}"
            directory.mkdir(parents=True)
            split_rows = {}
            for split in ("train","tune"):
                rows = []
                for step in range(1,depth+1):
                    error = .45
                    if step==depth:
                        error = .4 if depth==2 else ((.2,.25,.3,.35,.45)[NEW_SEEDS.index(seed)] if seed in NEW_SEEDS else .1)
                    for case,row in cohort(split).items():
                        for view,image in enumerate(row["image_ids"]):
                            offset = (-.025,.025)[view] if row["n_images"]==2 else 0.
                            rows.append({"case_id":case,"image_id":image,"label":row["label"],
                                         "probability":(error if row["label"]==0 else 1-error)+offset,
                                         "step":step,"arm":"SJ","seed":seed,"dice":.5+.02*step,"split":split})
                split_rows[split] = rows
                csv_write(directory/("training_predictions.csv" if split=="train" else "development_predictions.csv"),rows)
            observed = metrics(split_rows["tune"],depth)
            summary = {"status":"executed_cpu_engineering_pilot","clinical_validation":False,"device":"cpu",
                       "patient_mapping":"unverified_Case_grouped_only","evaluation_split":"tune_development",
                       "model_selection":"fixed_final_epoch_no_best_epoch_selection",
                       "config":{**FIXED_CONFIG,"seed":seed,"steps":depth,"supervision":"terminal",
                                  "unknown_field":"private_local_filename"},
                       "encoder_sha256":"a"*64,"manifest_sha256":"b"*64,"training_source_sha256":"c"*64,
                       "initialization_sha256":init(seed),"n_training_cases":2,"n_training_images":2,
                       "timing":{"updates":50},"cohort_selection":{"selection_seed":20261004,"sampling":"proportional",
                          "train_fingerprint":protocol["train"]["fingerprint"],"tune_fingerprint":protocol["tune"]["fingerprint"]},
                       "readout_supervision":{"mode":"terminal","directly_supervised_steps":[depth]},
                       "development_subset":observed,
                       "development_history":[{"epoch":epoch,"by_step":copy.deepcopy(observed)} for epoch in range(5,51,5)]}
            (directory/"summary.json").write_text(json.dumps(summary))
            runs.append(directory)
    return runs


@pytest.fixture
def experiment(tmp_path):
    old = make_protocol(REFERENCE_SEEDS,"fixed_before_depth_supervision_training")
    old["supervisions"] = ["all","terminal"]
    old_path = tmp_path/"reference_protocol.json"
    old_path.write_text(json.dumps(old))
    protocol = make_protocol(NEW_SEEDS,"fixed_before_seed_replication_training")
    protocol.update(phase="exploratory_training_randomness_replication_on_observed_tune",
                    reference_seeds=list(REFERENCE_SEEDS),supervisions=["terminal"],screening_gate=dict(GATE),
                    reference_protocol_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest())
    path = tmp_path/"protocol.json"
    path.write_text(json.dumps(protocol))
    return {"runs":make_runs(tmp_path/"new",NEW_SEEDS,protocol),"protocol":path,
            "old_runs":make_runs(tmp_path/"old",REFERENCE_SEEDS,old),"old_protocol":old_path}


def analyze(experiment,**kwargs):
    return analyze_seed_replication(experiment["runs"],protocol_path=experiment["protocol"],
                                    expected_training_cases=2,expected_development_cases=2,draws=100,**kwargs)


def test_five_new_seeds_primary_gain_stats_and_all_gate_conditions(experiment):
    report = analyze(experiment)
    block = report["primary_new_seeds"]
    expected = .4**2-np.array([.2,.25,.3,.35,.45])**2
    assert block["training_seeds"] == list(NEW_SEEDS)
    assert block["depth_benefit_Brier"]["mean"] == pytest.approx(expected.mean())
    assert block["depth_benefit_Brier"]["std_across_training_seeds"] == pytest.approx(expected.std(ddof=1))
    assert block["depth_benefit_Brier"]["positive_seed_count"] == 4
    assert block["gate"]["passed"] is True
    assert all(condition["passed"] for condition in block["gate"]["conditions"].values())
    assert block["gate"]["automatically_starts_experiments"] is False
    assert block["gate"]["is_statistical_significance_test"] is False
    assert block["Dice_change"]["estimate"] == pytest.approx(.04)
    assert report["secondary_combined_seeds"] is None


def test_endpoint_metrics_average_fits_not_ensemble_predictions(experiment):
    block = analyze(experiment)["primary_new_seeds"]
    expected = np.mean(np.array([.2,.25,.3,.35,.45])**2)
    assert block["regimes"]["depth4_terminal"]["endpoint"]["brier"]["mean"] == pytest.approx(expected)
    assert expected != pytest.approx(np.mean([.2,.25,.3,.35,.45])**2)


def test_case_bootstrap_conditions_on_fitted_seeds_not_ten_cases():
    values = np.stack([np.array([0.,1.])]*5)
    effect = _effect(values,NEW_SEEDS,draws=2000,bootstrap_seed=17)
    assert effect["ci95"] == [0.,1.]
    assert effect["bootstrap"]["n_cases"] == 2
    assert effect["bootstrap"]["n_fitted_seeds"] == 5
    assert effect["bootstrap"]["includes_training_seed_sampling_uncertainty"] is False


def test_high_confidence_error_bounds_are_inclusive_and_pathology_specific():
    y = np.array([1,1,0,0,1,0])
    p = np.array([.05,.051,.95,.949,.99,.01])
    diagnostic = _high_confidence_errors(y,p)
    assert diagnostic["n_high_confidence_predictions"] == 4
    assert diagnostic["n_high_confidence_errors"] == 2
    assert diagnostic["high_confidence_false_negatives"] == 1
    assert diagnostic["high_confidence_false_positives"] == 1
    assert diagnostic["fraction_of_all_cases"] == pytest.approx(2/6)


def test_saturation_audit_counts_exact_case_probability_boundaries():
    diagnostic = _high_confidence_errors(np.array([0,1,0,1]),np.array([0.,1.,1.,0.]))
    assert diagnostic["n_probability_exact_zero"] == 2
    assert diagnostic["n_probability_exact_one"] == 2
    assert diagnostic["n_wrong_at_probability_boundary"] == 2


def test_corrections_deteriorations_are_classification_transitions_not_brier_changes():
    y = np.array([0,1,0,1])
    early = np.array([.7,.7,.1,.1])
    late = np.array([.3,.3,.2,.2])
    diagnostic = _transitions(y,early,late)
    assert diagnostic == {"classification_threshold":.5,"corrected_cases":1,"deteriorated_cases":1,
                          "both_correct_cases":1,"both_wrong_cases":1}


def test_loss_concentration_reports_sensitivity_without_removing_primary_cases():
    diagnostic = _concentration(np.array([.9,.1,-.2,0.,0.,0.]))
    assert diagnostic["improved_brier_cases"] == 2
    assert diagnostic["worsened_brier_cases"] == 1
    assert diagnostic["largest_1_positive_gain_share"] == pytest.approx(.9)
    assert diagnostic["largest_1_absolute_change_share"] == pytest.approx(.9/1.2)
    assert diagnostic["mean_gain_after_removing_top_1_gain_cases"] == pytest.approx(-.02)
    assert diagnostic["primary_cases_removed"] is False


def test_old_three_cannot_hide_failed_new_seed_gate(experiment):
    for directory in experiment["runs"]:
        if "depth4" in directory.name:
            update_endpoint(directory,.42)
    report = analyze(experiment,reference_run_dirs=experiment["old_runs"],reference_protocol_path=experiment["old_protocol"])
    assert report["primary_new_seeds"]["gate"]["passed"] is False
    assert report["primary_new_seeds"]["depth_benefit_Brier"]["mean"] < 0
    combined = report["secondary_combined_seeds"]
    assert combined["depth_benefit_Brier"]["mean"] > 0
    assert combined["n_seeds"] == 8
    assert combined["gate"] is None
    assert combined["can_override_new_seed_gate"] is False
    assert combined["training_seeds"] == list(REFERENCE_SEEDS+NEW_SEEDS)


def test_four_positive_seeds_not_enough_if_one_seed_dominates_leave_one_out(experiment):
    for index,directory in enumerate(experiment["runs"]):
        if "depth4" in directory.name:
            error = (.001,.399,.399,.399,.8)[NEW_SEEDS.index(int(directory.name.split("seed")[-1]))]
            update_endpoint(directory,error)
    gate = analyze(experiment)["primary_new_seeds"]["gate"]
    assert gate["conditions"]["positive_gain_at_least_four_of_five"]["passed"] is True
    assert gate["conditions"]["all_five_leave_one_seed_out_means_positive"]["passed"] is False
    assert gate["passed"] is False


def test_brier_consistency_does_not_override_worse_nll_safeguard(experiment):
    for directory in experiment["runs"]:
        depth = json.loads((directory/"summary.json").read_text())["config"]["steps"]
        if depth==2:
            update_endpoint(directory,.71)
        else:
            fields,rows = csv_read(directory/"development_predictions.csv")
            for row in rows:
                if int(row["step"])==4:
                    row["probability"] = .99 if row["label"]=="0" else 1.
            csv_write(directory/"development_predictions.csv",rows,fields)
            values = metrics(rows,4)
            json_edit(directory/"summary.json",lambda s:s.update(development_subset=values))
            json_edit(directory/"summary.json",lambda s:s["development_history"][-1].update(by_step=values))
    primary = analyze(experiment)["primary_new_seeds"]
    assert primary["depth_benefit_Brier"]["positive_seed_count"] == 5
    gate = primary["gate"]
    assert gate["conditions"]["positive_gain_at_least_four_of_five"]["passed"] is True
    assert gate["conditions"]["all_five_leave_one_seed_out_means_positive"]["passed"] is True
    assert gate["conditions"]["mean_nll_not_worse"]["passed"] is False
    assert gate["passed"] is False


def test_no_private_identifiers_paths_vectors_or_unknown_config_in_public_output(experiment,tmp_path):
    report = analyze(experiment,reference_run_dirs=experiment["old_runs"],reference_protocol_path=experiment["old_protocol"])
    text = json.dumps(report,allow_nan=False)
    for private in ("private_train","private_tune","private_local_filename",str(tmp_path)):
        assert private not in text
    assert "unknown_field" not in report["config"]
    assert report["clinical_validation"] is False


@pytest.mark.parametrize("mutation",["missing","duplicate"])
def test_exactly_ten_distinct_new_fit_directories_required(experiment,mutation):
    experiment["runs"] = experiment["runs"][:-1] if mutation=="missing" else experiment["runs"][:-1]+experiment["runs"][:1]
    with pytest.raises(ValueError,match="distinct"):
        analyze(experiment)


@pytest.mark.parametrize("change",[lambda s:s["config"].update(seed=17),lambda s:s["config"].update(steps=3),
                                   lambda s:s["config"].update(supervision="all"),lambda s:s["config"].update(epochs=49),
                                   lambda s:s["config"].update(batch_size=True),lambda s:s["config"].update(learning_rate=.001),
                                   lambda s:s.update(training_source_sha256="e"*64),lambda s:s.update(initialization_sha256="f"*64),
                                   lambda s:s["timing"].update(updates=49),lambda s:s.update(evaluation_split="test"),
                                   lambda s:s.update(model_selection="best_tune_epoch"),
                                   lambda s:s["readout_supervision"].update(directly_supervised_steps=[1,2,3,4])])
def test_run_provenance_completion_config_and_initialization_are_locked(experiment,change):
    json_edit(experiment["runs"][-1]/"summary.json",change)
    with pytest.raises(ValueError):
        analyze(experiment)


@pytest.mark.parametrize("split",["train","cal","calibration","test",""])
def test_development_csv_must_explicitly_originate_only_from_tune(experiment,split):
    csv_edit(experiment["runs"][-1]/"development_predictions.csv",lambda rows:rows[0].update(split=split))
    with pytest.raises(ValueError,match="train/tune split"):
        analyze(experiment)


@pytest.mark.parametrize("field,value",[("case_id","private_different_case"),("image_id","private_different_view"),("label","1")])
def test_equal_counts_do_not_hide_different_case_view_label_sets(experiment,field,value):
    def change(rows):
        for row in rows:
            if row["case_id"]=="private_tune_case0" and (field!="image_id" or row["image_id"]=="private_tune_image0_0"):
                row[field] = value
    csv_edit(experiment["runs"][-1]/"development_predictions.csv",change)
    with pytest.raises(ValueError):
        analyze(experiment)


@pytest.mark.parametrize("change",[lambda rows:rows.append(copy.deepcopy(rows[0])),lambda rows:rows[0].update(probability="nan"),
                                   lambda rows:rows[0].update(step=5),lambda rows:rows[0].update(seed=53),
                                   lambda rows:rows[0].update(dice="")])
def test_invalid_prediction_trace_records_rejected(experiment,change):
    csv_edit(experiment["runs"][-1]/"development_predictions.csv",change)
    with pytest.raises(ValueError):
        analyze(experiment)


@pytest.mark.parametrize("change",[lambda s:s["development_subset"]["4"].update(brier=.123),
                                   lambda s:s["development_history"][-1]["by_step"]["4"].update(n_cases=3),
                                   lambda s:s["development_history"].pop(),
                                   lambda s:s["cohort_selection"].update(tune_fingerprint="e"*64)])
def test_declared_metrics_history_and_cohort_must_match_trace_and_protocol(experiment,change):
    json_edit(experiment["runs"][-1]/"summary.json",change)
    with pytest.raises(ValueError):
        analyze(experiment)


@pytest.mark.parametrize("change",[lambda p:p.update(seeds=[17,29,43,53,71]),lambda p:p.update(supervisions=["all"]),
                                   lambda p:p.update(warm_starts=True),lambda p:p.update(training_updates_per_run=49),
                                   lambda p:p["screening_gate"].update(minimum_positive_seeds=3),
                                   lambda p:p["screening_gate"].update(maximum_mean_nll_worsening=.1),
                                   lambda p:p["tune"].update(fingerprint="e"*64),
                                   lambda p:p["initialization_sha256_by_seed"].pop("131")])
def test_protocol_requires_exact_new_seeds_fixed_gate_and_cohort(experiment,change):
    json_edit(experiment["protocol"],change)
    with pytest.raises(ValueError):
        analyze(experiment)


def test_reference_fit_and_reference_protocol_must_be_supplied_together(experiment):
    with pytest.raises(ValueError,match="together"):
        analyze(experiment,reference_run_dirs=experiment["old_runs"])


def test_reference_protocol_bytes_are_pinned_before_replication(experiment):
    json_edit(experiment["old_protocol"],lambda p:p.update(unknown="changed"))
    with pytest.raises(ValueError,match="protocol bytes"):
        analyze(experiment,reference_run_dirs=experiment["old_runs"],reference_protocol_path=experiment["old_protocol"])


def test_old_reference_fits_are_validated_not_blindly_merged(experiment):
    json_edit(experiment["old_runs"][-1]/"summary.json",lambda s:s.update(training_source_sha256="e"*64))
    with pytest.raises(ValueError,match="source digests"):
        analyze(experiment,reference_run_dirs=experiment["old_runs"],reference_protocol_path=experiment["old_protocol"])


def test_cli_aggregate_output_refuses_overwrite(experiment,tmp_path,capsys):
    output = tmp_path/"analysis.json"
    args = ["--run-dirs",*map(str,experiment["runs"]),"--protocol",str(experiment["protocol"]),
            "--output",str(output),"--expected-training-cases","2","--expected-development-cases","2","--draws","50"]
    assert main(args)==0
    before = output.read_bytes()
    assert json.loads(before)["status"]=="executed_cpu_seed_replication_analysis"
    assert "private_tune" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(args)
    assert output.read_bytes()==before
