"""Paired sharing/depth signs, pathology tradeoffs, provenance and privacy."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from loop_ultrasound.analyze_architecture import (_paired_transitions,_pathology,_protocol,_trace,analyze_architecture,main)
from loop_ultrasound.analyze_depth_supervision import _metrics
from loop_ultrasound.analyze_tail_lr import SEEDS,DEPTHS,POLICIES
from test_analyze_tail_lr import (experiment,_csv_edit,_json_edit,_rows,_sha,_write_csv)


UNTIED_ERRORS = {"constant":{2:[.2,.45,.55,.6,.96],4:[.15,.2,.7,.97,.95]},
                "low_tail":{2:[.15,.4,.5,.5,.8],4:[.1,.15,.5,.96,.9]}}


def parameter_counts(arm,depth):
    core = 2 if arm=="SJ" else 2*depth
    return {"total":core+5,"trainable":core+5,"core":core,"core_and_classification":core+1,"mask_head":4,
            "independent_blocks":1 if arm=="SJ" else depth,"block_applications_at_full_depth":depth}


@pytest.fixture
def architecture(experiment,tmp_path):
    shared = experiment
    p_shared = json.loads(shared["protocol"].read_text())
    frozen = 100
    for directory in shared["runs"]:
        (directory/"checkpoint.pt").write_bytes(b"synthetic shared endpoint")
        _json_edit(directory/"summary.json",lambda s:s.update(parameters={"trainable":7,"encoder_frozen":frozen}))
    for seed in SEEDS:
        for depth in DEPTHS:
            constant = shared["root"]/f"SJ-depth{depth}-constant-seed{seed}"
            low = shared["root"]/f"SJ-depth{depth}-low_tail-seed{seed}"
            _json_edit(low/"summary.json",lambda s:s["constant_replay_verification"].update(constant_summary_sha256=_sha(constant/"summary.json")))
    root = tmp_path/"untied";root.mkdir()
    init = {f"seed{seed}_depth{depth}":hashlib.sha256(f"UJ-{seed}-{depth}".encode()).hexdigest() for seed in SEEDS for depth in DEPTHS}
    pairing = {f"seed{seed}_depth{depth}":{"canonical_shared_sha256":p_shared["initialization_sha256_by_seed"][str(seed)],
               "all_initial_block_and_head_values_match":True,"untied_parameter_storage_distinct":True,
               "torch_rng_after_initialization_matches":True} for seed in SEEDS for depth in DEPTHS}
    protocol = {**p_shared,"status":"fixed_before_cpu_architecture_comparison","phase":"exploratory_sharing_depth_comparison_on_observed_tune",
                "arms":["SJ","UJ"],"new_training_arms":["UJ"],"new_prefixes":10,"new_endpoint_trajectories":20,
                "reused_shared_endpoints":20,"effective_endpoint_trajectories":40,"launcher_source_sha256":"e"*64,
                "helper_source_sha256":"d"*64,"shared_launcher_source_sha256":"d"*64,
                "shared_protocol_sha256":_sha(shared["protocol"]),"shared_reference_protocol_sha256":_sha(shared["reference_protocol"]),
                "common_initialization_sha256_by_seed":p_shared["initialization_sha256_by_seed"],
                "initialization_sha256_by_seed_depth":init,"initial_component_pairing_by_seed_depth":pairing,
                "parameters_by_arm_depth":{f"{arm}{depth}":parameter_counts(arm,depth) for arm in ("SJ","UJ") for depth in DEPTHS},
                "encoder_frozen_parameters":frozen,"temperature_fits_or_threshold_search":False,"lesion_intervention_executed":False,
                "error_diagnostics":{"classification_threshold":.5,"high_confidence_false_negative_max":.05,
                                     "high_confidence_false_positive_min":.95,"bounds":"inclusive","unit":"Case",
                                     "cross_model_changes_are_not_internal_correction":True},
                "bootstrap":{"draws":2000,"seed":17,"unit":"Case","conditional_on_fitted_models":True,
                             "truth_specific_effects_resample_within_pathology_class":True}}
    bindings,prebindings = {},{}
    for seed in SEEDS:
        for depth in DEPTHS:
            prefix = shared["root"]/f"prefix-depth{depth}-seed{seed}"
            prebindings[f"seed{seed}_depth{depth}"] = {field:_sha(prefix/filename) for field,filename in
                (("prefix_summary_sha256","prefix_summary.json"),("checkpoint_sha256","checkpoint.pt"),
                 ("training_predictions_sha256","training_predictions.csv"),("development_predictions_sha256","development_predictions.csv"))}
            for policy in POLICIES:
                path = shared["root"]/f"SJ-depth{depth}-{policy}-seed{seed}"
                bindings[f"seed{seed}_depth{depth}_{policy}"] = {field:_sha(path/filename) for field,filename in
                    (("summary_sha256","summary.json"),("checkpoint_sha256","checkpoint.pt"),
                     ("training_predictions_sha256","training_predictions.csv"),("development_predictions_sha256","development_predictions.csv"),
                     ("history_sha256","history.json"))}
    protocol["shared_run_artifact_bindings"],protocol["shared_prefix_artifact_bindings"] = bindings,prebindings
    runs = []
    for index,seed in enumerate(SEEDS):
        for depth in DEPTHS:
            key = f"seed{seed}_depth{depth}"
            prefix = root/f"prefix-depth{depth}-seed{seed}";prefix.mkdir()
            shared_prefix = shared["root"]/prefix.name
            for split,filename in (("train","training_predictions.csv"),("tune","development_predictions.csv")):
                rows = _rows(seed,depth,split,.45 if depth==2 else .4)
                for row in rows:row["arm"]="UJ"
                _write_csv(prefix/filename,rows)
            (prefix/"checkpoint.pt").write_bytes(f"synthetic untied full state {seed} {depth}".encode())
            state = hashlib.sha256(f"untied-state-{seed}-{depth}".encode()).hexdigest()
            pre = json.loads((shared_prefix/"prefix_summary.json").read_text())
            pre.update(status="executed_cpu_architecture_prefix",launcher_source_sha256="e"*64,helper_source_sha256="d"*64,
                       initialization_sha256=init[key],common_initialization_sha256=pairing[key]["canonical_shared_sha256"],initial_component_pairing=pairing[key],
                       checkpoint_sha256=_sha(prefix/"checkpoint.pt"),state_sha256=state,
                       training_predictions_sha256=_sha(prefix/"training_predictions.csv"),development_predictions_sha256=_sha(prefix/"development_predictions.csv"))
            pre["config"]["arm"]="UJ"
            objective = json.loads((shared_prefix/"history.json").read_text())
            pre["history"]=objective
            (prefix/"prefix_summary.json").write_text(json.dumps(pre))
            constant_path = None
            for policy in POLICIES:
                path = root/f"UJ-depth{depth}-{policy}-seed{seed}";path.mkdir()
                shared_branch = shared["root"]/f"SJ-depth{depth}-{policy}-seed{seed}"
                summary = json.loads((shared_branch/"summary.json").read_text())
                error = UNTIED_ERRORS[policy][depth][index]
                for split,filename in (("train","training_predictions.csv"),("tune","development_predictions.csv")):
                    rows = _rows(seed,depth,split,error)
                    for row in rows:row["arm"]="UJ"
                    _write_csv(path/filename,rows)
                trace = _trace(path/"development_predictions.csv",arm="UJ",seed=seed,depth=depth,split="tune")
                observed = {str(step):_metrics(trace[step]) for step in trace}
                summary.update(status="executed_cpu_architecture_control",launcher_source_sha256="e"*64,helper_source_sha256="d"*64,
                               initialization_sha256=init[key],common_initialization_sha256=pairing[key]["canonical_shared_sha256"],initial_component_pairing=pairing[key],
                               parameters={**parameter_counts("UJ",depth),"encoder_frozen":frozen},development_subset=observed,
                               development_history=copy.deepcopy(pre["development_history"])+[{"epoch":25,"by_step":observed}])
                summary["config"]["arm"]="UJ"
                summary["branch_provenance"].update(prefix_checkpoint_sha256=pre["checkpoint_sha256"],prefix_state_sha256=state,resumed_state_sha256=state)
                if policy=="constant":
                    constant_path=path;summary["paired_policy_verification"]={"status":"first_constant_branch_from_verified_prefix"}
                else:
                    summary["paired_policy_verification"]={"status":"full_state_policy_pair_verified","constant_summary_sha256":_sha(constant_path/"summary.json")}
                (path/"summary.json").write_text(json.dumps(summary))
                (path/"history.json").write_bytes((shared_branch/"history.json").read_bytes())
                (path/"checkpoint.pt").write_bytes(f"synthetic endpoint {seed} {depth} {policy}".encode());runs.append(path)
    protocol_path=tmp_path/"architecture_protocol.json";protocol_path.write_text(json.dumps(protocol))
    return {"runs":runs,"protocol":protocol_path,"shared":shared,"root":root}


def analyze(e,**kwargs):
    s=e["shared"]
    return analyze_architecture(e["runs"],protocol_path=e["protocol"],shared_run_dirs=s["runs"],shared_protocol_path=s["protocol"],
                                shared_reference_run_dirs=s["reference_runs"],shared_reference_protocol_path=s["reference_protocol"],
                                expected_training_cases=4,expected_development_cases=2,draws=100,**kwargs)


def test_known_sharing_depth_and_theta_signs_from_separately_fitted_endpoints(architecture):
    report=analyze(architecture)
    sj2=np.full(5,.4);sj4=np.array([.2,.25,.3,.35,.4]);uj2=np.array(UNTIED_ERRORS["constant"][2]);uj4=np.array(UNTIED_ERRORS["constant"][4])
    sharing2=uj2**2-sj2**2
    assert report["sharing_benefit_by_depth_policy"]["depth2_constant"]["Brier_benefit"]["estimate"]==pytest.approx(sharing2.mean())
    assert report["sharing_benefit_by_depth_policy"]["depth2_constant"]["Brier_benefit"]["std_across_training_seeds"]==pytest.approx(sharing2.std(ddof=1))
    expected_theta=(sj2**2-sj4**2)-(uj2**2-uj4**2)
    assert report["theta_by_policy"]["constant"]["estimate"]==pytest.approx(expected_theta.mean())
    assert [row["value"] for row in report["theta_by_policy"]["constant"]["by_seed"]]==pytest.approx(expected_theta)
    assert report["theta_by_policy"]["constant"]["bootstrap"]["n_cases"]==2
    assert report["theta_by_policy"]["constant"]["bootstrap"]["n_fitted_seeds"]==5
    assert report["compute_budget"]["total_actual_new_updates"]==600


def test_endpoint_means_are_fit_metrics_not_ensembles(architecture):
    report=analyze(architecture)
    errors=np.array(UNTIED_ERRORS["constant"][4])
    assert report["regimes"]["UJ"]["depth4_constant"]["endpoint"]["brier"]["mean"]==pytest.approx(np.mean(errors**2))
    assert np.mean(errors**2)!=pytest.approx(errors.mean()**2)


def test_pathology_counts_and_high_confidence_bounds_are_label_specific():
    labels=np.array([0,0,0,1,1,1]);p=np.array([.95,.7,.1,.05,.3,.9])
    diagnostic=_pathology(labels,p)
    assert diagnostic["benign"]["error_count"]==2
    assert diagnostic["malignant"]["error_count"]==2
    assert diagnostic["benign"]["high_confidence_error_count"]==1
    assert diagnostic["malignant"]["high_confidence_error_count"]==1
    assert diagnostic["malignant"]["high_confidence_error_rate"]==pytest.approx(1/3)


def test_class_conditional_transition_rate_CIs_and_direction(architecture):
    report=analyze(architecture)
    change=report["paired_transitions"]["sharing"]["depth2_constant"]
    # UJ seeds89/107/131 are wrong for both labels; SJ is correct.
    for label in ("benign","malignant"):
        assert change[label]["summary"]["corrected_cases"]["mean"]==pytest.approx(.6)
        assert change[label]["error_rate_benefit"]["estimate"]==pytest.approx(.6)
        assert change[label]["error_rate_benefit"]["bootstrap"]["n_cases"]==1
        assert change[label]["error_rate_benefit"]["ci95"]==pytest.approx([.6,.6])
        assert change[label]["high_confidence_error_rate_benefit"]["estimate"]==pytest.approx(.2)
    assert change["benign"]["signed_probability_change"]["estimate"]<0
    assert change["malignant"]["signed_probability_change"]["estimate"]>0


def test_concentration_and_leave_one_seed_out_are_diagnostic_only(architecture):
    effect=analyze(architecture)["theta_by_policy"]["constant"]
    assert effect["case_concentration"]["primary_cases_removed"] is False
    values=np.array([row["value"] for row in effect["by_seed"]])
    assert [row["mean"] for row in effect["leave_one_seed_out"]]==pytest.approx([np.delete(values,i).mean() for i in range(5)])


def test_public_report_has_no_case_ids_local_paths_or_unknown_fields(architecture,tmp_path):
    report=analyze(architecture);encoded=json.dumps(report,allow_nan=False)
    for private in ("private_train","private_tune","private_local_filename",str(tmp_path)):
        assert private not in encoded
    assert report["clinical_validation"] is False
    assert report["decision_rule_automatically_applied"] is False
    assert report["compute_budget"]["optimizer_update_matching_is_not_FLOP_matching"] is True


@pytest.mark.parametrize("change",[lambda s:s["config"].update(arm="SJ"),lambda s:s["config"].update(seed=17),
                                   lambda s:s["config"].update(supervision="all"),lambda s:s["timing"].update(new_updates_in_branch=9),
                                   lambda s:s.update(common_initialization_sha256="f"*64),lambda s:s.update(initialization_sha256="f"*64),
                                   lambda s:s["initial_component_pairing"].update(untied_parameter_storage_distinct=False),
                                   lambda s:s["initial_component_pairing"].update(torch_rng_after_initialization_matches=False),
                                   lambda s:s.update(helper_source_sha256="f"*64),lambda s:s["parameters"].update(independent_blocks=1),
                                   lambda s:s["development_subset"]["4"].update(brier=.123)])
def test_untied_source_recipe_pairing_and_actual_model_schema_refused(architecture,change):
    _json_edit(architecture["runs"][-1]/"summary.json",change)
    with pytest.raises(ValueError):analyze(architecture)


@pytest.mark.parametrize("split",["train","test","cal","calibration",""])
def test_tune_prediction_provenance_is_explicit(architecture,split):
    _csv_edit(architecture["runs"][-1]/"development_predictions.csv",lambda rows:rows[0].update(split=split))
    with pytest.raises(ValueError,match="train/tune-only"):analyze(architecture)


@pytest.mark.parametrize("field,value",[("case_id","private_other_case"),("image_id","private_other_view"),("label","1"),("arm","SJ"),("seed","53")])
def test_aligned_case_views_labels_and_arm_seed_are_exact(architecture,field,value):
    def change(rows):
        for row in rows:
            if field in ("arm","seed") or row["case_id"]=="private_tune_case0":
                row[field]=value
    _csv_edit(architecture["runs"][-1]/"development_predictions.csv",change)
    with pytest.raises(ValueError):analyze(architecture)


@pytest.mark.parametrize("change",[lambda s:s["branch_provenance"].update(restored_rng=False),
                                   lambda s:s["branch_provenance"].update(resumed_state_sha256="f"*64),
                                   lambda s:s["sampling_trace_sha256"].update(tail="f"*64),
                                   lambda s:s["paired_policy_verification"].update(constant_summary_sha256="f"*64)])
def test_cloned_prefix_state_sampling_and_constant_pair_binding_refused(architecture,change):
    _json_edit(architecture["runs"][-1]/"summary.json",change)
    with pytest.raises(ValueError):analyze(architecture)


def test_equal_sampling_within_untied_pair_cannot_hide_different_architecture_trajectory(architecture):
    for path in architecture["runs"]:
        _json_edit(path/"summary.json",lambda s:s["sampling_trace_sha256"].update(tail="f"*64))
    # Re-bind low summaries to their modified constants so only cross-arm
    # sampling, rather than a stale pairing hash, rejects this mismatch.
    for seed in SEEDS:
        for depth in DEPTHS:
            constant=architecture["root"]/f"UJ-depth{depth}-constant-seed{seed}"
            low=architecture["root"]/f"UJ-depth{depth}-low_tail-seed{seed}"
            _json_edit(low/"summary.json",lambda s:s["paired_policy_verification"].update(constant_summary_sha256=_sha(constant/"summary.json")))
    with pytest.raises(ValueError,match="architectures/depths/policies"):analyze(architecture)


def test_changed_single_view_with_equal_case_counts_is_not_intersected(architecture):
    def change(rows):
        for row in rows:
            if row["image_id"]=="private_tune_view0_0":row["image_id"]="private_other_view"
    _csv_edit(architecture["runs"][-1]/"development_predictions.csv",change)
    with pytest.raises(ValueError,match="view sets"):analyze(architecture)


def test_pathology_case_weighted_effects_reconstruct_overall_sharing_contrast(architecture):
    report=analyze(architecture)
    change=report["paired_transitions"]["sharing"]["depth2_constant"]
    weighted=sum(value["n_cases"]*value["Brier_benefit"]["estimate"] for value in change.values())/sum(value["n_cases"] for value in change.values())
    assert weighted==pytest.approx(report["sharing_benefit_by_depth_policy"]["depth2_constant"]["Brier_benefit"]["estimate"])


def test_reused_shared_checkpoint_bytes_are_pinned(architecture):
    path=architecture["shared"]["runs"][-1]/"checkpoint.pt"
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="shared branch artifact"):analyze(architecture)


@pytest.mark.parametrize("change",[lambda p:p.update(effective_endpoint_trajectories=20),lambda p:p.update(total_actual_new_updates=1000),
                                   lambda p:p.update(temperature_fits_or_threshold_search=True),
                                   lambda p:p["initial_component_pairing_by_seed_depth"]["seed53_depth2"].update(all_initial_block_and_head_values_match=False)])
def test_protocol_preserves_budget_scope_and_locked_initial_certificate(architecture,change):
    _json_edit(architecture["protocol"],change)
    with pytest.raises(ValueError):analyze(architecture)


def test_exactly_twenty_new_endpoints_no_duplicates(architecture):
    architecture["runs"]=architecture["runs"][:-1]+architecture["runs"][:1]
    with pytest.raises(ValueError,match="distinct"):analyze(architecture)


def test_cli_publication_refuses_overwrite(architecture,tmp_path,capsys):
    s=architecture["shared"];output=tmp_path/"aggregate.json"
    args=["--run-dirs",*map(str,architecture["runs"]),"--protocol",str(architecture["protocol"]),
          "--shared-run-dirs",*map(str,s["runs"]),"--shared-protocol",str(s["protocol"]),
          "--shared-reference-run-dirs",*map(str,s["reference_runs"]),"--shared-reference-protocol",str(s["reference_protocol"]),
          "--expected-training-cases","4","--expected-development-cases","2","--draws","50","--output",str(output)]
    assert main(args)==0
    before=output.read_bytes()
    assert json.loads(before)["status"]=="executed_cpu_architecture_analysis"
    assert "private_tune" not in capsys.readouterr().out
    with pytest.raises(SystemExit):main(args)
    assert output.read_bytes()==before
