"""Paired-Case sharing, depth and tail-policy analysis on observed tune.

Forty separately fitted endpoints use identical Cases, labels and sampled
views. Existing shared runs undergo their original replay validation; new
untied runs certify cloned full-state policy forks and paired component values.
Only aggregate metrics, counts and artifact digests leave this analyzer.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .analyze import _hash,_integer,_match_cases,_read_predictions
from .analyze_depth_supervision import FINAL_SELECTION,SOURCE_KEYS,_declared_metrics,_json,_metrics
from .analyze_expanded import AGGREGATE_METRICS
from .analyze_seed_replication import (_cohort_check,_concentration,_effect,_high_confidence_errors,_nll_per_case,_stats)
from .analyze_tail_lr import (DEPTHS,POLICIES,RATES,SEEDS,FIXED_CONFIG,_cohort_declaration,
                              _exact_config,_file_hash,_history,_objective_history,_objective_values,
                              analyze_tail_lr)
from .metrics import aggregate_cases


ARMS = ("SJ","UJ")
LABELS = {"benign":0,"malignant":1}


def regime_name(depth,policy):
    return f"depth{depth}_{policy}"


def _bindings(value,keys,fields,message):
    if not isinstance(value,dict) or set(value)!=keys:
        raise ValueError(message)
    for item in value.values():
        if not isinstance(item,dict) or set(item)!=fields:
            raise ValueError(message)
        for digest in item.values():
            _hash(digest,"locked artifact digest")


def _protocol(path,training_cases,development_cases):
    p = _json(path,"A readable locked architecture protocol is required.")
    if (p.get("status")!="fixed_before_cpu_architecture_comparison"
            or p.get("phase")!="exploratory_sharing_depth_comparison_on_observed_tune"
            or p.get("arms")!=list(ARMS) or p.get("new_training_arms")!=["UJ"]
            or p.get("depths")!=list(DEPTHS) or p.get("policies")!=list(POLICIES) or p.get("seeds")!=list(SEEDS)
            or p.get("clinical_validation") is not False or p.get("device")!="cpu"
            or p.get("patient_mapping")!="unverified_Case_grouped_only" or p.get("model_selection")!=FINAL_SELECTION
            or p.get("calibration_and_test_predictions") is not False or p.get("full_state_fork") is not True
            or p.get("warm_starts") is not False or p.get("supervisions")!=["terminal"]
            or p.get("prefix_epochs")!=20 or p.get("tail_epochs")!=5 or p.get("tail_learning_rates")!=RATES
            or p.get("temperature_fits_or_threshold_search") is not False or p.get("lesion_intervention_executed") is not False):
        raise ValueError("Protocol must lock the forty-endpoint, paired terminal-only CPU architecture comparison.")
    _exact_config(p,{key:value for key,value in FIXED_CONFIG.items() if key not in {"arm","supervision"}},
                  "Architecture protocol differs from the fixed CPU recipe.")
    per_epoch = math.ceil(training_cases/2)
    for key,value in {"updates_per_epoch":per_epoch,"prefix_updates":20*per_epoch,"tail_updates":5*per_epoch,
                      "training_updates_per_run":25*per_epoch,"total_actual_new_updates":10*30*per_epoch}.items():
        if _integer(p.get(key),"architecture update budget",1)!=value:
            raise ValueError("Architecture prefix, branch or total actual-new-work budgets are inconsistent.")
    for key,value in {"new_prefixes":10,"new_endpoint_trajectories":20,"reused_shared_endpoints":20,"effective_endpoint_trajectories":40}.items():
        if _integer(p.get(key),"architecture trajectory count",1)!=value:raise ValueError("Architecture counts must separate twenty reused and twenty new endpoints.")
    for key in SOURCE_KEYS+("launcher_source_sha256","helper_source_sha256","shared_launcher_source_sha256","shared_protocol_sha256","shared_reference_protocol_sha256"):
        _hash(p.get(key),"locked architecture source digest")
    if p["helper_source_sha256"]!=p["shared_launcher_source_sha256"]:raise ValueError("The reused shared launcher must be the common locked helper.")
    for split,count in (("train",training_cases),("tune",development_cases)):
        _cohort_declaration(p.get(split),count)
        if min(p[split]["benign"],p[split]["malignant"])<1:
            raise ValueError("Both pathology classes are required in each locked architecture cohort.")
    _exact_config(p.get("error_diagnostics"),
                  {"classification_threshold":.5,"high_confidence_false_negative_max":.05,
                   "high_confidence_false_positive_min":.95,"bounds":"inclusive","unit":"Case",
                   "cross_model_changes_are_not_internal_correction":True},
                  "Pathology diagnostics must use the prespecified fixed thresholds and cross-model interpretation.")
    _exact_config(p.get("bootstrap"),{"draws":2000,"seed":17,"unit":"Case","conditional_on_fitted_models":True,
                                      "truth_specific_effects_resample_within_pathology_class":True},
                  "Architecture protocol must lock the Case-conditional bootstrap and within-class effects.")
    init = p.get("initialization_sha256_by_seed_depth")
    keys = {f"seed{seed}_depth{depth}" for seed in SEEDS for depth in DEPTHS}
    if not isinstance(init,dict) or set(init)!=keys:
        raise ValueError("Protocol must lock all ten whole-untied-model initialization digests.")
    for digest in init.values():_hash(digest,"untied initialization")
    common = p.get("common_initialization_sha256_by_seed")
    if not isinstance(common,dict) or set(common)!={str(seed) for seed in SEEDS}:
        raise ValueError("Protocol must lock canonical shared component initialization at every seed.")
    for digest in common.values():_hash(digest,"shared component initialization")
    pairing = p.get("initial_component_pairing_by_seed_depth")
    if not isinstance(pairing,dict) or set(pairing)!=keys:raise ValueError("Every untied seed/depth requires its locked component-pairing certificate.")
    for seed in SEEDS:
        for depth in DEPTHS:
            _exact_config(pairing[f"seed{seed}_depth{depth}"],
                          {"canonical_shared_sha256":common[str(seed)],"all_initial_block_and_head_values_match":True,
                           "untied_parameter_storage_distinct":True,"torch_rng_after_initialization_matches":True},
                          "Protocol component-value/storage/RNG pairing certificate is incomplete.")
    parameters = p.get("parameters_by_arm_depth")
    if not isinstance(parameters,dict) or set(parameters)!={f"{arm}{depth}" for arm in ARMS for depth in DEPTHS}:
        raise ValueError("Protocol must lock parameter counts at every architecture/depth.")
    param_fields = {"total","trainable","core","core_and_classification","mask_head","independent_blocks","block_applications_at_full_depth"}
    for arm in ARMS:
        for depth in DEPTHS:
            item = parameters[f"{arm}{depth}"]
            if not isinstance(item,dict) or set(item)!=param_fields:raise ValueError("Locked model parameter-count schema is incomplete.")
            for value in item.values():_integer(value,"locked model parameters",1)
            if item["total"]!=item["trainable"] or item["trainable"]!=item["core_and_classification"]+item["mask_head"]:
                raise ValueError("Locked trainable parameter counts are inconsistent.")
            if item["independent_blocks"]!=(1 if arm=="SJ" else depth) or item["block_applications_at_full_depth"]!=depth:
                raise ValueError("Locked block counts must reflect tied versus untied architecture.")
    _integer(p.get("encoder_frozen_parameters"),"frozen encoder parameters",1)
    _bindings(p.get("shared_run_artifact_bindings"),
              {f"seed{seed}_depth{depth}_{policy}" for seed in SEEDS for depth in DEPTHS for policy in POLICIES},
              {"summary_sha256","checkpoint_sha256","training_predictions_sha256","development_predictions_sha256","history_sha256"},
              "Protocol must bind all twenty shared branch artifact groups.")
    _bindings(p.get("shared_prefix_artifact_bindings"),keys,
              {"prefix_summary_sha256","checkpoint_sha256","training_predictions_sha256","development_predictions_sha256"},
              "Protocol must bind all ten shared prefix artifact groups.")
    return p


def _trace(path,*,arm,seed,depth,split):
    try:
        with Path(path).open(newline="") as stream:
            reader = csv.DictReader(stream)
            if "split" not in (reader.fieldnames or []) or any(row.get("split")!=split for row in reader):
                raise ValueError("Architecture prediction traces require explicit train/tune-only provenance.")
    except OSError:
        raise ValueError("Readable architecture prediction traces are required.") from None
    records = _read_predictions(Path(path))
    if any(row["arm"]!=arm or row["seed"]!=seed or row["step"] not in range(1,depth+1) for row in records):
        raise ValueError("Prediction arm, seed or readout disagrees with the separately fitted model.")
    try:
        rows = aggregate_cases(records)
    except ValueError:
        raise ValueError("Prediction values, duplicate views or Case pathology labels are invalid.") from None
    result = {step:{} for step in range(1,depth+1)}
    for row in rows:
        if row["dice"] is None or row["n_dice"]!=row["n_images"]:
            raise ValueError("Every architecture Case view and readout requires finite Dice.")
        result[row["step"]][row["case_id"]] = row
    if not result[1]:raise ValueError("Every configured readout is required.")
    for step in result:_match_cases(result[1],result[step])
    return result


def _untied_summary(summary,p,*,status):
    if (summary.get("status")!=status or summary.get("clinical_validation") is not False or summary.get("device")!="cpu"
            or summary.get("patient_mapping")!="unverified_Case_grouped_only" or summary.get("evaluation_split")!="tune_development"
            or summary.get("model_selection")!=FINAL_SELECTION):
        raise ValueError("Untied summaries must certify completed CPU, fixed-final, tune-only runs.")
    config = summary.get("config")
    _exact_config(config,{**FIXED_CONFIG,"arm":"UJ"},"Untied config differs from the locked terminal-only recipe.")
    seed,depth = _integer(config.get("seed"),"seed"),_integer(config.get("steps"),"depth",1)
    if seed not in SEEDS or depth not in DEPTHS:raise ValueError("Untied run is outside the locked seed/depth design.")
    for key in SOURCE_KEYS+("launcher_source_sha256","helper_source_sha256"):
        if _hash(summary.get(key),key)!=p[key]:raise ValueError("Untied source digests differ from the locked protocol.")
    if _hash(summary.get("initialization_sha256"),"untied initialization")!=p["initialization_sha256_by_seed_depth"][f"seed{seed}_depth{depth}"]:
        raise ValueError("Whole untied initialization differs from its locked seed/depth digest.")
    common = p["common_initialization_sha256_by_seed"][str(seed)]
    if _hash(summary.get("common_initialization_sha256"),"canonical shared initialization")!=common:
        raise ValueError("Untied initial components do not match the canonical paired shared model.")
    _exact_config(summary.get("initial_component_pairing"),
                  {"canonical_shared_sha256":common,"all_initial_block_and_head_values_match":True,
                   "untied_parameter_storage_distinct":True,"torch_rng_after_initialization_matches":True},
                  "Untied blocks/heads, storage and initialization RNG must certify the paired component construction.")
    if summary["initial_component_pairing"]!=p["initial_component_pairing_by_seed_depth"][f"seed{seed}_depth{depth}"]:
        raise ValueError("Untied pairing certificate differs from the exact certificate locked before training.")
    for key,value in (("n_training_cases",p["train"]["cases"]),("n_training_images",p["train"]["images"])):
        if _integer(summary.get(key),key,2)!=value:raise ValueError("Untied training counts differ from the locked cohort.")
    expected_selection = {"sampling":p["sampling"],"selection_seed":p["selection_seed"],
                          "train_fingerprint":p["train"]["fingerprint"],"tune_fingerprint":p["tune"]["fingerprint"]}
    if summary.get("cohort_selection")!=expected_selection:raise ValueError("Untied declared cohort selection differs from the locked protocol.")
    _exact_config(summary.get("readout_supervision"),{"mode":"terminal","directly_supervised_steps":[depth]},
                  "Only the separately trained endpoint may receive direct readout supervision.")
    return seed,depth


def _root(directories,prefix_root,message):
    if prefix_root is not None:return Path(prefix_root)
    parents = {Path(value).parent.resolve() for value in directories}
    if len(parents)!=1:raise ValueError(message)
    return next(iter(parents))


def _collect(directories,*,arm,p,train_reference,tune_reference):
    directories = [Path(value) for value in directories]
    if len(directories)!=20 or len({value.resolve() for value in directories})!=20:
        raise ValueError("Exactly twenty distinct endpoint branches are required per architecture.")
    runs,traces,metrics,paths,common_config = {},{}, {},{},None
    for directory in directories:
        summary = _json(directory/"summary.json","Readable endpoint summaries are required.")
        config = summary.get("config",{})
        if arm=="UJ":seed,depth = _untied_summary(summary,p,status="executed_cpu_architecture_control")
        else:seed,depth = config["seed"],config["steps"]
        policy = config.get("tail_policy")
        if policy not in POLICIES:raise ValueError("Endpoint policy differs from the locked constant/low-tail design.")
        key = (arm,seed,depth,policy)
        if key in runs:raise ValueError("Duplicate architecture/depth/policy/seed combination.")
        _exact_config(config,{"prefix_updates":p["prefix_updates"],"tail_learning_rate":RATES[policy]},
                      "Endpoint tail learning rate or prefix budget differs from the locked intervention.")
        comparable = {k:v for k,v in config.items() if k not in {"seed","steps","tail_policy","tail_learning_rate"}}
        if common_config is not None and comparable!=common_config:raise ValueError("Architecture endpoint configs differ beyond the locked factors.")
        common_config = comparable
        timing = summary.get("timing")
        if (not isinstance(timing,dict) or timing.get("updates")!=p["training_updates_per_run"]
                or timing.get("new_updates_in_branch")!=p["tail_updates"]
                or summary.get("prefix_reused",timing.get("prefix_reused")) is not True):
            raise ValueError("Endpoint conceptual and actual-new-work budgets are inconsistent.")
        tr = _trace(directory/"training_predictions.csv",arm=arm,seed=seed,depth=depth,split="train")
        dev = _trace(directory/"development_predictions.csv",arm=arm,seed=seed,depth=depth,split="tune")
        for step in tr:
            _match_cases(train_reference,tr[step]);_match_cases(tune_reference,dev[step])
        observed = {str(step):_metrics(dev[step]) for step in dev}
        _history(summary,observed,final_epoch=25)
        _objective_history(directory/"history.json",25)
        if "final_training_fit" in summary:
            _declared_metrics(summary["final_training_fit"],{str(step):_metrics(tr[step]) for step in tr},steps=range(1,depth+1))
        if not (directory/"checkpoint.pt").is_file():raise ValueError("Each architecture endpoint requires its local model checkpoint.")
        parameters = summary.get("parameters")
        if not isinstance(parameters,dict):raise ValueError("Endpoint summaries must record actual parameter counts.")
        _integer(parameters.get("trainable"),"trainable parameters",1)
        _integer(parameters.get("encoder_frozen"),"frozen encoder parameters",1)
        if arm=="UJ":
            if parameters.get("independent_blocks")!=depth or parameters.get("block_applications_at_full_depth")!=depth:
                raise ValueError("Untied parameter metadata must record distinct blocks at the configured depth.")
            if parameters.get("total")!=parameters["trainable"]:raise ValueError("All refinement-model parameters must be trainable in this control.")
            _exact_config(parameters,{**p["parameters_by_arm_depth"][f"UJ{depth}"],"encoder_frozen":p["encoder_frozen_parameters"]},
                          "Untied actual parameter counts differ from the locked architecture.")
        else:
            _exact_config(parameters,{"trainable":p["parameters_by_arm_depth"][f"SJ{depth}"]["trainable"],"encoder_frozen":p["encoder_frozen_parameters"]},
                          "Shared actual parameter counts differ from the locked architecture.")
        runs[key],traces[key],metrics[key],paths[key] = summary,dev,observed,directory
    if set(runs)!={(arm,seed,depth,policy) for seed in SEEDS for depth in DEPTHS for policy in POLICIES}:
        raise ValueError("Every expected endpoint architecture/depth/policy/seed combination is required.")
    for depth in DEPTHS:
        counts = [runs[arm,seed,depth,policy]["parameters"] for seed in SEEDS for policy in POLICIES]
        if any(count!=counts[0] for count in counts):raise ValueError("Parameter counts must agree across seeds/policies at a fixed architecture/depth.")
    return runs,traces,metrics,paths


def _prefixes(p,paths,prefix_root,train_reference,tune_reference):
    result = {}
    for seed in SEEDS:
        for depth in DEPTHS:
            directory = prefix_root/f"prefix-depth{depth}-seed{seed}"
            summary = _json(directory/"prefix_summary.json","Every untied policy pair requires its completed common-prefix summary.")
            if _untied_summary(summary,p,status="executed_cpu_architecture_prefix")!=(seed,depth):
                raise ValueError("Untied prefix belongs to a different paired seed/depth.")
            if summary["config"].get("prefix_updates")!=p["prefix_updates"] or summary.get("timing",{}).get("updates")!=p["prefix_updates"]:
                raise ValueError("Untied prefix must complete exactly the locked twenty-epoch budget.")
            for field,filename in (("checkpoint_sha256","checkpoint.pt"),("training_predictions_sha256","training_predictions.csv"),
                                   ("development_predictions_sha256","development_predictions.csv")):
                if _hash(summary.get(field),"prefix artifact digest")!=_file_hash(directory/filename):
                    raise ValueError("Untied prefix artifact bytes disagree with their declared bindings.")
            _hash(summary.get("state_sha256"),"complete prefix state")
            _hash(summary.get("sampling_trace_sha256"),"prefix sampling trajectory")
            tr = _trace(directory/"training_predictions.csv",arm="UJ",seed=seed,depth=depth,split="train")
            dev = _trace(directory/"development_predictions.csv",arm="UJ",seed=seed,depth=depth,split="tune")
            for step in tr:_match_cases(train_reference,tr[step]);_match_cases(tune_reference,dev[step])
            observed = {str(step):_metrics(dev[step]) for step in dev}
            _history(summary,observed,final_epoch=20)
            objective = _objective_values(summary["history"],20) if "history" in summary else _objective_history(directory/"history.json",20)
            for policy in POLICIES:
                path = paths["UJ",seed,depth,policy]
                branch = _json(path/"summary.json","Readable untied branch summaries are required.")
                _exact_config(branch.get("branch_provenance"),
                              {"prefix_checkpoint_sha256":summary["checkpoint_sha256"],"prefix_state_sha256":summary["state_sha256"],
                               "resumed_state_sha256":summary["state_sha256"],"prefix_updates":p["prefix_updates"],
                               "restored_sampler_epoch":20,"restored_loader_generator":True,"restored_rng":True},
                              "Untied branch must certify restoration of the full paired prefix state.")
                sampling = branch.get("sampling_trace_sha256")
                if not isinstance(sampling,dict) or set(sampling)!={"prefix","tail"} or sampling["prefix"]!=summary["sampling_trace_sha256"]:
                    raise ValueError("Untied branch sampling trace differs from its common prefix.")
                _hash(sampling["tail"],"untied tail sampling trace")
                if branch["development_history"][:-1]!=summary["development_history"]:
                    raise ValueError("Untied pre-intervention development histories must match the common prefix.")
                if _objective_history(path/"history.json",25)[:20]!=objective:
                    raise ValueError("Untied pre-intervention objective histories must match the common prefix.")
            constant_path = paths["UJ",seed,depth,"constant"]
            constant = _json(constant_path/"summary.json","Readable constant branch summary is required.")
            low = _json(paths["UJ",seed,depth,"low_tail"]/"summary.json","Readable low-tail branch summary is required.")
            if constant["sampling_trace_sha256"]!=low["sampling_trace_sha256"]:
                raise ValueError("Untied policy branches must use identical Case-view/order sampling.")
            _exact_config(constant.get("paired_policy_verification"),{"status":"first_constant_branch_from_verified_prefix"},
                          "Untied constant branch must certify its verified paired prefix.")
            _exact_config(low.get("paired_policy_verification"),{"status":"full_state_policy_pair_verified", "constant_summary_sha256":_file_hash(constant_path/"summary.json")},
                          "Untied low-tail branch must bind its full-state paired constant branch.")
            result[seed,depth] = summary
    return result


def _pathology(labels,probabilities):
    correct = (probabilities>=.5)==labels
    high_error = ((labels==1)&(probabilities<=.05))|((labels==0)&(probabilities>=.95))
    result = {}
    losses = (probabilities-labels)**2
    nll = _nll_per_case(labels,probabilities)
    for name,label in LABELS.items():
        selected = labels==label
        result[name] = {"n_cases":int(selected.sum()),"error_count":int((~correct&selected).sum()),
                        "error_rate":float((~correct)[selected].mean()),
                        "high_confidence_error_count":int((high_error&selected).sum()),
                        "high_confidence_error_rate":float(high_error[selected].mean()),
                        "mean_probability":float(probabilities[selected].mean()),"brier":float(losses[selected].mean()),"nll":float(nll[selected].mean())}
    return result


def _paired_transitions(labels,before,after,*,draws,bootstrap_seed):
    results = {}
    correct_before,correct_after = (before>=.5)==labels,(after>=.5)==labels
    high_before = ((labels==1)&(before<=.05))|((labels==0)&(before>=.95))
    high_after = ((labels==1)&(after<=.05))|((labels==0)&(after>=.95))
    for name,label in LABELS.items():
        selected = labels==label
        n_cases = int(selected.sum())
        by_seed = []
        for i,seed in enumerate(SEEDS):
            cb,ca,hb,ha = correct_before[i,selected],correct_after[i,selected],high_before[i,selected],high_after[i,selected]
            counts = {"corrected_cases":int((~cb&ca).sum()),"deteriorated_cases":int((cb&~ca).sum()),
                      "both_correct_cases":int((cb&ca).sum()),"both_wrong_cases":int((~cb&~ca).sum()),
                      "high_confidence_errors_removed":int((hb&~ha).sum()),"high_confidence_errors_added":int((~hb&ha).sum())}
            by_seed.append({"seed":seed,**counts,**{key.replace("cases","rate").replace("errors_removed","errors_removed_rate").replace("errors_added","errors_added_rate"):value/n_cases for key,value in counts.items()}})
        results[name] = {"n_cases":n_cases,"by_seed":by_seed,
                         "summary":{key:_stats([row[key] for row in by_seed]) for key in by_seed[0] if key!="seed"},
                         "Brier_benefit":_effect(((before-labels)**2-(after-labels)**2)[:,selected],SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
                         "NLL_benefit":_effect((_nll_per_case(labels,before)-_nll_per_case(labels,after))[:,selected],SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
                         "error_rate_benefit":_effect((correct_after.astype(float)-correct_before.astype(float))[:,selected],SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
                         "high_confidence_error_rate_benefit":_effect((high_before.astype(float)-high_after.astype(float))[:,selected],SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
                         "signed_probability_change":_effect((after-before)[:,selected],SEEDS,draws=draws,bootstrap_seed=bootstrap_seed)}
    return results


def _brier_effect(values,*,draws,bootstrap_seed):
    seed_means = np.asarray(values).mean(axis=1)
    return {**_effect(values,SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
            "case_concentration":_concentration(np.asarray(values).mean(axis=0)),
            "leave_one_seed_out":[{"omitted_seed":seed,"mean":float(np.delete(seed_means,i).mean())} for i,seed in enumerate(SEEDS)]}


def analyze_architecture(run_dirs,*,protocol_path,shared_run_dirs,shared_protocol_path,
                         shared_reference_run_dirs,shared_reference_protocol_path,prefix_root=None,shared_prefix_root=None,
                         draws=2000,bootstrap_seed=17,expected_training_cases=256,expected_development_cases=159):
    draws,bootstrap_seed = _integer(draws,"Case bootstrap draws",2),_integer(bootstrap_seed,"Case bootstrap seed")
    p = _protocol(protocol_path,expected_training_cases,expected_development_cases)
    run_dirs,shared_run_dirs,shared_reference_run_dirs = list(run_dirs),list(shared_run_dirs),list(shared_reference_run_dirs)
    if _file_hash(shared_protocol_path)!=p["shared_protocol_sha256"] or _file_hash(shared_reference_protocol_path)!=p["shared_reference_protocol_sha256"]:
        raise ValueError("Shared control/reference protocol bytes differ from the architecture protocol pins.")
    shared_p = _json(shared_protocol_path,"Readable locked shared-control protocol is required.")
    for key in SOURCE_KEYS+("train","tune","training_updates_per_run","prefix_updates","tail_updates"):
        if shared_p.get(key)!=p.get(key):raise ValueError("Shared/untied sources, cohorts or budgets differ.")
    if shared_p.get("initialization_sha256_by_seed")!=p["common_initialization_sha256_by_seed"]:
        raise ValueError("Shared initial values differ from canonical component pairing.")
    if shared_p.get("launcher_source_sha256")!=p["helper_source_sha256"]:
        raise ValueError("The reused shared trainer must equal the locked common training helper.")
    shared_root = _root(shared_run_dirs,shared_prefix_root,"Shared prefix root is required for split branch directories.")
    new_root = _root(run_dirs,prefix_root,"Untied prefix root is required for split branch directories.")
    analyze_tail_lr(shared_run_dirs,protocol_path=shared_protocol_path,reference_run_dirs=shared_reference_run_dirs,
                    reference_protocol_path=shared_reference_protocol_path,prefix_root=shared_root,draws=draws,bootstrap_seed=bootstrap_seed,
                    expected_training_cases=expected_training_cases,expected_development_cases=expected_development_cases)
    reference_summary = _json(Path(shared_run_dirs[0])/"summary.json","Readable shared source summary is required.")
    seed,depth = reference_summary["config"]["seed"],reference_summary["config"]["steps"]
    tr = _trace(Path(shared_run_dirs[0])/"training_predictions.csv",arm="SJ",seed=seed,depth=depth,split="train")[1]
    dev = _trace(Path(shared_run_dirs[0])/"development_predictions.csv",arm="SJ",seed=seed,depth=depth,split="tune")[1]
    train_images,tune_images = _cohort_check(tr,p["train"]),_cohort_check(dev,p["tune"])
    if set(tr)&set(dev) or train_images&tune_images:raise ValueError("Architecture train/tune Cases or views overlap.")
    sj_runs,sj_dev,sj_metrics,sj_paths = _collect(shared_run_dirs,arm="SJ",p=p,train_reference=tr,tune_reference=dev)
    for key,path in sj_paths.items():
        _,seed,depth,policy = key
        binding = p["shared_run_artifact_bindings"][f"seed{seed}_depth{depth}_{policy}"]
        for field,filename in (("summary_sha256","summary.json"),("checkpoint_sha256","checkpoint.pt"),
                               ("training_predictions_sha256","training_predictions.csv"),("development_predictions_sha256","development_predictions.csv"),
                               ("history_sha256","history.json")):
            if _file_hash(path/filename)!=binding[field]:raise ValueError("Reused shared branch artifact bytes differ from the locked architecture bindings.")
    for seed in SEEDS:
        for depth in DEPTHS:
            path = shared_root/f"prefix-depth{depth}-seed{seed}"
            binding = p["shared_prefix_artifact_bindings"][f"seed{seed}_depth{depth}"]
            for field,filename in (("prefix_summary_sha256","prefix_summary.json"),("checkpoint_sha256","checkpoint.pt"),
                                   ("training_predictions_sha256","training_predictions.csv"),("development_predictions_sha256","development_predictions.csv")):
                if _file_hash(path/filename)!=binding[field]:raise ValueError("Shared prefix artifact bytes differ from their locked architecture bindings.")
    uj_runs,uj_dev,uj_metrics,uj_paths = _collect(run_dirs,arm="UJ",p=p,train_reference=tr,tune_reference=dev)
    _prefixes(p,uj_paths,new_root,tr,dev)
    runs,traces,metrics = {**sj_runs,**uj_runs},{**sj_dev,**uj_dev},{**sj_metrics,**uj_metrics}
    for seed in SEEDS:
        canonical = runs["SJ",seed,2,"constant"]["sampling_trace_sha256"]
        if any(runs[arm,seed,depth,policy]["sampling_trace_sha256"]!=canonical for arm in ARMS for depth in DEPTHS for policy in POLICIES):
            raise ValueError("All paired architectures/depths/policies must share identical prefix/tail sampled-view trajectories.")
    cases = sorted(dev);labels = np.array([dev[case]["label"] for case in cases])
    probabilities = {(arm,depth,policy):np.array([[traces[arm,seed,depth,policy][depth][case]["probability"] for case in cases] for seed in SEEDS])
                     for arm in ARMS for depth in DEPTHS for policy in POLICIES}
    dice = {(arm,depth,policy):np.array([[traces[arm,seed,depth,policy][depth][case]["dice"] for case in cases] for seed in SEEDS])
            for arm in ARMS for depth in DEPTHS for policy in POLICIES}
    seed_results = []
    for i,seed in enumerate(SEEDS):
        seed_results.append({"seed":seed,"regimes":{arm:{regime_name(depth,policy):{
            "endpoint":metrics[arm,seed,depth,policy][str(depth)],
            "pathology":_pathology(labels,probabilities[arm,depth,policy][i]),
            "endpoint_diagnostics":_high_confidence_errors(labels,probabilities[arm,depth,policy][i])}
            for depth in DEPTHS for policy in POLICIES} for arm in ARMS}})
    regimes = {arm:{} for arm in ARMS}
    for arm in ARMS:
        for depth in DEPTHS:
            for policy in POLICIES:
                name = regime_name(depth,policy)
                by_step = {str(step):{metric:_stats([metrics[arm,seed,depth,policy][str(step)][metric] for seed in SEEDS])
                                      for metric in AGGREGATE_METRICS} for step in range(1,depth+1)}
                pathology = {label_name:{"n_cases":int((labels==label).sum()),
                             **{metric:_stats([row["regimes"][arm][name]["pathology"][label_name][metric] for row in seed_results])
                                for metric in ("error_count","error_rate","high_confidence_error_count","high_confidence_error_rate","mean_probability","brier","nll")}}
                             for label_name,label in LABELS.items()}
                regimes[arm][name] = {"arm":arm,"depth":depth,"tail_policy":policy,"by_step":by_step,"endpoint":by_step[str(depth)],
                                     "pathology":pathology,"parameters":{**p["parameters_by_arm_depth"][f"{arm}{depth}"],"encoder_frozen":p["encoder_frozen_parameters"]}}
    sharing,depth_effects,theta,policy_effects = {},{arm:{} for arm in ARMS},{},{arm:{} for arm in ARMS}
    gains = {}
    transitions = {"sharing":{},"depth":{arm:{} for arm in ARMS},"policy":{arm:{} for arm in ARMS}}
    for policy in POLICIES:
        for depth in DEPTHS:
            name = regime_name(depth,policy)
            s,u = probabilities["SJ",depth,policy],probabilities["UJ",depth,policy]
            sharing[name] = {"Brier_benefit":_brier_effect((u-labels)**2-(s-labels)**2,draws=draws,bootstrap_seed=bootstrap_seed),
                             "NLL_benefit":_effect(_nll_per_case(labels,u)-_nll_per_case(labels,s),SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
                             "AUROC_change":_stats([metrics["SJ",seed,depth,policy][str(depth)]["auroc"]-metrics["UJ",seed,depth,policy][str(depth)]["auroc"] for seed in SEEDS]),
                             "Dice_change":_effect(dice["SJ",depth,policy]-dice["UJ",depth,policy],SEEDS,draws=draws,bootstrap_seed=bootstrap_seed)}
            transitions["sharing"][name] = _paired_transitions(labels,u,s,draws=draws,bootstrap_seed=bootstrap_seed)
        for arm in ARMS:
            early,late = probabilities[arm,2,policy],probabilities[arm,4,policy]
            gains[arm,policy] = (early-labels)**2-(late-labels)**2
            depth_effects[arm][policy] = {"Brier_benefit":_brier_effect(gains[arm,policy],draws=draws,bootstrap_seed=bootstrap_seed),
                                          "NLL_worsening":_effect(_nll_per_case(labels,late)-_nll_per_case(labels,early),SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
                                          "AUROC_change":_stats([metrics[arm,seed,4,policy]["4"]["auroc"]-metrics[arm,seed,2,policy]["2"]["auroc"] for seed in SEEDS]),
                                          "Dice_change":_effect(dice[arm,4,policy]-dice[arm,2,policy],SEEDS,draws=draws,bootstrap_seed=bootstrap_seed)}
            transitions["depth"][arm][policy] = _paired_transitions(labels,early,late,draws=draws,bootstrap_seed=bootstrap_seed)
        theta[policy] = _brier_effect(gains["SJ",policy]-gains["UJ",policy],draws=draws,bootstrap_seed=bootstrap_seed)
    for arm in ARMS:
        for depth in DEPTHS:
            constant,low = probabilities[arm,depth,"constant"],probabilities[arm,depth,"low_tail"]
            policy_effects[arm][str(depth)] = {"Brier_benefit":_effect((constant-labels)**2-(low-labels)**2,SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
                                              "NLL_benefit":_effect(_nll_per_case(labels,constant)-_nll_per_case(labels,low),SEEDS,draws=draws,bootstrap_seed=bootstrap_seed)}
            transitions["policy"][arm][str(depth)] = _paired_transitions(labels,constant,low,draws=draws,bootstrap_seed=bootstrap_seed)
    learning = [{"epoch":epoch,"updates":epoch*math.ceil(p["train"]["cases"]/2),"regimes":{arm:{regime_name(depth,policy):{
                 metric:_stats([next(row for row in runs[arm,seed,depth,policy]["development_history"] if row["epoch"]==epoch)
                                ["by_step"][str(depth)][metric] for seed in SEEDS]) for metric in AGGREGATE_METRICS}
                 for depth in DEPTHS for policy in POLICIES} for arm in ARMS}} for epoch in range(5,26,5)]
    return {"status":"executed_cpu_architecture_analysis","clinical_validation":False,"mechanism_identified":False,
            "decision_rule_automatically_applied":False,"phase":p["phase"],"patient_mapping":"unverified_Case_grouped_only",
            "evaluation_split":"tune_development","model_selection":FINAL_SELECTION,"training_seeds":list(SEEDS),
            "arms":list(ARMS),"depths":list(DEPTHS),"policies":list(POLICIES),
            "n_training_cases":len(tr),"n_development_cases":len(dev),"n_training_images":p["train"]["images"],"n_development_images":p["tune"]["images"],
            "n_development_positive":int(labels.sum()),"n_development_negative":int((labels==0).sum()),
            "sources":{**{key:p[key] for key in SOURCE_KEYS+("launcher_source_sha256","helper_source_sha256","shared_protocol_sha256","shared_reference_protocol_sha256")},
                       "protocol_sha256":_file_hash(protocol_path)},
            "cohort_fingerprints":{split:p[split]["fingerprint"] for split in ("train","tune")},
            "alignment_checks":{"forty_exact_architecture_depth_policy_seed_endpoints":True,"shared_control_exact_replay_revalidated":True,
                                "reused_shared_branch_and_prefix_artifact_bindings_verified":True,"full_untied_policy_prefix_state_restoration_certified":True,
                                "common_initial_block_head_values_and_distinct_untied_storage_certified":True,"paired_sampling_all_architectures_depths_policies":True,
                                "exact_case_view_label_alignment_all_readouts":True,"no_train_tune_case_or_image_overlap":True,"no_calibration_test_or_temperature_fit":True},
            "compute_budget":{"reused_shared_endpoints":20,"new_untied_endpoints":20,"new_untied_prefixes":10,
                              "conceptual_updates_per_endpoint":p["training_updates_per_run"],"total_actual_new_updates":p["total_actual_new_updates"],
                              "same_width_untied_models_use_more_parameters":True,"optimizer_update_matching_is_not_FLOP_matching":True},
            "definitions":{"sharing_benefit":"Brier/NLL_UJ - Brier/NLL_SJ at the same trained depth and tail policy; positive favors sharing",
                           "G":"Brier_trained2 - Brier_trained4 within architecture and policy; positive favors depth4",
                           "Theta":"G_SJ - G_UJ; positive means larger extra-depth Brier benefit with sharing",
                           "pathology_error":"Fixed threshold0.5; benign errors are false positives, malignant errors are false negatives",
                           "high_confidence_error":"Benign p>=0.95 or malignant p<=0.05; inclusive bounds",
                           "transition_directions":"sharing UJ→SJ; depth2→4 within arm/policy; policy constant→low_tail within arm/depth",
                           "signed_probability_change":"mean p_after - p_before; smaller favors benign label, larger favors malignant label",
                           "bootstrap":"Average five fitted-seed effects within aligned Case before resampling Cases; no network refitting",
                           "seed_SD":"sample SD ddof=1 across five fitted seeds; not a confidence interval"},
            "regimes":regimes,"seed_results":seed_results,"sharing_benefit_by_depth_policy":sharing,
            "depth_benefit_by_arm_policy":depth_effects,"theta_by_policy":theta,"policy_benefit_by_arm_depth":policy_effects,
            "paired_transitions":transitions,"learning_curves":learning,
            "limitations":["Observed tune is reused; this is exploratory architecture/optimizer comparison, not independent clinical validation.",
                           "Untied models use more trainable parameters at the same width; equal optimizer updates do not imply equal parameters or FLOPs.",
                           "Untied blocks start from copied shared-block values with independent storage; this differs from independently randomized ordinary layers and is not a fully tuned architecture comparison.",
                           "A sharing/depth interaction alone does not imply positive absolute depth benefit or clinical utility.",
                           "Full-state restoration, initial tensor equality and storage distinction are certified by the isolated runner; this analyzer verifies certificate provenance, artifact hashes and raw traces.",
                           "Case bootstrap intervals condition on the fitted networks and fixed observed Cases; training-seed sampling uncertainty and patient independence are not established.",
                           "All classification and pathology transitions use fixed threshold0.5; no threshold search or temperature fitting is performed.",
                           "Frozen features, no augmentation, resized Dice and already observed development labels restrict generalization."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("run-dirs","shared-run-dirs","shared-reference-run-dirs"):
        parser.add_argument(f"--{field}",nargs="+",required=True)
    for field in ("protocol","shared-protocol","shared-reference-protocol"):
        parser.add_argument(f"--{field}",required=True)
    parser.add_argument("--prefix-root")
    parser.add_argument("--shared-prefix-root")
    parser.add_argument("--draws",type=int,default=2000)
    parser.add_argument("--bootstrap-seed",type=int,default=17)
    parser.add_argument("--expected-training-cases",type=int,default=256)
    parser.add_argument("--expected-development-cases",type=int,default=159)
    parser.add_argument("--output",default="outputs/cpu-architecture256/architecture_analysis.json")
    args = parser.parse_args(argv);destination = Path(args.output)
    if destination.exists():parser.error("Architecture analysis output exists; choose a fresh path.")
    try:
        report = analyze_architecture(args.run_dirs,protocol_path=args.protocol,shared_run_dirs=args.shared_run_dirs,
                                      shared_protocol_path=args.shared_protocol,shared_reference_run_dirs=args.shared_reference_run_dirs,
                                      shared_reference_protocol_path=args.shared_reference_protocol,prefix_root=args.prefix_root,
                                      shared_prefix_root=args.shared_prefix_root,draws=args.draws,bootstrap_seed=args.bootstrap_seed,
                                      expected_training_cases=args.expected_training_cases,expected_development_cases=args.expected_development_cases)
    except ValueError as error:parser.error(str(error))
    destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        with destination.open("x") as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    except FileExistsError:parser.error("Architecture analysis output exists; choose a fresh path.")
    print(json.dumps({"status":report["status"],"theta_by_policy":report["theta_by_policy"]},allow_nan=False))
    return 0


if __name__=="__main__":raise SystemExit(main())
