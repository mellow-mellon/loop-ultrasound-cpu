"""Nested training-Case expansion at a fixed optimizer-update budget.

Compare separately fitted terminal-depth endpoints at 256 and 128 training
Cases. All paired uncertainty intervals condition on the three fitted seeds
and the previously observed development cohort. No scaling gate is applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .analyze import _hash, _integer, _match_cases
from .analyze_depth_supervision import (DEPTHS, FINAL_SELECTION, FIXED_CONFIG, SOURCE_KEYS,
                                       _declared_metrics, _json, _metrics, _protocol as _reference_protocol,
                                       _trace, regime_name)
from .analyze_expanded import AGGREGATE_METRICS, BASELINES, _baseline_data, _fingerprint
from .analyze_seed_replication import (_cohort_check, _effect, _high_confidence_errors,
                                      _load_runs, _nll_per_case, _stats, _transitions)
from .metrics import classification_metrics


SEEDS = (17,29,43)
NEW_EPOCHS = 25


def _protocol(path, training_cases, reference_cases, development_cases):
    protocol = _json(path,"A readable locked Case-expansion protocol is required.")
    if (protocol.get("status")!="fixed_before_case_expansion_training"
            or protocol.get("phase")!="exploratory_nested_training_case_expansion_on_observed_tune"
            or protocol.get("clinical_validation") is not False or protocol.get("device")!="cpu"
            or protocol.get("patient_mapping")!="unverified_Case_grouped_only"
            or protocol.get("model_selection")!=FINAL_SELECTION
            or protocol.get("warm_starts") is not False or protocol.get("calibration_and_test_predictions") is not False):
        raise ValueError("Protocol must lock a fresh CPU, observed-tune, fixed-final Case expansion.")
    if (protocol.get("seeds")!=list(SEEDS) or protocol.get("arms")!=["SJ"]
            or protocol.get("depths")!=list(DEPTHS) or protocol.get("supervisions")!=["terminal"]):
        raise ValueError("Protocol must lock all three paired seeds and terminal-only depths 2/4.")
    config = {**FIXED_CONFIG,"epochs":NEW_EPOCHS}
    for key,expected in config.items():
        if key!="arm" and (protocol.get(key)!=expected or type(protocol.get(key)) is not type(expected)):
            raise ValueError("Expansion protocol differs from the fixed 25-epoch configuration.")
    if protocol.get("reference_epochs")!=50:
        raise ValueError("The nested reference requires 50-epoch original fits.")
    if (protocol.get("updates_per_epoch")!=math.ceil(training_cases/2)
            or protocol.get("reference_updates_per_epoch")!=math.ceil(reference_cases/2)
            or protocol.get("matched_update_points")!=[epoch*math.ceil(training_cases/2) for epoch in range(5,26,5)]):
        raise ValueError("Protocol matched-update coordinates disagree with the fixed Case/batch/epoch budgets.")
    expected_updates = math.ceil(training_cases/2)*NEW_EPOCHS
    if (expected_updates != math.ceil(reference_cases/2)*50
            or _integer(protocol.get("training_updates_per_run"),"fixed update budget",1)!=expected_updates):
        raise ValueError("New and reference fits must share the same completed optimizer-update budget.")
    for key in SOURCE_KEYS+("reference_protocol_sha256","launcher_source_sha256"):
        protocol[key] = _hash(protocol.get(key),f"protocol {key}")
    init = protocol.get("initialization_sha256_by_seed")
    if not isinstance(init,dict) or set(init)!={str(seed) for seed in SEEDS}:
        raise ValueError("Protocol must lock paired initial parameter digests at each seed.")
    protocol["initialization_sha256_by_seed"] = {str(seed):_hash(init[str(seed)],"initialization digest") for seed in SEEDS}
    for split,count in (("train",training_cases),("reference_train",reference_cases),("tune",development_cases)):
        item = protocol.get(split)
        if not isinstance(item,dict) or _integer(item.get("cases"),"protocol Case count",2)!=count:
            raise ValueError("Protocol cohort Case counts disagree with the declared expansion.")
        _integer(item.get("images"),"protocol image count",count)
        if (_integer(item.get("benign"),"protocol benign")+_integer(item.get("malignant"),"protocol malignant"))!=count:
            raise ValueError("Protocol pathology counts are inconsistent.")
        item["fingerprint"] = _hash(item.get("fingerprint"),"protocol cohort fingerprint")
    return protocol


def _new_summary(directory,protocol):
    summary = _json(directory/"summary.json","Each new fit requires a readable completed summary.")
    if (summary.get("status")!="executed_cpu_engineering_pilot" or summary.get("clinical_validation") is not False
            or summary.get("device")!="cpu" or summary.get("patient_mapping")!="unverified_Case_grouped_only"
            or summary.get("evaluation_split")!="tune_development" or summary.get("model_selection")!=FINAL_SELECTION):
        raise ValueError("New fits must be completed CPU Case-grouped, tune-only, fixed-final runs.")
    config = summary.get("config")
    if not isinstance(config,dict):
        raise ValueError("New fits require a training config mapping.")
    for key,expected in {**FIXED_CONFIG,"epochs":NEW_EPOCHS}.items():
        if config.get(key)!=expected or type(config.get(key)) is not type(expected):
            raise ValueError("New fit config differs from the locked 25-epoch training configuration.")
    seed,depth = _integer(config.get("seed"),"training seed"),_integer(config.get("steps"),"depth",1)
    if seed not in SEEDS or depth not in DEPTHS or config.get("supervision")!="terminal":
        raise ValueError("New fits must match each expected paired seed and terminal depth.")
    for key in SOURCE_KEYS:
        if _hash(summary.get(key),key)!=protocol[key]:
            raise ValueError("New fit source digests differ from the locked protocol.")
    if _hash(summary.get("initialization_sha256"),"initialization digest")!=protocol["initialization_sha256_by_seed"][str(seed)]:
        raise ValueError("New fit initial parameters differ from the paired locked digest.")
    readout = summary.get("readout_supervision")
    if (not isinstance(readout,dict) or readout.get("mode")!="terminal"
            or readout.get("directly_supervised_steps")!=[depth]):
        raise ValueError("Only the final configured readout may have direct supervision.")
    for key,expected in (("n_training_cases",protocol["train"]["cases"]),("n_training_images",protocol["train"]["images"])):
        if _integer(summary.get(key),key,2)!=expected:
            raise ValueError("New fit training counts differ from the locked cohort.")
    timing = summary.get("timing")
    if (not isinstance(timing,dict) or _integer(timing.get("updates"),"completed updates",1)
            !=protocol["training_updates_per_run"]):
        raise ValueError("Every new fit must complete the fixed optimizer-update budget.")
    return summary


def _new_runs(directories,protocol):
    directories = [Path(path) for path in directories]
    if len(directories)!=6 or len({path.resolve() for path in directories})!=6:
        raise ValueError("Supply six distinct completed new terminal-depth/seed fits.")
    runs,training,development,metrics = {},{},{},{}
    common_config = None
    for directory in directories:
        summary = _new_summary(directory,protocol)
        config = summary["config"]
        key = (config["seed"],config["steps"])
        if key in runs:
            raise ValueError("Exactly one new fit per expected paired seed/depth combination is required.")
        common = {k:v for k,v in config.items() if k not in ("seed","steps","epochs")}
        if common_config is not None and common!=common_config:
            raise ValueError("New source configs must match apart from paired seed and depth.")
        common_config = common
        runs[key] = summary
        training[key] = _trace(directory/"training_predictions.csv",split="train",seed=key[0],depth=key[1],require_dice=False)
        development[key] = _trace(directory/"development_predictions.csv",split="tune",seed=key[0],depth=key[1],require_dice=True)
        metrics[key] = {str(step):_metrics(development[key][step]) for step in range(1,key[1]+1)}
        _declared_metrics(summary.get("development_subset"),metrics[key],steps=range(1,key[1]+1))
        history = summary.get("development_history")
        if (not isinstance(history,list) or [item.get("epoch") for item in history if isinstance(item,dict)]
                !=list(range(5,26,5))):
            raise ValueError("New histories must record fixed evaluation epochs through epoch 25.")
        _declared_metrics(history[-1].get("by_step"),metrics[key],steps=range(1,key[1]+1))
        selection = {"sampling":"proportional","selection_seed":20261004,
                     "train_fingerprint":protocol["train"]["fingerprint"],"tune_fingerprint":protocol["tune"]["fingerprint"]}
        if summary.get("cohort_selection")!=selection:
            raise ValueError("Declared new cohort selection differs from the locked protocol.")
    if set(runs)!={(seed,depth) for seed in SEEDS for depth in DEPTHS}:
        raise ValueError("All six expected new paired seed/depth fits are required.")
    train_reference,tune_reference = training[(17,2)][1],development[(17,2)][1]
    train_images = _cohort_check(train_reference,protocol["train"])
    tune_images = _cohort_check(tune_reference,protocol["tune"])
    if set(train_reference)&set(tune_reference) or train_images&tune_images:
        raise ValueError("New training/development Cases or images overlap.")
    for seed,depth in runs:
        for step in range(1,depth+1):
            _match_cases(train_reference,training[(seed,depth)][step])
            _match_cases(tune_reference,development[(seed,depth)][step])
    return {"runs":runs,"training":training,"development":development,"metrics":metrics,
            "train_reference":train_reference,"tune_reference":tune_reference,"common_config":common_config}


def _train_classification(index):
    cases = sorted(index)
    return classification_metrics([index[case]["label"] for case in cases],[index[case]["probability"] for case in cases])


def _block(loaded,*,draws,bootstrap_seed):
    cases = sorted(loaded["tune_reference"])
    labels = np.array([loaded["tune_reference"][case]["label"] for case in cases])
    probabilities = {depth:np.array([[loaded["development"][(seed,depth)][depth][case]["probability"] for case in cases]
                                     for seed in SEEDS]) for depth in DEPTHS}
    dices = {depth:np.array([[loaded["development"][(seed,depth)][depth][case]["dice"] for case in cases]
                            for seed in SEEDS]) for depth in DEPTHS}
    gains = (probabilities[2]-labels)**2-(probabilities[4]-labels)**2
    nll = _nll_per_case(labels,probabilities[4])-_nll_per_case(labels,probabilities[2])
    regimes,seed_results = {},[]
    train_metrics = {(seed,depth):_train_classification(loaded["training"][(seed,depth)][depth])
                     for seed in SEEDS for depth in DEPTHS}
    for depth in DEPTHS:
        by_step = {str(step):{metric:_stats([loaded["metrics"][(seed,depth)][str(step)][metric] for seed in SEEDS])
                              for metric in AGGREGATE_METRICS} for step in range(1,depth+1)}
        gap = [loaded["metrics"][(seed,depth)][str(depth)]["brier"]-train_metrics[(seed,depth)]["brier"] for seed in SEEDS]
        regimes[regime_name(depth,"terminal")] = {"depth":depth,"supervision":"terminal","by_step":by_step,
                                                  "endpoint":by_step[str(depth)],"development_minus_training_brier":_stats(gap),
                                                  "training_endpoint":{metric:_stats([train_metrics[(seed,depth)][metric] for seed in SEEDS])
                                                                       for metric in AGGREGATE_METRICS if metric!="mean_case_dice"}}
    for i,seed in enumerate(SEEDS):
        seed_results.append({"seed":seed,"depth_benefit_Brier":float(gains[i].mean()),"NLL_worsening":float(nll[i].mean()),
                             "regimes":{regime_name(depth,"terminal"):{"depth":depth,"supervision":"terminal",
                                           "endpoint":loaded["metrics"][(seed,depth)][str(depth)],
                                           "training_endpoint":train_metrics[(seed,depth)],
                                           "development_minus_training_brier":loaded["metrics"][(seed,depth)][str(depth)]["brier"]-train_metrics[(seed,depth)]["brier"],
                                           "endpoint_diagnostics":_high_confidence_errors(labels,probabilities[depth][i])} for depth in DEPTHS},
                             "classification_transitions":_transitions(labels,probabilities[2][i],probabilities[4][i])})
    return {"training_seeds":list(SEEDS),"n_seeds":3,"n_training_cases":len(loaded["train_reference"]),
            "regimes":regimes,"seed_results":seed_results,
            "depth_benefit_Brier":_effect(gains,SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
            "NLL_worsening":_effect(nll,SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
            "Dice_change":_effect(dices[4]-dices[2],SEEDS,draws=draws,bootstrap_seed=bootstrap_seed)},gains


def _history_endpoint(summary,epoch,depth,n_cases):
    item = next((item for item in summary["development_history"] if item["epoch"]==epoch),None)
    candidate = None if item is None else item.get("by_step",{}).get(str(depth))
    if not isinstance(candidate,dict) or candidate.get("n_cases")!=n_cases:
        raise ValueError("Matched-update histories require the full development cohort at each endpoint.")
    for metric in AGGREGATE_METRICS:
        value = candidate.get(metric)
        if value is None and metric in ("auroc","sensitivity","specificity"):
            continue
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError("Matched-update history endpoint metrics must be finite and valid.")
        if metric!="nll" and value>1:
            raise ValueError("Matched-update history bounded metrics must remain in [0,1].")
    return candidate


def _classification_baselines(directory,loaded,protocol):
    """Validate a separately fitted baseline on the exact size-specific cohort."""
    summary = _json(Path(directory)/"baselines_summary.json","Readable completed baseline summaries are required.")
    indexed = _baseline_data(directory,sources=(protocol["encoder_sha256"],protocol["manifest_sha256"]),
                             training=loaded["train_reference"],development=loaded["tune_reference"])
    if summary.get("selection_seed")!=20261004:
        raise ValueError("Baseline cohort selection must match the fixed Case selection seed.")
    method = summary.get("method")
    expected_method = {"feature_shape_per_image":[196,192],
                       "feature_pooling":"mean spatial tokens per image, then mean all available image features per Case",
                       "fit_split":"train","scaler_fit_split":"train",
                       "hyperparameters_selected_using_tune":False,"threshold":.5}
    if not isinstance(method,dict) or any(method.get(key)!=value or type(method.get(key)) is not type(value)
                                        for key,value in expected_method.items()):
        raise ValueError("Baseline method must use fixed all-view features and train-only fitting/scaling without tune selection.")
    logistic = method.get("logistic_regression")
    expected_logistic = {"C":1.,"solver":"lbfgs","max_iter":2000,"random_state":17,
                         "class_weight":None,"fit_intercept":True}
    if not isinstance(logistic,dict) or any(logistic.get(key)!=value or type(logistic.get(key)) is not type(value)
                                          for key,value in expected_logistic.items()):
        raise ValueError("Baseline logistic regression must retain the fixed C=1 and fit configuration.")
    _integer(logistic.get("n_iter"),"baseline converged fit iterations",1)
    if logistic["n_iter"]>2000:
        raise ValueError("Baseline fit iterations exceed the fixed solver budget.")
    train_reference = loaded["train_reference"]
    prevalence = float(np.mean([row["label"] for row in train_reference.values()]))
    declared_prevalence = method.get("train_case_prevalence")
    if (isinstance(declared_prevalence,bool) or not isinstance(declared_prevalence,(int,float))
            or not math.isfinite(declared_prevalence)
            or not math.isclose(declared_prevalence,prevalence,rel_tol=0,abs_tol=1e-12)):
        raise ValueError("Constant baseline prevalence must be derived from its exact training Case cohort.")
    declared = summary.get("baselines")
    if not isinstance(declared,dict) or set(declared)!=set(BASELINES):
        raise ValueError("Both fixed constant and logistic baseline summaries are required.")
    metrics = {name:{} for name in BASELINES}
    for name in BASELINES:
        for split in ("train","tune"):
            reference = indexed[name][split]
            cases = sorted(reference)
            labels = [reference[case]["label"] for case in cases]
            probabilities = [reference[case]["probability"] for case in cases]
            if name=="constant" and any(not math.isclose(value,prevalence,rel_tol=0,abs_tol=1e-12) for value in probabilities):
                raise ValueError("Constant baseline predictions must use its train-only Case prevalence.")
            metrics[name][split] = classification_metrics(labels,probabilities)
            try:
                _declared_metrics({"1":declared[name].get(split)}, {"1":metrics[name][split]},steps=[1])
            except (ValueError,AttributeError):
                raise ValueError("Declared baseline metrics disagree with its exact paired Case predictions.") from None
    return {"fit_seed":17,"n_training_cases":len(train_reference),"n_development_cases":len(loaded["tune_reference"]),
            "cohort_fingerprints":summary["cohort_fingerprints"],
            "method":{**expected_method,"logistic_regression":expected_logistic,"train_case_prevalence":prevalence,
                      "view_difference_from_loop_training":"all-view Case mean; loop training samples one view per Case per epoch"},
            "baselines":metrics}


def analyze_case_expansion(run_dirs,*,protocol_path,reference_run_dirs,reference_protocol_path,
                          draws=2000,bootstrap_seed=17,expected_training_cases=256,
                          expected_reference_training_cases=128,expected_development_cases=159,
                          baseline_dir=None,reference_baseline_dir=None):
    draws = _integer(draws,"bootstrap draws",2)
    bootstrap_seed = _integer(bootstrap_seed,"bootstrap seed")
    training_cases = _integer(expected_training_cases,"new training Cases",2)
    reference_cases = _integer(expected_reference_training_cases,"reference training Cases",2)
    development_cases = _integer(expected_development_cases,"development Cases",2)
    protocol = _protocol(protocol_path,training_cases,reference_cases,development_cases)
    run_dirs,reference_run_dirs = list(run_dirs),list(reference_run_dirs)
    reference_digest = hashlib.sha256(Path(reference_protocol_path).read_bytes()).hexdigest()
    if reference_digest!=protocol["reference_protocol_sha256"]:
        raise ValueError("Reference protocol bytes differ from the digest locked before expansion.")
    reference_protocol = _reference_protocol(reference_protocol_path,reference_cases,development_cases)
    if (any(reference_protocol[key]!=protocol[key] for key in SOURCE_KEYS)
            or reference_protocol["train"]!=protocol["reference_train"] or reference_protocol["tune"]!=protocol["tune"]
            or reference_protocol["initialization_sha256_by_seed"]!=protocol["initialization_sha256_by_seed"]):
        raise ValueError("Reference source/cohort and paired initialization must match the expansion protocol.")
    new,old = _new_runs(run_dirs,protocol),_load_runs(reference_run_dirs,reference_protocol,SEEDS)
    old_common = {k:v for k,v in old["common_config"].items() if k!="epochs"}
    if new["common_config"]!=old_common:
        raise ValueError("New/reference fitting configs must match apart from epochs, paired seed and depth.")
    if {Path(path).resolve() for path in run_dirs}&{Path(path).resolve() for path in reference_run_dirs}:
        raise ValueError("New/reference model fit directories must be distinct.")
    _match_cases(new["tune_reference"],old["tune_reference"])
    if not set(old["train_reference"])<set(new["train_reference"]):
        raise ValueError("Reference training Cases must form a strict nested subset of new training Cases.")
    for case,row in old["train_reference"].items():
        candidate = new["train_reference"][case]
        if row["label"]!=candidate["label"] or row["image_ids"]!=candidate["image_ids"]:
            raise ValueError("Every nested reference Case must retain its pathology label and all shared views.")
    primary,new_gains = _block(new,draws=draws,bootstrap_seed=bootstrap_seed)
    reference,old_gains = _block(old,draws=draws,bootstrap_seed=bootstrap_seed)
    # A depth interaction can change because either fitted endpoint improves.
    # Report each depth's direct data-size contrast as a secondary diagnostic.
    cases = sorted(new["tune_reference"])
    labels = np.array([new["tune_reference"][case]["label"] for case in cases])
    data_size_benefit = {}
    for depth in DEPTHS:
        probabilities = {size:np.array([[loaded["development"][(seed,depth)][depth][case]["probability"]
                                        for case in cases] for seed in SEEDS])
                         for size,loaded in (("new",new),("old",old))}
        dice_change = np.array([[new["development"][(seed,depth)][depth][case]["dice"]
                                -old["development"][(seed,depth)][depth][case]["dice"] for case in cases] for seed in SEEDS])
        data_size_benefit[regime_name(depth,"terminal")] = {
            "Brier_benefit":_effect((probabilities["old"]-labels)**2-(probabilities["new"]-labels)**2,
                                    SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
            "NLL_benefit":_effect(_nll_per_case(labels,probabilities["old"])-_nll_per_case(labels,probabilities["new"]),
                                  SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
            "Dice_change":_effect(dice_change,SEEDS,draws=draws,bootstrap_seed=bootstrap_seed)}
    learning = []
    for epoch in range(5,26,5):
        entry = {"updates":epoch*math.ceil(training_cases/2),"new_epoch":epoch,"reference_epoch":epoch*2,
                 "new256":{},"reference128":{}}
        for depth in DEPTHS:
            name = regime_name(depth,"terminal")
            for label,loaded,matched_epoch in (("new256",new,epoch),("reference128",old,epoch*2)):
                history = [_history_endpoint(loaded["runs"][(seed,depth)],matched_epoch,depth,development_cases) for seed in SEEDS]
                entry[label][name] = {metric:_stats([value[metric] for value in history]) for metric in AGGREGATE_METRICS}
        learning.append(entry)
    baseline_reports = None
    if baseline_dir is not None or reference_baseline_dir is not None:
        baseline_reports = {
            "new256":_classification_baselines(baseline_dir,new,protocol) if baseline_dir is not None else None,
            "reference128":_classification_baselines(reference_baseline_dir,old,reference_protocol)
                           if reference_baseline_dir is not None else None}
    return {"status":"executed_cpu_case_expansion_analysis","clinical_validation":False,"mechanism_identified":False,
            "patient_mapping":"unverified_Case_grouped_only","evaluation_split":"tune_development","model_selection":FINAL_SELECTION,
            "phase":protocol["phase"],"training_seeds":list(SEEDS),"config":{**FIXED_CONFIG,"epochs":25,"reference_epochs":50,"supervision":"terminal"},
            "training_updates_per_run":protocol["training_updates_per_run"],
            "n_training_cases":training_cases,"n_reference_training_cases":reference_cases,"n_development_cases":development_cases,
            "n_training_images":protocol["train"]["images"],"n_reference_training_images":protocol["reference_train"]["images"],
            "n_development_images":protocol["tune"]["images"],"n_development_positive":protocol["tune"]["malignant"],
            "n_development_negative":protocol["tune"]["benign"],
            "sources":{**{key:protocol[key] for key in SOURCE_KEYS},"launcher_source_sha256":protocol["launcher_source_sha256"],
                       "protocol_sha256":hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest(),"reference_protocol_sha256":reference_digest},
            "cohort_fingerprints":{"train":protocol["train"]["fingerprint"],"reference_train":protocol["reference_train"]["fingerprint"],"tune":protocol["tune"]["fingerprint"]},
            "paired_initialization":{"same_across_data_sizes_and_depths_within_seed":True,
                                     "by_seed":[{"seed":seed,"initialization_sha256":protocol["initialization_sha256_by_seed"][str(seed)]} for seed in SEEDS]},
            "alignment_checks":{"exactly_six_new_and_six_reference_terminal_fits":True,"strictly_nested_training_cases":True,
                                "all_shared_training_views_and_labels_preserved":True,"same_development_case_view_label_sets":True,
                                "train_tune_only_records":True,"no_cross_split_case_or_image_overlap":True,
                                "matching_locked_source_and_init_digests":True,"equal_completed_optimizer_updates":True,
                                "new_25_reference_50_fixed_epoch_histories":True,"endpoint_metrics_recomputed_from_case_traces":True},
            "definitions":{"G":"Brier_separately_trained_terminal2_endpoint - Brier_separately_trained_terminal4_endpoint; positive favors trained4",
                           "interaction_data_size_depth":"G256 - G128 paired within identical development Cases and fitted seeds",
                           "data_size_benefit_by_depth":"Brier/NLL128 - Brier/NLL256 at the same separately trained depth; positive favors the fixed-update 256-Case fit",
                           "NLL_worsening":"NLL_trained4_endpoint - NLL_trained2_endpoint; positive is worse trained4 NLL",
                           "training_gap":"development endpoint Brier minus training endpoint Brier, computed separately per fitted seed",
                           "matched_update_learning":"new epoch5/10/15/20/25 vs reference epoch10/20/30/40/50; equal completed optimizer updates",
                           "bootstrap":"average seed-specific paired effects within each aligned Case before resampling Cases; conditional on fitted models",
                           "seed_std":"sample SD ddof=1 over three fitted seeds; not a seed-population confidence interval"},
            "primary_256":primary,"reference_128":reference,
            "data_size_benefit_by_depth":data_size_benefit,
            "baselines":baseline_reports,
            "interaction_data_size_depth":_effect(new_gains-old_gains,SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
            "matched_update_learning":learning,"decision_rule_automatically_applied":False,
            "limitations":["This nested training expansion reuses the already observed development cohort; it is exploratory and adds no new validation patients.",
                           "Matched optimizer updates do not equalize all computation or stochastic training paths; trained4 requires more recurrent applications than trained2.",
                           "The 256-Case models see more distinct Cases for fewer epochs; the contrast studies this fixed-update tradeoff rather than an isolated data-size effect under equal epochs.",
                           "The Case bootstrap conditions on these three fitted seeds and excludes population uncertainty over training randomness.",
                           "Case-to-patient mapping remains unverified; clinical utility and patient independence are not established.",
                           "Primary comparison uses separately trained endpoints; terminal-only intermediate readouts are not directly supervised.",
                           "No success gate or automatic expansion/GPU action is applied; calibration/test predictions are absent.",
                           "Frozen encoder features, no augmentation and resized Dice limit the scope of this CPU screen."]}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dirs",nargs="+",required=True)
    p.add_argument("--protocol",required=True)
    p.add_argument("--reference-run-dirs",nargs="+",required=True)
    p.add_argument("--reference-protocol",required=True)
    p.add_argument("--baseline-dir")
    p.add_argument("--reference-baseline-dir")
    p.add_argument("--output",default="outputs/cpu-case-expansion256/case_expansion_analysis.json")
    p.add_argument("--draws",type=int,default=2000)
    p.add_argument("--bootstrap-seed",type=int,default=17)
    p.add_argument("--expected-training-cases",type=int,default=256)
    p.add_argument("--expected-reference-training-cases",type=int,default=128)
    p.add_argument("--expected-development-cases",type=int,default=159)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    destination = Path(args.output)
    if destination.exists():
        p.error("Analysis output already exists; choose a fresh output path.")
    try:
        report = analyze_case_expansion(args.run_dirs,protocol_path=args.protocol,reference_run_dirs=args.reference_run_dirs,
                                       reference_protocol_path=args.reference_protocol,draws=args.draws,bootstrap_seed=args.bootstrap_seed,
                                       expected_training_cases=args.expected_training_cases,
                                       expected_reference_training_cases=args.expected_reference_training_cases,
                                       expected_development_cases=args.expected_development_cases,
                                       baseline_dir=args.baseline_dir,reference_baseline_dir=args.reference_baseline_dir)
    except (ValueError,OSError) as error:
        p.error(str(error) if isinstance(error,ValueError) else "A required local reference artifact is unreadable.")
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        with destination.open("x",encoding="utf-8") as stream:
            stream.write(payload)
    except FileExistsError:
        p.error("Analysis output already exists; choose a fresh output path.")
    print(json.dumps({"status":report["status"],"G256":report["primary_256"]["depth_benefit_Brier"],
                      "G256_minus_G128":report["interaction_data_size_depth"]},allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
