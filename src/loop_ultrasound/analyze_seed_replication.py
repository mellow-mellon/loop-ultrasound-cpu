"""Five-new-seed replication of separately fitted terminal-only loop depths.

The resource screening gate applies only to the five new seeds. Optional old
fits are reported as a clearly secondary eight-seed description. Paired Case
bootstrap intervals condition on fitted models, not a population of seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .analyze import _hash, _integer, _match_cases
from .analyze_depth_supervision import (DEPTHS, FINAL_SELECTION, FIXED_CONFIG,
                                       SOURCE_KEYS, _declared_metrics, _json,
                                       _metrics, _protocol as _reference_protocol,
                                       _trace, regime_name)
from .analyze_expanded import AGGREGATE_METRICS, BASELINES, _baseline_data, _fingerprint
from .metrics import NLL_CLIP_EPSILON, classification_metrics


NEW_SEEDS = (53, 71, 89, 107, 131)
REFERENCE_SEEDS = (17, 29, 43)
GATE = {"minimum_positive_seeds": 4, "require_all_leave_one_seed_out_means_positive": True,
        "maximum_mean_nll_worsening": 0.0}


def _stats(values):
    if any(value is None for value in values):
        return {"mean": None, "std_across_training_seeds": None, "min": None, "max": None}
    numbers = np.asarray(values, dtype=np.float64)
    if numbers.ndim != 1 or numbers.size < 2 or not np.isfinite(numbers).all():
        raise ValueError("Use at least two finite fitted-seed aggregate values.")
    return {"mean": float(numbers.mean()), "std_across_training_seeds": float(numbers.std(ddof=1)),
            "min": float(numbers.min()), "max": float(numbers.max())}


def _effect(values, seeds, *, draws, bootstrap_seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(seeds) or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Paired effects must align fitted seeds by nonempty finite Cases.")
    within_case = values.mean(axis=0)
    seed_means = values.mean(axis=1)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_means = np.empty(draws)
    for start in range(0, draws, 256):
        stop = min(start+256, draws)
        indices = rng.integers(0, within_case.size, size=(stop-start, within_case.size))
        bootstrap_means[start:stop] = within_case[indices].mean(axis=1)
    interval = np.quantile(bootstrap_means, [.025, .975]).tolist()
    return {**_stats(seed_means), "estimate": float(within_case.mean()), "ci95": interval,
            "by_seed": [{"seed": seed, "value": float(value)} for seed, value in zip(seeds, seed_means)],
            "positive_seed_count": int(np.count_nonzero(seed_means > 0)),
            "negative_seed_count": int(np.count_nonzero(seed_means < 0)),
            "bootstrap": {"unit": "Case", "n_cases": int(within_case.size), "n_fitted_seeds": len(seeds),
                          "draws": draws, "seed": bootstrap_seed, "method": "percentile",
                          "fitted_seed_effects_averaged_within_case": True,
                          "same_case_resample_across_seeds_and_depths": True,
                          "includes_training_seed_sampling_uncertainty": False}}


def _replication_protocol(path, training_cases, development_cases):
    protocol = _json(path, "A readable locked replication protocol is required.")
    if (protocol.get("status") != "fixed_before_seed_replication_training"
            or protocol.get("phase") != "exploratory_training_randomness_replication_on_observed_tune"
            or protocol.get("clinical_validation") is not False or protocol.get("device") != "cpu"
            or protocol.get("patient_mapping") != "unverified_Case_grouped_only"
            or protocol.get("calibration_and_test_predictions") is not False
            or protocol.get("model_selection") != FINAL_SELECTION or protocol.get("warm_starts") is not False):
        raise ValueError("Protocol must lock a fresh CPU, observed-tune, fixed-final replication.")
    if (protocol.get("seeds") != list(NEW_SEEDS) or protocol.get("reference_seeds") != list(REFERENCE_SEEDS)
            or protocol.get("arms") != ["SJ"] or protocol.get("depths") != list(DEPTHS)
            or protocol.get("supervisions") != ["terminal"]):
        raise ValueError("Replication protocol must lock the five new seeds and terminal-only depths 2/4.")
    for key, expected in FIXED_CONFIG.items():
        if key != "arm" and (protocol.get(key) != expected or type(protocol.get(key)) is not type(expected)):
            raise ValueError("Replication protocol differs from the fixed CPU training configuration.")
    for key in SOURCE_KEYS + ("reference_protocol_sha256", "launcher_source_sha256"):
        protocol[key] = _hash(protocol.get(key), f"protocol {key}")
    initialization = protocol.get("initialization_sha256_by_seed")
    if not isinstance(initialization, dict) or set(initialization) != {str(seed) for seed in NEW_SEEDS}:
        raise ValueError("Protocol must lock initial parameter digests for all five new seeds.")
    protocol["initialization_sha256_by_seed"] = {
        str(seed): _hash(initialization[str(seed)], "protocol initialization digest") for seed in NEW_SEEDS}
    expected_updates = math.ceil(training_cases/FIXED_CONFIG["batch_size"])*FIXED_CONFIG["epochs"]
    if _integer(protocol.get("training_updates_per_run"), "protocol updates", 1) != expected_updates:
        raise ValueError("Protocol optimizer updates must cover all 50 fixed-cohort epochs.")
    screening = protocol.get("screening_gate")
    if not isinstance(screening, dict) or any(screening.get(key) != expected or type(screening.get(key)) is not type(expected)
                                            for key, expected in GATE.items()):
        raise ValueError("Protocol must lock all three prespecified engineering screening conditions.")
    for split, count in (("train", training_cases), ("tune", development_cases)):
        item = protocol.get(split)
        if not isinstance(item, dict) or _integer(item.get("cases"), "protocol Case count", 2) != count:
            raise ValueError("Protocol training/development Case counts disagree with the expected experiment.")
        _integer(item.get("images"), "protocol image count", count)
        benign = _integer(item.get("benign"), "protocol benign count")
        malignant = _integer(item.get("malignant"), "protocol malignant count")
        if benign+malignant != count:
            raise ValueError("Protocol pathology counts are inconsistent.")
        item["fingerprint"] = _hash(item.get("fingerprint"), "protocol cohort fingerprint")
    return protocol


def _run_summary(directory, protocol, seeds):
    summary = _json(directory/"summary.json", "Each fit requires a readable completed summary.")
    if (summary.get("status") != "executed_cpu_engineering_pilot"
            or summary.get("clinical_validation") is not False or summary.get("device") != "cpu"
            or summary.get("patient_mapping") != "unverified_Case_grouped_only"
            or summary.get("evaluation_split") != "tune_development"
            or summary.get("model_selection") != FINAL_SELECTION):
        raise ValueError("Only completed CPU Case-grouped, tune-only, fixed-final fits are accepted.")
    config = summary.get("config")
    if not isinstance(config, dict):
        raise ValueError("Each fit requires a training config mapping.")
    for key, expected in FIXED_CONFIG.items():
        if config.get(key) != expected or type(config.get(key)) is not type(expected):
            raise ValueError("Run config differs from the locked CPU training configuration.")
    seed, depth = _integer(config.get("seed"), "training seed"), _integer(config.get("steps"), "trained depth", 1)
    if seed not in seeds or depth not in DEPTHS or config.get("supervision") != "terminal":
        raise ValueError("Every fit must match an expected seed and terminal-only depth 2/4.")
    for key in SOURCE_KEYS:
        if _hash(summary.get(key), f"run {key}") != protocol[key]:
            raise ValueError("Run source digests differ from the locked protocol.")
    if _hash(summary.get("initialization_sha256"), "run initialization") != protocol["initialization_sha256_by_seed"][str(seed)]:
        raise ValueError("Run initial parameters differ from the paired seed digest locked before training.")
    readout = summary.get("readout_supervision")
    if (not isinstance(readout, dict) or readout.get("mode") != "terminal"
            or readout.get("directly_supervised_steps") != [depth]):
        raise ValueError("Only the configured final readout may have direct supervision.")
    for key, expected in (("n_training_cases", protocol["train"]["cases"]),
                          ("n_training_images", protocol["train"]["images"])):
        if _integer(summary.get(key), key, 2) != expected:
            raise ValueError("Training counts differ from the locked cohort.")
    timing = summary.get("timing")
    if (not isinstance(timing, dict) or _integer(timing.get("updates"), "completed updates", 1)
            != protocol["training_updates_per_run"]):
        raise ValueError("Fit must complete every prespecified optimizer update.")
    return summary


def _cohort_check(index, declared):
    images = {image for row in index.values() for image in row["image_ids"]}
    labels = [row["label"] for row in index.values()]
    actual = {"cases": len(index), "images": len(images), "benign": labels.count(0),
              "malignant": sum(labels), "fingerprint": _fingerprint(index)}
    if any(declared.get(key) != value for key, value in actual.items()):
        raise ValueError("Observed Cases, views and pathology labels disagree with the locked cohort.")
    return images


def _load_runs(run_dirs, protocol, seeds):
    directories = [Path(path) for path in run_dirs]
    if len(directories) != 2*len(seeds) or len({path.resolve() for path in directories}) != 2*len(seeds):
        raise ValueError("Supply exactly two distinct completed terminal-depth fits per expected seed.")
    runs, training, development, metrics = {}, {}, {}, {}
    common_config = None
    for directory in directories:
        summary = _run_summary(directory, protocol, seeds)
        config = summary["config"]
        key = (config["seed"], config["steps"])
        if key in runs:
            raise ValueError("Exactly one terminal-depth fit per expected seed/depth combination is required.")
        common = {key: value for key, value in config.items() if key not in ("seed", "steps")}
        if common_config is not None and common != common_config:
            raise ValueError("Source configs must match apart from fitted seed and terminal depth.")
        common_config = common
        runs[key] = summary
        training[key] = _trace(directory/"training_predictions.csv", split="train", seed=key[0], depth=key[1], require_dice=False)
        development[key] = _trace(directory/"development_predictions.csv", split="tune", seed=key[0], depth=key[1], require_dice=True)
        metrics[key] = {str(step): _metrics(development[key][step]) for step in range(1, key[1]+1)}
        _declared_metrics(summary.get("development_subset"), metrics[key], steps=range(1,key[1]+1))
        history = summary.get("development_history")
        if (not isinstance(history, list) or [item.get("epoch") for item in history if isinstance(item, dict)]
                != list(range(5,51,5))):
            raise ValueError("Every fit requires the fixed development history through epoch 50.")
        _declared_metrics(history[-1].get("by_step"), metrics[key], steps=range(1,key[1]+1))
        selection = {"sampling": "proportional", "selection_seed": 20261004,
                     "train_fingerprint": protocol["train"]["fingerprint"], "tune_fingerprint": protocol["tune"]["fingerprint"]}
        if summary.get("cohort_selection") != selection:
            raise ValueError("Declared cohort selection disagrees with the locked experiment.")
    if set(runs) != {(seed,depth) for seed in seeds for depth in DEPTHS}:
        raise ValueError("Every expected seed/depth combination is required.")
    train_reference, tune_reference = training[(seeds[0],2)][1], development[(seeds[0],2)][1]
    train_images = _cohort_check(train_reference, protocol["train"])
    tune_images = _cohort_check(tune_reference, protocol["tune"])
    if set(train_reference)&set(tune_reference) or train_images&tune_images:
        raise ValueError("Training/development Cases or images overlap.")
    for seed,depth in runs:
        for step in range(1,depth+1):
            _match_cases(train_reference, training[(seed,depth)][step])
            _match_cases(tune_reference, development[(seed,depth)][step])
    return {"runs": runs, "training": training, "development": development, "metrics": metrics,
            "train_reference": train_reference, "tune_reference": tune_reference, "common_config": common_config}


def _high_confidence_errors(labels, probabilities):
    negative_error = (labels==1)&(probabilities <= .05)
    positive_error = (labels==0)&(probabilities >= .95)
    high_confidence = (probabilities <= .05)|(probabilities >= .95)
    return {"n_high_confidence_predictions": int(high_confidence.sum()),
            "n_high_confidence_errors": int((negative_error|positive_error).sum()),
            "high_confidence_false_negatives": int(negative_error.sum()),
            "high_confidence_false_positives": int(positive_error.sum()),
            "n_probability_exact_zero": int((probabilities==0).sum()),
            "n_probability_exact_one": int((probabilities==1).sum()),
            "n_wrong_at_probability_boundary": int((((probabilities==0)&(labels==1))|
                                                     ((probabilities==1)&(labels==0))).sum()),
            "fraction_of_all_cases": float((negative_error|positive_error).mean()),
            "definition": "malignant Case with p<=0.05 or benign Case with p>=0.95; inclusive bounds"}


def _transitions(labels, early, late):
    correct2, correct4 = (early>=.5)==labels, (late>=.5)==labels
    return {"classification_threshold": .5, "corrected_cases": int((~correct2&correct4).sum()),
            "deteriorated_cases": int((correct2&~correct4).sum()),
            "both_correct_cases": int((correct2&correct4).sum()),
            "both_wrong_cases": int((~correct2&~correct4).sum())}


def _concentration(gains):
    """No Cases are removed from the primary metric; this is sensitivity only."""
    gains = np.asarray(gains)
    absolute, positive = np.abs(gains), np.clip(gains,0,None)
    absolute_sum, positive_sum = float(absolute.sum()), float(positive.sum())
    descending = np.sort(gains)[::-1]
    result = {"improved_brier_cases": int((gains>0).sum()), "worsened_brier_cases": int((gains<0).sum()),
              "equal_brier_cases": int((gains==0).sum()), "total_positive_gain": positive_sum,
              "total_negative_gain": float(np.clip(gains,None,0).sum()),
              "absolute_change_sum": absolute_sum, "primary_cases_removed": False}
    for number in (1,5):
        take = min(number, gains.size)
        result[f"largest_{number}_absolute_change_share"] = (float(np.sort(absolute)[-take:].sum()/absolute_sum)
                                                              if absolute_sum else 0.)
        result[f"largest_{number}_positive_gain_share"] = (float(np.sort(positive)[-take:].sum()/positive_sum)
                                                           if positive_sum else 0.)
        remove = min(number, gains.size-1)
        result[f"mean_gain_after_removing_top_{number}_gain_cases"] = float(descending[remove:].mean())
        result[f"n_removed_for_top_{number}_sensitivity"] = int(remove)
    return result


def _nll_per_case(labels, probabilities):
    p = np.clip(probabilities, NLL_CLIP_EPSILON, 1-NLL_CLIP_EPSILON)
    return -(labels*np.log(p)+(1-labels)*np.log1p(-p))


def _block(loaded, seeds, *, draws, bootstrap_seed, apply_gate):
    cases = sorted(loaded["tune_reference"])
    labels = np.array([loaded["tune_reference"][case]["label"] for case in cases])
    probabilities = {depth: np.array([[loaded["development"][(seed,depth)][depth][case]["probability"] for case in cases]
                                      for seed in seeds]) for depth in DEPTHS}
    dices = {depth: np.array([[loaded["development"][(seed,depth)][depth][case]["dice"] for case in cases]
                              for seed in seeds]) for depth in DEPTHS}
    gains = (probabilities[2]-labels)**2-(probabilities[4]-labels)**2
    nll_worsening = _nll_per_case(labels, probabilities[4])-_nll_per_case(labels, probabilities[2])
    effect = _effect(gains,seeds,draws=draws,bootstrap_seed=bootstrap_seed)
    nll_effect = _effect(nll_worsening,seeds,draws=draws,bootstrap_seed=bootstrap_seed)
    regimes = {}
    for depth in DEPTHS:
        by_step = {str(step): {metric: _stats([loaded["metrics"][(seed,depth)][str(step)][metric] for seed in seeds])
                              for metric in AGGREGATE_METRICS} for step in range(1,depth+1)}
        regimes[regime_name(depth,"terminal")] = {"depth":depth,"supervision":"terminal",
                                                   "by_step":by_step,"endpoint":by_step[str(depth)]}
    seed_results = []
    for index,seed in enumerate(seeds):
        seed_results.append({"seed":seed,
                             "regimes": {regime_name(depth,"terminal"): {"depth":depth,"supervision":"terminal",
                                           "by_step":loaded["metrics"][(seed,depth)],
                                           "endpoint":loaded["metrics"][(seed,depth)][str(depth)]} for depth in DEPTHS},
                             "depth_benefit_Brier":float(gains[index].mean()),
                             "NLL_worsening":float(nll_worsening[index].mean()),
                             "AUROC_change": (loaded["metrics"][(seed,4)]["4"]["auroc"]-loaded["metrics"][(seed,2)]["2"]["auroc"]
                                                if loaded["metrics"][(seed,4)]["4"]["auroc"] is not None else None),
                             "endpoint_diagnostics": {regime_name(depth,"terminal"):
                                                      _high_confidence_errors(labels,probabilities[depth][index]) for depth in DEPTHS},
                             "classification_transitions":_transitions(labels,probabilities[2][index],probabilities[4][index]),
                             "loss_concentration":_concentration(gains[index])})
    values = gains.mean(axis=1)
    loo = [{"omitted_seed":seed,"mean_gain":float(np.delete(values,index).mean())} for index,seed in enumerate(seeds)]
    gate = None
    if apply_gate:
        conditions = {
            "positive_gain_at_least_four_of_five": {"passed": effect["positive_seed_count"] >= 4,
                                                    "observed_positive_seeds":effect["positive_seed_count"],"minimum":4},
            "all_five_leave_one_seed_out_means_positive": {"passed": all(item["mean_gain"]>0 for item in loo),
                                                           "minimum_observed_mean":min(item["mean_gain"] for item in loo),
                                                           "strictly_greater_than":0.0},
            "mean_nll_not_worse": {"passed":nll_effect["mean"]<=0,"observed_nll4_minus_nll2":nll_effect["mean"],
                                    "maximum":0.0}}
        passed = all(condition["passed"] for condition in conditions.values())
        gate = {"conditions":conditions,"passed":passed,"applies_only_to_new_five_seeds":True,
                "recommendation":"eligible_for_scientific_review_of_next_experiment" if passed else "do_not_expand_from_this_screen",
                "automatically_starts_experiments":False,"is_statistical_significance_test":False,
                "is_clinical_validation":False,"interpretation":"Prespecified CPU engineering resource screen only; passing does not establish clinical utility or independently authorize expansion."}
    return {"training_seeds":list(seeds),"n_seeds":len(seeds),"regimes":regimes,
            "depth_benefit_Brier":effect,"NLL_worsening":nll_effect,
            "Dice_change":_effect(dices[4]-dices[2],seeds,draws=draws,bootstrap_seed=bootstrap_seed),
            "AUROC_change":_stats([row["AUROC_change"] for row in seed_results]),
            "seed_results":seed_results,"leave_one_seed_out":loo,"gate":gate,
            "seed_averaged_case_loss_concentration":_concentration(gains.mean(axis=0)),
            "primary_case_count":len(cases)}


def analyze_seed_replication(run_dirs, *, protocol_path, reference_run_dirs=None,
                            reference_protocol_path=None, baseline_dir=None, draws=2000,
                            bootstrap_seed=17, expected_training_cases=128, expected_development_cases=159):
    draws = _integer(draws,"bootstrap draws",2)
    bootstrap_seed = _integer(bootstrap_seed,"bootstrap seed")
    expected_training_cases = _integer(expected_training_cases,"expected training Cases",2)
    expected_development_cases = _integer(expected_development_cases,"expected development Cases",2)
    protocol = _replication_protocol(protocol_path,expected_training_cases,expected_development_cases)
    loaded = _load_runs(run_dirs,protocol,NEW_SEEDS)
    primary = _block(loaded,NEW_SEEDS,draws=draws,bootstrap_seed=bootstrap_seed,apply_gate=True)
    combined, reference_digest = None,None
    if (reference_run_dirs is None) != (reference_protocol_path is None):
        raise ValueError("Reference fits and their locked protocol must be provided together.")
    if reference_run_dirs is not None:
        reference_digest = hashlib.sha256(Path(reference_protocol_path).read_bytes()).hexdigest()
        if reference_digest != protocol["reference_protocol_sha256"]:
            raise ValueError("Reference protocol bytes differ from the digest locked before replication.")
        reference_protocol = _reference_protocol(reference_protocol_path,expected_training_cases,expected_development_cases)
        if (any(reference_protocol[key]!=protocol[key] for key in SOURCE_KEYS)
                or any(reference_protocol[split]!=protocol[split] for split in ("train","tune"))):
            raise ValueError("Reference and new fits must use identical training-code/source digests and cohorts.")
        old = _load_runs(reference_run_dirs,reference_protocol,REFERENCE_SEEDS)
        if old["common_config"] != loaded["common_config"]:
            raise ValueError("Reference and new fitting configs must match apart from seed and depth.")
        _match_cases(loaded["train_reference"],old["train_reference"])
        _match_cases(loaded["tune_reference"],old["tune_reference"])
        if {Path(path).resolve() for path in run_dirs}&{Path(path).resolve() for path in reference_run_dirs}:
            raise ValueError("New replication and reference fit directories must be distinct.")
        merged = {**loaded, **{key:{**old[key],**loaded[key]} for key in ("runs","training","development","metrics")}}
        combined = _block(merged,REFERENCE_SEEDS+NEW_SEEDS,draws=draws,bootstrap_seed=bootstrap_seed,apply_gate=False)
        combined["role"] = "secondary_combined_eight_fitted_seeds_descriptive_only"
        combined["can_override_new_seed_gate"] = False
    baselines = None
    if baseline_dir is not None:
        baseline = _baseline_data(baseline_dir,sources=(protocol["encoder_sha256"],protocol["manifest_sha256"]),
                                  training=loaded["train_reference"],development=loaded["tune_reference"])
        cases = sorted(loaded["tune_reference"])
        labels = np.array([loaded["tune_reference"][case]["label"] for case in cases])
        baselines = {}
        for name in BASELINES:
            probabilities = np.array([baseline[name]["tune"][case]["probability"] for case in cases])
            baselines[name] = {"fit_seed":17,"tune_metrics":classification_metrics(labels,probabilities)}
    return {"status":"executed_cpu_seed_replication_analysis","clinical_validation":False,"mechanism_identified":False,
            "patient_mapping":"unverified_Case_grouped_only","evaluation_split":"tune_development",
            "model_selection":FINAL_SELECTION,"phase":protocol["phase"],"config":{**FIXED_CONFIG,"supervision":"terminal"},
            "n_training_cases":protocol["train"]["cases"],"n_training_images":protocol["train"]["images"],
            "n_development_cases":protocol["tune"]["cases"],"n_development_images":protocol["tune"]["images"],
            "n_development_positive":protocol["tune"]["malignant"],"n_development_negative":protocol["tune"]["benign"],
            "sources":{**{key:protocol[key] for key in SOURCE_KEYS},"launcher_source_sha256":protocol["launcher_source_sha256"],
                       "protocol_sha256":hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest(),
                       "reference_protocol_sha256":protocol["reference_protocol_sha256"]},
            "cohort_fingerprints":{split:protocol[split]["fingerprint"] for split in ("train","tune")},
            "paired_initialization":{"identical_within_each_new_seed":True,
                                     "by_seed":[{"seed":seed,"initialization_sha256":protocol["initialization_sha256_by_seed"][str(seed)]} for seed in NEW_SEEDS]},
            "alignment_checks":{"exactly_ten_fresh_terminal_depth_seed_fits":True,"every_configured_readout_present":True,
                                "identical_case_view_label_sets_all_fits_and_steps":True,
                                "explicit_train_tune_only_provenance":True,"no_training_development_case_or_image_overlap":True,
                                "locked_source_init_and_cohort_digests":True,"fixed_50_epochs_and_all_optimizer_updates":True,
                                "declared_endpoint_and_final_history_match_traces":True,
                                "reference_models_validated_separately":reference_run_dirs is not None},
            "definitions":{"G":"Brier_separately_trained_terminal2_endpoint - Brier_separately_trained_terminal4_endpoint; positive favors trained4",
                           "NLL_worsening":"NLL_trained4_endpoint - NLL_trained2_endpoint; positive means worse trained4 NLL",
                           "Dice_change":"mean_Case_Dice_trained4 - mean_Case_Dice_trained2",
                           "AUROC_change":"AUROC_trained4 - AUROC_trained2, summarized over fitted seeds",
                           "high_confidence_error":"Case p<=0.05 with malignant label, or p>=0.95 with benign label; inclusive",
                           "probability_boundary_counts":"Exact Case-average p=0 and p=1 counts, and wrong-label predictions at those boundaries; saturation audit only",
                           "classification_transitions":"Case correct/wrong transition between separately fitted endpoints at fixed threshold 0.5",
                           "loss_concentration":"Descriptive sensitivity to largest per-Case gains/absolute changes; no Case removed from primary metrics",
                           "seed_std":"sample SD ddof=1 of fitted-seed metrics, not a seed-population interval",
                           "bootstrap":"first average paired seed-specific effects within each Case, then resample aligned Cases; no ensemble prediction averaging"},
            "primary_new_seeds":primary,"secondary_combined_seeds":combined,"baselines":baselines,
            "limitations":["The five new seeds replicate training randomness on the already observed fixed tune cohort; they do not supply new patients or untouched validation.",
                           "The resource gate is an engineering rule, not statistical significance or clinical validation; passing never automatically authorizes expansion.",
                           "The secondary eight-seed description cannot override a failed five-new-seed gate.",
                           "Paired Case bootstrap intervals condition on fitted models; they exclude training-seed population uncertainty.",
                           "Case-to-patient mapping remains unverified; patient independence is not established.",
                           "Equal optimizer updates and Case exposure do not equalize recurrent compute; trained4 uses more block applications.",
                           "Intermediate terminal-only readouts are diagnostic and not directly supervised; primary comparisons use trained endpoints only.",
                           "Calibration and test predictions are absent; frozen features, no augmentation and resized Dice limit the scope."]}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dirs",nargs="+",required=True)
    p.add_argument("--protocol",required=True)
    p.add_argument("--reference-run-dirs",nargs="+")
    p.add_argument("--reference-protocol")
    p.add_argument("--baseline-dir")
    p.add_argument("--output",default="outputs/cpu-seed-replication128/seed_replication_analysis.json")
    p.add_argument("--draws",type=int,default=2000)
    p.add_argument("--bootstrap-seed",type=int,default=17)
    p.add_argument("--expected-training-cases",type=int,default=128)
    p.add_argument("--expected-development-cases",type=int,default=159)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    destination = Path(args.output)
    if destination.exists():
        p.error("Analysis output already exists; choose a fresh output path.")
    try:
        report = analyze_seed_replication(args.run_dirs,protocol_path=args.protocol,
                                         reference_run_dirs=args.reference_run_dirs,reference_protocol_path=args.reference_protocol,
                                         baseline_dir=args.baseline_dir,draws=args.draws,bootstrap_seed=args.bootstrap_seed,
                                         expected_training_cases=args.expected_training_cases,
                                         expected_development_cases=args.expected_development_cases)
    except (ValueError,OSError) as error:
        p.error(str(error) if isinstance(error,ValueError) else "A required local reference artifact is unreadable.")
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    destination.parent.mkdir(parents=True,exist_ok=True)
    try:
        with destination.open("x",encoding="utf-8") as stream:
            stream.write(payload)
    except FileExistsError:
        p.error("Analysis output already exists; choose a fresh output path.")
    print(json.dumps({"status":report["status"],"primary_new_seed_gain":report["primary_new_seeds"]["depth_benefit_Brier"],
                      "gate":report["primary_new_seeds"]["gate"]},allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
