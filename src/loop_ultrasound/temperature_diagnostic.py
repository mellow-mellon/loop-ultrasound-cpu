"""Case-grouped out-of-fold temperature diagnostic on already observed tune.

Prepare locks source artifacts and folds without fitting. Execute is a separate
action so the protocol can be committed before fitting any temperature. Only
aggregate diagnostics are public; Case-level folds and OOF predictions are local.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

from .analyze import _integer
from .analyze_case_expansion import NEW_SEEDS, _new_runs, _paired_runs, _protocol as _expansion_protocol
from .analyze_depth_supervision import _json
from .analyze_expanded import AGGREGATE_METRICS, _fingerprint
from .analyze_seed_replication import _effect, _stats
from .metrics import classification_metrics


FOLD_SEED = 20261011
CLIP_EPSILON = 1e-15
LOG_T_BOUNDS = (-4.,4.)
TIE_TOLERANCE = 1e-12
OPTIMIZER_XATOL = 1e-10
OPTIMIZER_MAXITER = 1000


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _probability_vectors(labels,probabilities):
    y,p = np.asarray(labels,dtype=np.float64),np.asarray(probabilities,dtype=np.float64)
    if (y.ndim!=1 or p.shape!=y.shape or y.size<2 or not np.isfinite(y).all()
            or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any()):
        raise ValueError("Use finite, equally sized Case vectors with binary labels and probabilities in [0,1].")
    return y,p


def case_logits(probabilities):
    probabilities = np.asarray(probabilities,dtype=np.float64)
    if not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any():
        raise ValueError("Case probabilities must be finite values in [0,1].")
    clipped = np.clip(probabilities,CLIP_EPSILON,1-CLIP_EPSILON)
    return np.log(clipped)-np.log1p(-clipped)


def _stable_nll(labels,logits,log_temperature):
    scores = logits/np.exp(log_temperature)
    # This objective does not clip the calibrated probabilities. Stable
    # softplus remains finite even where the sigmoid numerically reaches 0/1.
    return float(np.mean(np.logaddexp(0,scores)-labels*scores))


def fit_temperature(labels,probabilities):
    """One fixed bounded scalar fit; identity and both endpoints are candidates."""
    y,p = _probability_vectors(labels,probabilities)
    if np.unique(y).size!=2:
        raise ValueError("Each temperature fitting fold requires both pathology classes.")
    logits = case_logits(p)
    objective = lambda value:_stable_nll(y,logits,float(value))
    result = minimize_scalar(objective,bounds=LOG_T_BOUNDS,method="bounded",
                             options={"xatol":OPTIMIZER_XATOL,"maxiter":OPTIMIZER_MAXITER})
    if not result.success or not math.isfinite(result.fun) or not math.isfinite(result.x):
        raise ValueError("The prespecified scalar temperature optimizer did not converge.")
    candidates = [0.,LOG_T_BOUNDS[0],LOG_T_BOUNDS[1],float(result.x)]
    values = [(value,objective(value)) for value in candidates]
    minimum = min(value for _,value in values)
    near = [(log_t,value) for log_t,value in values if value<=minimum+TIE_TOLERANCE]
    log_t,value = min(near,key=lambda item:(abs(item[0]),item[0]))
    return {"temperature":float(np.exp(log_t)),"log_temperature":float(log_t),
            "fit_nll_stable_softplus":value,"identity_fit_nll_stable_softplus":objective(0.),
            "selected_identity":log_t==0.,"selected_exact_lower_bound":log_t==LOG_T_BOUNDS[0],
            "selected_exact_upper_bound":log_t==LOG_T_BOUNDS[1],
            "near_lower_bound":abs(log_t-LOG_T_BOUNDS[0])<=1e-6,
            "near_upper_bound":abs(log_t-LOG_T_BOUNDS[1])<=1e-6,
            "optimizer_converged":True,"optimizer_function_evaluations":int(result.nfev)}


def calibrate_probabilities(probabilities,temperature):
    if isinstance(temperature,bool) or not isinstance(temperature,(int,float)) or not math.isfinite(temperature) or temperature<=0:
        raise ValueError("Temperature must be a finite positive scalar.")
    raw = np.asarray(probabilities,dtype=np.float64)
    logits = case_logits(raw)  # Validates without changing the caller's array.
    if temperature==1.:
        return raw.copy()  # Preserve exact 0/1 and identity Brier without a roundtrip.
    calibrated = expit(logits/temperature)
    # Positive T preserves the mathematical side of threshold .5. Guard the
    # rare floating-point rounding of a negative tiny score to exactly .5.
    return np.where((raw<.5)&(calibrated>=.5),np.nextafter(.5,0.),calibrated)


def assign_folds(index,*,n_splits=5,fold_seed=FOLD_SEED):
    n_splits = _integer(n_splits,"Case-fold count",2)
    fold_seed = _integer(fold_seed,"Case-fold seed")
    cases = sorted(index)
    labels = np.array([index[case]["label"] for case in cases],dtype=np.int64)
    if set(labels)!={0,1} or min(np.count_nonzero(labels==0),np.count_nonzero(labels==1))<n_splits:
        raise ValueError("Both pathology classes need at least one Case per stratified fold.")
    folds = np.full(len(cases),-1,dtype=np.int64)
    splitter = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=fold_seed)
    for fold,(_,holdout) in enumerate(splitter.split(np.zeros((len(cases),1)),labels)):
        folds[holdout] = fold
    triples = [[case,int(label),int(fold)] for case,label,fold in zip(cases,labels,folds)]
    fingerprint = hashlib.sha256(json.dumps(triples,ensure_ascii=True,separators=(",",":")).encode()).hexdigest()
    counts = [{"fold":fold,"n_fit_cases":int((folds!=fold).sum()),"n_holdout_cases":int((folds==fold).sum()),
               "n_fit_positive":int(labels[folds!=fold].sum()),"n_holdout_positive":int(labels[folds==fold].sum())}
              for fold in range(n_splits)]
    return cases,labels,folds,fingerprint,counts


def cross_fit_temperature(labels,probabilities,folds):
    y,p = _probability_vectors(labels,probabilities)
    folds = np.asarray(folds)
    if folds.shape!=y.shape or not np.issubdtype(folds.dtype,np.integer) or (folds<0).any():
        raise ValueError("Fold assignments must be an aligned integer Case vector.")
    unique = np.unique(folds)
    if unique.size<2 or unique.tolist()!=list(range(unique.size)):
        raise ValueError("Use at least two nonempty consecutive Case folds.")
    oof = np.empty_like(p)
    records = []
    for fold in unique:
        holdout = folds==fold
        fit = ~holdout
        fitted = fit_temperature(y[fit],p[fit])
        oof[holdout] = calibrate_probabilities(p[holdout],fitted["temperature"])
        records.append({"fold":int(fold),"n_fit_cases":int(fit.sum()),"n_holdout_cases":int(holdout.sum()),**fitted})
    if not np.isfinite(oof).all():
        raise ValueError("Out-of-fold calibrated probabilities must remain finite.")
    return oof,records


def _artifact_bindings(directories):
    result = []
    for directory in directories:
        directory = Path(directory)
        summary = _json(directory/"summary.json","Readable source summaries are required.")
        result.append({"seed":summary["config"]["seed"],"depth":summary["config"]["steps"],
                       "summary_sha256":_digest(directory/"summary.json"),
                       "training_predictions_sha256":_digest(directory/"training_predictions.csv"),
                       "development_predictions_sha256":_digest(directory/"development_predictions.csv")})
    return sorted(result,key=lambda row:(row["seed"],row["depth"]))


def _sources(run_dirs,source_protocol_path,reference_run_dirs,reference_protocol_path,*,training_cases,reference_cases,development_cases):
    source = _expansion_protocol(source_protocol_path,training_cases,reference_cases,development_cases)
    if tuple(source["seeds"])!=NEW_SEEDS:
        raise ValueError("The primary diagnostic requires only the five new 256-Case seed fits.")
    if (reference_run_dirs is None)!=(reference_protocol_path is None):
        raise ValueError("Secondary matched128 fits and their protocol must be provided together.")
    if reference_run_dirs is not None:
        primary,secondary,reference,_ = _paired_runs(run_dirs,source,reference_run_dirs,reference_protocol_path,
                                                    reference_cases,development_cases)
    else:
        primary,secondary,reference = _new_runs(run_dirs,source),None,None
    return primary,secondary,source,reference


def build_protocol(run_dirs,*,source_protocol_path,reference_run_dirs=None,reference_protocol_path=None,
                   expected_training_cases=256,expected_reference_training_cases=128,expected_development_cases=159,
                   n_splits=5,draws=2000,bootstrap_seed=17):
    """Validate source runs and fix shared Case folds; this performs no fitting."""
    run_dirs = list(run_dirs)
    reference_run_dirs = list(reference_run_dirs) if reference_run_dirs is not None else None
    training_cases = _integer(expected_training_cases,"primary training Cases",2)
    reference_cases = _integer(expected_reference_training_cases,"secondary training Cases",2)
    development_cases = _integer(expected_development_cases,"development Cases",2)
    n_splits = _integer(n_splits,"number of Case folds",2)
    draws = _integer(draws,"Case bootstrap draws",2)
    bootstrap_seed = _integer(bootstrap_seed,"Case bootstrap seed")
    primary,secondary,source,reference = _sources(run_dirs,source_protocol_path,reference_run_dirs,reference_protocol_path,
                                                 training_cases=training_cases,reference_cases=reference_cases,development_cases=development_cases)
    cases,labels,folds,fold_hash,fold_counts = assign_folds(primary["tune_reference"],n_splits=n_splits)
    protocol = {"status":"fixed_before_case_grouped_crossfit_temperature_diagnostic",
                "phase":"exploratory_crossfit_on_previously_observed_tune","clinical_validation":False,
                "patient_mapping":"unverified_Case_grouped_only","calibration_and_test_predictions":False,
                "primary_training_seeds":list(NEW_SEEDS),"depths":[2,4],"primary_training_cases":training_cases,
                "secondary_training_cases":reference_cases if secondary is not None else None,
                "n_development_cases":development_cases,"n_splits":n_splits,"stratified_case_fold_seed":FOLD_SEED,
                "fold_assignment_sha256":fold_hash,"fold_counts":fold_counts,
                "temperature_input":"logit_of_mean_view_case_probability","probability_clip_epsilon":CLIP_EPSILON,
                "log_temperature_bounds":list(LOG_T_BOUNDS),"fit_criterion":"stable_softplus_NLL_on_clipped_input_logits",
                "optimizer":{"implementation":"scipy_minimize_scalar_bounded","xatol":OPTIMIZER_XATOL,"maxiter":OPTIMIZER_MAXITER},
                "identity_candidate":True,"endpoints_candidates":True,
                "tie_rule":"identity_then_zero_nearest_logT_within_1e-12","tie_nll_tolerance":TIE_TOLERANCE,
                "report_nll":"classification_metrics_clips_calibrated_probabilities_at_1e-15",
                "identity_prediction_policy":"T1_returns_exact_raw_Case_probability_copy",
                "threshold_roundoff_policy":"preserve_raw_side_of_0.5_using_nextafter_if_sigmoid_rounds_up",
                "bootstrap":{"draws":draws,"seed":bootstrap_seed,"unit":"Case","refit_temperatures":False},
                "sources":{"training_source_sha256":source["training_source_sha256"],"encoder_sha256":source["encoder_sha256"],
                           "manifest_sha256":source["manifest_sha256"],"primary_protocol_sha256":_digest(source_protocol_path),
                           "secondary_protocol_sha256":_digest(reference_protocol_path) if secondary is not None else None,
                           "diagnostic_module_sha256":_digest(__file__)},
                "cohort_fingerprints":{"primary_train":source["train"]["fingerprint"],"tune":_fingerprint(primary["tune_reference"]),
                                       "secondary_train":reference["train"]["fingerprint"] if reference is not None else None},
                "paired_initialization_sha256_by_seed":source["initialization_sha256_by_seed"],
                "run_artifact_bindings":{"primary256":_artifact_bindings(run_dirs),
                                         "secondary128":_artifact_bindings(reference_run_dirs) if secondary is not None else None},
                "environment":{"numpy":np.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__},
                "interpretation":"Exploratory OOF calibrator diagnostic on already observed tune; no untouched clinical evaluation or model-selection gate."}
    return protocol,(primary,secondary,cases,labels,folds)


def _fold_csv(cases,labels,folds):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream,fieldnames=("case_id","label","fold"),lineterminator="\n")
    writer.writeheader()
    writer.writerows({"case_id":case,"label":int(label),"fold":int(fold)} for case,label,fold in zip(cases,labels,folds))
    return stream.getvalue()


def _write_once_or_match(path,payload):
    path = Path(path)
    if path.exists():
        if path.read_text()!=payload:
            raise ValueError("Existing prepared artifact differs; use a fresh diagnostic directory.")
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="") as stream:
        stream.write(payload)


def prepare(run_dirs,*,output_dir,**kwargs):
    protocol,loaded = build_protocol(run_dirs,**kwargs)
    _,_,cases,labels,folds = loaded
    root = Path(output_dir)
    _write_once_or_match(root/"protocol.json",json.dumps(protocol,sort_keys=True,indent=2,allow_nan=False)+"\n")
    _write_once_or_match(root/"fold_assignments.csv",_fold_csv(cases,labels,folds))
    return protocol


def _nll_report(labels,probabilities):
    p = np.clip(probabilities,CLIP_EPSILON,1-CLIP_EPSILON)
    return -(labels*np.log(p)+(1-labels)*np.log1p(-p))


def _block(loaded,cases,labels,folds,*,size,draws,bootstrap_seed):
    raw,oof,fold_fits = {},{},{}
    seed_results,local_rows = [],[]
    for depth in (2,4):
        raw[depth] = np.array([[loaded["development"][(seed,depth)][depth][case]["probability"] for case in cases] for seed in NEW_SEEDS])
        oof[depth] = np.empty_like(raw[depth])
        fold_fits[depth] = []
        for index,seed in enumerate(NEW_SEEDS):
            oof[depth][index],fits = cross_fit_temperature(labels,raw[depth][index],folds)
            fold_fits[depth].append(fits)
            temperatures = {row["fold"]:row["temperature"] for row in fits}
            local_rows.extend({"source_size":size,"seed":seed,"depth":depth,"case_id":case,"label":int(label),"fold":int(fold),
                               "raw_probability":float(p),"temperature":temperatures[int(fold)],"oof_probability":float(q)}
                              for case,label,fold,p,q in zip(cases,labels,folds,raw[depth][index],oof[depth][index]))
    regimes = {}
    metric_names = tuple(metric for metric in AGGREGATE_METRICS if metric!="mean_case_dice")
    for index,seed in enumerate(NEW_SEEDS):
        row = {"seed":seed,"regimes":{}}
        for depth in (2,4):
            raw_metrics = classification_metrics(labels,raw[depth][index])
            calibrated_metrics = classification_metrics(labels,oof[depth][index])
            if any(raw_metrics[key]!=calibrated_metrics[key] for key in ("tp","tn","fp","fn")):
                raise ValueError("Positive-temperature OOF calibration must preserve classification at fixed threshold 0.5.")
            row["regimes"][f"depth{depth}_terminal"] = {
                "raw":raw_metrics,"oof":calibrated_metrics,"temperature_fits":fold_fits[depth][index],
                "oof_exact_probability_zero_count":int((oof[depth][index]==0).sum()),
                "oof_exact_probability_one_count":int((oof[depth][index]==1).sum())}
        row["raw_depth_gain_Brier"] = row["regimes"]["depth2_terminal"]["raw"]["brier"]-row["regimes"]["depth4_terminal"]["raw"]["brier"]
        row["oof_depth_gain_Brier"] = row["regimes"]["depth2_terminal"]["oof"]["brier"]-row["regimes"]["depth4_terminal"]["oof"]["brier"]
        seed_results.append(row)
    for depth in (2,4):
        name = f"depth{depth}_terminal"
        all_fits = [fit for seed_fits in fold_fits[depth] for fit in seed_fits]
        brier_improvement = (raw[depth]-labels)**2-(oof[depth]-labels)**2
        nll_improvement = _nll_report(labels,raw[depth])-_nll_report(labels,oof[depth])
        regimes[name] = {
            "raw":{metric:_stats([row["regimes"][name]["raw"][metric] for row in seed_results]) for metric in metric_names},
            "oof":{metric:_stats([row["regimes"][name]["oof"][metric] for row in seed_results]) for metric in metric_names},
            "raw_minus_oof_Brier":_effect(brier_improvement,NEW_SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
            "raw_minus_oof_NLL":_effect(nll_improvement,NEW_SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
            "oof_minus_raw_AUROC":_stats([row["regimes"][name]["oof"]["auroc"]-row["regimes"][name]["raw"]["auroc"] for row in seed_results]),
            "temperature_fit_summary":{"n_fits":len(all_fits),"min_temperature":min(fit["temperature"] for fit in all_fits),
                                       "max_temperature":max(fit["temperature"] for fit in all_fits),
                                       "n_identity":sum(fit["selected_identity"] for fit in all_fits),
                                       "n_exact_lower_bound":sum(fit["selected_exact_lower_bound"] for fit in all_fits),
                                       "n_exact_upper_bound":sum(fit["selected_exact_upper_bound"] for fit in all_fits),
                                       "scope":"descriptive over five fits per network; folds are not independent patients"},
            "leave_one_seed_out_Brier_improvement":[{"omitted_seed":seed,"mean":float(np.delete(brier_improvement.mean(axis=1),i).mean())} for i,seed in enumerate(NEW_SEEDS)]}
    raw_gains = (raw[2]-labels)**2-(raw[4]-labels)**2
    oof_gains = (oof[2]-labels)**2-(oof[4]-labels)**2
    result = {"role":"primary_five_seed_256_diagnostic" if size==256 else "secondary_matched_five_seed_128_diagnostic",
              "training_seeds":list(NEW_SEEDS),"n_cases":len(cases),"regimes":regimes,"seed_results":seed_results,
              "raw_depth_gain_Brier":_effect(raw_gains,NEW_SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
              "oof_depth_gain_Brier":_effect(oof_gains,NEW_SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
              "oof_minus_raw_depth_gain":_effect(oof_gains-raw_gains,NEW_SEEDS,draws=draws,bootstrap_seed=bootstrap_seed),
              "leave_one_seed_out_oof_depth_gain":[{"omitted_seed":seed,"mean":float(np.delete(oof_gains.mean(axis=1),i).mean())} for i,seed in enumerate(NEW_SEEDS)]}
    return result,local_rows


def execute(run_dirs,*,output_dir,**kwargs):
    root = Path(output_dir)
    locked = _json(root/"protocol.json","Prepare and lock the diagnostic protocol before fitting.")
    current,loaded = build_protocol(run_dirs,**kwargs)
    if current!=locked:
        raise ValueError("Diagnostic recipe, source artifacts or folds changed after preparation.")
    primary,secondary,cases,labels,folds = loaded
    fold_path = root/"fold_assignments.csv"
    if not fold_path.exists() or fold_path.read_text()!=_fold_csv(cases,labels,folds):
        raise ValueError("Prepared local Case folds differ from the locked assignment.")
    report_path,oof_path = root/"temperature_diagnostic_analysis.json",root/"oof_case_predictions.csv"
    if report_path.exists() or oof_path.exists():
        raise ValueError("Diagnostic results already exist; output is never overwritten.")
    primary_result,local_rows = _block(primary,cases,labels,folds,size=256,
                                      draws=locked["bootstrap"]["draws"],bootstrap_seed=locked["bootstrap"]["seed"])
    secondary_result = None
    if secondary is not None:
        secondary_result,secondary_rows = _block(secondary,cases,labels,folds,size=128,
                                                draws=locked["bootstrap"]["draws"],bootstrap_seed=locked["bootstrap"]["seed"])
        local_rows.extend(secondary_rows)
    report = {"status":"executed_case_grouped_crossfit_temperature_diagnostic","clinical_validation":False,
              "patient_mapping":"unverified_Case_grouped_only","phase":locked["phase"],"evaluation_split":"observed_tune_crossfit",
              "calibration_and_test_predictions":False,"decision_rule_automatically_applied":False,
              "protocol_sha256":_digest(root/"protocol.json"),"sources":locked["sources"],
              "cohort_fingerprints":locked["cohort_fingerprints"],"fold_assignment_sha256":locked["fold_assignment_sha256"],
              "primary_training_cases":locked["primary_training_cases"],"secondary_training_cases":locked["secondary_training_cases"],
              "fold_counts":locked["fold_counts"],"recipe":{key:locked[key] for key in ("n_splits","stratified_case_fold_seed","temperature_input",
                "probability_clip_epsilon","log_temperature_bounds","fit_criterion","optimizer","identity_candidate","endpoints_candidates",
                 "tie_rule","tie_nll_tolerance","report_nll","identity_prediction_policy","threshold_roundoff_policy","bootstrap")},
              "alignment_checks":{"identical_case_folds_across_seeds_depths_and_sizes":True,"all_temperatures_fit_on_complementary_cases_only":True,
                                  "locked_source_run_and_fold_artifacts_verified":True,"no_calibration_test_records":True,
                                  "fixed_threshold_0_5_classification_preserved":True},
              "primary256":primary_result,"secondary128":secondary_result,
              "definitions":{"raw_minus_oof":"Raw loss minus out-of-fold calibrated loss; positive means lower OOF loss",
                             "depth_gain_Brier":"Brier_trained2_endpoint - Brier_trained4_endpoint, each with its own complementary-fold temperature",
                             "OOF_AUROC":"Pooled Case AUROC per fitted network after fold-specific calibration; ranking can change across folds",
                             "bootstrap":"Average five fitted-seed Case effects before Case resampling; networks and fold temperatures are not refit"},
              "limitations":["Tune has already informed earlier model and experiment choices; OOF calibration is a diagnostic, not untouched validation.",
                             "Positive temperature is monotone within one fold, but different fold temperatures can change pooled OOF Case ranking and AUROC.",
                             "Temperature rescales Case logits only; it does not demonstrate an architecture benefit or a causal training mechanism.",
                             "Bootstrap intervals condition on fitted networks, fixed folds and fitted temperatures; they exclude calibrator-refit and seed-population uncertainty.",
                             "Case-to-patient mapping is unverified, so Case folds do not establish patient-independent evaluation.",
                             "The fitting objective uses stable softplus NLL; reporting NLL clips calibrated probabilities at 1e-15 and can differ numerically at saturation.",
                             "Identity is a fitting-fold candidate only; it cannot guarantee that held-out-fold Brier or NLL improves.",
                             "Calibration/test remain sealed; no best seed, temperature grid or best checkpoint is selected."]}
    fields = ("source_size","seed","depth","case_id","label","fold","raw_probability","temperature","oof_probability")
    with oof_path.open("x",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(local_rows)
    with report_path.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare",action="store_true")
    modes.add_argument("--execute",action="store_true")
    p.add_argument("--run-dirs",nargs="+",required=True)
    p.add_argument("--source-protocol",required=True)
    p.add_argument("--reference-run-dirs",nargs="+")
    p.add_argument("--reference-protocol")
    p.add_argument("--output-dir",default="outputs/cpu-temperature-diagnostic")
    p.add_argument("--expected-training-cases",type=int,default=256)
    p.add_argument("--expected-reference-training-cases",type=int,default=128)
    p.add_argument("--expected-development-cases",type=int,default=159)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    operation = prepare if args.prepare else execute
    try:
        result = operation(args.run_dirs,output_dir=args.output_dir,source_protocol_path=args.source_protocol,
                           reference_run_dirs=args.reference_run_dirs,reference_protocol_path=args.reference_protocol,
                           expected_training_cases=args.expected_training_cases,
                           expected_reference_training_cases=args.expected_reference_training_cases,
                           expected_development_cases=args.expected_development_cases)
    except (ValueError,OSError) as error:
        p.error(str(error) if isinstance(error,ValueError) else "A required local diagnostic artifact is unreadable.")
    print(json.dumps({"status":result["status"],"temperatures_fitted":bool(args.execute)},allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
