"""Aggregate-only figure for the primary five-seed OOF temperature diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


SEEDS = (53,71,89,107,131)


def _read(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError,ValueError):
        raise ValueError("Readable diagnostic analysis and locked protocol are required.") from None
    if not isinstance(value,dict):
        raise ValueError("Diagnostic artifacts must be aggregate mappings.")
    return value


def _validate(report,protocol,protocol_path):
    if (report.get("status")!="executed_case_grouped_crossfit_temperature_diagnostic"
            or report.get("clinical_validation") is not False or report.get("calibration_and_test_predictions") is not False
            or report.get("patient_mapping")!="unverified_Case_grouped_only"
            or report.get("evaluation_split")!="observed_tune_crossfit"):
        raise ValueError("Plot requires the completed exploratory Case-grouped OOF diagnostic.")
    if (protocol.get("status")!="fixed_before_case_grouped_crossfit_temperature_diagnostic"
            or protocol.get("n_splits")!=5 or protocol.get("primary_training_seeds")!=list(SEEDS)
            or protocol.get("n_development_cases")!=159 or protocol.get("primary_training_cases")!=256):
        raise ValueError("Plot requires the fixed 256-training/159-tune, five-seed, five-fold protocol.")
    if report.get("protocol_sha256")!=hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest():
        raise ValueError("Analysis and locked protocol digests differ.")
    if (report.get("sources")!=protocol.get("sources") or report.get("cohort_fingerprints")!=protocol.get("cohort_fingerprints")
            or report.get("fold_assignment_sha256")!=protocol.get("fold_assignment_sha256")):
        raise ValueError("Analysis provenance, cohorts or shared Case folds differ from the locked protocol.")
    block = report.get("primary256")
    if (not isinstance(block,dict) or block.get("training_seeds")!=list(SEEDS) or block.get("n_cases")!=159
            or block.get("role")!="primary_five_seed_256_diagnostic"):
        raise ValueError("Only the five new primary 256-Case fitted seeds can enter this figure.")
    rows = block.get("seed_results")
    if not isinstance(rows,list) or [row.get("seed") for row in rows]!=list(SEEDS):
        raise ValueError("All five separate fitted-seed results are required.")
    for depth in (2,4):
        name = f"depth{depth}_terminal"
        for method in ("raw","oof"):
            for metric in ("brier","nll"):
                values = np.array([row["regimes"][name][method][metric] for row in rows])
                declared = block["regimes"][name][method][metric]
                if (not np.isfinite(values).all() or (values<0).any()
                        or not math.isclose(declared["mean"],float(values.mean()),abs_tol=1e-12)
                        or not math.isclose(declared["std_across_training_seeds"],float(values.std(ddof=1)),abs_tol=1e-12)):
                    raise ValueError("Endpoint means and seed SD must match the five fitted-seed metrics.")
        for row in rows:
            fits = row["regimes"][name].get("temperature_fits")
            if not isinstance(fits,list) or [fit.get("fold") for fit in fits]!=list(range(5)):
                raise ValueError("Each fitted network requires all five complementary-fold temperatures.")
            for fit in fits:
                temperature = fit.get("temperature")
                if (isinstance(temperature,bool) or not isinstance(temperature,(int,float))
                        or not math.isfinite(temperature) or not np.exp(-4)<=temperature<=np.exp(4)):
                    raise ValueError("All displayed temperatures must satisfy the locked positive bounds.")
    for row in rows:
        for method in ("raw","oof"):
            gain = row["regimes"]["depth2_terminal"][method]["brier"]-row["regimes"]["depth4_terminal"][method]["brier"]
            if not math.isclose(row[f"{method}_depth_gain_Brier"],gain,abs_tol=1e-12):
                raise ValueError("Seed-level depth benefit disagrees with its separately fitted endpoints.")
    return block


def plot(analysis_path,*,protocol_path,output_dir):
    report,protocol = _read(analysis_path),_read(protocol_path)
    try:
        block = _validate(report,protocol,protocol_path)
    except (KeyError,TypeError,AttributeError):
        raise ValueError("Diagnostic aggregate schema is incomplete.") from None
    root = Path(output_dir)
    names = ("temperature_diagnostic.png","temperature_diagnostic.svg")
    if any((root/name).exists() for name in names):
        raise ValueError("Figure output already exists; choose a fresh output directory.")
    root.mkdir(parents=True,exist_ok=True)
    old_mpl,old_cache = os.environ.get("MPLCONFIGDIR"),os.environ.get("XDG_CACHE_HOME")
    with tempfile.TemporaryDirectory(prefix="temperature-plot-") as cache:
        os.environ["MPLCONFIGDIR"],os.environ["XDG_CACHE_HOME"] = cache,cache
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch
            plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
            fig,axes = plt.subplots(2,2,figsize=(12,8))
            colors = {"raw":"#3178a8","oof":"#d37735"}
            x = np.array([0.,1.])
            for axis,metric,title in ((axes[0,0],"brier","Case Brier (lower is better)"),
                                      (axes[0,1],"nll","Case NLL (lower is better)")):
                for method,offset in (("raw",-.16),("oof",.16)):
                    stats = [block["regimes"][f"depth{depth}_terminal"][method][metric] for depth in (2,4)]
                    axis.bar(x+offset,[s["mean"] for s in stats],width=.29,
                             yerr=[s["std_across_training_seeds"] for s in stats],capsize=4,
                             color=colors[method],alpha=.85,label="Raw" if method=="raw" else "OOF temperature")
                axis.set_xticks(x,["Trained depth 2","Trained depth 4"])
                axis.set_title(title)
                axis.grid(axis="y",alpha=.18)
                axis.set_axisbelow(True)
            axis = axes[1,0]
            rows = block["seed_results"]
            for i,row in enumerate(rows):
                raw_gain,oof_gain = row["raw_depth_gain_Brier"],row["oof_depth_gain_Brier"]
                axis.plot([i-.1,i+.1],[raw_gain,oof_gain],color="#8b8b8b",lw=1)
                axis.scatter(i-.1,raw_gain,color=colors["raw"],s=45,label="Raw" if i==0 else None)
                axis.scatter(i+.1,oof_gain,color=colors["oof"],s=45,label="OOF temperature" if i==0 else None)
            axis.axhline(0,color="black",lw=.8,linestyle="--")
            axis.set_xticks(range(5),[str(seed) for seed in SEEDS])
            axis.set_xlabel("Fitted network seed")
            axis.set_ylabel("Brier depth 2 - depth 4")
            axis.set_title("Depth benefit per fitted seed (positive favors depth 4)")
            axis.grid(axis="y",alpha=.18)
            axis = axes[1,1]
            for i,row in enumerate(rows):
                for depth,offset,color in ((2,-.15,"#458c78"),(4,.15,"#8a5c9f")):
                    fits = row["regimes"][f"depth{depth}_terminal"]["temperature_fits"]
                    values = [fit["temperature"] for fit in fits]
                    axis.plot([i+offset]*2,[min(values),max(values)],color=color,lw=1.1)
                    axis.scatter(i+offset+np.linspace(-.035,.035,5),values,color=color,s=20,
                                 label=f"Depth {depth}" if i==0 else None)
            axis.axhline(1,color="black",lw=.8,linestyle="--")
            axis.set_yscale("log")
            axis.set_xticks(range(5),[str(seed) for seed in SEEDS])
            axis.set_xlabel("Fitted network seed; five fold fits per network")
            axis.set_ylabel("Positive temperature, log scale")
            axis.set_title("Complementary-fold temperature ranges")
            axis.legend(frameon=False,fontsize=9)
            axis.grid(axis="y",alpha=.18)
            fig.suptitle("OOF temperature diagnostic: 256 training Cases / 159 observed tune Cases",fontsize=14,y=.985)
            fig.legend(handles=[Patch(facecolor=colors["raw"],label="Raw"),
                                Patch(facecolor=colors["oof"],label="OOF temperature")],
                       loc="upper center",bbox_to_anchor=(.5,.95),ncol=2,frameon=False,fontsize=10)
            fig.text(.5,.022,"Five new fitted seeds; bars show mean ± seed sample SD, not confidence intervals.\n"
                     "Shared stratified Case folds; fold temperatures are dependent fits. Exploratory diagnostic, no clinical validation.",
                     ha="center",va="bottom",fontsize=9,color="#555555")
            fig.tight_layout(rect=(0,.09,1,.89))
            for name in names:
                fig.savefig(root/name,dpi=180,bbox_inches="tight")
            plt.close(fig)
        finally:
            for key,prior in (("MPLCONFIGDIR",old_mpl),("XDG_CACHE_HOME",old_cache)):
                if prior is None:
                    os.environ.pop(key,None)
                else:
                    os.environ[key] = prior
    return [root/name for name in names]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis",required=True)
    p.add_argument("--protocol",required=True)
    p.add_argument("--output-dir",default="outputs/cpu-temperature-diagnostic/figures")
    args = p.parse_args(argv)
    try:
        plot(args.analysis,protocol_path=args.protocol,output_dir=args.output_dir)
    except (ValueError,OSError) as error:
        p.error(str(error) if isinstance(error,ValueError) else "A required local figure artifact is unreadable.")
    print(json.dumps({"status":"aggregate_temperature_diagnostic_figure_rendered","figures":2}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
