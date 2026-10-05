"""Scientific figures from aggregate expanded-screen JSON reports only.

No image, Case-level CSV, or checkpoint is read. Matplotlib is an optional
dependency and is imported only after all input reports have been validated.
Figures show means of seed-specific metrics, never ensemble predictions.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


ARMS = ("SC", "SJ", "UC", "UJ")
SEEDS = (17, 29, 43)
DEPTHS = (1, 2, 3, 4)
COLORS = {"SC": "#0072B2", "SJ": "#009E73", "UC": "#D55E00", "UJ": "#CC79A7"}
LABELS = {"SC": "SC: shared, detached mask", "SJ": "SJ: shared, joint mask",
          "UC": "UC: untied, detached mask", "UJ": "UJ: untied, joint mask"}
BASELINES = {"constant": ("Constant (train prevalence)", "#5D5D5D", "--"),
             "logistic_regression": ("Linear feature classifier", "#292929", ":")}
DISCLAIMER = ("Development screening only; no clinical validation. Case-to-patient mapping unverified.\n"
              "Lines: means of 3 fitted runs, not an ensemble. Bands: +/- sample SD across seeds (not confidence intervals).\n"
              "Final epoch 50 fixed; no best-epoch selection. Dice: 224 x 224 letterbox valid pixels.")


def _read_report(path):
    if not path.is_file():
        raise ValueError(f"Missing aggregate report: {path.name}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Cannot read aggregate report: {path.name}") from error
    if not isinstance(record, dict):
        raise ValueError(f"Aggregate report must be a JSON object: {path.name}")
    return record


def _metric(record, metric, context):
    value = record.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing/nonfinite {metric} in {context}")
    if not 0 <= value <= 1:
        raise ValueError(f"Out-of-range {metric} in {context}")
    return float(value)


def load_reports(report_dir):
    """Load exactly the declared 12 fitted runs and their aggregate analysis."""
    report_dir = Path(report_dir)
    # Check run paths first so a partial experiment fails clearly, even when
    # Matplotlib is absent or its combined analysis has not yet been exported.
    runs = {(arm, seed): _read_report(report_dir / f"{arm}-seed{seed}.json")
            for arm in ARMS for seed in SEEDS}
    analysis = _read_report(report_dir / "expanded_analysis.json")
    if (analysis.get("clinical_validation") is not False
            or analysis.get("evaluation_split") != "tune_development"
            or analysis.get("model_selection") != "fixed_final_epoch_no_best_epoch_selection"
            or analysis.get("training_seeds") != list(SEEDS)
            or analysis.get("n_training_cases") != 128
            or analysis.get("n_development_cases") != 159):
        raise ValueError("Analysis is not the fixed 128/159 Case, three-seed development screen.")
    fingerprints = analysis.get("cohort_fingerprints", {})
    sources = analysis.get("sources", {})
    if set(fingerprints) != {"train", "tune"} or not all(fingerprints.values()):
        raise ValueError("Analysis is missing fixed-cohort fingerprints.")
    if not sources.get("encoder_sha256") or not sources.get("manifest_sha256"):
        raise ValueError("Analysis is missing encoder/manifest source digests.")
    epochs = None
    for (arm, seed), run in runs.items():
        config, cohort = run.get("config", {}), run.get("cohort_selection", {})
        if (config.get("arm") != arm or config.get("seed") != seed
                or config.get("steps") != 4 or config.get("epochs") != 50
                or config.get("eval_every") != 5 or run.get("n_training_cases") != 128
                or run.get("clinical_validation") is not False or run.get("device") != "cpu"
                or run.get("evaluation_split") != "tune_development"
                or run.get("model_selection") != "fixed_final_epoch_no_best_epoch_selection"):
            raise ValueError(f"Unexpected fixed-screen configuration in {arm}-seed{seed}.json")
        if (cohort.get("train_fingerprint") != fingerprints["train"]
                or cohort.get("tune_fingerprint") != fingerprints["tune"]
                or any(run.get(key) != sources[key] for key in ("encoder_sha256", "manifest_sha256"))):
            raise ValueError(f"Cohort/source mismatch in {arm}-seed{seed}.json")
        history = run.get("development_history")
        if not isinstance(history, list) or not history:
            raise ValueError(f"Missing development history in {arm}-seed{seed}.json")
        actual_epochs = [entry.get("epoch") for entry in history]
        if actual_epochs != list(range(5, 51, 5)):
            raise ValueError(f"Expected recorded evaluation epochs 5 through 50 in {arm}-seed{seed}.json")
        if epochs is not None and actual_epochs != epochs:
            raise ValueError("All fitted runs must have the same recorded evaluation epochs.")
        epochs = actual_epochs
        for entry in history:
            final_step = entry.get("by_step", {}).get("4", {})
            if final_step.get("n_cases") != 159:
                raise ValueError(f"Development cohort count changed in {arm}-seed{seed}.json")
            for metric in ("brier", "auroc"):
                _metric(final_step, metric, f"{arm}, seed {seed}, epoch {entry['epoch']}")
        final = run.get("development_subset", {})
        for depth in DEPTHS:
            value = final.get(str(depth), {})
            if value.get("n_cases") != 159:
                raise ValueError(f"Wrong final development cohort count in {arm}-seed{seed}.json")
            for metric in ("brier", "mean_case_dice"):
                _metric(value, metric, f"{arm}, seed {seed}, depth {depth}")
        for metric in ("brier", "auroc"):
            recorded = _metric(history[-1]["by_step"]["4"], metric, "final recorded epoch")
            terminal = _metric(final["4"], metric, "final model")
            if not math.isclose(recorded, terminal, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Epoch-50 and final model metrics disagree in {arm}-seed{seed}.json")
    baseline_values = {}
    for name in BASELINES:
        record = (analysis.get("baselines") or {}).get(name, {}).get("tune_metrics", {})
        if record.get("n_cases") != 159:
            raise ValueError(f"Missing/wrong-cohort {name} baseline in analysis.")
        baseline_values[name] = {metric: _metric(record, metric, name) for metric in ("brier", "auroc")}
    return runs, epochs, baseline_values


def _mean_sd(values):
    values = np.asarray(values, dtype=np.float64)
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def _seed_lines(axis, x, values, arm):
    mean, sd = _mean_sd(values)
    axis.plot(x, mean, color=COLORS[arm], marker="o", markersize=3.5,
              linewidth=1.8, label=LABELS[arm])
    # Keep the exact mean +/- sample SD; do not turn it into a confidence
    # interval or silently clip it to a probability range.
    axis.fill_between(x, mean-sd, mean+sd, color=COLORS[arm], alpha=.13, linewidth=0)


def _baseline_lines(axis, baselines, metric):
    for name, (label, color, linestyle) in BASELINES.items():
        axis.axhline(baselines[name][metric], color=color, linestyle=linestyle,
                     linewidth=1.4, label=label)


def _decorate(figure, axes, title):
    figure.suptitle(title, fontsize=13, fontweight="bold", y=.975)
    for axis in axes:
        axis.grid(alpha=.20, linewidth=.6)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .19),
                  ncol=3, frameon=False, fontsize=8.5)
    figure.text(.5, .035, DISCLAIMER, ha="center", va="bottom", fontsize=8,
                color="#333333", linespacing=1.45)
    figure.subplots_adjust(left=.08, right=.97, top=.83, bottom=.39, wspace=.27)


def _render_figures(runs, epochs, baselines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    learning, axes = plt.subplots(1, 2, figsize=(12, 7))
    for axis, metric, ylabel in zip(axes, ("brier", "auroc"),
                                     ("Development Brier score (lower is better)", "Development AUROC (higher is better)")):
        for arm in ARMS:
            values = [[entry["by_step"]["4"][metric] for entry in runs[arm, seed]["development_history"]]
                      for seed in SEEDS]
            _seed_lines(axis, epochs, values, arm)
        _baseline_lines(axis, baselines, metric)
        axis.set(xlabel="Recorded training epoch", ylabel=ylabel, xticks=epochs)
        axis.set_title("Depth-4 readout", fontsize=10)
    _decorate(learning, axes, "Learning curves | 128 training / 159 development Cases")

    depth, axes = plt.subplots(1, 2, figsize=(12, 7))
    for axis, metric, ylabel in zip(axes, ("brier", "mean_case_dice"),
                                     ("Development Brier score (lower is better)", "Mean Case Dice (higher is better)")):
        for arm in ARMS:
            values = [[runs[arm, seed]["development_subset"][str(r)][metric] for r in DEPTHS]
                      for seed in SEEDS]
            _seed_lines(axis, DEPTHS, values, arm)
        if metric == "brier":
            _baseline_lines(axis, baselines, metric)
        axis.set(xlabel="Readout depth of the trained 4-step model", ylabel=ylabel, xticks=DEPTHS)
        axis.set_title("Fixed final epoch 50", fontsize=10)
    _decorate(depth, axes, "Depth curves | 128 training / 159 development Cases")
    payloads = {}
    try:
        for name, figure in (("learning_curves", learning), ("depth_curves", depth)):
            for extension in ("png", "svg"):
                buffer = BytesIO()
                figure.savefig(buffer, format=extension, dpi=180)
                payloads[f"{name}.{extension}"] = buffer.getvalue()
    finally:
        plt.close(learning)
        plt.close(depth)
    return payloads


def create_figures(report_dir, output_dir):
    runs, epochs, baselines = load_reports(report_dir)
    output_dir = Path(output_dir)
    names = [f"{name}.{suffix}" for name in ("learning_curves", "depth_curves") for suffix in ("png", "svg")]
    if any((output_dir / name).exists() for name in names):
        raise ValueError("Figure output already exists; choose a fresh output directory.")
    # Keep font/cache artifacts outside the reports; only the four figures are
    # written to the requested destination.
    previous_config = {key: os.environ.get(key) for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME")}
    try:
        with tempfile.TemporaryDirectory(prefix="loop-ultrasound-plot-") as config_dir:
            os.environ["MPLCONFIGDIR"] = config_dir
            os.environ["XDG_CACHE_HOME"] = config_dir
            try:
                payloads = _render_figures(runs, epochs, baselines)
            except ModuleNotFoundError as error:
                if error.name == "matplotlib":
                    raise ValueError("Optional plotting dependency missing; install matplotlib in the project environment.") from error
                raise
    finally:
        for key, value in previous_config.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        try:
            with (output_dir / name).open("xb") as stream:
                stream.write(payload)
        except FileExistsError as error:
            raise ValueError("Figure output already exists; choose a fresh output directory.") from error
    return [str(output_dir / name) for name in names]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="reports/expanded128")
    parser.add_argument("--output-dir", default="reports/expanded128/figures")
    args = parser.parse_args(argv)
    try:
        paths = create_figures(args.report_dir, args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"status": "aggregate_development_figures_created", "files": paths}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
