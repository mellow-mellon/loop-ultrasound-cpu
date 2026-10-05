"""Aggregate-only scientific figures for the depth/supervision CPU screen.

Only the 12 declared summary JSON files and their combined analysis are read.
The plotted endpoints belong to separately trained two- and four-step models;
terminal-only intermediate readouts are not treated as trained short models.
Matplotlib is optional and loaded only after report validation.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np


DEPTHS = (2, 4)
SUPERVISIONS = ("all", "terminal")
SEEDS = (17, 29, 43)
REGIMES = tuple((depth, supervision) for supervision in SUPERVISIONS for depth in DEPTHS)
COLORS = {(2, "all"): "#0072B2", (4, "all"): "#56B4E9",
          (2, "terminal"): "#D55E00", (4, "terminal"): "#E69F00"}
LABELS = {(depth, supervision): f"{depth}-step: {('every step' if supervision == 'all' else 'final step only')} supervised"
          for depth, supervision in REGIMES}
SUPERVISION_COLORS = {"all": "#0072B2", "terminal": "#D55E00"}
BASELINES = {"constant": ("Constant (train prevalence)", "#5D5D5D", "--"),
             "logistic_regression": ("Linear feature classifier", "#292929", ":")}
METRICS = ("brier", "auroc", "mean_case_dice")
DISCLAIMER = (
    "Development screening only; Case-to-patient mapping unverified; no clinical validation.\n"
    "Frozen image features; no augmentation; same 50 epochs and data exposure, different compute costs.\n"
    "Means of 3 fitted runs, not an ensemble. Bands/error bars: +/- sample SD across seeds, not confidence intervals.\n"
    "Final epoch 50 fixed; no best-epoch selection. Dice: 224 x 224 letterbox valid pixels."
)


def _reject_case_records(value, filename):
    if isinstance(value, dict):
        forbidden = {"case_id", "case_ids", "image_id", "image_ids", "image_path", "mask_path"}
        if forbidden.intersection(value):
            raise ValueError(f"Only aggregate reports are accepted: {filename}")
        for child in value.values():
            _reject_case_records(child, filename)
    elif isinstance(value, list):
        for child in value:
            _reject_case_records(child, filename)


def _read_report(path):
    if not path.is_file():
        raise ValueError(f"Missing aggregate report: {path.name}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Cannot read aggregate report: {path.name}") from error
    if not isinstance(record, dict):
        raise ValueError(f"Aggregate report must be a JSON object: {path.name}")
    _reject_case_records(record, path.name)
    return record


def _metric(record, metric, context):
    value = record.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing/nonfinite {metric} in {context}")
    if not 0 <= value <= 1:
        raise ValueError(f"Out-of-range {metric} in {context}")
    return float(value)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _regime_key(depth, supervision):
    return f"depth{depth}_{supervision}"


def _mean_sd(values):
    values = np.asarray(values, dtype=np.float64)
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def load_reports(report_dir):
    """Validate the fixed screen, paired initialization, and endpoint summaries."""
    report_dir = Path(report_dir)
    runs = {(depth, supervision, seed): _read_report(
        report_dir / f"SJ-depth{depth}-{supervision}-seed{seed}.json")
        for depth, supervision in REGIMES for seed in SEEDS}
    analysis = _read_report(report_dir / "depth_supervision_analysis.json")
    if (analysis.get("clinical_validation") is not False
            or analysis.get("evaluation_split") != "tune_development"
            or analysis.get("model_selection") != "fixed_final_epoch_no_best_epoch_selection"
            or analysis.get("training_seeds") != list(SEEDS)
            or analysis.get("n_training_cases") != 128
            or analysis.get("n_development_cases") != 159
            or analysis.get("patient_mapping") != "unverified_Case_grouped_only"):
        raise ValueError("Analysis is not the fixed 128/159 Case, three-seed development screen.")
    fingerprints = analysis.get("cohort_fingerprints", {})
    sources = analysis.get("sources", {})
    source_keys = ("encoder_sha256", "manifest_sha256", "training_source_sha256")
    if (set(fingerprints) != {"train", "tune"}
            or not all(_digest(value) for value in fingerprints.values())):
        raise ValueError("Analysis is missing fixed-cohort fingerprints.")
    if not all(_digest(sources.get(key)) for key in source_keys):
        raise ValueError("Analysis is missing encoder/manifest/training-source digests.")
    regimes = analysis.get("regimes", {})
    if set(regimes) != {_regime_key(*regime) for regime in REGIMES}:
        raise ValueError("Analysis must describe exactly the four depth/supervision regimes.")
    paired = analysis.get("paired_initialization", {})
    paired_rows = paired.get("by_seed", [])
    if (paired.get("identical_within_each_seed") is not True
            or not isinstance(paired_rows, list) or len(paired_rows) != len(SEEDS)
            or not all(isinstance(row, dict) for row in paired_rows)
            or {row.get("seed") for row in paired_rows} != set(SEEDS)):
        raise ValueError("Analysis does not verify paired initialization for all three seeds.")
    declared_initialization = {row["seed"]: row.get("initialization_sha256") for row in paired_rows}
    epochs = list(range(5, 51, 5))
    initialization_by_seed, shared_config = {}, None
    for (depth, supervision, seed), run in runs.items():
        name = f"SJ-depth{depth}-{supervision}-seed{seed}.json"
        config, cohort = run.get("config", {}), run.get("cohort_selection", {})
        if (config.get("arm") != "SJ" or config.get("seed") != seed
                or config.get("steps") != depth or config.get("supervision") != supervision
                or config.get("epochs") != 50 or config.get("eval_every") != 5
                or config.get("cached_frozen_features") is not True
                or config.get("augment") is not False
                or run.get("n_training_cases") != 128
                or run.get("clinical_validation") is not False or run.get("device") != "cpu"
                or run.get("patient_mapping") != "unverified_Case_grouped_only"
                or run.get("evaluation_split") != "tune_development"
                or run.get("model_selection") != "fixed_final_epoch_no_best_epoch_selection"):
            raise ValueError(f"Unexpected fixed-screen configuration in {name}")
        comparable_config = {key: value for key, value in config.items()
                             if key not in {"steps", "supervision", "seed"}}
        if shared_config is not None and comparable_config != shared_config:
            raise ValueError(f"Non-experimental configuration differs in {name}")
        shared_config = comparable_config
        if (cohort.get("train_fingerprint") != fingerprints["train"]
                or cohort.get("tune_fingerprint") != fingerprints["tune"]
                or any(run.get(key) != sources[key] for key in source_keys)):
            raise ValueError(f"Cohort/source mismatch in {name}")
        initial = run.get("initialization_sha256")
        if not _digest(initial) or declared_initialization[seed] != initial:
            raise ValueError(f"Missing initialization digest in {name}")
        if seed in initialization_by_seed and initialization_by_seed[seed] != initial:
            raise ValueError(f"Initialization differs across regimes for seed {seed}")
        initialization_by_seed[seed] = initial
        readouts = run.get("readout_supervision", {})
        expected_steps = list(range(1, depth+1)) if supervision == "all" else [depth]
        expected_scope = ("directly_supervised" if supervision == "all"
                          else "diagnostic_only_no_direct_intermediate_readout_loss")
        if (readouts.get("mode") != supervision
                or readouts.get("directly_supervised_steps") != expected_steps
                or readouts.get("intermediate_predictions_scope") != expected_scope):
            raise ValueError(f"Readout supervision metadata disagree with configuration in {name}")
        history = run.get("development_history")
        if (not isinstance(history, list)
                or [entry.get("epoch") for entry in history] != epochs):
            raise ValueError(f"Expected recorded evaluation epochs 5 through 50 in {name}")
        final = run.get("development_subset", {})
        if set(final) != {str(step) for step in range(1, depth+1)}:
            raise ValueError(f"Wrong readout depths in {name}")
        endpoint = final[str(depth)]
        if endpoint.get("n_cases") != 159:
            raise ValueError(f"Wrong final development cohort count in {name}")
        for metric in METRICS:
            _metric(endpoint, metric, name)
        for entry in history:
            by_step = entry.get("by_step", {})
            if set(by_step) != {str(step) for step in range(1, depth+1)}:
                raise ValueError(f"Wrong recorded readout depths in {name}")
            value = by_step[str(depth)]
            if value.get("n_cases") != 159:
                raise ValueError(f"Development cohort count changed in {name}")
            for metric in METRICS:
                _metric(value, metric, f"{name}, epoch {entry['epoch']}")
        for metric in METRICS:
            if not math.isclose(history[-1]["by_step"][str(depth)][metric], endpoint[metric],
                                rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Epoch-50 and final endpoint metrics disagree in {name}")
    # The combined analysis and plotted summaries must describe the same fits.
    # Plot directly from the summaries, while verifying the analyzer's means/SD.
    for depth, supervision in REGIMES:
        regime = regimes[_regime_key(depth, supervision)]
        if regime.get("depth") != depth or regime.get("supervision") != supervision:
            raise ValueError("Analysis regime labels disagree with fitted configurations.")
        for metric in METRICS:
            values = [runs[depth, supervision, seed]["development_subset"][str(depth)][metric]
                      for seed in SEEDS]
            mean, sd = _mean_sd(values)
            stats = regime.get("endpoint", {}).get(metric, {})
            for name, expected in (("mean", mean), ("std_across_training_seeds", sd)):
                actual = stats.get(name)
                if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                        or not math.isfinite(actual)
                        or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)):
                    raise ValueError(f"Analysis endpoint {metric}/{name} disagrees with run summaries.")
    baseline_values = {}
    for name in BASELINES:
        value = (analysis.get("baselines") or {}).get(name, {}).get("tune_metrics", {})
        if value.get("n_cases") != 159:
            raise ValueError(f"Missing/wrong-cohort {name} baseline in analysis.")
        baseline_values[name] = {metric: _metric(value, metric, name) for metric in ("brier", "auroc")}
    return runs, epochs, baseline_values


def _seed_lines(axis, x, values, regime):
    mean, sd = _mean_sd(values)
    axis.plot(x, mean, color=COLORS[regime], marker="o", markersize=3.5,
              linestyle="-" if regime[1] == "all" else "--", linewidth=1.8,
              label=LABELS[regime])
    # Exact mean +/- sample SD: no confidence-interval conversion or clipping.
    axis.fill_between(x, mean-sd, mean+sd, color=COLORS[regime], alpha=.14, linewidth=0)


def _baseline_lines(axis, baselines, metric):
    for name, (label, color, linestyle) in BASELINES.items():
        axis.axhline(baselines[name][metric], color=color, linestyle=linestyle,
                     linewidth=1.3, label=label)


def _decorate(figure, axes, title, subtitle):
    figure.suptitle(title, fontsize=13, fontweight="bold", y=.975)
    figure.text(.5, .91, subtitle, ha="center", fontsize=9.5, color="#333333")
    for axis in axes:
        axis.grid(alpha=.20, linewidth=.6)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    labels = [LABELS[regime] for regime in REGIMES] + [value[0] for value in BASELINES.values()]
    figure.legend([by_label[label] for label in labels], labels,
                  loc="lower center", bbox_to_anchor=(.5, .205),
                  ncol=3, frameon=False, fontsize=8.5)
    figure.text(.5, .025, DISCLAIMER, ha="center", va="bottom", fontsize=8,
                color="#333333", linespacing=1.45)
    figure.subplots_adjust(left=.07, right=.97, top=.82, bottom=.41, wspace=.28)


def _render_figures(runs, epochs, baselines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    learning, axes = plt.subplots(1, 2, figsize=(12, 7))
    for axis, metric, ylabel in zip(axes, ("brier", "auroc"),
                                   ("Development Brier (lower is better)", "Development AUROC (higher is better)")):
        for depth, supervision in REGIMES:
            values = [[entry["by_step"][str(depth)][metric]
                       for entry in runs[depth, supervision, seed]["development_history"]]
                      for seed in SEEDS]
            _seed_lines(axis, epochs, values, (depth, supervision))
        _baseline_lines(axis, baselines, metric)
        axis.set(xlabel="Recorded training epoch", ylabel=ylabel, xticks=epochs)
        axis.set_title("Each model's trained endpoint", fontsize=10)
    _decorate(learning, axes, "Learning curves | shared Transformer + joint segmentation",
              "128 training / 159 development Cases | separately trained depth and supervision settings")

    endpoint, axes = plt.subplots(1, 3, figsize=(15, 7))
    for axis, metric, ylabel in zip(axes, METRICS,
                                   ("Development Brier (lower is better)",
                                    "Development AUROC (higher is better)",
                                    "Mean Case Dice (higher is better)")):
        for supervision in SUPERVISIONS:
            values = [[runs[depth, supervision, seed]["development_subset"][str(depth)][metric]
                       for depth in DEPTHS] for seed in SEEDS]
            means, sd = _mean_sd(values)
            axis.plot(DEPTHS, means, color=SUPERVISION_COLORS[supervision], linewidth=1.4,
                      linestyle="-" if supervision == "all" else "--", alpha=.7)
            for index, depth in enumerate(DEPTHS):
                axis.errorbar(depth, means[index], yerr=sd[index], color=COLORS[depth, supervision],
                              marker="o" if supervision == "all" else "s", markersize=6,
                              capsize=4, elinewidth=1.6, linestyle="none",
                              label=LABELS[depth, supervision])
        if metric in ("brier", "auroc"):
            _baseline_lines(axis, baselines, metric)
        axis.set(xlabel="Trained model depth", ylabel=ylabel, xticks=DEPTHS,
                 xticklabels=("2-step model", "4-step model"), xlim=(1.6, 4.4))
        axis.set_title("Separately optimized endpoints", fontsize=10)
    _decorate(endpoint, axes, "Endpoint comparison | separately trained 2-step and 4-step models",
              "128 training / 159 development Cases | fixed final epoch 50 | connecting lines compare distinct fitted models")
    payloads = {}
    try:
        for name, figure in (("learning_curves", learning), ("endpoint_comparison", endpoint)):
            for extension in ("png", "svg"):
                buffer = BytesIO()
                figure.savefig(buffer, format=extension, dpi=180)
                payloads[f"{name}.{extension}"] = buffer.getvalue()
    finally:
        plt.close(learning)
        plt.close(endpoint)
    return payloads


def create_figures(report_dir, output_dir):
    runs, epochs, baselines = load_reports(report_dir)
    output_dir = Path(output_dir)
    names = [f"{name}.{suffix}" for name in ("learning_curves", "endpoint_comparison")
             for suffix in ("png", "svg")]
    if any((output_dir / name).exists() for name in names):
        raise ValueError("Figure output already exists; choose a fresh output directory.")
    previous_config = {key: os.environ.get(key) for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME")}
    try:
        with tempfile.TemporaryDirectory(prefix="loop-ultrasound-depth-plot-") as config_dir:
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
    parser.add_argument("--report-dir", default="reports/depth_supervision128")
    parser.add_argument("--output-dir", default="reports/depth_supervision128/figures")
    args = parser.parse_args(argv)
    try:
        paths = create_figures(args.report_dir, args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"status": "aggregate_development_figures_created", "files": paths}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
