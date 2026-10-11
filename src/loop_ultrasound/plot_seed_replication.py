"""Aggregate-only figures for five-new-seed terminal-supervision replication.

The fixed protocol and ten explicitly named run-summary JSON files are the
only inputs. Earlier seeds are never pooled into the primary figures. Plotted
readouts are endpoints of separately fitted two- and four-step models.
Matplotlib is optional and is imported only after validation.
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
SEEDS = (53, 71, 89, 107, 131)
SOURCE_KEYS = ("encoder_sha256", "manifest_sha256", "training_source_sha256")
FINAL_SELECTION = "fixed_final_epoch_no_best_epoch_selection"
DEFAULT_PROTOCOL = "configs/cpu_seed_replication128.json"
DEPTH_COLORS = {2: "#0072B2", 4: "#D55E00"}
SEED_COLORS = dict(zip(SEEDS, ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")))
FIXED_CONFIG = {"arm": "SJ", "supervision": "terminal", "epochs": 50,
                "batch_size": 2, "learning_rate": .0003, "segmentation_weight": 1.,
                "cached_frozen_features": True, "augment": False,
                "selection_seed": 20261004, "sampling": "proportional", "eval_every": 5}
DISCLAIMER = (
    "Exploratory follow-up on an already observed development cohort; Case-to-patient mapping unverified; no clinical validation.\n"
    "Frozen image features; no augmentation; final epoch 50 fixed. Same updates/data exposure; four steps use more compute.\n"
    "Only five NEW seeds in these figures; earlier three excluded. Means are not ensembles; sample SD is not a confidence interval."
)


def _reject_case_records(value, filename):
    if isinstance(value, dict):
        if {"case_id", "case_ids", "image_id", "image_ids", "image_path", "mask_path"}.intersection(value):
            raise ValueError(f"Only aggregate inputs are accepted: {filename}")
        for child in value.values():
            _reject_case_records(child, filename)
    elif isinstance(value, list):
        for child in value:
            _reject_case_records(child, filename)


def _read_json(path):
    if not path.is_file():
        raise ValueError(f"Missing aggregate input: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read aggregate input: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Aggregate input must be a JSON object: {path.name}")
    _reject_case_records(value, path.name)
    return value


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _metric(record, metric, context):
    value = record.get(metric)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            or (metric != "nll" and value > 1)):
        raise ValueError(f"Missing/nonfinite/out-of-range {metric} in {context}")
    return float(value)


def _mean_sd(values):
    values = np.asarray(values, dtype=np.float64)
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def load_reports(report_dir, *, protocol_path=DEFAULT_PROTOCOL):
    """Validate all ten fits against the locked protocol, without raw data."""
    protocol = _read_json(Path(protocol_path))
    if (protocol.get("status") != "fixed_before_seed_replication_training"
            or protocol.get("seeds") != list(SEEDS) or protocol.get("depths") != list(DEPTHS)
            or protocol.get("supervisions") != ["terminal"] or protocol.get("arms") != ["SJ"]
            or protocol.get("device") != "cpu" or protocol.get("clinical_validation") is not False
            or protocol.get("calibration_and_test_predictions") is not False
            or protocol.get("warm_starts") is not False
            or protocol.get("patient_mapping") != "unverified_Case_grouped_only"
            or protocol.get("model_selection") != FINAL_SELECTION
            or protocol.get("cached_frozen_features") is not True or protocol.get("augment") is not False
            or protocol.get("training_updates_per_run") != 3200
            or any(protocol.get(key) != value for key, value in FIXED_CONFIG.items()
                   if key not in {"arm", "supervision"})):
        raise ValueError("Protocol is not the fixed five-new-seed CPU terminal-supervision replication.")
    initialization = protocol.get("initialization_sha256_by_seed", {})
    if (not isinstance(initialization, dict) or set(initialization) != {str(seed) for seed in SEEDS}
            or not all(_digest(value) for value in initialization.values())
            or not all(_digest(protocol.get(key)) for key in SOURCE_KEYS)):
        raise ValueError("Protocol is missing source or paired-initialization digests.")
    expected_cohorts = {"train": {"cases": 128, "images": 227, "benign": 87, "malignant": 41},
                        "tune": {"cases": 159, "images": 281, "benign": 108, "malignant": 51}}
    for split, counts in expected_cohorts.items():
        cohort = protocol.get(split, {})
        if (not isinstance(cohort, dict) or not _digest(cohort.get("fingerprint"))
                or any(cohort.get(key) != count for key, count in counts.items())):
            raise ValueError(f"Protocol has an unexpected fixed {split} cohort.")
    report_dir = Path(report_dir)
    runs = {(depth, seed): _read_json(report_dir / f"SJ-depth{depth}-terminal-seed{seed}.json")
            for depth in DEPTHS for seed in SEEDS}
    epochs = list(range(5, 51, 5))
    common_config = None
    for (depth, seed), run in runs.items():
        name = f"SJ-depth{depth}-terminal-seed{seed}.json"
        config = run.get("config", {})
        timing = run.get("timing", {})
        if (not isinstance(config, dict)
                or any(config.get(key) != value for key, value in FIXED_CONFIG.items())
                or config.get("cached_frozen_features") is not True or config.get("augment") is not False
                or config.get("steps") != depth or config.get("seed") != seed
                or run.get("status") != "executed_cpu_engineering_pilot"
                or run.get("clinical_validation") is not False or run.get("device") != "cpu"
                or run.get("patient_mapping") != "unverified_Case_grouped_only"
                or run.get("evaluation_split") != "tune_development"
                or run.get("model_selection") != FINAL_SELECTION
                or run.get("n_training_cases") != 128 or run.get("n_training_images") != 227
                or not isinstance(timing, dict) or timing.get("updates") != 3200):
            raise ValueError(f"Unexpected configuration, provenance, or incomplete training in {name}")
        comparable_config = {key: value for key, value in config.items() if key not in {"steps", "seed"}}
        if common_config is not None and comparable_config != common_config:
            raise ValueError(f"Non-experimental configuration differs in {name}")
        common_config = comparable_config
        cohort = run.get("cohort_selection", {})
        if (not isinstance(cohort, dict) or cohort.get("sampling") != protocol["sampling"]
                or cohort.get("selection_seed") != protocol["selection_seed"]
                or any(cohort.get(f"{split}_fingerprint") != protocol[split]["fingerprint"]
                       for split in ("train", "tune"))
                or any(run.get(key) != protocol[key] for key in SOURCE_KEYS)
                or run.get("initialization_sha256") != initialization[str(seed)]):
            raise ValueError(f"Locked cohort/source/initialization mismatch in {name}")
        readout = run.get("readout_supervision", {})
        if (not isinstance(readout, dict) or readout.get("mode") != "terminal"
                or readout.get("directly_supervised_steps") != [depth]
                or readout.get("intermediate_predictions_scope")
                != "diagnostic_only_no_direct_intermediate_readout_loss"):
            raise ValueError(f"Terminal-only supervision metadata disagree in {name}")
        history = run.get("development_history")
        if (not isinstance(history, list)
                or not all(isinstance(entry, dict) for entry in history)
                or [entry.get("epoch") for entry in history] != epochs):
            raise ValueError(f"Expected all ten recorded epochs 5 through 50 in {name}")
        final = run.get("development_subset", {})
        if not isinstance(final, dict) or set(final) != {str(step) for step in range(1, depth+1)}:
            raise ValueError(f"Wrong final readout depths in {name}")
        endpoint = final[str(depth)]
        snapshots = [entry.get("by_step", {}) for entry in history]
        for snapshot in snapshots + [final]:
            if not isinstance(snapshot, dict) or set(snapshot) != {str(step) for step in range(1, depth+1)}:
                raise ValueError(f"Wrong recorded readout depths in {name}")
            value = snapshot[str(depth)]
            if (not isinstance(value, dict) or value.get("n_cases") != 159
                    or value.get("n_positive") != 51 or value.get("n_negative") != 108):
                raise ValueError(f"Development cohort count changed in {name}")
            for metric in ("brier", "auroc", "nll"):
                _metric(value, metric, name)
        for metric in ("brier", "auroc", "nll"):
            if not math.isclose(snapshots[-1][str(depth)][metric], endpoint[metric], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Recorded epoch 50 and final endpoint disagree in {name}")
    return runs, epochs


def _decorate(figure, axes, title, subtitle, *, legend_handles=None, legend_labels=None):
    figure.suptitle(title, fontsize=13, fontweight="bold", y=.975)
    figure.text(.5, .91, subtitle, ha="center", fontsize=9.5, color="#333333")
    for axis in axes:
        axis.grid(alpha=.20, linewidth=.6)
        axis.spines[["top", "right"]].set_visible(False)
    if legend_handles is None:
        legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(legend_handles, legend_labels, loc="lower center", bbox_to_anchor=(.5, .185),
                  ncol=3, frameon=False, fontsize=8.5)
    figure.text(.5, .025, DISCLAIMER, ha="center", va="bottom", fontsize=8,
                color="#333333", linespacing=1.45)
    figure.subplots_adjust(left=.075, right=.97, top=.81, bottom=.36, wspace=.30)


def _render_figures(runs, epochs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    learning, axes = plt.subplots(1, 2, figsize=(12, 7))
    for axis, metric, ylabel in zip(axes, ("brier", "auroc"),
                                   ("Development Brier (lower is better)", "Development AUROC (higher is better)")):
        for depth in DEPTHS:
            values = [[entry["by_step"][str(depth)][metric]
                       for entry in runs[depth, seed]["development_history"]] for seed in SEEDS]
            means, sd = _mean_sd(values)
            axis.plot(epochs, means, color=DEPTH_COLORS[depth], marker="o", markersize=3.5,
                      linewidth=1.8, label=f"Separately trained {depth}-step endpoint")
            # Exact sample SD bands, neither clipped nor interpreted as CIs.
            axis.fill_between(epochs, means-sd, means+sd, color=DEPTH_COLORS[depth], alpha=.14, linewidth=0)
        axis.set(xlabel="Recorded training epoch", ylabel=ylabel, xticks=epochs)
        axis.set_title("Endpoint mean +/- sample SD across five new seeds", fontsize=10)
    _decorate(learning, axes, "New-seed learning curves | terminal-only supervision",
              "128 training / 159 development Cases | shared Transformer + joint segmentation")

    benefit, axes = plt.subplots(1, 3, figsize=(15, 7))
    brier = np.array([[runs[depth, seed]["development_subset"][str(depth)]["brier"]
                       for depth in DEPTHS] for seed in SEEDS])
    nll = np.array([[runs[depth, seed]["development_subset"][str(depth)]["nll"]
                     for depth in DEPTHS] for seed in SEEDS])
    for index, seed in enumerate(SEEDS):
        axes[0].plot(DEPTHS, brier[index], color=SEED_COLORS[seed], marker="o", markersize=4,
                     linewidth=1.4, alpha=.85, label=f"New seed {seed}")
    means, sd = _mean_sd(brier)
    axes[0].errorbar(DEPTHS, means, yerr=sd, color="#111111", marker="D", markersize=6,
                    linewidth=2, capsize=4, label="Five-new-seed mean +/- sample SD")
    axes[0].set(xlabel="Separately trained model depth", ylabel="Endpoint Brier (lower is better)",
                xticks=DEPTHS, xticklabels=("2-step model", "4-step model"), xlim=(1.65, 4.35))
    axes[0].set_title("Paired endpoint results by new seed", fontsize=10)
    gains = brier[:, 0] - brier[:, 1]
    worsening = nll[:, 1] - nll[:, 0]
    for axis, values, ylabel, title, positive_is_good in (
            (axes[1], gains, "G = Brier2 - Brier4 (positive favors 4 steps)", "Primary: Brier depth benefit", True),
            (axes[2], worsening, "NLL4 - NLL2 (positive is worse)", "Secondary: NLL change", False)):
        colors = ["#0072B2" if (value >= 0) == positive_is_good else "#D55E00" for value in values]
        axis.bar(range(len(SEEDS)), values, color=colors, alpha=.85, width=.6)
        axis.axhline(0, color="#555555", linewidth=.9)
        axis.axhline(float(np.mean(values)), color="#111111", linestyle="--", linewidth=1.4)
        axis.text(.04, .96, f"Five-new-seed mean: {np.mean(values):+.4f}", transform=axis.transAxes,
                  va="top", fontsize=9, bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        axis.set(xlabel="New training seed", ylabel=ylabel, xticks=range(len(SEEDS)),
                 xticklabels=[str(seed) for seed in SEEDS])
        axis.set_title(title, fontsize=10)
        axis.margins(y=.2)
    _decorate(benefit, axes, "Depth-benefit replication | five new seeds only",
              "Separately trained 2-step / 4-step endpoints | terminal-only supervision | fixed final epoch 50")
    payloads = {}
    try:
        for name, figure in (("learning_curves", learning), ("depth_benefit", benefit)):
            for extension in ("png", "svg"):
                buffer = BytesIO()
                figure.savefig(buffer, format=extension, dpi=180)
                payloads[f"{name}.{extension}"] = buffer.getvalue()
    finally:
        plt.close(learning)
        plt.close(benefit)
    return payloads


def create_figures(report_dir, output_dir, *, protocol_path=DEFAULT_PROTOCOL):
    runs, epochs = load_reports(report_dir, protocol_path=protocol_path)
    output_dir = Path(output_dir)
    names = [f"{name}.{extension}" for name in ("depth_benefit", "learning_curves")
             for extension in ("png", "svg")]
    if any((output_dir / name).exists() for name in names):
        raise ValueError("Figure output already exists; choose a fresh output directory.")
    previous_config = {key: os.environ.get(key) for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME")}
    try:
        with tempfile.TemporaryDirectory(prefix="loop-ultrasound-replication-plot-") as config_dir:
            os.environ["MPLCONFIGDIR"] = config_dir
            os.environ["XDG_CACHE_HOME"] = config_dir
            try:
                payloads = _render_figures(runs, epochs)
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
    parser.add_argument("--report-dir", default="reports/seed_replication128")
    parser.add_argument("--output-dir", default="reports/seed_replication128/figures")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    try:
        paths = create_figures(args.report_dir, args.output_dir, protocol_path=args.protocol)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"status": "aggregate_seed_replication_figures_created", "files": paths}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
