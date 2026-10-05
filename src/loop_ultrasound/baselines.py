"""Train-only, Case-weighted classification controls for the CPU pre-experiment.

Input features are frozen image-encoder tokens. The linear control averages
tokens within each image and averages all available images within each Case,
then fits a scaler and logistic regression on training Cases only. It therefore
uses a different view aggregation from the loop models' one-view-per-epoch
training sampler; this difference is recorded in the aggregate report.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
import json
import math
from numbers import Integral
from pathlib import Path
import time
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .cohort import cohort_fingerprint
from .metrics import classification_metrics


def _identifier(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise ValueError(f"{name} must be a nonempty string or integer.")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} cannot be empty.")
    return result


def _label(value):
    if isinstance(value, bool):
        raise ValueError("Labels must be numeric binary values, malignant=1.")
    try:
        scalar = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Labels must be numeric binary values, malignant=1.") from error
    if scalar.ndim != 0 or not math.isfinite(float(scalar)) or float(scalar) not in (0., 1.):
        raise ValueError("Labels must be finite binary scalars, malignant=1.")
    return int(scalar)


def _validate_rows(rows, expected_split):
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("Supply a nonempty sequence of image rows for each cohort.")
    groups, labels, seen_images = defaultdict(list), {}, set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("Each image row must be a mapping.")
        if not {"case_id", "image_id", "label", "split"}.issubset(row):
            raise ValueError("Image rows require case_id, image_id, label and split.")
        if row["split"] != expected_split:
            raise ValueError(f"Only split={expected_split!r} rows are allowed in this cohort.")
        case_id = _identifier(row["case_id"], "case_id")
        image_id = _identifier(row["image_id"], "image_id")
        label = _label(row["label"])
        if image_id in seen_images:
            raise ValueError("Duplicate image ID within a cohort.")
        seen_images.add(image_id)
        if case_id in labels and labels[case_id] != label:
            raise ValueError("Conflicting pathology labels within a Case.")
        labels[case_id] = label
        groups[case_id].append(index)
    if set(labels.values()) != {0, 1}:
        raise ValueError("Each pre-experiment cohort must contain both pathology classes.")
    return groups, labels, seen_images


def _case_features(rows, features, expected_split):
    groups, labels, image_ids = _validate_rows(rows, expected_split)
    if not isinstance(features, torch.Tensor) or features.device.type != "cpu":
        raise ValueError("Features must be CPU torch tensors.")
    if features.shape != (len(rows), 196, 192) or not features.is_floating_point():
        raise ValueError("Use floating-point features shaped [n_images, 196, 192].")
    if not torch.isfinite(features).all().item():
        raise ValueError("All encoder features must be finite.")
    # Float64 averaging avoids changing the mean merely through accumulation
    # precision. The input features and their order are never modified.
    image_features = features.detach().to(dtype=torch.float64).mean(dim=1).numpy()
    case_ids = sorted(groups)
    x = np.stack([image_features[groups[case_id]].mean(axis=0) for case_id in case_ids])
    y = np.asarray([labels[case_id] for case_id in case_ids], dtype=np.int64)
    image_counts = [len(groups[case_id]) for case_id in case_ids]
    return case_ids, x, y, image_counts, image_ids


def run_baselines(train_rows, tune_rows, train_features, tune_features, output_dir,
                  *, selection_seed=20261004):
    """Fit two fixed controls; return and save only aggregate performance.

    Local ``baseline_case_predictions.csv`` additionally contains Case IDs for
    explicitly aligned paired analyses. Keep that CSV in the ignored outputs
    directory. Neither control fits on tune labels/features, selects a threshold
    from tune data, or reads test/calibration data. ``selection_seed`` documents
    the caller's cohort selection; it does not select rows or tune parameters.
    """
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, Integral) or selection_seed < 0:
        raise ValueError("selection_seed must be a nonnegative integer.")
    train_ids, train_x, train_y, train_counts, train_images = _case_features(
        train_rows, train_features, "train")
    tune_ids, tune_x, tune_y, tune_counts, tune_images = _case_features(
        tune_rows, tune_features, "tune")
    if set(train_ids) & set(tune_ids):
        raise ValueError("Training and tune Case IDs must be disjoint.")
    if train_images & tune_images:
        raise ValueError("Training and tune image IDs must be disjoint.")
    output_dir = Path(output_dir)
    summary_path = output_dir / "baselines_summary.json"
    predictions_path = output_dir / "baseline_case_predictions.csv"
    if summary_path.exists() or predictions_path.exists():
        raise FileExistsError("Use a fresh baseline output directory; results are never overwritten.")

    start = time.perf_counter()
    prevalence = float(train_y.mean())  # Every Case has one equal vote.
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_x)
    scaled_tune = scaler.transform(tune_x)
    logistic = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=17)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        logistic.fit(scaled_train, train_y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("The fixed logistic baseline did not converge; no results were saved.")
    positive_index = list(logistic.classes_).index(1)
    probabilities = {
        "constant": {"train": np.full(len(train_y), prevalence),
                     "tune": np.full(len(tune_y), prevalence)},
        "logistic_regression": {
            "train": logistic.predict_proba(scaled_train)[:, positive_index],
            "tune": logistic.predict_proba(scaled_tune)[:, positive_index],
        },
    }
    elapsed = time.perf_counter() - start
    cohorts = {
        "train": (train_ids, train_y, train_counts),
        "tune": (tune_ids, tune_y, tune_counts),
    }
    metrics, records = {}, []
    for name, split_probabilities in probabilities.items():
        metrics[name] = {}
        for split, probability in split_probabilities.items():
            ids, y, counts = cohorts[split]
            metrics[name][split] = classification_metrics(y, probability, threshold=.5)
            records.extend({"baseline": name, "split": split, "seed": 17,
                            "case_id": case_id, "label": int(label),
                            "probability": float(p), "n_images": count}
                           for case_id, label, p, count in zip(ids, y, probability, counts))
    summary = {
        "status": "executed_cpu_preexperiment_baselines",
        "clinical_validation": False,
        "patient_mapping": "unverified_Case_grouped_only",
        "device": "cpu",
        "selection_seed": int(selection_seed),
        "cohort_fingerprints": {"train": cohort_fingerprint(train_rows),
                                "tune": cohort_fingerprint(tune_rows)},
        "fingerprint_definition": "sha256 of UTF-8 compact ensure_ascii JSON sorted [str(case_id),str(image_id),int(label)] rows; paths excluded",
        "cohorts": {split: {"n_cases": len(ids), "n_images": int(sum(counts)),
                             "n_positive": int(y.sum()), "n_negative": int(len(y)-y.sum())}
                    for split, (ids, y, counts) in cohorts.items()},
        "method": {
            "feature_shape_per_image": [196, 192],
            "feature_pooling": "mean spatial tokens per image, then mean all available image features per Case",
            "view_difference_from_loop_training": "all-view Case mean here; loop training samples one image per Case per epoch",
            "fit_split": "train",
            "scaler_fit_split": "train",
            "hyperparameters_selected_using_tune": False,
            "train_case_prevalence": prevalence,
            "logistic_regression": {"C": 1., "solver": "lbfgs", "max_iter": 2000,
                                    "random_state": 17, "class_weight": None,
                                    "fit_intercept": True, "n_iter": int(logistic.n_iter_.max())},
            "threshold": .5,
            "segmentation": "not performed by classification controls",
        },
        "timing_seconds": {"fit_and_predict_excluding_encoder": elapsed},
        "baselines": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["baseline", "split", "seed", "case_id", "label", "probability", "n_images"])
        writer.writeheader()
        writer.writerows(records)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
