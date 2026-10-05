"""NumPy-only, case-level research metrics for the CPU engineering pilot.

These functions evaluate already aligned predictions. They do not establish
patient independence, clinical validity, or image/mask eligibility. Case IDs must
be merged upstream if several Cases belong to one verified patient.
"""

from collections.abc import Mapping
import math
from numbers import Integral
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


NLL_CLIP_EPSILON = 1e-15


def _identifier(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise ValueError(f"{name} must be a nonempty string or integer ID.")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} cannot be empty.")
    return result


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _unit_scalar(value: Any, name: str) -> float:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar in [0, 1].") from error
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar.")
    result = float(array)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite scalar in [0, 1].")
    return result


def _vectors(labels: Sequence[float], probabilities: Sequence[float]):
    try:
        y = np.asarray(labels, dtype=np.float64)
        p = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Labels and probabilities must be numeric vectors.") from error
    if y.ndim != 1 or p.ndim != 1 or y.size == 0 or y.shape != p.shape:
        raise ValueError("Use nonempty, equally sized one-dimensional vectors.")
    if not np.isfinite(y).all() or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("Labels must be finite binary values, with malignant=1.")
    if not np.isfinite(p).all() or ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    return y, p


def aggregate_cases(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Average image probabilities within (arm, seed, step, case_id).

    Required fields: case_id, image_id, label, probability, step, arm, seed.
    ``dice`` is optional or None; its case value is the mean of available image
    Dice values, not a Dice score of a case-averaged segmentation. ``n_dice``
    makes that denominator explicit. Image IDs must be unique within each group.
    Labels are checked across *all* arms, seeds, and depths for each Case.
    """
    required = {"case_id", "image_id", "label", "probability", "step", "arm", "seed"}
    grouped: Dict[Any, Dict[str, Any]] = {}
    case_labels: Dict[str, int] = {}
    image_cases: Dict[str, str] = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("Each prediction record must be a mapping.")
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Prediction record is missing: {', '.join(sorted(missing))}.")
        case_id = _identifier(row["case_id"], "case_id")
        image_id = _identifier(row["image_id"], "image_id")
        if not isinstance(row["arm"], str) or not row["arm"].strip():
            raise ValueError("arm must be a nonempty string.")
        arm = row["arm"].strip()
        seed = _integer(row["seed"], "seed")
        step = _integer(row["step"], "step", minimum=1)
        label_value = _unit_scalar(row["label"], "label")
        if label_value not in (0.0, 1.0):
            raise ValueError("Case pathology labels must be binary.")
        label = int(label_value)
        probability = _unit_scalar(row["probability"], "probability")
        dice = row.get("dice")
        if dice is not None:
            dice = _unit_scalar(dice, "dice")
        if case_id in case_labels and case_labels[case_id] != label:
            raise ValueError(f"Inconsistent pathology labels for Case {case_id}.")
        case_labels[case_id] = label
        if image_id in image_cases and image_cases[image_id] != case_id:
            raise ValueError(f"Image {image_id} is assigned to different Cases.")
        image_cases[image_id] = case_id
        key = (arm, seed, step, case_id)
        group = grouped.setdefault(key, {"label": label, "images": {}})
        if image_id in group["images"]:
            raise ValueError(f"Duplicate image {image_id} within prediction group {key}.")
        group["images"][image_id] = (probability, dice)

    result = []
    for (arm, seed, step, case_id), group in sorted(grouped.items()):
        image_ids = sorted(group["images"])
        values = [group["images"][image_id] for image_id in image_ids]
        dices = [value[1] for value in values if value[1] is not None]
        result.append({
            "arm": arm,
            "seed": seed,
            "step": step,
            "case_id": case_id,
            "label": group["label"],
            "probability": float(np.mean([value[0] for value in values])),
            "n_images": len(image_ids),
            "image_ids": image_ids,
            "dice": float(np.mean(dices)) if dices else None,
            "n_dice": len(dices),
        })
    return result


def _auroc(y: np.ndarray, p: np.ndarray):
    """Mann--Whitney rank statistic, with average ranks for tied scores."""
    positives = int(y.sum())
    negatives = y.size - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    sorted_p, sorted_y = p[order], y[order]
    positive_rank_sum = 0.0
    start = 0
    while start < y.size:
        stop = start + 1
        while stop < y.size and sorted_p[stop] == sorted_p[start]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        positive_rank_sum += average_rank * float(sorted_y[start:stop].sum())
        start = stop
    value = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(np.clip(value, 0.0, 1.0))


def classification_metrics(labels: Sequence[float], probabilities: Sequence[float],
                           threshold: float = 0.5) -> Dict[str, Any]:
    """Unweighted case metrics; predict malignant when probability >= threshold.

    Undefined single-class AUROC/sensitivity/specificity return None, not NaN.
    NLL uses an explicitly reported 1e-15 clip for finite JSON at exact 0/1.
    The caller must supply one probability per eligible, independent Case.
    """
    y, p = _vectors(labels, probabilities)
    threshold = _unit_scalar(threshold, "threshold")
    prediction = p >= threshold
    positive = y == 1.0
    tp = int(np.count_nonzero(prediction & positive))
    fn = int(np.count_nonzero(~prediction & positive))
    fp = int(np.count_nonzero(prediction & ~positive))
    tn = int(np.count_nonzero(~prediction & ~positive))
    clipped = np.clip(p, NLL_CLIP_EPSILON, 1.0 - NLL_CLIP_EPSILON)
    nll = -np.mean(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))
    return {
        "n_cases": int(y.size),
        "n_positive": tp + fn,
        "n_negative": tn + fp,
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(nll),
        "nll_clip_epsilon": NLL_CLIP_EPSILON,
        "auroc": _auroc(y, p),
        "threshold": threshold,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def paired_brier_change(labels: Sequence[float], p_early: Sequence[float],
                        p_late: Sequence[float], draws: int = 2000,
                        seed: int = 17) -> Dict[str, Any]:
    """Paired Case bootstrap of Brier_early - Brier_late; positive is improvement.

    Each vector must have the SAME Case order, aligned upstream by Case ID.
    Resample Cases, not images or individual early/late scores. The percentile
    interval is conditional on this observed sample, not on training randomness.
    """
    y, early = _vectors(labels, p_early)
    _, late = _vectors(labels, p_late)
    draws = _integer(draws, "draws", minimum=2)
    seed = _integer(seed, "seed")
    changes = (early - y) ** 2 - (late - y) ** 2
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 256):
        stop = min(draws, start + 256)
        indices = rng.integers(0, y.size, size=(stop - start, y.size))
        bootstrap_means[start:stop] = changes[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, (0.025, 0.975))
    return {
        "mean": float(changes.mean()),
        "ci95": [float(low), float(high)],
        "n_cases": int(y.size),
        "draws": draws,
        "seed": seed,
        "definition": "Brier_early - Brier_late",
        "positive_means": "lower late-step Brier",
    }


def select_sensitivity_threshold(labels: Sequence[float], probabilities: Sequence[float],
                                 target_sensitivity: float = 0.95, *,
                                 source_split: str) -> Dict[str, Any]:
    """Select the largest observed threshold meeting calibration sensitivity.

    Explicit calibration provenance is mandatory; test/train/tune sources are
    rejected. This checks the declared source, not actual dataset lineage, which
    the caller must audit. Apply the returned threshold unchanged to held-out
    predictions. 0.95 is a research setting, not a clinical guarantee/standard.
    """
    if source_split != "calibration":
        raise ValueError("Threshold selection is allowed only on the calibration split.")
    y, p = _vectors(labels, probabilities)
    target = _unit_scalar(target_sensitivity, "target_sensitivity")
    if target == 0.0:
        raise ValueError("target_sensitivity must be greater than zero.")
    positive_scores = np.sort(p[y == 1.0])
    if positive_scores.size == 0:
        raise ValueError("Calibration requires at least one positive Case.")
    # nextafter avoids ceil turning an integer target count into count+1 because
    # of a single floating-point rounding unit (e.g., a decimal target fraction).
    required = max(1, int(math.ceil(float(np.nextafter(
        target * positive_scores.size, -np.inf)))))
    threshold = float(positive_scores[positive_scores.size - required])
    metrics = classification_metrics(y, p, threshold)
    return {
        "source_split": source_split,
        "threshold": threshold,
        "target_sensitivity": target,
        "achieved_sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "n_cases": metrics["n_cases"],
        "n_positive": metrics["n_positive"],
        "n_negative": metrics["n_negative"],
        "rule": "probability >= threshold",
    }
