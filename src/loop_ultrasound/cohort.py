"""Fixed train-only Case cohorts, independent of optimization randomness."""
from collections import defaultdict
import hashlib
import json
import random
from pathlib import Path

from .data import validate_case_groups


def cohort_fingerprint(rows):
    values = sorted([[str(r["case_id"]), str(r["image_id"]), int(r["label"])] for r in rows])
    return hashlib.sha256(json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def training_source_fingerprint():
    directory = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("train.py", "models.py", "data.py", "cohort.py", "feature_cache.py"):
        digest.update(name.encode()+b"\0"+(directory/name).read_bytes())
    return digest.hexdigest()


def select_training_rows(rows, max_cases, *, sampling="proportional", selection_seed=20261004):
    validate_case_groups(rows)
    if sampling not in ("balanced", "proportional") or max_cases < 2:
        raise ValueError("Choose balanced/proportional sampling and >=2 Cases.")
    groups = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            groups[str(row["case_id"])].append(row)
    by_label = {label: sorted(k for k, group in groups.items() if int(group[0]["label"]) == label)
                for label in (0, 1)}
    if not all(by_label.values()) or max_cases > len(groups):
        raise ValueError("Requested training cohort is unavailable.")
    if sampling == "balanced":
        if max_cases % 2:
            raise ValueError("Balanced sampling needs an even Case count.")
        n_positive = max_cases // 2
    else:
        n_positive = min(max_cases-1, max(1, round(max_cases * len(by_label[1]) / len(groups))))
    counts = {0: max_cases-n_positive, 1: n_positive}
    if any(counts[label] > len(by_label[label]) for label in (0, 1)):
        raise ValueError("Insufficient Cases for the specified stratification.")
    rng, chosen = random.Random(selection_seed), set()
    for label in (0, 1):
        rng.shuffle(by_label[label])
        chosen.update(by_label[label][:counts[label]])
    return [r for r in rows if r["split"] == "train" and str(r["case_id"]) in chosen]
