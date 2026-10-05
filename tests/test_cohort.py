import pytest
from loop_ultrasound.cohort import cohort_fingerprint, select_training_rows


def rows():
    return [{"case_id": str(c), "image_id": f"image{c}_{view}", "label": int(c >= 397),
             "image_path": "unused", "mask_path": "unused", "split": "train", "device": "test"}
            for c in range(585) for view in range(1 + (c % 2))]


def test_proportional_selection_counts_cases_not_views_and_keeps_all_views():
    source = rows()
    selected = select_training_rows(source, 128, selection_seed=20261004)
    cases = {r["case_id"]: r["label"] for r in selected}
    assert len(cases) == 128 and sum(cases.values()) == 41
    assert {r["image_id"] for r in selected} == {r["image_id"] for r in source if r["case_id"] in cases}
    assert selected == select_training_rows(source, 128, selection_seed=20261004)
    assert cohort_fingerprint(selected) == cohort_fingerprint(list(reversed(selected)))
    assert cohort_fingerprint(selected) != cohort_fingerprint(select_training_rows(source, 128, selection_seed=17))


def test_selection_never_includes_nontraining_cases():
    source = rows()
    source.append(dict(source[0], case_id="heldout", image_id="heldoutimage", split="test"))
    assert all(r["split"] == "train" for r in select_training_rows(source, 128))
    with pytest.raises(ValueError):
        select_training_rows(source, 700)
    with pytest.raises(ValueError):
        select_training_rows(source, 127, sampling="balanced")
