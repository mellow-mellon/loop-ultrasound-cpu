"""Case-weighting and padding-aware segmentation regressions."""
import torch

from loop_ultrasound.train import CaseViewSampler, case_subset, mask_dice


def test_case_sampler_equal_weights_and_reproducible_views():
    rows = [{"case_id": "a"}, {"case_id": "a"}, {"case_id": "b"},
            {"case_id": "c"}, {"case_id": "c"}, {"case_id": "c"}]
    left, right = CaseViewSampler(rows, seed=17), CaseViewSampler(rows, seed=17)
    seen_a = set()
    for _ in range(20):
        indices = list(left)
        assert indices == list(right)
        assert len(indices) == 3
        assert {rows[i]["case_id"] for i in indices} == {"a", "b", "c"}
        seen_a.update(i for i in indices if rows[i]["case_id"] == "a")
    assert seen_a == {0, 1}


def test_development_subset_keeps_all_views_and_ignores_labels():
    rows = [{"case_id": str(c), "label": c % 2, "view": v} for c in range(10) for v in range(2)]
    selected = case_subset(rows, 3, seed=17)
    assert len(selected) == 6
    relabeled = [dict(r, label=1-r["label"]) for r in rows]
    assert {r["case_id"] for r in selected} == {r["case_id"] for r in case_subset(relabeled, 3, 17)}


def test_mask_dice_ignores_predictions_in_padding():
    target = torch.tensor([[[[1., 0.], [0., 0.]]]])
    valid = torch.tensor([[[[1., 1.], [0., 0.]]]])
    logits = torch.tensor([[[[20., -20.], [20., 20.]]]])
    assert mask_dice(logits, target, valid).item() == 1.
