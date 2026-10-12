from argparse import Namespace

import pytest
import torch

from loop_ultrasound import architecture_comparison as architecture
from loop_ultrasound.models import make_model
from loop_ultrasound.train import initialization_fingerprint


@pytest.mark.parametrize("seed", [53, 71, 89, 107, 131])
@pytest.mark.parametrize("depth", [2, 4])
def test_untied_initial_values_and_rng_match_shared_but_storage_is_independent(seed, depth):
    untied, full_hash, certificate = architecture.paired_initialization(seed, depth)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    shared = make_model("SJ", steps=depth)
    assert torch.equal(torch.get_rng_state(), rng)
    assert certificate == {
        "canonical_shared_sha256": initialization_fingerprint(shared),
        "all_initial_block_and_head_values_match": True,
        "untied_parameter_storage_distinct": True,
        "torch_rng_after_initialization_matches": True}
    assert full_hash == initialization_fingerprint(untied)
    assert full_hash != initialization_fingerprint(shared)
    assert len(untied.blocks) == depth
    for block in untied.blocks:
        for name, value in block.state_dict().items():
            assert torch.equal(value, shared.blocks[0].state_dict()[name])
    pointers = [p.data_ptr() for block in untied.blocks for p in block.parameters()]
    assert len(pointers) == len(set(pointers))
    # One independent layer can change without changing the other layers.
    original = untied.blocks[0].linear1.weight.detach().clone()
    with torch.no_grad():
        untied.blocks[1].linear1.weight.add_(1)
    assert torch.equal(untied.blocks[0].linear1.weight, original)
    assert not torch.equal(untied.blocks[1].linear1.weight, original)


def test_depths_share_common_initial_components_but_have_distinct_full_untied_hashes():
    _, two, cert2 = architecture.paired_initialization(53, 2)
    _, four, cert4 = architecture.paired_initialization(53, 4)
    assert two != four
    assert cert2 == cert4


@pytest.mark.parametrize("depth", [2, 4])
def test_paired_initial_models_compute_the_same_endpoint_before_training(depth):
    untied, _, _ = architecture.paired_initialization(53, depth)
    torch.manual_seed(53)
    shared = make_model("SJ", steps=depth)
    tokens = torch.randn(1, 196, 192)
    shared.eval(); untied.eval()
    with torch.no_grad():
        a = shared(tokens, return_all=False)[0]
        b = untied(tokens, return_all=False)[0]
    assert torch.equal(a["cls_logits"], b["cls_logits"])
    assert torch.equal(a["mask_logits"], b["mask_logits"])


def test_fixed_policy_configuration_changes_neither_arm_depth_nor_endpoint():
    constant = architecture.branch_config(53, 4, "constant")
    low = architecture.branch_config(53, 4, "low_tail")
    assert constant["arm"] == low["arm"] == "UJ"
    assert constant["steps"] == low["steps"] == 4
    assert constant["epochs"] == low["epochs"] == 25
    assert constant["prefix_updates"] == low["prefix_updates"] == 2560
    assert constant["tail_learning_rate"] == .0003
    assert low["tail_learning_rate"] == .00003
    assert {k:v for k,v in constant.items() if k not in {"tail_policy","tail_learning_rate"}} == {
        k:v for k,v in low.items() if k not in {"tail_policy","tail_learning_rate"}}


@pytest.mark.parametrize("changed", ["core", "runner", "helper"])
def test_changed_locked_execution_source_refused(monkeypatch, changed):
    monkeypatch.setattr(architecture, "training_source_fingerprint", lambda: "core")
    monkeypatch.setattr(architecture, "launcher_digest", lambda: "runner")
    monkeypatch.setattr(architecture, "helper_digest", lambda: "helper")
    protocol = {"training_source_sha256":"core", "launcher_source_sha256":"runner",
                "helper_source_sha256":"helper"}
    architecture.verify_locked_sources(protocol)
    key = {"core":"training_source_sha256", "runner":"launcher_source_sha256",
           "helper":"helper_source_sha256"}[changed]
    protocol[key] = "changed"
    with pytest.raises(ValueError, match="changed after lock"):
        architecture.verify_locked_sources(protocol)


def test_claimed_prefix_refused_before_encoder_or_training(tmp_path):
    prefix = tmp_path/"prefix-depth4-seed53"
    prefix.mkdir()
    (prefix/"pair_claim.json").write_text("already claimed")
    with pytest.raises(FileExistsError):
        architecture.run_pair(Namespace(output_dir=tmp_path,worker_index=0), {}, [], [], 53, 4)


def test_worker_assignment_covers_ten_new_prefixes_and_twenty_tails():
    a,b = set(architecture.assigned_pairs(2,0)),set(architecture.assigned_pairs(2,1))
    assert not a&b
    assert a|b == {(s,d) for s in architecture.SEEDS for d in (2,4)}
    assert len(a|b)*(architecture.PREFIX_UPDATES+2*architecture.TAIL_UPDATES) == 38400
