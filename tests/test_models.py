"""Meaningful CPU regression checks for study isolation and loss geometry."""

from copy import deepcopy

import pytest
import torch

from loop_ultrasound.models import (ClinicalDepthModel, FrozenEncoderAdapter,
                                    make_model, parameter_counts, trajectory_loss)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(17)


def small_model(arm):
    return make_model(arm, width=12, heads=3, grid_size=2, image_size=16)


@pytest.mark.parametrize("arm", ["SC", "SJ", "UC", "UJ"])
def test_supported_depths_and_endpoint(arm):
    model = small_model(arm)
    tokens = torch.randn(2, 4, 12)
    for depth in (1, 2, 3, 4):
        outputs = model(tokens, max_steps=depth)
        endpoint = model(tokens, max_steps=depth, return_all=False)
        assert len(outputs) == depth
        assert outputs[-1]["cls_logits"].shape == (2,)
        assert outputs[-1]["mask_logits"].shape == (2, 1, 16, 16)
        assert torch.allclose(outputs[-1]["mask_logits"], endpoint[0]["mask_logits"])
    with pytest.raises(ValueError):
        model(tokens, max_steps=5)
    with pytest.raises(ValueError):
        model(torch.randn(2, 4, 13))


@pytest.mark.parametrize("arm", ["SC", "SJ", "UC", "UJ"])
def test_mask_gradient_isolation_includes_normalization(arm):
    model = small_model(arm)
    outputs = model(torch.randn(1, 4, 12))
    sum(output["mask_logits"].square().mean() for output in outputs).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.mask_head.parameters())
    assert all(p.grad is None for p in model.cls_head.parameters())
    if arm.endswith("C"):
        assert all(p.grad is None for p in model.blocks.parameters())
        assert all(p.grad is None for p in model.readout_norm.parameters())
    else:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.blocks.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.readout_norm.parameters())


@pytest.mark.parametrize("arm", ["SC", "UC"])
def test_detached_mask_weight_cannot_change_core_gradient(arm):
    model = small_model(arm)
    tokens = torch.randn(2, 4, 12)
    pathology = torch.tensor([0, 1])
    mask = torch.zeros(2, 1, 16, 16)
    mask[:, :, 4:10, 5:11] = 1
    valid = torch.ones_like(mask)
    gradients = []
    for weight in (0, 3):
        model.zero_grad(set_to_none=True)
        trajectory_loss(model(tokens), pathology, mask, valid, weight).backward()
        gradients.append([p.grad.clone() for p in model.training_parameter_groups()["core_and_classification"]])
    assert all(torch.equal(a, b) for a, b in zip(*gradients))


def test_tied_untied_storage_counts_and_paired_initialization():
    torch.manual_seed(17)
    shared = small_model("SC")
    torch.manual_seed(17)
    untied = small_model("UC")
    assert len(shared.blocks) == 1 and len(untied.blocks) == 4
    reference = dict(shared.blocks[0].named_parameters())
    for block in untied.blocks:
        for name, parameter in block.named_parameters():
            assert torch.equal(reference[name], parameter)
    storages = [{p.data_ptr() for p in block.parameters()} for block in untied.blocks]
    assert all(not a & b for i, a in enumerate(storages) for b in storages[i + 1:])
    assert parameter_counts(untied)["core"] == 4 * parameter_counts(shared)["core"]
    groups = shared.training_parameter_groups()
    assert not {id(p) for p in groups["core_and_classification"]} & {id(p) for p in groups["mask_head"]}


def test_padding_cannot_change_loss_or_receive_mask_gradient():
    logits = torch.randn(1, 1, 4, 4, requires_grad=True)
    outputs = [{"cls_logits": torch.tensor([0.5], requires_grad=True), "mask_logits": logits}]
    pathology = torch.tensor([1])
    mask = torch.zeros_like(logits)
    valid = torch.ones_like(mask)
    valid[:, :, 0, :] = 0
    loss = trajectory_loss(outputs, pathology, mask, valid)
    changed = deepcopy(outputs)
    changed[0]["mask_logits"].data[:, :, 0, :] = 100
    changed_mask = mask.clone()
    changed_mask[:, :, 0, :] = 1
    assert torch.equal(loss, trajectory_loss(changed, pathology, changed_mask, valid))
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :, 0, :]) == 0


@pytest.mark.parametrize("bad_kind", ["pathology", "mask", "valid", "empty_valid", "broadcast"])
def test_loss_rejects_invalid_targets(bad_kind):
    outputs = [{"cls_logits": torch.zeros(1), "mask_logits": torch.zeros(1, 1, 4, 4)}]
    pathology = torch.tensor([1.0])
    mask = torch.zeros(1, 1, 4, 4)
    valid = torch.ones_like(mask)
    if bad_kind == "pathology":
        pathology[0] = 0.5
    elif bad_kind == "mask":
        mask[0, 0, 0, 0] = 255
    elif bad_kind == "valid":
        valid[0, 0, 0, 0] = float("nan")
    elif bad_kind == "empty_valid":
        valid.zero_()
    else:
        valid = valid.squeeze(1)
    with pytest.raises(ValueError):
        trajectory_loss(outputs, pathology, mask, valid)


def test_frozen_encoder_stays_eval_and_permits_downstream_backward():
    class FakeEncoder(torch.nn.Module):
        num_prefix_tokens = 1

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward_features(self, images):
            return images.mean(dim=(1, 2, 3))[:, None, None].expand(-1, 197, 192) * self.scale

    encoder = FrozenEncoderAdapter(FakeEncoder())
    parent = torch.nn.Sequential(encoder)
    parent.train()
    assert not encoder.encoder.training and not encoder.encoder.scale.requires_grad
    images = torch.randn(2, 3, 224, 224, requires_grad=True)
    features = encoder(images)
    assert features.shape == (2, 196, 192) and not features.requires_grad
    head = torch.nn.Linear(192, 1)
    head(features.mean(1)).sum().backward()
    assert head.weight.grad is not None
    assert encoder.encoder.scale.grad is None and images.grad is None


def test_factory_and_configuration_rejections():
    with pytest.raises(ValueError):
        make_model("bad")
    with pytest.raises(ValueError):
        ClinicalDepthModel(True, False, width=13, heads=3)
    with pytest.raises(ValueError):
        ClinicalDepthModel(True, False, steps=0)
