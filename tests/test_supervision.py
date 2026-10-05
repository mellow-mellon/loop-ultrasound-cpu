"""Loss scaling, recurrent graph, and paired initialization in the new ablation."""
import pytest
import torch

from loop_ultrasound.models import make_model, trajectory_loss, parameter_counts
from loop_ultrasound.train import initialization_fingerprint, parser


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(17)


def model(depth):
    return make_model("SJ", steps=depth, width=12, heads=3, grid_size=2, image_size=16)


def targets():
    y = torch.tensor([0., 1.])
    mask = torch.zeros(2, 1, 16, 16)
    mask[:, :, 4:10, 5:11] = 1
    return y, mask, torch.ones_like(mask)


def gradients(net):
    return [p.grad.clone() for p in net.parameters()]


def test_one_step_modes_have_identical_loss_and_parameter_gradients():
    net, x = model(1), torch.randn(2, 4, 12)
    values = []
    for mode in ("all", "terminal"):
        net.zero_grad(set_to_none=True)
        loss = trajectory_loss(net(x), *targets(), supervision=mode)
        loss.backward()
        values.append((loss.detach(), gradients(net)))
    assert torch.equal(values[0][0], values[1][0])
    assert all(torch.equal(a, b) for a, b in zip(values[0][1], values[1][1]))


def test_duplicate_readouts_do_not_scale_all_depth_loss_or_gradient():
    net, x = model(1), torch.randn(2, 4, 12)
    values = []
    for repeats in (1, 2, 4):
        net.zero_grad(set_to_none=True)
        out = net(x)[0]
        loss = trajectory_loss([out]*repeats, *targets())
        loss.backward()
        values.append((loss.detach(), gradients(net)))
    assert all(torch.allclose(v[0], values[0][0], atol=1e-7, rtol=1e-6) for v in values)
    assert all(torch.allclose(a, b, atol=1e-7, rtol=1e-6)
               for v in values[1:] for a, b in zip(v[1], values[0][1]))


@pytest.mark.parametrize("depth", [2, 4])
def test_terminal_loss_has_no_intermediate_head_loss_but_full_recurrent_gradient(depth):
    net, x = model(depth), torch.randn(2, 4, 12)
    hidden = []
    def retain(module, inputs, output):
        output.retain_grad()
        hidden.append(output)
    handle = net.blocks[0].register_forward_hook(retain)
    outs = net(x)
    handle.remove()
    for out in outs:
        out["cls_logits"].retain_grad()
        out["mask_logits"].retain_grad()
    loss = trajectory_loss(outs, *targets(), supervision="terminal")
    loss.backward()
    assert len(hidden) == depth
    assert all(h.grad is not None and torch.count_nonzero(h.grad) > 0 for h in hidden)
    for out in outs[:-1]:
        assert out["cls_logits"].grad is None and out["mask_logits"].grad is None
    assert outs[-1]["cls_logits"].grad is not None
    assert outs[-1]["mask_logits"].grad is not None
    assert all(p.grad is not None for p in net.parameters())


@pytest.mark.parametrize("depth", [2, 4])
def test_skipping_intermediate_heads_preserves_terminal_loss_and_gradients(depth):
    net, x = model(depth), torch.randn(2, 4, 12)
    values = []
    for return_all in (True, False):
        net.zero_grad(set_to_none=True)
        out = net(x, return_all=return_all)
        loss = trajectory_loss(out, *targets(), supervision="terminal")
        loss.backward()
        values.append((loss.detach(), gradients(net)))
    assert torch.equal(values[0][0], values[1][0])
    assert all(torch.equal(a, b) for a, b in zip(values[0][1], values[1][1]))


@pytest.mark.parametrize("seed", [17, 29, 43])
def test_depth_does_not_change_shared_initial_parameters_or_count(seed):
    torch.manual_seed(seed)
    two = model(2)
    torch.manual_seed(seed)
    four = model(4)
    assert initialization_fingerprint(two) == initialization_fingerprint(four)
    assert parameter_counts(two)["trainable"] == parameter_counts(four)["trainable"]
    with torch.no_grad():
        next(four.parameters()).add_(.01)
    assert initialization_fingerprint(two) != initialization_fingerprint(four)


def test_supervision_parser_and_invalid_mode():
    assert parser().parse_args([]).supervision == "all"
    assert parser().parse_args(["--supervision", "terminal", "--steps", "2"]).steps == 2
    with pytest.raises(ValueError):
        trajectory_loss(model(1)(torch.randn(2, 4, 12)), *targets(), supervision="bad")
