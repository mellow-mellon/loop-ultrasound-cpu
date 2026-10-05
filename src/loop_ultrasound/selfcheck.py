"""Execute CPU implementation checks on synthetic tensors, never diagnosis tests.

Run ``python -m loop_ultrasound.selfcheck``. The default encoder has random
weights for shape/freeze checks; ``--pretrained`` explicitly requests real timm
pretrained weights and may download them. JSON reports what actually ran.
"""

import argparse
import json
import time
from typing import Callable, Dict, List

import torch
from torch.nn import functional as F

from .models import create_encoder, make_model, parameter_counts, trajectory_loss


def _nonzero_grad(parameters) -> bool:
    return any(p.grad is not None and torch.count_nonzero(p.grad).item() > 0
               for p in parameters)


def _no_grad(parameters) -> bool:
    return all(p.grad is None or torch.count_nonzero(p.grad).item() == 0
               for p in parameters)


def run_selfcheck(pretrained: bool = False, threads: int = 2,
                  weights_path: str = None) -> Dict:
    """Run default-size models on CPU and return an aggregate, including failures."""
    if not 1 <= threads <= 2:
        raise ValueError("CPU selfcheck uses one or two threads only.")
    started = time.perf_counter()
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261004)
    checks: List[Dict] = []
    counts = {}
    execution = {"model_forward_calls": 0, "shape_and_identity_forward_cases": 0, "endpoint_cases": 0,
                 "backward_passes": 0, "encoder_forward_cases": 0}

    def check(name: str, operation: Callable):
        begin = time.perf_counter()
        try:
            details = operation()
            checks.append({"name": name, "passed": True,
                           "runtime_seconds": round(time.perf_counter() - begin, 6),
                           "details": details})
        except Exception as error:
            checks.append({"name": name, "passed": False,
                           "runtime_seconds": round(time.perf_counter() - begin, 6),
                           "error": f"{type(error).__name__}: {error}"})

    models = {}
    hooks = []

    def count_forward(module, inputs, output):
        execution["model_forward_calls"] += 1

    for arm in ("SC", "SJ", "UC", "UJ"):
        torch.manual_seed(17)  # Equal initial values across paired arms.
        models[arm] = make_model(arm).cpu()
        hooks.append(models[arm].register_forward_hook(count_forward))
        counts[arm] = parameter_counts(models[arm])

    for arm, model in models.items():
        def sharing_check(model=model):
            if model.shared:
                assert len(model.blocks) == 1
                called = []
                hook = model.blocks[0].register_forward_pre_hook(
                    lambda module, inputs: called.append(id(module)))
                try:
                    with torch.no_grad():
                        model(torch.randn(1, 196, 192))
                finally:
                    hook.remove()
                assert len(called) == 4 and len(set(called)) == 1
                execution["shape_and_identity_forward_cases"] += 1
            else:
                assert len(model.blocks) == 4
                storages = [{p.data_ptr() for p in block.parameters()}
                            for block in model.blocks]
                for i in range(4):
                    for j in range(i + 1, 4):
                        assert not storages[i] & storages[j]
                reference = dict(model.blocks[0].named_parameters())
                for block in model.blocks[1:]:
                    for name, parameter in block.named_parameters():
                        assert torch.equal(reference[name], parameter)
            groups = model.training_parameter_groups()
            ids = [{id(p) for p in parameters} for parameters in groups.values()]
            assert not ids[0] & ids[1]
            assert ids[0] | ids[1] == {id(p) for p in model.parameters()}
            return {"independent_blocks": len(model.blocks), "disjoint_clip_groups": True}
        check(f"{arm}/parameter_identity_and_clip_groups", sharing_check)

        for batch in (1, 2):
            tokens = torch.randn(batch, 196, 192)
            for depth in (1, 2, 3, 4):
                def shape_check(model=model, tokens=tokens, batch=batch, depth=depth):
                    with torch.no_grad():
                        outputs = model(tokens, max_steps=depth, return_all=True)
                        execution["shape_and_identity_forward_cases"] += 1
                        endpoint = model(tokens, max_steps=depth, return_all=False)
                        execution["endpoint_cases"] += 1
                    assert len(outputs) == depth and len(endpoint) == 1
                    for output in outputs:
                        assert output["cls_logits"].shape == (batch,)
                        assert output["mask_logits"].shape == (batch, 1, 224, 224)
                        assert all(torch.isfinite(value).all() for value in output.values())
                    assert torch.allclose(outputs[-1]["cls_logits"], endpoint[0]["cls_logits"])
                    assert torch.allclose(outputs[-1]["mask_logits"], endpoint[0]["mask_logits"])
                    return {"batch": batch, "steps": depth,
                            "mask_shape": [batch, 1, 224, 224]}
                check(f"{arm}/batch{batch}/depth{depth}/forward_and_endpoint", shape_check)

        def mask_gradient_check(model=model):
            model.zero_grad(set_to_none=True)
            outputs = model(torch.randn(1, 196, 192))
            mask_only = torch.stack([F.binary_cross_entropy_with_logits(
                output["mask_logits"], torch.zeros_like(output["mask_logits"]))
                for output in outputs]).mean()
            mask_only.backward()
            execution["backward_passes"] += 1
            assert _nonzero_grad(model.mask_head.parameters())
            assert _no_grad(model.cls_head.parameters())
            core_has_gradient = _nonzero_grad(model.blocks.parameters())
            norm_has_gradient = _nonzero_grad(model.readout_norm.parameters())
            if model.seg_gradient:
                assert core_has_gradient and norm_has_gradient
            else:
                assert _no_grad(model.blocks.parameters()) and _no_grad(model.readout_norm.parameters())
            model.zero_grad(set_to_none=True)
            return {"mask_head_has_gradient": True,
                    "core_has_gradient": core_has_gradient,
                    "norm_has_gradient": norm_has_gradient}
        check(f"{arm}/mask_only_gradient_routing", mask_gradient_check)

        if not model.seg_gradient:
            def lambda_check(model=model):
                tokens = torch.randn(1, 196, 192)
                pathology = torch.tensor([1.0])
                mask = torch.zeros(1, 1, 224, 224)
                mask[:, :, 60:160, 70:150] = 1
                valid = torch.ones_like(mask)
                gradients = []
                for weight in (0.0, 0.1, 3.0):
                    model.zero_grad(set_to_none=True)
                    loss = trajectory_loss(model(tokens), pathology, mask, valid, weight)
                    loss.backward()
                    execution["backward_passes"] += 1
                    gradients.append({name: p.grad.detach().clone() for name, p in model.named_parameters()
                                      if not name.startswith("mask_head.")})
                for gradient in gradients[1:]:
                    for name in gradients[0]:
                        assert torch.equal(gradients[0][name], gradient[name]), name
                model.zero_grad(set_to_none=True)
                return {"weights": [0.0, 0.1, 3.0], "core_and_class_gradients_identical": True}
            check(f"{arm}/lambda_does_not_change_classification_update_gradient", lambda_check)

    def encoder_check():
        encoder = create_encoder(pretrained=pretrained, weights_path=weights_path).cpu()
        encoder.train()
        assert encoder.training and not encoder.encoder.training
        assert all(not p.requires_grad for p in encoder.encoder.parameters())
        shape_results = []
        for batch in (1, 2):
            images = torch.randn(batch, 3, 224, 224, requires_grad=True)
            features = encoder(images)
            execution["encoder_forward_cases"] += 1
            assert features.shape == (batch, 196, 192)
            assert not features.requires_grad
            output = models["SC"](features, max_steps=1)
            output[0]["cls_logits"].sum().backward()
            execution["backward_passes"] += 1
            assert images.grad is None
            assert _no_grad(encoder.encoder.parameters())
            assert _nonzero_grad(models["SC"].blocks.parameters())
            models["SC"].zero_grad(set_to_none=True)
            shape_results.append([batch, 196, 192])
        return {"encoder_name": encoder.model_name,
                "pretrained_requested_and_loaded": encoder.pretrained,
                "weights_source": encoder.weights_source,
                "weights_role": "pretrained structure check" if pretrained else "random weights; shape/freeze check only",
                "spatial_shapes": shape_results,
                "total_parameters": sum(p.numel() for p in encoder.encoder.parameters()),
                "trainable_parameters": sum(p.numel() for p in encoder.encoder.parameters() if p.requires_grad)}
    check("encoder/shape_eval_freeze_and_downstream_backward", encoder_check)

    for hook in hooks:
        hook.remove()

    failed = [item["name"] for item in checks if not item["passed"]]
    return {"status": "passed" if not failed else "failed", "device": "cpu",
            "purpose": "synthetic implementation checks, not training or diagnostic validation",
            "torch_version": torch.__version__, "threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "encoder_pretrained_requested": pretrained,
            "checks_executed": len(checks), "checks_passed": len(checks) - len(failed),
            "checks_failed": failed, "execution_counts": execution,
            "parameter_counts": counts, "checks": checks,
            "runtime_seconds": round(time.perf_counter() - started, 6)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", action="store_true",
                        help="Request actual pretrained encoder weights; may download.")
    parser.add_argument("--threads", type=int, default=2, choices=(1, 2))
    parser.add_argument("--weights-path", help="Complete local pretrained state dict; requires --pretrained.")
    args = parser.parse_args()
    if args.weights_path and not args.pretrained:
        parser.error("--weights-path requires --pretrained")
    report = run_selfcheck(pretrained=args.pretrained, threads=args.threads,
                           weights_path=args.weights_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
