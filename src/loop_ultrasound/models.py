"""Controlled recurrent/untied refinement models; no diagnostic validation.

Inputs to the refinement model are spatial tokens from a frozen image encoder.
Neither pathology labels nor predicted/true masks enter ``forward``. The two
classification-only representation arms still train a detached mask probe.
"""

from copy import deepcopy
import math
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MaskDecoder(nn.Module):
    """Small spatial decoder, with one identical instance across readout steps."""

    def __init__(self, width: int = 192, image_size: int = 224):
        super().__init__()
        self.image_size = image_size
        self.proj = nn.Conv2d(width, 96, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Conv2d(96, 64, 3, padding=1),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.Conv2d(32, 16, 3, padding=1),
        ])
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, tokens: Tensor, grid_size: int) -> Tensor:
        batch, count, width = tokens.shape
        if count != grid_size * grid_size:
            raise ValueError("Only spatial patch tokens may enter the decoder.")
        x = tokens.transpose(1, 2).reshape(batch, width, grid_size, grid_size)
        x = F.gelu(self.proj(x))
        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = F.gelu(block(x))
        return F.interpolate(self.out(x), size=(self.image_size, self.image_size),
                             mode="bilinear", align_corners=False)


class ClinicalDepthModel(nn.Module):
    """Parameter sharing x segmentation gradient into the representation.

    ``seg_gradient=False`` detaches *after* representation normalization. Mask
    gradients therefore reach neither the core, normalization nor classifier.
    Untied blocks start at equal values but own independent Parameter storage.
    Readouts without direct supervision are diagnostic intermediate predictions.
    """

    def __init__(self, shared: bool, seg_gradient: bool, steps: int = 4,
                 width: int = 192, heads: int = 3, grid_size: int = 14,
                 image_size: int = 224):
        super().__init__()
        if not isinstance(shared, bool) or not isinstance(seg_gradient, bool):
            raise TypeError("shared and seg_gradient must be booleans.")
        for name, value in (("steps", steps), ("width", width),
                            ("heads", heads), ("grid_size", grid_size),
                            ("image_size", image_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if width % heads:
            raise ValueError("Representation width must be divisible by heads.")
        self.shared = shared
        self.seg_gradient = seg_gradient
        self.steps = steps
        self.width = width
        self.grid_size = grid_size
        self.image_size = image_size
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.ModuleList(
            [block] if shared else [deepcopy(block) for _ in range(steps)]
        )
        self.readout_norm = nn.LayerNorm(width)
        self.cls_head = nn.Sequential(nn.Linear(width, 128), nn.GELU(),
                                      nn.Linear(128, 1))
        self.mask_head = MaskDecoder(width, image_size)

    def forward(self, spatial_tokens: Tensor, max_steps: Optional[int] = None,
                return_all: bool = True) -> List[Dict[str, Tensor]]:
        if spatial_tokens.ndim != 3 or spatial_tokens.shape[1:] != (
                self.grid_size ** 2, self.width):
            raise ValueError("Expected B x spatial-patches x representation-width.")
        if not spatial_tokens.is_floating_point():
            raise ValueError("Spatial tokens must be floating point features.")
        max_steps = self.steps if max_steps is None else max_steps
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not (
                1 <= max_steps <= self.steps):
            raise ValueError("Only configured depths 1 through steps are supported.")
        h = spatial_tokens
        outputs = []
        for r in range(max_steps):
            block = self.blocks[0] if self.shared else self.blocks[r]
            h = block(h)  # Keep the graph through every recurrent application.
            if not return_all and r != max_steps - 1:
                continue  # Endpoint timing need not execute intermediate heads.
            z = self.readout_norm(h)
            outputs.append({
                "cls_logits": self.cls_head(z.mean(dim=1)).squeeze(-1),
                "mask_logits": self.mask_head(
                    z if self.seg_gradient else z.detach(), self.grid_size),
            })
        return outputs

    def training_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Disjoint groups for separate clipping; never clip all groups together.

        The shared readout normalization belongs to the representation group.
        Joint mask supervision may update it; detached mask supervision cannot.
        """
        return {
            "core_and_classification": list(self.blocks.parameters())
            + list(self.readout_norm.parameters()) + list(self.cls_head.parameters()),
            "mask_head": list(self.mask_head.parameters()),
        }


def make_model(arm: str, steps: int = 4, **kwargs) -> ClinicalDepthModel:
    """Create SC/SJ/UC/UJ; reset the same seed before paired constructions."""
    definitions = {"SC": (True, False), "SJ": (True, True),
                   "UC": (False, False), "UJ": (False, True)}
    if arm not in definitions:
        raise ValueError("arm must be one of SC, SJ, UC, UJ.")
    shared, seg_gradient = definitions[arm]
    return ClinicalDepthModel(shared, seg_gradient, steps=steps, **kwargs)


def parameter_counts(model: ClinicalDepthModel) -> Dict[str, int]:
    """Actual counts for this refinement model, excluding its separate encoder."""
    groups = model.training_parameter_groups()
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "core": sum(p.numel() for p in model.blocks.parameters()),
        "core_and_classification": sum(p.numel() for p in groups["core_and_classification"]),
        "mask_head": sum(p.numel() for p in groups["mask_head"]),
        "independent_blocks": len(model.blocks),
        "block_applications_at_full_depth": model.steps,
    }


def trajectory_loss(outputs: List[Dict[str, Tensor]], pathology: Tensor,
                    mask: Tensor, valid_pixels: Tensor,
                    segmentation_weight: float = 1.0, *,
                    supervision: str = "all") -> Tensor:
    """Mean selected-readout BCE plus masked (0.5 pixel BCE + 0.5 soft Dice).

    Binary malignancy is 1 and benign is 0. Letterbox padding is excluded. Case
    sampling and patient grouping are responsibilities of the data pipeline.
    Labels supervise/evaluate outputs and are never inputs to the model.
    Terminal supervision selects only the final readout, with no depth divisor;
    its gradient still traverses every preceding recurrent application.
    """
    if supervision not in ("all", "terminal"):
        raise ValueError("supervision must be all or terminal.")
    if not math.isfinite(segmentation_weight) or segmentation_weight < 0:
        raise ValueError("segmentation_weight must be finite and nonnegative.")
    pathology, mask, valid = pathology.float(), mask.float(), valid_pixels.float()
    if not outputs:
        raise ValueError("The loss requires at least one readout.")
    outputs = outputs if supervision == "all" else outputs[-1:]
    if mask.ndim != 4 or mask.shape[1] != 1 or valid.shape != mask.shape:
        raise ValueError("Mask and validity must both be B x 1 x H x W.")
    if pathology.shape != (mask.shape[0],):
        raise ValueError("Pathology must be a B-vector aligned with image IDs.")
    for name, value in (("pathology", pathology), ("mask", mask), ("valid_pixels", valid)):
        if not torch.isfinite(value).all() or not ((value == 0) | (value == 1)).all():
            raise ValueError(f"{name} must contain finite binary 0/1 values.")
    valid_count = valid.flatten(1).sum(1)
    if (valid_count == 0).any().item():
        raise ValueError("Every image must contain valid ultrasound pixels.")
    total = pathology.new_zeros(())
    for out in outputs:
        if out["mask_logits"].shape != mask.shape:
            raise ValueError("Mask logits must exactly match target geometry.")
        if out["cls_logits"].shape != pathology.shape:
            raise ValueError("Classification logits must match pathology shape.")
        classification = F.binary_cross_entropy_with_logits(out["cls_logits"], pathology)
        pixel_bce = F.binary_cross_entropy_with_logits(out["mask_logits"], mask, reduction="none")
        pixel_bce = (pixel_bce * valid).flatten(1).sum(1) / valid_count
        probabilities = out["mask_logits"].sigmoid()
        intersection = (probabilities * mask * valid).flatten(1).sum(1)
        denominator = ((probabilities + mask) * valid).flatten(1).sum(1)
        dice_loss = 1 - (2 * intersection + 1e-6) / (denominator + 1e-6)
        segmentation = 0.5 * pixel_bce.mean() + 0.5 * dice_loss.mean()
        total = total + classification + segmentation_weight * segmentation
    return total / len(outputs)


class FrozenEncoderAdapter(nn.Module):
    """Pinned non-distilled DeiT-Tiny spatial features, with encoder always eval.

    Random encoder weights are allowed for structure checks only. ``pretrained``
    metadata records whether timm was asked for pretrained weights; it is not a
    clinical performance claim. Encode after synchronized pixel augmentation.
    """

    def __init__(self, encoder: nn.Module, pretrained: Optional[bool] = None,
                 model_name: Optional[str] = None):
        super().__init__()
        self.encoder = encoder
        self.pretrained = pretrained
        self.model_name = model_name
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        if getattr(encoder, "num_prefix_tokens", None) != 1:
            raise ValueError("This adapter expects a non-distilled DeiT with one prefix token.")

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, augmented_images: Tensor) -> Tensor:
        if augmented_images.ndim != 4 or augmented_images.shape[1:] != (3, 224, 224):
            raise ValueError("Expected B x 3 x 224 x 224 augmented encoder images.")
        self.encoder.eval()
        # no_grad, not inference_mode: trainable modules save features for backward.
        with torch.no_grad():
            features = self.encoder.forward_features(augmented_images)
        if features.ndim != 3 or features.shape[1:] != (197, 192):
            raise ValueError("Encoder output must be B x 197 x 192 before CLS removal.")
        return features[:, 1:, :]


def create_encoder(name: str = "deit_tiny_patch16_224.fb_in1k",
                   pretrained: bool = True,
                   weights_path: Optional[str] = None) -> FrozenEncoderAdapter:
    """Create timm encoder; ``pretrained=False`` is explicitly a random shape check.

    Pretrained downloads are timm's responsibility and failures are propagated;
    there is no silent fallback to random weights. A local ``weights_path`` must
    be a complete state dict saved from the stated encoder; its provenance/hash
    belongs in the caller's artifact manifest. Loading is strict and weights-only.
    """
    import timm
    if not isinstance(pretrained, bool):
        raise TypeError("pretrained must be a boolean.")
    if weights_path is not None and not pretrained:
        raise ValueError("Local pretrained weights and pretrained=False are contradictory.")
    encoder = timm.create_model(name, pretrained=pretrained if weights_path is None else False)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state, strict=True)
    adapter = FrozenEncoderAdapter(encoder, pretrained=pretrained, model_name=name)
    adapter.weights_source = weights_path or ("timm pretrained" if pretrained else "random initialization")
    return adapter
