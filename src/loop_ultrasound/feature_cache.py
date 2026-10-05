"""Local CPU feature cache bound to exact images, order, encoder and transform."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch
from torch.utils.data import DataLoader

from .data import UltrasoundDataset, IMAGENET_MEAN, IMAGENET_STD
from .download import file_digest


def validate_features(features, n_images):
    if (not isinstance(features, torch.Tensor) or features.shape != (n_images, 196, 192)
            or features.dtype != torch.float32 or features.device.type != "cpu"
            or features.requires_grad or not torch.isfinite(features).all()):
        raise ValueError("Cached features have invalid shape, dtype, device or values.")


def feature_signature(rows, weights):
    record = {"format": 1, "encoder_sha256": file_digest(weights),
              "transform": {"version": "fullframe-letterbox-v1", "size": 224,
                            "interpolation": "PIL_bilinear", "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
              "images": [[str(r["image_id"]), file_digest(r["image_path"])] for r in rows]}
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_or_encode(encoder, rows, weights, cache_dir, batch_size):
    if not rows or any(r["split"] not in ("train", "tune") for r in rows):
        raise ValueError("Feature caching is restricted to nonempty train/tune rows.")
    begin = time.perf_counter()
    signature = feature_signature(rows, weights)
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"features-{signature}.pt"
    reused = target.exists()
    if reused:
        record = torch.load(target, map_location="cpu", weights_only=True)
        if record.get("signature") != signature or record.get("image_ids") != [r["image_id"] for r in rows]:
            raise ValueError("Feature cache identity/order mismatch.")
        features = record["features"]
    else:
        loader = DataLoader(UltrasoundDataset(rows, augment=False), batch_size=batch_size, num_workers=0,
                            generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            features = torch.cat([encoder(batch["image"]) for batch in loader]).detach().contiguous()
        validate_features(features, len(rows))
        handle, partial = tempfile.mkstemp(prefix="features-", suffix=".pt.part", dir=directory)
        os.close(handle)
        try:
            torch.save({"signature": signature, "image_ids": [r["image_id"] for r in rows],
                        "features": features}, partial)
            os.replace(partial, target)
        finally:
            Path(partial).unlink(missing_ok=True)
    validate_features(features, len(rows))
    return features, time.perf_counter()-begin, reused
