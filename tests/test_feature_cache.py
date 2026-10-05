from pathlib import Path
import pytest
import torch
from PIL import Image

from loop_ultrasound.feature_cache import load_or_encode


class FakeEncoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        return image.mean(dim=(1, 2, 3))[:, None, None].expand(-1, 196, 192).contiguous()


def fixture(tmp_path):
    image, mask, weights = tmp_path / "image.png", tmp_path / "mask.png", tmp_path / "weights.pt"
    Image.new("RGB", (12, 8), "gray").save(image)
    Image.new("L", (12, 8), 1).save(mask)
    weights.write_bytes(b"synthetic weights")
    row = {"case_id": "synthetic", "image_id": "synthetic", "label": 0,
           "image_path": str(image), "mask_path": str(mask), "split": "train", "device": "test"}
    return [row], weights


def test_cache_reuses_only_matching_content_and_weights(tmp_path):
    rows, weights = fixture(tmp_path)
    encoder, cache = FakeEncoder(), tmp_path / "cache"
    first, _, reused = load_or_encode(encoder, rows, weights, cache, 1)
    assert not reused and encoder.calls == 1
    second, _, reused = load_or_encode(encoder, rows, weights, cache, 1)
    assert reused and encoder.calls == 1 and torch.equal(first, second)
    Image.new("RGB", (12, 8), "white").save(rows[0]["image_path"])
    third, _, reused = load_or_encode(encoder, rows, weights, cache, 1)
    assert not reused and encoder.calls == 2 and not torch.equal(first, third)
    weights.write_bytes(b"changed encoder")
    load_or_encode(encoder, rows, weights, cache, 1)
    assert encoder.calls == 3


def test_cache_rejects_invalid_loaded_features_and_heldout(tmp_path):
    rows, weights = fixture(tmp_path)
    cache = tmp_path / "cache"
    load_or_encode(FakeEncoder(), rows, weights, cache, 1)
    target = next(cache.glob("*.pt"))
    record = torch.load(target, weights_only=True)
    record["features"] = torch.full((1, 196, 192), float("nan"))
    torch.save(record, target)
    with pytest.raises(ValueError, match="invalid shape"):
        load_or_encode(FakeEncoder(), rows, weights, cache, 1)
    with pytest.raises(ValueError, match="train/tune"):
        load_or_encode(FakeEncoder(), [dict(rows[0], split="cal")], weights, cache, 1)
