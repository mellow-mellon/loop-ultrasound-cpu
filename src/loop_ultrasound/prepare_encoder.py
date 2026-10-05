"""Acquire real timm ImageNet weights; verify cached state, never use a fallback."""
import argparse
import json
import os
from pathlib import Path

from .download import file_digest

MODEL = "deit_tiny_patch16_224.fb_in1k"
SOURCE = "https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str((args.data_dir / "hf-cache").resolve()))
    os.environ.setdefault("TORCH_HOME", str((args.data_dir / "torch-cache").resolve()))
    import timm
    import torch
    torch.set_num_threads(2)
    target = args.data_dir / "deit_tiny_encoder.pt"
    provenance_path = args.data_dir / "encoder_provenance.json"
    if target.exists():
        if not provenance_path.exists():
            raise ValueError("Existing weights have no provenance; use a fresh data directory.")
        provenance = json.loads(provenance_path.read_text())
        recorded = provenance.get("sha256", provenance.get("local_weights_sha256"))
        name = provenance.get("model_name", provenance.get("model", provenance.get("name")))
        if recorded != file_digest(target) or name != MODEL or provenance.get("pretrained") is not True:
            raise ValueError("Existing encoder weights failed provenance/name/hash verification.")
        from .models import create_encoder
        create_encoder(pretrained=True, weights_path=str(target))
    else:
        if provenance_path.exists():
            raise ValueError("Encoder provenance exists without weights; use a fresh data directory.")
        encoder = timm.create_model(MODEL, pretrained=True).cpu().eval()
        partial = target.with_suffix(".pt.part")
        try:
            torch.save(encoder.state_dict(), partial)
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
        provenance = {"model_name": MODEL, "source_url": SOURCE, "pretrained": True,
                      "sha256": file_digest(target), "torch": torch.__version__, "timm": timm.__version__,
                      "parameters": sum(p.numel() for p in encoder.parameters()),
                      "note": "Local serialized state hash; upstream default weights may change. No random fallback."}
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"model_name": MODEL, "sha256": file_digest(target), "pretrained": True}))


if __name__ == "__main__":
    main()
