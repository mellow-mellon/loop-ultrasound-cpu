"""Download, checksum, make Case splits, and audit all released BUS-BRA pixels."""
import argparse
import json
from pathlib import Path

from .data import audit_images, build_manifest
from .download import download_bus_bra


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--seed", type=int, default=20261004)
    p.add_argument("--outer-fold", type=int, choices=range(1, 6), default=1)
    args = p.parse_args()
    root = download_bus_bra(args.data_dir)
    rows = build_manifest(root, args.data_dir / "manifest.csv", args.seed, args.outer_fold)
    audit = audit_images(rows, args.data_dir / "image_audit.json")
    print(json.dumps({"images_checked": audit["images_checked"],
                      "valid_pairs": audit["valid_pairs"],
                      "errors": len(audit["errors"]), "warnings": len(audit["warnings"]),
                      "exact_duplicate_images": len(audit["exact_duplicate_images"]),
                      "patient_mapping_verified": False}), flush=True)
    if audit["errors"] or any(g["cross_split"] for g in audit["exact_duplicate_images"]):
        raise SystemExit("Data audit failed; inspect the local image_audit.json before training.")


if __name__ == "__main__":
    main()
