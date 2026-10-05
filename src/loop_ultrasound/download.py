"""Verified BUS-BRA acquisition; all data stay in a local data directory.

The fixed public archive is CC-BY-4.0. This module contains original code, not
the authors' Matlab code. Downloading never accepts an unverified ZIP, and ZIP
extraction rejects traversal/symlink entries before writing any archive member.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import urllib.request
import zipfile

BUS_BRA_URL = "https://zenodo.org/api/records/8231412/files/BUSBRA.zip/content"
BUS_BRA_MD5 = "1f8b2be6476d58fc97bfb5e5a1ea9bab"
BUS_BRA_ARCHIVE_BYTES = 133917740
BUS_BRA_DOI = "https://doi.org/10.5281/zenodo.8231412"
CHUNK_BYTES = 1024 * 1024


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_zip(archive: str | Path, destination: str | Path) -> None:
    """Extract only regular files/directories after validating every ZIP name.

    Use a new staging directory as destination. Reading each ZIP member verifies
    its CRC through zipfile; errors propagate rather than accepting partial data.
    """
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if len(members) > 10000 or sum(i.file_size for i in members) > 2 * 1024**3:
            raise ValueError("Unexpected archive size or member count")
        seen = set()
        for member in members:
            name = member.filename
            parts = PurePosixPath(name).parts
            if (not name or name.startswith(("/", "\\")) or "\\" in name
                    or ":" in name or ".." in parts or not parts):
                raise ValueError(f"Unsafe ZIP path: {name!r}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"ZIP symlink rejected: {name!r}")
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"ZIP special file rejected: {name!r}")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"ZIP path escapes destination: {name!r}")
            if target in seen:
                raise ValueError(f"Duplicate ZIP target: {name!r}")
            seen.add(target)
            if target.exists() and not member.is_dir():
                raise FileExistsError(target)
        for member in members:
            target = destination / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as source, target.open("xb") as output:
                for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
                    output.write(chunk)


def _complete_bus_tree(path: Path) -> bool:
    return (all((path / name).is_file() for name in ("bus_data.csv", "5-fold-cv.csv", "10-fold-cv.csv"))
            and len(list((path / "Images").glob("*.png"))) == 1875
            and len(list((path / "Masks").glob("*.png"))) == 1875)


def download_bus_bra(data_dir: str | Path = "data") -> Path:
    """Use/download data/BUSBRA.zip, verify fixed MD5, and extract atomically.

    Existing verified archives are reused, including a ZIP downloaded by curl.
    Existing incomplete extracted trees are never silently overwritten.
    """
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "BUSBRA.zip"
    if not archive.exists():
        partial = data_dir / "BUSBRA.zip.part"
        req = urllib.request.Request(BUS_BRA_URL, headers={"User-Agent": "loop-ultrasound-cpu/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=60) as response, partial.open("wb") as output:
                downloaded = 0
                for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
                    downloaded += len(chunk)
                    if downloaded > BUS_BRA_ARCHIVE_BYTES:
                        raise ValueError("Archive download exceeds pinned size")
                    output.write(chunk)
            if partial.stat().st_size != BUS_BRA_ARCHIVE_BYTES or file_digest(partial, "md5") != BUS_BRA_MD5:
                raise ValueError("Downloaded BUS-BRA archive failed size/MD5 verification")
            os.replace(partial, archive)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    if archive.stat().st_size != BUS_BRA_ARCHIVE_BYTES or file_digest(archive, "md5") != BUS_BRA_MD5:
        raise ValueError(f"Existing archive failed pinned size/MD5 verification: {archive}")
    target = data_dir / "BUSBRA"
    if target.exists():
        if not _complete_bus_tree(target):
            raise ValueError(f"Existing extracted BUS-BRA tree is incomplete: {target}")
    else:
        with tempfile.TemporaryDirectory(prefix="bus-extract-", dir=data_dir) as temporary:
            staging = Path(temporary)
            safe_extract_zip(archive, staging)
            extracted = staging / "BUSBRA"
            if not _complete_bus_tree(extracted):
                raise ValueError("Verified archive has unexpected BUS-BRA layout")
            os.replace(extracted, target)
    provenance = {
        "source_url": BUS_BRA_URL, "doi": BUS_BRA_DOI, "license": "CC-BY-4.0",
        "archive_bytes": BUS_BRA_ARCHIVE_BYTES, "verified_md5": BUS_BRA_MD5,
        "local_archive_sha256": file_digest(archive),
        "patient_mapping": "Case-to-patient mapping remains unverified",
    }
    (data_dir / "download_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(download_bus_bra(args.data_dir))


if __name__ == "__main__":
    main()
