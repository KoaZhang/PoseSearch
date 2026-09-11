#!/usr/bin/env python3
"""Explicit model setup helper.

This is intentionally not called by the service at runtime. It downloads the two
archives named in models.lock.json, extracts exactly one ONNX model from each,
renames them to the stable local filenames, computes SHA-256, and resolves the
lock. Review model/weight licensing before redistribution or commercial use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lock", default="models.lock.json")
    p.add_argument("--models-dir", default="models")
    args = p.parse_args()

    lock_path = Path(args.lock)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="posesearch-models-") as tmp:
        tmpdir = Path(tmp)
        for key in ("detector", "pose"):
            entry = lock[key]
            archive = tmpdir / entry["archive_name"]
            print(f"Downloading {key}: {entry['source_url']}")
            with urllib.request.urlopen(entry["source_url"], timeout=60) as src, archive.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            entry["archive_sha256"] = sha256(archive)

            with zipfile.ZipFile(archive) as zf:
                onnx_members = [n for n in zf.namelist() if n.lower().endswith(".onnx") and not n.endswith("/")]
                if len(onnx_members) != 1:
                    raise SystemExit(
                        f"{key}: expected exactly one ONNX in archive, found {len(onnx_members)}: {onnx_members}. "
                        "Inspect the upstream archive and update this script/lock explicitly."
                    )
                target = models_dir / entry["onnx_file"]
                with zf.open(onnx_members[0]) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            entry["onnx_sha256"] = sha256(target)
            entry["archive_onnx_member"] = onnx_members[0]
            print(f"Resolved {key}: {target} sha256={entry['onnx_sha256']}")

    lock["status"] = "resolved"
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {lock_path}")


if __name__ == "__main__":
    main()
