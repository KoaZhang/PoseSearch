#!/usr/bin/env python3
"""Download, verify, extract, and optionally resolve PoseSearch model hashes.

The service never downloads models at runtime. This helper is an explicit setup
step. Every archive is checked against its pinned SHA-256 before extraction.
When ONNX hashes are already pinned, they are verified; otherwise --update-lock
is required to record them.
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


def download(url: str, target: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "PoseSearch/0.2 model-resolver"})
    with urllib.request.urlopen(req, timeout=90) as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def fetch_with_fallback(entry: dict, target: Path) -> str:
    errors: list[str] = []
    for url in (entry.get("source_url"), entry.get("mirror_url")):
        if not url:
            continue
        try:
            print(f"Downloading: {url}")
            download(url, target)
            return url
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            target.unlink(missing_ok=True)
    raise RuntimeError("all model download sources failed: " + " | ".join(errors))


def resolve_entry(key: str, entry: dict, tmpdir: Path, models_dir: Path, *, update_lock: bool) -> None:
    archive = tmpdir / entry["archive_name"]
    source_used = fetch_with_fallback(entry, archive)
    archive_hash = sha256(archive)
    expected_archive = entry.get("archive_sha256")
    if not expected_archive:
        raise RuntimeError(f"{key}: archive_sha256 is not pinned")
    if archive_hash.lower() != expected_archive.lower():
        raise RuntimeError(
            f"{key}: archive SHA-256 mismatch: expected {expected_archive}, got {archive_hash}"
        )
    print(f"Verified {key} archive sha256={archive_hash} from {source_used}")

    with zipfile.ZipFile(archive) as zf:
        onnx_members = [n for n in zf.namelist() if n.lower().endswith(".onnx") and not n.endswith("/")]
        if len(onnx_members) != 1:
            raise RuntimeError(
                f"{key}: expected exactly one ONNX in archive, found {len(onnx_members)}: {onnx_members}"
            )
        target = models_dir / entry["onnx_file"]
        tmp_target = target.with_suffix(target.suffix + ".tmp")
        with zf.open(onnx_members[0]) as src, tmp_target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        tmp_target.replace(target)

    onnx_hash = sha256(target)
    expected_onnx = entry.get("onnx_sha256")
    if expected_onnx:
        if onnx_hash.lower() != expected_onnx.lower():
            raise RuntimeError(
                f"{key}: ONNX SHA-256 mismatch: expected {expected_onnx}, got {onnx_hash}"
            )
    elif update_lock:
        entry["onnx_sha256"] = onnx_hash
    else:
        raise RuntimeError(
            f"{key}: ONNX hash is unresolved; rerun with --update-lock after reviewing the pinned archive"
        )
    entry["archive_onnx_member"] = onnx_members[0]
    print(f"Verified {key} ONNX: {target} sha256={onnx_hash}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lock", default="models.lock.json")
    p.add_argument("--models-dir", default="models")
    p.add_argument(
        "--update-lock", action="store_true",
        help="fill unresolved ONNX SHA-256 fields after the pinned archive hash verifies",
    )
    args = p.parse_args()

    lock_path = Path(args.lock)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="posesearch-models-") as tmp:
        tmpdir = Path(tmp)
        for key in ("detector", "pose"):
            resolve_entry(key, lock[key], tmpdir, models_dir, update_lock=args.update_lock)

    if args.update_lock:
        lock["status"] = "resolved"
        lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Updated {lock_path}")
    elif lock.get("status") != "resolved":
        raise RuntimeError("model lock status is not resolved")


if __name__ == "__main__":
    main()
