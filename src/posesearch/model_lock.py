from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelEntry:
    name: str
    onnx_file: str
    onnx_sha256: str | None
    input_wh: tuple[int, int]
    source_url: str


class ModelLockError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelLock:
    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.rtmlib_commit = raw["rtmlib"]["commit"]
        self.rtmlib_version = raw["rtmlib"]["version"]
        self.models: dict[str, ModelEntry] = {}
        for key in ("detector", "pose"):
            m = raw[key]
            self.models[key] = ModelEntry(
                name=m["name"], onnx_file=m["onnx_file"], onnx_sha256=m.get("onnx_sha256"),
                input_wh=tuple(m["input_wh"]), source_url=m["source_url"],
            )

    @classmethod
    def load(cls, path: str | Path) -> "ModelLock":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def verify(self, models_dir: str | Path, require_hashes: bool = True) -> dict[str, str]:
        root = Path(models_dir)
        verified: dict[str, str] = {}
        for key, entry in self.models.items():
            path = root / entry.onnx_file
            if not path.is_file():
                raise ModelLockError(f"missing {key} model: {path}")
            actual = sha256_file(path)
            if entry.onnx_sha256:
                if actual.lower() != entry.onnx_sha256.lower():
                    raise ModelLockError(f"{key} SHA-256 mismatch")
            elif require_hashes:
                raise ModelLockError(f"{key} hash is unresolved in models.lock.json")
            verified[key] = actual
        return verified
