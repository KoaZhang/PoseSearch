from __future__ import annotations

from pathlib import Path


class PathSecurityError(ValueError):
    pass


def resolve_library_path(root: str | Path, relative_path: str) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute():
        raise PathSecurityError("absolute paths are not allowed")
    if any(part == ".." for part in rel.parts):
        raise PathSecurityError("parent traversal is not allowed")

    root_path = Path(root).resolve(strict=True)
    candidate = (root_path / rel).resolve(strict=True)
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise PathSecurityError("path escapes configured root") from exc
    if not candidate.is_file():
        raise PathSecurityError("source is not a regular file")
    return candidate
