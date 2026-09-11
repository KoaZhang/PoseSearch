from pathlib import Path

import pytest

from posesearch.security import PathSecurityError, resolve_library_path


def test_root_path_resolution(tmp_path: Path):
    root = tmp_path / "lib"; root.mkdir()
    f = root / "a.jpg"; f.write_bytes(b"x")
    assert resolve_library_path(root, "a.jpg") == f.resolve()
    with pytest.raises(PathSecurityError):
        resolve_library_path(root, "../outside.jpg")
    with pytest.raises(PathSecurityError):
        resolve_library_path(root, str(f.resolve()))


def test_symlink_escape_rejected(tmp_path: Path):
    root = tmp_path / "lib"; root.mkdir()
    outside = tmp_path / "secret.jpg"; outside.write_bytes(b"x")
    link = root / "link.jpg"
    link.symlink_to(outside)
    with pytest.raises(PathSecurityError):
        resolve_library_path(root, "link.jpg")
