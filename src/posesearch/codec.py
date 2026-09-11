from __future__ import annotations

import io
import json

import numpy as np

from .types import PoseGeometry


def encode_array(array: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, np.asarray(array), allow_pickle=False)
    return bio.getvalue()


def decode_array(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


def encode_geometry(g: PoseGeometry) -> bytes:
    bio = io.BytesIO()
    np.savez_compressed(
        bio,
        points=g.points,
        point_mask=g.point_mask,
        bones=g.bones,
        bone_mask=g.bone_mask,
        angles=g.angles,
        angle_mask=g.angle_mask,
    )
    return bio.getvalue()


def decode_geometry(blob: bytes) -> PoseGeometry:
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        return PoseGeometry(z["points"], z["point_mask"], z["bones"], z["bone_mask"], z["angles"], z["angle_mask"])


def dumps_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
