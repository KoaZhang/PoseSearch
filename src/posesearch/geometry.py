from __future__ import annotations

import math

import numpy as np

from .constants import (
    ANGLES,
    BODY12_COCO_INDICES,
    BODY12_LEFT_RIGHT_PAIRS,
    BONES,
    COCO17_LEFT_RIGHT_PAIRS,
)
from .types import PoseGeometry

EPS = 1e-8


def _unit_vector(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, bool]:
    v = np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n <= EPS:
        return np.zeros(2, dtype=np.float32), False
    return (v / n).astype(np.float32), True


def _angle_cos(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[float, bool]:
    ba = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    bc = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    n1 = float(np.linalg.norm(ba))
    n2 = float(np.linalg.norm(bc))
    if not (math.isfinite(n1) and math.isfinite(n2)) or n1 <= EPS or n2 <= EPS:
        return 0.0, False
    val = float(np.dot(ba, bc) / (n1 * n2))
    return float(np.clip(val, -1.0, 1.0)), True


def build_geometry_from_body12(points: np.ndarray, point_mask: np.ndarray) -> PoseGeometry:
    points = np.asarray(points, dtype=np.float32).reshape(12, 2)
    point_mask = np.asarray(point_mask, dtype=bool).reshape(12)

    bones = np.zeros((12, 2), dtype=np.float32)
    bone_mask = np.zeros(12, dtype=bool)
    for i, (a, b) in enumerate(BONES):
        if point_mask[a] and point_mask[b]:
            bones[i], bone_mask[i] = _unit_vector(points[a], points[b])

    angles = np.zeros(8, dtype=np.float32)
    angle_mask = np.zeros(8, dtype=bool)
    for i, (a, b, c) in enumerate(ANGLES):
        if point_mask[a] and point_mask[b] and point_mask[c]:
            angles[i], angle_mask[i] = _angle_cos(points[a], points[b], points[c])

    clean_points = points.copy()
    clean_points[~point_mask] = 0.0
    return PoseGeometry(clean_points, point_mask, bones, bone_mask, angles, angle_mask)


def build_geometry_from_coco17(keypoints: np.ndarray, valid_mask: np.ndarray) -> PoseGeometry:
    keypoints = np.asarray(keypoints, dtype=np.float32).reshape(17, 2)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(17)
    return build_geometry_from_body12(keypoints[BODY12_COCO_INDICES], valid_mask[BODY12_COCO_INDICES])


def mirror_body12(points: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror a BODY12 pose and swap left/right semantics.

    Reflection uses x -> -x. Translation is intentionally omitted because
    geometry-v1 centers each compared pose using common valid joints.
    """
    out = np.asarray(points, dtype=np.float32).reshape(12, 2).copy()
    out[:, 0] *= -1.0
    out_mask = np.asarray(mask, dtype=bool).reshape(12).copy()
    for left, right in BODY12_LEFT_RIGHT_PAIRS:
        out[[left, right]] = out[[right, left]]
        out_mask[[left, right]] = out_mask[[right, left]]
    return out, out_mask


def mirror_geometry(geometry: PoseGeometry) -> PoseGeometry:
    points, mask = mirror_body12(geometry.points, geometry.point_mask)
    return build_geometry_from_body12(points, mask)


def mirror_coco17(
    keypoints: np.ndarray,
    scores: np.ndarray,
    valid_mask: np.ndarray,
    image_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keypoints = np.asarray(keypoints, dtype=np.float32).reshape(17, 2).copy()
    scores = np.asarray(scores, dtype=np.float32).reshape(17).copy()
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(17).copy()
    if image_width is None:
        keypoints[:, 0] *= -1.0
    else:
        keypoints[:, 0] = (float(image_width) - 1.0) - keypoints[:, 0]
    for left, right in COCO17_LEFT_RIGHT_PAIRS:
        keypoints[[left, right]] = keypoints[[right, left]]
        scores[[left, right]] = scores[[right, left]]
        valid_mask[[left, right]] = valid_mask[[right, left]]
    return keypoints, scores, valid_mask
