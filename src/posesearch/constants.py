from __future__ import annotations

import numpy as np

COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

# geometry-v1 storage order: L arm, R arm, L leg, R leg.
BODY12_COCO_INDICES = np.asarray([5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16], dtype=np.int64)
BODY12_NAMES = tuple(COCO17_NAMES[i] for i in BODY12_COCO_INDICES)

BODY12_LEFT_RIGHT_PAIRS = ((0, 3), (1, 4), (2, 5), (6, 9), (7, 10), (8, 11))
COCO17_LEFT_RIGHT_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))

BONES = (
    (0, 1), (1, 2),
    (3, 4), (4, 5),
    (6, 7), (7, 8),
    (9, 10), (10, 11),
    (0, 6), (3, 9),
    (0, 3), (6, 9),
)

ANGLES = (
    (0, 1, 2), (3, 4, 5),
    (1, 0, 6), (4, 3, 9),
    (0, 6, 7), (3, 9, 10),
    (6, 7, 8), (9, 10, 11),
)

SCOPE_POINTS = {
    "full": np.asarray(range(12), dtype=np.int64),
    "upper": np.asarray(range(6), dtype=np.int64),
    "lower": np.asarray(range(6, 12), dtype=np.int64),
}


def _feature_scope_mask(features: tuple[tuple[int, ...], ...], scope: str) -> np.ndarray:
    allowed = set(int(x) for x in SCOPE_POINTS[scope])
    return np.asarray([all(p in allowed for p in feature) for feature in features], dtype=bool)


SCOPE_BONE_MASK = {scope: _feature_scope_mask(BONES, scope) for scope in SCOPE_POINTS}
SCOPE_ANGLE_MASK = {scope: _feature_scope_mask(ANGLES, scope) for scope in SCOPE_POINTS}

UPPER_BONE_INDICES = np.asarray([0, 1, 2, 3], dtype=np.int64)
LOWER_BONE_INDICES = np.asarray([4, 5, 6, 7], dtype=np.int64)
TORSO_BONE_INDICES = np.asarray([8, 9, 10, 11], dtype=np.int64)

FEATURE_VERSION = "geometry-v1.0"
KEYPOINT_SCHEMA = "coco17"
