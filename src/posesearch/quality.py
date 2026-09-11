from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import BODY12_COCO_INDICES


@dataclass(slots=True)
class QualityResult:
    valid_mask: np.ndarray
    quality: str
    available_scopes: tuple[str, ...]
    reasons: tuple[str, ...]


def evaluate_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray,
    image_width: int,
    image_height: int,
    bbox: np.ndarray | None = None,
    score_threshold: float = 0.25,
) -> QualityResult:
    """Conservative validity filter for COCO17 keypoints.

    Model scores are used as a gate only; they are not interpreted as calibrated
    correctness probabilities.
    """
    pts = np.asarray(keypoints, dtype=np.float32).reshape(17, 2)
    sc = np.asarray(scores, dtype=np.float32).reshape(17)
    finite = np.isfinite(pts).all(axis=1) & np.isfinite(sc)
    in_canvas = (
        (pts[:, 0] >= 0) & (pts[:, 0] < image_width) &
        (pts[:, 1] >= 0) & (pts[:, 1] < image_height)
    )
    valid = finite & in_canvas & (sc >= float(score_threshold))

    reasons: list[str] = []
    if bbox is not None:
        x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox).reshape(4)]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        margin_x, margin_y = 0.15 * bw, 0.15 * bh
        near_box = (
            (pts[:, 0] >= x1 - margin_x) & (pts[:, 0] <= x2 + margin_x) &
            (pts[:, 1] >= y1 - margin_y) & (pts[:, 1] <= y2 + margin_y)
        )
        body = BODY12_COCO_INDICES
        valid[body] &= near_box[body]
        if int((~near_box[body]).sum()) >= 4:
            reasons.append("bbox_keypoint_conflict")

    body_valid = valid[BODY12_COCO_INDICES]
    upper = int(body_valid[:6].sum())
    lower = int(body_valid[6:].sum())
    shoulders_hips = int(body_valid[[0, 3, 6, 9]].sum())

    scopes: list[str] = []
    if upper >= 5:
        scopes.append("upper")
    if lower >= 5:
        scopes.append("lower")
    if upper >= 5 and lower >= 5 and shoulders_hips >= 3 and int(body_valid.sum()) >= 10:
        scopes.insert(0, "full")

    if "full" in scopes:
        quality = "usable_full"
    elif "upper" in scopes:
        quality = "usable_upper"
    elif "lower" in scopes:
        quality = "usable_lower"
    else:
        quality = "low_quality"
        reasons.append("insufficient_joints")

    if not finite.all():
        reasons.append("non_finite_keypoints")
    if int(in_canvas[BODY12_COCO_INDICES].sum()) < 6:
        reasons.append("out_of_frame")

    return QualityResult(valid, quality, tuple(dict.fromkeys(scopes)), tuple(dict.fromkeys(reasons)))


def choose_auto_scope(quality: str, geometry_point_mask: np.ndarray) -> tuple[str, str | None]:
    mask = np.asarray(geometry_point_mask, dtype=bool).reshape(12)
    if quality == "usable_full" and int(mask.sum()) >= 10:
        return "full", None
    if int(mask[:6].sum()) >= 5:
        return "upper", "lower_body_not_usable"
    if int(mask[6:].sum()) >= 5:
        return "lower", "upper_body_not_usable"
    raise ValueError("QUERY_POSE_NOT_USABLE")
