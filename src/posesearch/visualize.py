from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

COCO17_BODY_EDGES: tuple[tuple[int, int], ...] = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
)


def draw_pose_overlay(
    image_bgr: np.ndarray,
    persons: list[dict[str, Any]],
    *,
    point_radius: int = 4,
    line_thickness: int = 2,
) -> np.ndarray:
    canvas = np.asarray(image_bgr).copy()
    for person in persons:
        bbox = np.asarray(person["bbox"], dtype=np.float32).reshape(4)
        keypoints = np.asarray(person["keypoints"], dtype=np.float32).reshape(17, 2)
        valid = np.asarray(person["valid_mask"], dtype=bool).reshape(17)
        quality = str(person.get("quality", "unknown"))

        x1, y1, x2, y2 = [int(round(float(x))) for x in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 220, 40), line_thickness)
        cv2.putText(
            canvas, quality, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (40, 220, 40), 1, cv2.LINE_AA,
        )

        for a, b in COCO17_BODY_EDGES:
            if valid[a] and valid[b]:
                pa = tuple(int(round(float(v))) for v in keypoints[a])
                pb = tuple(int(round(float(v))) for v in keypoints[b])
                cv2.line(canvas, pa, pb, (0, 190, 255), line_thickness, cv2.LINE_AA)
        for idx, (x, y) in enumerate(keypoints):
            if valid[idx]:
                cv2.circle(
                    canvas, (int(round(float(x))), int(round(float(y)))), point_radius,
                    (255, 90, 30), -1, cv2.LINE_AA,
                )
    return canvas


def save_pose_overlay(path: str | Path, image_bgr: np.ndarray, persons: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    overlay = draw_pose_overlay(image_bgr, persons)
    ok, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("failed to encode diagnostic overlay")
    target.write_bytes(encoded.tobytes())
    return target


def encode_pose_overlay(image_bgr: np.ndarray, persons: list[dict[str, Any]]) -> bytes:
    overlay = draw_pose_overlay(image_bgr, persons)
    ok, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("failed to encode diagnostic overlay")
    return encoded.tobytes()
