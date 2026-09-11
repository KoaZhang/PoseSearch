from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .geometry import build_geometry_from_coco17
from .inference.base import PoseRuntime
from .preprocess import DecodedImage, decode_image
from .quality import evaluate_keypoints


@dataclass(slots=True)
class AnalysisResult:
    extraction_id: str
    content_sha256: str
    image_width: int
    image_height: int
    coordinate_space: str
    status: str
    persons_truncated: bool
    persons: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class PosePipeline:
    def __init__(self, runtime: PoseRuntime, config: PipelineConfig):
        self.runtime = runtime
        self.config = config

    def _decode(self, path: str | Path) -> DecodedImage:
        return decode_image(
            path,
            max_file_bytes=self.config.max_file_bytes,
            max_pixels=self.config.max_pixels,
            max_long_edge=self.config.analysis_max_long_edge,
        )

    def analyze_file(self, path: str | Path) -> AnalysisResult:
        decoded = self._decode(path)
        digest = sha256_file(path)
        boxes = self.runtime.detect(decoded.bgr)
        if boxes.size == 0:
            return AnalysisResult(
                extraction_id=str(uuid.uuid4()), content_sha256=digest,
                image_width=decoded.image_width, image_height=decoded.image_height,
                coordinate_space=decoded.coordinate_space, status="no_person",
                persons_truncated=False, persons=[], diagnostics={"detected_persons": 0},
            )

        areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        order = np.argsort(-areas, kind="stable")
        max_people = int(self.config.max_persons_per_image)
        persons_truncated = len(order) > max_people
        selected = boxes[order[:max_people]]

        keypoints, scores = self.runtime.pose(decoded.bgr, selected)
        if keypoints.shape[0] != selected.shape[0]:
            raise RuntimeError("pose output count does not match selected detection boxes")

        sx, sy = decoded.scale_x_to_original, decoded.scale_y_to_original
        persons: list[dict[str, Any]] = []
        usable = 0
        for i, (bbox_work, kpts_work, score) in enumerate(zip(selected, keypoints, scores, strict=True)):
            bbox = np.asarray(bbox_work, np.float32).copy()
            bbox[[0, 2]] *= sx
            bbox[[1, 3]] *= sy
            kpts = np.asarray(kpts_work, np.float32).copy()
            kpts[:, 0] *= sx
            kpts[:, 1] *= sy

            quality = evaluate_keypoints(
                kpts, score, decoded.image_width, decoded.image_height,
                bbox=bbox, score_threshold=self.config.score_threshold,
            )
            geometry = build_geometry_from_coco17(kpts, quality.valid_mask)
            if quality.quality != "low_quality":
                usable += 1
            persons.append({
                "person_id": str(uuid.uuid4()),
                "person_index": i,
                "bbox": bbox,
                "keypoints": kpts,
                "scores": np.asarray(score, np.float32),
                "valid_mask": quality.valid_mask,
                "geometry": geometry,
                "quality": quality.quality,
                "available_scopes": quality.available_scopes,
                "diagnostics": {"reasons": list(quality.reasons)},
            })

        if usable == 0:
            status = "no_usable_pose"
        elif persons_truncated or usable < len(persons):
            status = "partial"
        else:
            status = "ready"

        return AnalysisResult(
            extraction_id=str(uuid.uuid4()), content_sha256=digest,
            image_width=decoded.image_width, image_height=decoded.image_height,
            coordinate_space=decoded.coordinate_space, status=status,
            persons_truncated=persons_truncated, persons=persons,
            diagnostics={"detected_persons": int(len(boxes)), "analyzed_persons": len(persons)},
        )
