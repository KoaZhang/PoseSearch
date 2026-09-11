from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PoseRuntime(ABC):
    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return human boxes in xyxy format, shape [N,4]."""

    @abstractmethod
    def pose(self, image_bgr: np.ndarray, bboxes_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return COCO17 keypoints [N,17,2] and raw scores [N,17]."""
