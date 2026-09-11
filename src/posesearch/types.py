from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class PoseGeometry:
    points: np.ndarray          # float32 [12, 2]
    point_mask: np.ndarray      # bool [12]
    bones: np.ndarray           # float32 [12, 2]
    bone_mask: np.ndarray       # bool [12]
    angles: np.ndarray          # float32 [8]
    angle_mask: np.ndarray      # bool [8]

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32).reshape(12, 2)
        self.point_mask = np.asarray(self.point_mask, dtype=bool).reshape(12)
        self.bones = np.asarray(self.bones, dtype=np.float32).reshape(12, 2)
        self.bone_mask = np.asarray(self.bone_mask, dtype=bool).reshape(12)
        self.angles = np.asarray(self.angles, dtype=np.float32).reshape(8)
        self.angle_mask = np.asarray(self.angle_mask, dtype=bool).reshape(8)


@dataclass(slots=True)
class PoseRecord:
    person_id: str
    collection_id: str
    external_id: str
    geometry: PoseGeometry
    quality: str = "usable_full"
    character_id: str | None = None
    album_id: str | None = None
    work_id: str | None = None
    content_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchHit:
    external_id: str
    person_id: str
    distance: float
    similarity: float
    coverage: float
    matched_joint_count: int
    mirror_applied: bool
    character_id: str | None = None
    album_id: str | None = None
    work_id: str | None = None
