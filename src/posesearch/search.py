from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .constants import (
    FEATURE_VERSION,
    LOWER_BONE_INDICES,
    SCOPE_ANGLE_MASK,
    SCOPE_BONE_MASK,
    SCOPE_POINTS,
    TORSO_BONE_INDICES,
    UPPER_BONE_INDICES,
)
from .geometry import mirror_geometry
from .types import PoseGeometry, PoseRecord, SearchHit

EPS = 1e-8


@dataclass(slots=True)
class SearchConfig:
    min_coverage: float = 0.75
    block_size: int = 4096
    max_top_k: int = 100
    weight_bone: float = 0.50
    weight_angle: float = 0.20
    weight_position: float = 0.30
    coverage_penalty: float = 0.20


@dataclass(slots=True)
class _IndexArrays:
    records: list[PoseRecord]
    points: np.ndarray
    point_masks: np.ndarray
    bones: np.ndarray
    bone_masks: np.ndarray
    angles: np.ndarray
    angle_masks: np.ndarray

    @classmethod
    def from_records(cls, records: list[PoseRecord]) -> "_IndexArrays":
        if not records:
            return cls(
                [],
                np.empty((0, 12, 2), np.float32), np.empty((0, 12), bool),
                np.empty((0, 12, 2), np.float32), np.empty((0, 12), bool),
                np.empty((0, 8), np.float32), np.empty((0, 8), bool),
            )
        return cls(
            records,
            np.stack([r.geometry.points for r in records]).astype(np.float32),
            np.stack([r.geometry.point_mask for r in records]).astype(bool),
            np.stack([r.geometry.bones for r in records]).astype(np.float32),
            np.stack([r.geometry.bone_mask for r in records]).astype(bool),
            np.stack([r.geometry.angles for r in records]).astype(np.float32),
            np.stack([r.geometry.angle_mask for r in records]).astype(bool),
        )


class ExactMaskedSearcher:
    """NumPy exact search for geometry-v1.

    Scoring is vectorized by block. Python loops are used only for blocks and
    final result diversification, not per-candidate geometry calculations.
    """

    metric_version = FEATURE_VERSION

    def __init__(self, records: Iterable[PoseRecord] = (), config: SearchConfig | None = None):
        self.config = config or SearchConfig()
        self.replace(records)

    def replace(self, records: Iterable[PoseRecord]) -> None:
        self._idx = _IndexArrays.from_records(list(records))

    @property
    def size(self) -> int:
        return len(self._idx.records)

    def _scope_masks(self, scope: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if scope not in SCOPE_POINTS:
            raise ValueError(f"unsupported scope: {scope}")
        point_scope = np.zeros(12, dtype=bool)
        point_scope[SCOPE_POINTS[scope]] = True
        return point_scope, SCOPE_BONE_MASK[scope], SCOPE_ANGLE_MASK[scope]

    def _score_block(
        self,
        query: PoseGeometry,
        points: np.ndarray,
        point_masks: np.ndarray,
        bones: np.ndarray,
        bone_masks: np.ndarray,
        angles: np.ndarray,
        angle_masks: np.ndarray,
        scope: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        point_scope, bone_scope, angle_scope = self._scope_masks(scope)
        q_point_mask = query.point_mask & point_scope
        q_valid_count = int(q_point_mask.sum())
        min_common = 6 if scope == "full" else 4
        if q_valid_count < min_common:
            n = points.shape[0]
            return np.full(n, np.inf), np.zeros(n), np.zeros(n, dtype=np.int16)

        common = point_masks & q_point_mask[None, :]
        count = common.sum(axis=1).astype(np.int16)
        coverage = count.astype(np.float32) / float(q_valid_count)
        eligible = (coverage >= cfg.min_coverage) & (count >= min_common)

        # Candidate-specific common-point centering and RMS scaling.
        denom = np.maximum(count, 1).astype(np.float32)[:, None]
        w = common.astype(np.float32)
        c_center = (points * w[..., None]).sum(axis=1) / denom
        q_center = (query.points[None, :, :] * w[..., None]).sum(axis=1) / denom
        c_delta = points - c_center[:, None, :]
        q_delta = query.points[None, :, :] - q_center[:, None, :]
        c_sq = (c_delta * c_delta).sum(axis=2)
        q_sq = (q_delta * q_delta).sum(axis=2)
        c_scale = np.sqrt((c_sq * w).sum(axis=1) / np.maximum(count, 1))
        q_scale = np.sqrt((q_sq * w).sum(axis=1) / np.maximum(count, 1))
        nondeg = (c_scale > EPS) & (q_scale > EPS)
        eligible &= nondeg
        c_norm = c_delta / np.maximum(c_scale, EPS)[:, None, None]
        q_norm = q_delta / np.maximum(q_scale, EPS)[:, None, None]
        pos_sq = ((c_norm - q_norm) ** 2).sum(axis=2)
        d_position = np.clip((pos_sq * w).sum(axis=1) / np.maximum(count, 1) / 4.0, 0.0, 1.0)

        common_bones = bone_masks & query.bone_mask[None, :] & bone_scope[None, :]
        bone_count = common_bones.sum(axis=1)
        dots = np.clip((bones * query.bones[None, :, :]).sum(axis=2), -1.0, 1.0)
        bone_terms = (1.0 - dots) / 2.0
        d_bone = (bone_terms * common_bones).sum(axis=1) / np.maximum(bone_count, 1)

        common_angles = angle_masks & query.angle_mask[None, :] & angle_scope[None, :]
        angle_count = common_angles.sum(axis=1)
        angle_terms = np.abs(angles - query.angles[None, :]) / 2.0
        d_angle = (angle_terms * common_angles).sum(axis=1) / np.maximum(angle_count, 1)

        # A valid comparison needs position plus at least one structural class.
        available_bone = bone_count > 0
        available_angle = angle_count > 0
        eligible &= available_bone | available_angle

        if scope == "full":
            # Full-body comparisons must contain comparable upper, torso and lower information.
            eligible &= (
                common_bones[:, UPPER_BONE_INDICES].any(axis=1)
                & common_bones[:, TORSO_BONE_INDICES].any(axis=1)
                & common_bones[:, LOWER_BONE_INDICES].any(axis=1)
            )

        # Re-normalize weights only over feature classes available for each candidate.
        wb = cfg.weight_bone * available_bone.astype(np.float32)
        wa = cfg.weight_angle * available_angle.astype(np.float32)
        wp = np.full(points.shape[0], cfg.weight_position, dtype=np.float32)
        wsum = wb + wa + wp
        d_shape = (wb * d_bone + wa * d_angle + wp * d_position) / np.maximum(wsum, EPS)
        distance = np.clip(d_shape + cfg.coverage_penalty * (1.0 - coverage), 0.0, 1.0)
        distance[~eligible] = np.inf
        return distance.astype(np.float32), coverage.astype(np.float32), count

    def search(
        self,
        query: PoseRecord,
        *,
        scope: str,
        mirror: str = "equivalent",
        top_k: int = 30,
        exclude_self: bool = True,
        exclude_same_sha256: bool = False,
        exclude_character_id: str | None = None,
        max_per_character: int | None = None,
        max_per_album: int | None = None,
    ) -> list[SearchHit]:
        if mirror not in {"equivalent", "strict"}:
            raise ValueError("mirror must be 'equivalent' or 'strict'")
        top_k = max(1, min(int(top_k), self.config.max_top_k))
        n = self.size
        if n == 0:
            return []

        all_dist: list[np.ndarray] = []
        all_cov: list[np.ndarray] = []
        all_count: list[np.ndarray] = []
        all_mirror: list[np.ndarray] = []
        mirrored_query = mirror_geometry(query.geometry) if mirror == "equivalent" else None

        bs = self.config.block_size
        for start in range(0, n, bs):
            end = min(start + bs, n)
            sl = slice(start, end)
            d0, c0, j0 = self._score_block(
                query.geometry,
                self._idx.points[sl], self._idx.point_masks[sl],
                self._idx.bones[sl], self._idx.bone_masks[sl],
                self._idx.angles[sl], self._idx.angle_masks[sl], scope,
            )
            mirrored = np.zeros(end - start, dtype=bool)
            if mirrored_query is not None:
                d1, c1, j1 = self._score_block(
                    mirrored_query,
                    self._idx.points[sl], self._idx.point_masks[sl],
                    self._idx.bones[sl], self._idx.bone_masks[sl],
                    self._idx.angles[sl], self._idx.angle_masks[sl], scope,
                )
                use_mirror = d1 < d0
                d0 = np.where(use_mirror, d1, d0)
                c0 = np.where(use_mirror, c1, c0)
                j0 = np.where(use_mirror, j1, j0)
                mirrored = use_mirror
            all_dist.append(d0)
            all_cov.append(c0)
            all_count.append(j0)
            all_mirror.append(mirrored)

        dist = np.concatenate(all_dist)
        cov = np.concatenate(all_cov)
        count = np.concatenate(all_count)
        mirrored = np.concatenate(all_mirror)

        # Metadata filters are applied as an index mask, not geometry loops.
        meta_ok = np.ones(n, dtype=bool)
        for i, rec in enumerate(self._idx.records):
            if rec.collection_id != query.collection_id:
                meta_ok[i] = False
            elif exclude_self and rec.person_id == query.person_id:
                meta_ok[i] = False
            elif exclude_self and rec.external_id == query.external_id:
                meta_ok[i] = False
            elif exclude_same_sha256 and query.content_sha256 and rec.content_sha256 == query.content_sha256:
                meta_ok[i] = False
            elif exclude_character_id and rec.character_id == exclude_character_id:
                meta_ok[i] = False
            elif rec.quality == "low_quality":
                meta_ok[i] = False
        dist[~meta_ok] = np.inf

        finite = np.isfinite(dist)
        if not finite.any():
            return []
        candidate_idx = np.flatnonzero(finite)
        order = candidate_idx[np.argsort(dist[candidate_idx], kind="stable")]

        hits: list[SearchHit] = []
        seen_images: set[str] = set()
        per_character: dict[str, int] = {}
        per_album: dict[str, int] = {}
        for i in order:
            rec = self._idx.records[int(i)]
            if rec.external_id in seen_images:
                continue
            if max_per_character is not None and rec.character_id is not None:
                if per_character.get(rec.character_id, 0) >= max_per_character:
                    continue
            if max_per_album is not None and rec.album_id is not None:
                if per_album.get(rec.album_id, 0) >= max_per_album:
                    continue
            d = float(dist[i])
            hits.append(SearchHit(
                external_id=rec.external_id,
                person_id=rec.person_id,
                distance=d,
                similarity=float(1.0 - d),
                coverage=float(cov[i]),
                matched_joint_count=int(count[i]),
                mirror_applied=bool(mirrored[i]),
                character_id=rec.character_id,
                album_id=rec.album_id,
                work_id=rec.work_id,
            ))
            seen_images.add(rec.external_id)
            if rec.character_id is not None:
                per_character[rec.character_id] = per_character.get(rec.character_id, 0) + 1
            if rec.album_id is not None:
                per_album[rec.album_id] = per_album.get(rec.album_id, 0) + 1
            if len(hits) >= top_k:
                break
        return hits
