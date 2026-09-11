from __future__ import annotations

import numpy as np

from posesearch.geometry import build_geometry_from_body12, mirror_geometry
from posesearch.search import ExactMaskedSearcher
from posesearch.types import PoseRecord


def body_pose() -> np.ndarray:
    # Asymmetric but non-degenerate full-body pose in pixel coordinates.
    return np.array([
        [100, 100], [75, 140], [55, 195],       # left arm
        [160, 110], [205, 115], [245, 85],      # right arm
        [112, 205], [95, 280], [80, 360],       # left leg
        [155, 210], [175, 280], [215, 345],     # right leg
    ], dtype=np.float32)


def rec(person_id: str, points: np.ndarray, mask=None, external_id=None) -> PoseRecord:
    if mask is None:
        mask = np.ones(12, bool)
    return PoseRecord(
        person_id=person_id,
        collection_id="c",
        external_id=external_id or person_id,
        geometry=build_geometry_from_body12(points, mask),
    )


def distance(q: PoseRecord, c: PoseRecord, scope="full", mirror="strict") -> float:
    s = ExactMaskedSearcher([c])
    hits = s.search(q, scope=scope, mirror=mirror, exclude_self=False, top_k=1)
    assert hits
    return hits[0].distance


def test_translation_invariant():
    p = body_pose()
    assert distance(rec("q", p), rec("c", p + np.array([700, -300], np.float32))) < 1e-6


def test_uniform_scale_invariant():
    p = body_pose()
    assert distance(rec("q", p), rec("c", p * 2.75)) < 1e-6


def test_rotation_is_not_normalized_away():
    p = body_pose()
    center = p.mean(axis=0)
    r = np.array([[0, -1], [1, 0]], dtype=np.float32)
    rotated = (p - center) @ r.T + center
    assert distance(rec("q", p), rec("c", rotated)) > 0.15


def test_missing_joint_does_not_become_origin():
    p = body_pose()
    mask = np.ones(12, bool)
    mask[2] = False
    c1 = p.copy(); c1[2] = [0, 0]
    c2 = p.copy(); c2[2] = [99999, -99999]
    d1 = distance(rec("q", p), rec("c1", c1, mask), scope="full")
    d2 = distance(rec("q", p), rec("c2", c2, mask), scope="full")
    assert abs(d1 - d2) < 1e-7


def test_mirror_equivalence_swaps_semantics():
    q = rec("q", body_pose())
    mirrored = mirror_geometry(q.geometry)
    c = PoseRecord("c", "c", "c", mirrored)
    strict = ExactMaskedSearcher([c]).search(q, scope="full", mirror="strict", exclude_self=False, top_k=1)[0]
    equiv = ExactMaskedSearcher([c]).search(q, scope="full", mirror="equivalent", exclude_self=False, top_k=1)[0]
    assert strict.distance > 0.05
    assert equiv.distance < 1e-6
    assert equiv.mirror_applied is True


def test_half_body_cannot_pass_full():
    p = body_pose()
    mask = np.zeros(12, bool); mask[:6] = True
    q = rec("q", p, mask)
    c = rec("c", p, mask)
    assert ExactMaskedSearcher([c]).search(q, scope="full", mirror="strict", exclude_self=False) == []
    assert ExactMaskedSearcher([c]).search(q, scope="upper", mirror="strict", exclude_self=False)


def test_degenerate_scale_rejected():
    p = np.ones((12, 2), dtype=np.float32) * 42
    q = rec("q", p)
    c = rec("c", p)
    assert ExactMaskedSearcher([c]).search(q, scope="full", mirror="strict", exclude_self=False) == []


def test_best_person_per_image_only():
    p = body_pose()
    q = rec("q", p, external_id="query")
    close = rec("a", p + 1, external_id="same-image")
    farther = rec("b", p + np.array([[0,0]]*11 + [[30,0]], np.float32), external_id="same-image")
    other = rec("c", p + 2, external_id="other")
    hits = ExactMaskedSearcher([farther, close, other]).search(q, scope="full", mirror="strict", top_k=10)
    assert [h.external_id for h in hits].count("same-image") == 1
    assert next(h for h in hits if h.external_id == "same-image").person_id == "a"
