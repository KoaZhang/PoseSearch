import numpy as np

from posesearch.constants import BODY12_COCO_INDICES
from posesearch.quality import choose_auto_scope, evaluate_keypoints
from posesearch.geometry import build_geometry_from_body12


def test_auto_scope_downgrades_to_upper():
    mask = np.zeros(12, bool); mask[:6] = True
    g = build_geometry_from_body12(np.arange(24, dtype=np.float32).reshape(12,2), mask)
    scope, reason = choose_auto_scope("usable_upper", g.point_mask)
    assert scope == "upper"
    assert reason == "lower_body_not_usable"


def test_quality_keeps_scores_as_gate_not_probability():
    k = np.zeros((17,2), np.float32)
    s = np.ones(17, np.float32) * 0.9
    # Put all body points inside canvas at distinct positions.
    for j, idx in enumerate(BODY12_COCO_INDICES):
        k[idx] = [20 + j * 3, 20 + j * 4]
    r = evaluate_keypoints(k, s, 300, 300, score_threshold=0.25)
    assert r.quality == "usable_full"
