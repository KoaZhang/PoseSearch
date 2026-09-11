import numpy as np

from posesearch.visualize import draw_pose_overlay, encode_pose_overlay


def test_overlay_draws_valid_skeleton_only():
    image = np.zeros((128, 128, 3), np.uint8)
    keypoints = np.zeros((17, 2), np.float32)
    keypoints[5] = [30, 30]
    keypoints[7] = [50, 60]
    valid = np.zeros(17, bool)
    valid[[5, 7]] = True
    person = {
        "bbox": np.array([20, 20, 90, 100], np.float32),
        "keypoints": keypoints,
        "valid_mask": valid,
        "quality": "usable_upper",
    }
    overlay = draw_pose_overlay(image, [person])
    assert overlay.shape == image.shape
    assert int(overlay.sum()) > 0
    encoded = encode_pose_overlay(image, [person])
    assert encoded[:2] == b"\xff\xd8"
