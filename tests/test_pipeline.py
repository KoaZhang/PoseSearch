from pathlib import Path

import numpy as np
from PIL import Image

from posesearch.config import PipelineConfig
from posesearch.pipeline import PosePipeline
from posesearch.inference.base import PoseRuntime


class NoPerson(PoseRuntime):
    def detect(self, image_bgr):
        return np.empty((0,4), np.float32)
    def pose(self, image_bgr, bboxes_xyxy):
        raise AssertionError("pose must not be called when detector returned no boxes")


def test_no_person_never_calls_pose(tmp_path: Path):
    path = tmp_path / "x.png"
    Image.new("RGB", (64, 64), "white").save(path)
    r = PosePipeline(NoPerson(), PipelineConfig()).analyze_file(path)
    assert r.status == "no_person"
    assert r.persons == []
