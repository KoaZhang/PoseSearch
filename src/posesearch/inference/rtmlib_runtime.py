from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import RuntimeConfig
from ..model_lock import ModelLock
from .base import PoseRuntime


class RTMLibRuntime(PoseRuntime):
    """Pinned RTMLib preprocessing/postprocessing with PoseSearch-owned ORT sessions.

    This intentionally does not call RTMLib Body(image): we need the detector boxes,
    must reject empty detections instead of triggering RTMPose's whole-image fallback,
    and must pass explicit ONNX Runtime SessionOptions.
    """

    def __init__(self, models_dir: str, lock_path: str, runtime: RuntimeConfig):
        try:
            import onnxruntime as ort
            from rtmlib import RTMPose, YOLOX
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError("Install PoseSearch inference dependencies first") from exc

        self.lock = ModelLock.load(lock_path)
        self.lock.verify(models_dir, require_hashes=True)
        det_entry = self.lock.models["detector"]
        pose_entry = self.lock.models["pose"]
        det_path = str(Path(models_dir) / det_entry.onnx_file)
        pose_path = str(Path(models_dir) / pose_entry.onnx_file)

        def make_options():
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(runtime.intra_op_threads)
            opts.inter_op_num_threads = int(runtime.inter_op_threads)
            opts.execution_mode = (
                ort.ExecutionMode.ORT_SEQUENTIAL
                if runtime.execution_mode.lower() == "sequential"
                else ort.ExecutionMode.ORT_PARALLEL
            )
            spin = "1" if runtime.allow_spinning else "0"
            opts.add_session_config_entry("session.intra_op.allow_spinning", spin)
            opts.add_session_config_entry("session.inter_op.allow_spinning", spin)
            return opts

        class _YOLOX(YOLOX):
            def __init__(self, model_path: str):
                # Reproduce BaseTool state without invoking its automatic downloader/session.
                self.onnx_model = model_path
                self.model_input_size = (416, 416)  # RTMLib uses (H,W) here; square detector.
                self.mean = None
                self.std = None
                self.backend = "onnxruntime"
                self.device = "cpu"
                self.det_mode = "human"
                self.nms_thr = 0.45
                self.score_thr = 0.7
                self.session = ort.InferenceSession(
                    model_path, sess_options=make_options(), providers=["CPUExecutionProvider"]
                )

        class _RTMPose(RTMPose):
            def __init__(self, model_path: str):
                self.onnx_model = model_path
                self.model_input_size = tuple(pose_entry.input_wh)  # RTMLib expects (W,H).
                self.mean = (123.675, 116.28, 103.53)
                self.std = (58.395, 57.12, 57.375)
                self.backend = "onnxruntime"
                self.device = "cpu"
                self.to_openpose = False
                self.session = ort.InferenceSession(
                    model_path, sess_options=make_options(), providers=["CPUExecutionProvider"]
                )

        self.detector = _YOLOX(det_path)
        self.pose_model = _RTMPose(pose_path)

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        boxes = np.asarray(self.detector(image_bgr), dtype=np.float32)
        if boxes.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        return boxes.reshape(-1, 4)

    def pose(self, image_bgr: np.ndarray, bboxes_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boxes = np.asarray(bboxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            # Critical: never call RTMPose with an empty list; upstream interprets it as whole image.
            return np.empty((0, 17, 2), np.float32), np.empty((0, 17), np.float32)
        keypoints, scores = self.pose_model(image_bgr, bboxes=boxes.tolist())
        return np.asarray(keypoints, np.float32), np.asarray(scores, np.float32)
