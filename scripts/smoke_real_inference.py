#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from posesearch.config import PipelineConfig, RuntimeConfig
from posesearch.inference import RTMLibRuntime
from posesearch.pipeline import PosePipeline
from posesearch.visualize import save_pose_overlay


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--lock", default="models.lock.json")
    p.add_argument("--overlay")
    args = p.parse_args()

    runtime = RTMLibRuntime(args.models_dir, args.lock, RuntimeConfig(intra_op_threads=2, inter_op_threads=1))
    pipeline = PosePipeline(runtime, PipelineConfig())
    result = pipeline.analyze_file(args.image)
    usable = [p for p in result.persons if p["quality"] != "low_quality"]
    summary = {
        "status": result.status,
        "image_width": result.image_width,
        "image_height": result.image_height,
        "person_count": len(result.persons),
        "usable_person_count": len(usable),
        "qualities": [p["quality"] for p in result.persons],
        "available_scopes": [p["available_scopes"] for p in result.persons],
    }
    print(json.dumps(summary, indent=2))
    if not result.persons:
        raise SystemExit("real-model smoke failed: detector returned no persons")
    if not usable:
        raise SystemExit("real-model smoke failed: no usable pose")

    if args.overlay:
        decoded = pipeline.decode_for_overlay(args.image)
        people = []
        for person in result.persons:
            p2 = dict(person)
            bbox = np.asarray(person["bbox"], np.float32).copy()
            bbox[[0, 2]] /= decoded.scale_x_to_original
            bbox[[1, 3]] /= decoded.scale_y_to_original
            kpts = np.asarray(person["keypoints"], np.float32).copy()
            kpts[:, 0] /= decoded.scale_x_to_original
            kpts[:, 1] /= decoded.scale_y_to_original
            p2["bbox"] = bbox
            p2["keypoints"] = kpts
            people.append(p2)
        save_pose_overlay(Path(args.overlay), decoded.bgr, people)
        print(f"overlay={args.overlay}")


if __name__ == "__main__":
    main()
