from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from .config import Settings
from .inference import RTMLibRuntime
from .pipeline import AnalysisResult, PosePipeline
from .preprocess import ImageInputError
from .security import resolve_library_path
from .temporary_storage import PoseRepository
from .visualize import save_pose_overlay


def _model_hashes(runtime: RTMLibRuntime) -> tuple[str | None, str | None]:
    return (runtime.lock.models["detector"].onnx_sha256, runtime.lock.models["pose"].onnx_sha256)


def _person_summaries(result: AnalysisResult) -> list[dict[str, object]]:
    return [
        {
            "person_id": p["person_id"], "person_index": p["person_index"],
            "quality": p["quality"], "available_scopes": p.get("available_scopes", []),
        }
        for p in result.persons
    ]


def _working_space_persons(pipeline: PosePipeline, source_path: Path, result: AnalysisResult):
    decoded = pipeline.decode_for_overlay(source_path)
    sx = decoded.scale_x_to_original
    sy = decoded.scale_y_to_original
    persons = []
    for person in result.persons:
        p = dict(person)
        bbox = np.asarray(person["bbox"], np.float32).copy()
        bbox[[0, 2]] /= sx
        bbox[[1, 3]] /= sy
        keypoints = np.asarray(person["keypoints"], np.float32).copy()
        keypoints[:, 0] /= sx
        keypoints[:, 1] /= sy
        p["bbox"] = bbox
        p["keypoints"] = keypoints
        persons.append(p)
    return decoded.bgr, persons


def _remove_file(path: str | Path | None) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _cleanup_expired(repo: PoseRepository, settings: Settings) -> None:
    for row in repo.cleanup_expired_temporary_analyses():
        _remove_file(row.get("upload_path"))
        overlay_dir = Path(settings.storage.temp_dir) / "overlays" / str(row["analysis_id"])
        if overlay_dir.is_dir():
            for child in overlay_dir.iterdir():
                _remove_file(child)
            try:
                overlay_dir.rmdir()
            except OSError:
                pass


def _process_asset(
    repo: PoseRepository, pipeline: PosePipeline, runtime: RTMLibRuntime,
    settings: Settings, job, payload: dict[str, object],
) -> None:
    root_id = str(payload["root_id"])
    if root_id not in settings.roots:
        repo.fail_job(job["job_id"], "UNKNOWN_ROOT", root_id, retryable=False)
        return
    source_path = resolve_library_path(settings.roots[root_id], str(payload["relative_path"]))
    result = pipeline.analyze_file(source_path)
    detector_hash, pose_hash = _model_hashes(runtime)
    committed = repo.save_extraction(
        collection_id=str(payload["collection_id"]), external_id=str(payload["external_id"]),
        expected_asset_version=int(job["asset_version"]), extraction_id=result.extraction_id,
        content_sha256=result.content_sha256, image_width=result.image_width,
        image_height=result.image_height, status=result.status,
        diagnostics={**result.diagnostics, "persons_truncated": result.persons_truncated,
                     "coordinate_space": result.coordinate_space}, persons=result.persons,
        detector_hash=detector_hash, pose_hash=pose_hash,
    )
    if not committed:
        repo.finish_job(job["job_id"], {"discarded": True, "reason": "stale_asset_version"})
    else:
        repo.finish_job(job["job_id"], {
            "extraction_id": result.extraction_id, "status": result.status,
            "person_count": len(result.persons), "persons_truncated": result.persons_truncated,
            "persons": _person_summaries(result),
        })


def _process_temporary_analysis(
    repo: PoseRepository, pipeline: PosePipeline, runtime: RTMLibRuntime,
    settings: Settings, job, payload: dict[str, object],
) -> None:
    analysis_id = str(payload["analysis_id"])
    collection_id = str(payload["collection_id"])
    upload_path = Path(str(payload["upload_path"]))
    include_overlay = bool(payload.get("include_overlay", True))
    repo.mark_temporary_analysis_running(analysis_id)

    result = pipeline.analyze_file(upload_path)
    overlay_paths: list[Path] = []
    if include_overlay:
        image_bgr, overlay_people = _working_space_persons(pipeline, upload_path, result)
        for person in overlay_people:
            overlay_path = Path(settings.storage.temp_dir) / "overlays" / analysis_id / f"{person['person_id']}.jpg"
            save_pose_overlay(overlay_path, image_bgr, [person])
            overlay_paths.append(overlay_path)

    detector_hash, pose_hash = _model_hashes(runtime)
    committed = repo.save_temporary_extraction(
        analysis_id=analysis_id, collection_id=collection_id,
        extraction_id=result.extraction_id, content_sha256=result.content_sha256,
        image_width=result.image_width, image_height=result.image_height, status=result.status,
        diagnostics={**result.diagnostics, "persons_truncated": result.persons_truncated,
                     "coordinate_space": result.coordinate_space}, persons=result.persons,
        detector_hash=detector_hash, pose_hash=pose_hash,
    )
    if not committed:
        for overlay_path in overlay_paths:
            _remove_file(overlay_path)
        repo.finish_job(job["job_id"], {"discarded": True, "reason": "analysis_expired"})
        _remove_file(upload_path)
        return

    _remove_file(upload_path)
    repo.finish_job(job["job_id"], {
        "analysis_id": analysis_id, "extraction_id": result.extraction_id,
        "status": result.status, "person_count": len(result.persons),
        "persons_truncated": result.persons_truncated, "overlay_available": bool(overlay_paths),
        "persons": _person_summaries(result),
    })


def run_worker() -> None:
    settings = Settings.load()
    settings.ensure_dirs()
    repo = PoseRepository(settings.storage.sqlite_path)
    repo.migrate()
    lock_path = os.getenv("POSESEARCH_MODEL_LOCK", "models.lock.json")
    runtime = RTMLibRuntime(settings.models_dir, lock_path, settings.runtime)
    pipeline = PosePipeline(runtime, settings.pipeline)

    last_cleanup = 0.0
    while True:
        if time.monotonic() - last_cleanup >= 60.0:
            _cleanup_expired(repo, settings)
            last_cleanup = time.monotonic()
        job = repo.claim_job()
        if job is None:
            time.sleep(0.5)
            continue
        job_id = job["job_id"]
        payload = json.loads(job["payload_json"])
        try:
            if job["job_type"] == "asset_extract":
                _process_asset(repo, pipeline, runtime, settings, job, payload)
            elif job["job_type"] == "temporary_analysis":
                _process_temporary_analysis(repo, pipeline, runtime, settings, job, payload)
            else:
                repo.fail_job(job_id, "UNKNOWN_JOB_TYPE", job["job_type"], retryable=False)
        except ImageInputError as exc:
            repo.fail_job(job_id, getattr(exc, "code", "INVALID_IMAGE"), str(exc), retryable=False)
            if job["job_type"] == "temporary_analysis":
                repo.mark_temporary_analysis_failed(str(payload["analysis_id"]))
                _remove_file(payload.get("upload_path"))
        except Exception as exc:
            traceback.print_exc()
            repo.fail_job(job_id, "PROCESSING_FAILED", str(exc), retryable=True)
            if job["job_type"] == "temporary_analysis":
                updated = repo.get_job(job_id)
                if updated is not None and updated["status"] == "failed":
                    repo.mark_temporary_analysis_failed(str(payload["analysis_id"]))
                    _remove_file(payload.get("upload_path"))


if __name__ == "__main__":
    run_worker()
