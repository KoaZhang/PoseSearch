from __future__ import annotations

import json
import os
import time
import traceback

from .config import Settings
from .inference import RTMLibRuntime
from .pipeline import PosePipeline
from .preprocess import ImageInputError
from .security import resolve_library_path
from .storage import Repository


def run_worker() -> None:
    settings = Settings.load()
    settings.ensure_dirs()
    repo = Repository(settings.storage.sqlite_path)
    repo.migrate()
    lock_path = os.getenv("POSESEARCH_MODEL_LOCK", "models.lock.json")
    runtime = RTMLibRuntime(settings.models_dir, lock_path, settings.runtime)
    pipeline = PosePipeline(runtime, settings.pipeline)

    while True:
        job = repo.claim_job()
        if job is None:
            time.sleep(0.5)
            continue
        job_id = job["job_id"]
        try:
            payload = json.loads(job["payload_json"])
            if job["job_type"] != "asset_extract":
                repo.fail_job(job_id, "UNKNOWN_JOB_TYPE", job["job_type"], retryable=False)
                continue
            root_id = payload["root_id"]
            if root_id not in settings.roots:
                repo.fail_job(job_id, "UNKNOWN_ROOT", root_id, retryable=False)
                continue
            source_path = resolve_library_path(settings.roots[root_id], payload["relative_path"])
            result = pipeline.analyze_file(source_path)
            committed = repo.save_extraction(
                collection_id=payload["collection_id"], external_id=payload["external_id"],
                expected_asset_version=int(job["asset_version"]), extraction_id=result.extraction_id,
                content_sha256=result.content_sha256, image_width=result.image_width,
                image_height=result.image_height, status=result.status,
                diagnostics={**result.diagnostics, "persons_truncated": result.persons_truncated,
                             "coordinate_space": result.coordinate_space}, persons=result.persons,
            )
            if not committed:
                repo.finish_job(job_id, {"discarded": True, "reason": "stale_asset_version"})
            else:
                repo.finish_job(job_id, {
                    "extraction_id": result.extraction_id, "status": result.status,
                    "person_count": len(result.persons), "persons_truncated": result.persons_truncated,
                })
        except ImageInputError as exc:
            repo.fail_job(job_id, getattr(exc, "code", "INVALID_IMAGE"), str(exc), retryable=False)
        except Exception as exc:  # worker boundary; internal trace stays server-side
            traceback.print_exc()
            repo.fail_job(job_id, "PROCESSING_FAILED", str(exc), retryable=True)


if __name__ == "__main__":
    run_worker()
