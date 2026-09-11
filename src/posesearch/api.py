from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, model_validator

from .config import Settings
from .preprocess import ImageInputError, decode_image
from .search_service import SearchService
from .security import PathSecurityError, resolve_library_path
from .temporary_storage import PoseRepository
from .visualize import encode_pose_overlay


class SourceRef(BaseModel):
    root_id: str
    relative_path: str


class AssetMetadata(BaseModel):
    character_id: str | None = None
    album_id: str | None = None
    work_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class AssetUpsertItem(BaseModel):
    external_id: str
    source: SourceRef
    source_revision: str | None = None
    metadata: AssetMetadata = Field(default_factory=AssetMetadata)


class BatchUpsertRequest(BaseModel):
    collection_id: str
    items: list[AssetUpsertItem] = Field(min_length=1, max_length=100)


class SearchQueryRef(BaseModel):
    external_id: str | None = None
    person_id: str | None = None
    analysis_id: str | None = None

    @model_validator(mode="after")
    def one_identity(self):
        if bool(self.external_id) == bool(self.analysis_id):
            raise ValueError("provide exactly one of external_id or analysis_id")
        return self


class SearchFilters(BaseModel):
    exclude_character_id: str | None = None


class Diversify(BaseModel):
    max_per_character: int | None = Field(default=None, ge=1)
    max_per_album: int | None = Field(default=None, ge=1)


class SearchRequest(BaseModel):
    collection_id: str
    query: SearchQueryRef
    scope: Literal["full", "upper", "lower", "auto"] = "auto"
    mirror: Literal["equivalent", "strict"] = "equivalent"
    top_k: int = Field(default=30, ge=1, le=100)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    diversify: Diversify = Field(default_factory=Diversify)


def error(code: str, message: str, http_status: int = 400) -> HTTPException:
    return HTTPException(status_code=http_status, detail={"error": {"code": code, "message": message}})


def _cleanup_expired_temp_files(repo: PoseRepository, settings: Settings) -> None:
    for row in repo.cleanup_expired_temporary_analyses():
        upload_path = row.get("upload_path")
        if upload_path:
            try:
                Path(upload_path).unlink(missing_ok=True)
            except OSError:
                pass
        overlay_dir = Path(settings.storage.temp_dir) / "overlays" / str(row["analysis_id"])
        if overlay_dir.is_dir():
            for child in overlay_dir.iterdir():
                try:
                    child.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                overlay_dir.rmdir()
            except OSError:
                pass


def _scale_person_to_working(person: dict[str, Any], sx: float, sy: float) -> dict[str, Any]:
    out = dict(person)
    bbox = np.asarray(person["bbox"], dtype=np.float32).copy()
    bbox[[0, 2]] /= sx
    bbox[[1, 3]] /= sy
    keypoints = np.asarray(person["keypoints"], dtype=np.float32).copy()
    keypoints[:, 0] /= sx
    keypoints[:, 1] /= sy
    out["bbox"] = bbox
    out["keypoints"] = keypoints
    return out


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.load()
    settings.ensure_dirs()
    repo = PoseRepository(settings.storage.sqlite_path)
    search_service = SearchService(repo, settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repo.migrate()
        _cleanup_expired_temp_files(repo, settings)
        search_service.refresh_if_needed()
        app.state.repo = repo
        app.state.settings = settings
        app.state.search_service = search_service
        yield

    app = FastAPI(title="PoseSearch", version="0.2.0", lifespan=lifespan)

    def authorize(collection_id: str, x_api_key: str | None) -> None:
        if not x_api_key or x_api_key not in settings.auth.api_keys:
            raise error("UNAUTHORIZED", "invalid API key", status.HTTP_401_UNAUTHORIZED)
        allowed = settings.auth.api_keys[x_api_key]
        if "*" not in allowed and collection_id not in allowed:
            raise error("FORBIDDEN_COLLECTION", "API key cannot access this collection", status.HTTP_403_FORBIDDEN)

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready():
        try:
            schema = repo.schema_version()
            if schema is None:
                raise RuntimeError("schema missing")
            return {
                "status": "ready",
                "db_schema_version": schema,
                "index_revision": search_service.index_revision,
                "indexed_persons": search_service.searcher.size,
            }
        except Exception as exc:
            raise error("NOT_READY", str(exc), status.HTTP_503_SERVICE_UNAVAILABLE) from exc

    @app.post("/v1/assets:batch-upsert", status_code=202)
    def batch_upsert(req: BatchUpsertRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        authorize(req.collection_id, x_api_key)
        results = []
        for item in req.items:
            if item.source.root_id not in settings.roots:
                results.append({"external_id": item.external_id, "error": {"code": "UNKNOWN_ROOT"}})
                continue
            try:
                source_path = resolve_library_path(settings.roots[item.source.root_id], item.source.relative_path)
            except (PathSecurityError, FileNotFoundError) as exc:
                results.append({"external_id": item.external_id, "error": {"code": "INVALID_SOURCE_PATH", "message": str(exc)}})
                continue
            st = source_path.stat()
            version, changed = repo.upsert_asset(
                collection_id=req.collection_id,
                external_id=item.external_id,
                root_id=item.source.root_id,
                relative_path=item.source.relative_path,
                source_revision=item.source_revision,
                character_id=item.metadata.character_id,
                album_id=item.metadata.album_id,
                work_id=item.metadata.work_id,
                metadata=item.metadata.extra,
                size=st.st_size,
                mtime_ns=st.st_mtime_ns,
            )
            if not changed:
                results.append({"external_id": item.external_id, "unchanged": True})
                continue
            job_id = repo.enqueue_job(
                "asset_extract",
                {
                    "collection_id": req.collection_id,
                    "external_id": item.external_id,
                    "root_id": item.source.root_id,
                    "relative_path": item.source.relative_path,
                },
                source_revision=item.source_revision,
                asset_version=version,
            )
            results.append({"external_id": item.external_id, "job_id": job_id, "asset_version": version})
        return {"items": results}

    @app.post("/v1/analyses", status_code=202)
    async def create_analysis(
        collection_id: str = Form(...),
        image: UploadFile = File(...),
        include_overlay: bool = Form(True),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ):
        authorize(collection_id, x_api_key)
        _cleanup_expired_temp_files(repo, settings)
        analysis_id = str(uuid.uuid4())
        upload_dir = Path(settings.storage.temp_dir) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / f"{analysis_id}.upload"
        total = 0
        try:
            with upload_path.open("wb") as dst:
                while True:
                    chunk = await image.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > settings.pipeline.max_file_bytes:
                        raise error(
                            "IMAGE_LIMIT_EXCEEDED",
                            f"upload exceeds {settings.pipeline.max_file_bytes} bytes",
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        )
                    dst.write(chunk)
            if total == 0:
                raise error("INVALID_IMAGE", "empty upload", 400)
            repo.create_temporary_analysis(
                analysis_id=analysis_id,
                collection_id=collection_id,
                upload_path=str(upload_path),
                ttl_seconds=settings.storage.temporary_analysis_ttl_seconds,
            )
            job_id = repo.enqueue_job(
                "temporary_analysis",
                {
                    "analysis_id": analysis_id,
                    "collection_id": collection_id,
                    "upload_path": str(upload_path),
                    "include_overlay": include_overlay,
                },
                priority=10,
                max_attempts=3,
            )
        except HTTPException:
            upload_path.unlink(missing_ok=True)
            raise
        except Exception:
            upload_path.unlink(missing_ok=True)
            raise
        finally:
            await image.close()
        return {"job_id": job_id}

    @app.get("/v1/analyses/{analysis_id}")
    def get_analysis(analysis_id: str, collection_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        authorize(collection_id, x_api_key)
        row = repo.get_temporary_analysis(collection_id, analysis_id)
        if row is None:
            raise error("ANALYSIS_NOT_FOUND", "analysis not found or expired", 404)
        return {
            "analysis_id": row["analysis_id"], "collection_id": row["collection_id"],
            "status": row["status"], "extraction_id": row["extraction_id"], "expires_at": row["expires_at"],
        }

    @app.get("/v1/assets/{external_id}")
    def get_asset(external_id: str, collection_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        authorize(collection_id, x_api_key)
        row = repo.get_asset(collection_id, external_id)
        if row is None or row["deleted_at"] is not None:
            raise error("ASSET_NOT_FOUND", "asset not found", 404)
        return {
            "collection_id": collection_id, "external_id": external_id, "root_id": row["root_id"],
            "relative_path": row["relative_path"], "source_revision": row["source_revision"],
            "asset_version": row["asset_version"], "active_extraction_id": row["active_extraction_id"],
            "character_id": row["character_id"], "album_id": row["album_id"], "work_id": row["work_id"],
        }

    @app.delete("/v1/assets/{external_id}")
    def delete_asset(external_id: str, collection_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        authorize(collection_id, x_api_key)
        repo.delete_asset(collection_id, external_id)
        search_service.refresh_if_needed()
        return {"deleted": True}

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        row = repo.get_job(job_id)
        if row is None:
            raise error("JOB_NOT_FOUND", "job not found", 404)
        payload = json.loads(row["payload_json"])
        collection_id = payload.get("collection_id")
        if collection_id:
            authorize(collection_id, x_api_key)
        elif not x_api_key or x_api_key not in settings.auth.api_keys:
            raise error("UNAUTHORIZED", "invalid API key", 401)
        return {
            "job_id": row["job_id"], "job_type": row["job_type"], "status": row["status"],
            "attempts": row["attempts"], "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": ({"code": row["error_code"], "message": row["error_message"]} if row["error_code"] else None),
        }

    @app.post("/v1/search")
    def search(req: SearchRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        authorize(req.collection_id, x_api_key)
        try:
            if req.query.analysis_id is not None:
                query, effective_scope, reason, hits = search_service.query_analysis(
                    collection_id=req.collection_id, analysis_id=req.query.analysis_id,
                    person_id=req.query.person_id, scope=req.scope, mirror=req.mirror, top_k=req.top_k,
                    exclude_character_id=req.filters.exclude_character_id,
                    max_per_character=req.diversify.max_per_character, max_per_album=req.diversify.max_per_album,
                )
                query_identity = {"analysis_id": req.query.analysis_id}
            else:
                assert req.query.external_id is not None
                query, effective_scope, reason, hits = search_service.query_asset(
                    collection_id=req.collection_id, external_id=req.query.external_id,
                    person_id=req.query.person_id, scope=req.scope, mirror=req.mirror, top_k=req.top_k,
                    exclude_character_id=req.filters.exclude_character_id,
                    max_per_character=req.diversify.max_per_character, max_per_album=req.diversify.max_per_album,
                )
                query_identity = {"external_id": req.query.external_id}
        except ValueError as exc:
            if str(exc) == "QUERY_POSE_NOT_USABLE":
                raise error("QUERY_POSE_NOT_USABLE", "query has no usable pose", 422) from exc
            if str(exc) == "ANALYSIS_NOT_FOUND":
                raise error("ANALYSIS_NOT_FOUND", "analysis not found or expired", 404) from exc
            raise
        return {
            "query": {**query_identity, "person_id": query.person_id, "effective_scope": effective_scope, "scope_reason": reason},
            "metric_version": search_service.searcher.metric_version, "index_revision": search_service.index_revision,
            "results": [
                {
                    "external_id": h.external_id, "person_id": h.person_id, "similarity": h.similarity,
                    "distance": h.distance, "coverage": h.coverage, "matched_joint_count": h.matched_joint_count,
                    "mirror_applied": h.mirror_applied, "character_id": h.character_id,
                    "album_id": h.album_id, "work_id": h.work_id,
                }
                for h in hits
            ],
        }

    @app.get("/v1/persons/{person_id}/keypoints")
    def person_keypoints(person_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        context = repo.get_person_context(person_id)
        if context is None:
            raise error("PERSON_NOT_FOUND", "person not found", 404)
        authorize(str(context["collection_id"]), x_api_key)
        return {
            "person_id": person_id, "collection_id": context["collection_id"],
            "external_id": context["external_id"], "analysis_id": context["analysis_id"],
            "coordinate_space": context["extraction_diagnostics"].get("coordinate_space", "exif_oriented_pixels"),
            "image_width": context["image_width"], "image_height": context["image_height"],
            "bbox_xyxy": np.asarray(context["bbox"], dtype=np.float32).tolist(),
            "keypoints_coco17_xy": np.asarray(context["keypoints"], dtype=np.float32).tolist(),
            "scores": np.asarray(context["scores"], dtype=np.float32).tolist(),
            "valid_mask": np.asarray(context["valid_mask"], dtype=bool).tolist(),
            "quality": context["quality"], "available_scopes": context["available_scopes"],
            "diagnostics": context["diagnostics"],
        }

    @app.get("/v1/persons/{person_id}/overlay", response_class=Response)
    def person_overlay(person_id: str, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        context = repo.get_person_context(person_id)
        if context is None:
            raise error("PERSON_NOT_FOUND", "person not found", 404)
        authorize(str(context["collection_id"]), x_api_key)
        if context["analysis_id"]:
            overlay_path = Path(settings.storage.temp_dir) / "overlays" / str(context["analysis_id"]) / f"{person_id}.jpg"
            if not overlay_path.is_file():
                raise error("OVERLAY_NOT_AVAILABLE", "overlay was not requested or has expired", 404)
            return FileResponse(overlay_path, media_type="image/jpeg", filename=f"{person_id}.jpg")

        root_id = context["root_id"]
        if not root_id or root_id not in settings.roots or not context["relative_path"]:
            raise error("OVERLAY_SOURCE_UNAVAILABLE", "asset source is unavailable", 404)
        try:
            source_path = resolve_library_path(settings.roots[root_id], context["relative_path"])
            decoded = decode_image(
                source_path, max_file_bytes=settings.pipeline.max_file_bytes,
                max_pixels=settings.pipeline.max_pixels, max_long_edge=settings.pipeline.analysis_max_long_edge,
            )
        except (PathSecurityError, FileNotFoundError, ImageInputError) as exc:
            raise error("OVERLAY_SOURCE_UNAVAILABLE", str(exc), 404) from exc
        person = _scale_person_to_working(
            {"bbox": context["bbox"], "keypoints": context["keypoints"], "scores": context["scores"],
             "valid_mask": context["valid_mask"], "quality": context["quality"]},
            decoded.scale_x_to_original, decoded.scale_y_to_original,
        )
        return Response(content=encode_pose_overlay(decoded.bgr, [person]), media_type="image/jpeg")

    @app.get("/v1/stats")
    def stats(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
        if not x_api_key or x_api_key not in settings.auth.api_keys:
            raise error("UNAUTHORIZED", "invalid API key", 401)
        search_service.refresh_if_needed()
        return {**repo.stats(), "index_revision": search_service.index_revision, "index_persons": search_service.searcher.size}

    return app


app = create_app()
