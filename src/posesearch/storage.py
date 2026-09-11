from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .codec import decode_geometry, dumps_json, encode_array, encode_geometry
from .constants import FEATURE_VERSION, KEYPOINT_SCHEMA
from .types import PoseGeometry, PoseRecord


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime | None = None) -> str:
    return (dt or utcnow()).isoformat()


class Repository:
    def __init__(self, sqlite_path: str):
        self.sqlite_path = str(sqlite_path)
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.sqlite_path, timeout=5.0, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA busy_timeout=5000")
        return con

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        con = self.connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            yield con
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.close()

    def migrate(self) -> None:
        sql = resources.files("posesearch.migrations").joinpath("0001_init.sql").read_text(encoding="utf-8")
        con = self.connect()
        try:
            con.executescript(sql)
        finally:
            con.close()

    def schema_version(self) -> str | None:
        con = self.connect()
        try:
            row = con.execute("SELECT value FROM schema_meta WHERE key='db_schema_version'").fetchone()
            return row[0] if row else None
        finally:
            con.close()

    def upsert_asset(
        self,
        *,
        collection_id: str,
        external_id: str,
        root_id: str,
        relative_path: str,
        source_revision: str | None,
        character_id: str | None,
        album_id: str | None,
        work_id: str | None,
        metadata: dict[str, Any] | None,
        size: int | None,
        mtime_ns: int | None,
    ) -> tuple[int, bool]:
        """Return (asset_version, changed). source_revision is opaque, never ordered."""
        metadata_json = dumps_json(metadata or {})
        with self.transaction() as con:
            row = con.execute(
                "SELECT * FROM assets WHERE collection_id=? AND external_id=?",
                (collection_id, external_id),
            ).fetchone()
            if row is None:
                con.execute(
                    """INSERT INTO assets(
                    collection_id, external_id, root_id, relative_path, source_revision,
                    asset_version, character_id, album_id, work_id, metadata_json, size, mtime_ns, deleted_at
                    ) VALUES(?,?,?,?,?,1,?,?,?,?,?,?,NULL)""",
                    (collection_id, external_id, root_id, relative_path, source_revision,
                     character_id, album_id, work_id, metadata_json, size, mtime_ns),
                )
                version = 1
                changed = True
            else:
                same = (
                    row["root_id"] == root_id and row["relative_path"] == relative_path
                    and row["source_revision"] == source_revision
                    and row["character_id"] == character_id and row["album_id"] == album_id
                    and row["work_id"] == work_id and row["metadata_json"] == metadata_json
                    and row["size"] == size and row["mtime_ns"] == mtime_ns
                    and row["deleted_at"] is None
                )
                if same:
                    return int(row["asset_version"]), False
                version = int(row["asset_version"]) + 1
                con.execute(
                    """UPDATE assets SET root_id=?, relative_path=?, source_revision=?, asset_version=?,
                    character_id=?, album_id=?, work_id=?, metadata_json=?, size=?, mtime_ns=?,
                    deleted_at=NULL, updated_at=CURRENT_TIMESTAMP
                    WHERE collection_id=? AND external_id=?""",
                    (root_id, relative_path, source_revision, version, character_id, album_id,
                     work_id, metadata_json, size, mtime_ns, collection_id, external_id),
                )
                changed = True
            con.execute(
                "INSERT INTO changes(entity_type, entity_id, operation) VALUES('asset', ?, 'upsert')",
                (f"{collection_id}:{external_id}",),
            )
            return version, changed

    def get_asset(self, collection_id: str, external_id: str) -> sqlite3.Row | None:
        con = self.connect()
        try:
            return con.execute(
                "SELECT * FROM assets WHERE collection_id=? AND external_id=?",
                (collection_id, external_id),
            ).fetchone()
        finally:
            con.close()

    def delete_asset(self, collection_id: str, external_id: str) -> bool:
        with self.transaction() as con:
            row = con.execute(
                "SELECT asset_version, deleted_at FROM assets WHERE collection_id=? AND external_id=?",
                (collection_id, external_id),
            ).fetchone()
            if row is None or row["deleted_at"] is not None:
                return False
            con.execute(
                """UPDATE assets SET deleted_at=?, asset_version=asset_version+1,
                active_extraction_id=NULL, updated_at=CURRENT_TIMESTAMP
                WHERE collection_id=? AND external_id=?""",
                (iso(), collection_id, external_id),
            )
            con.execute(
                "INSERT INTO changes(entity_type, entity_id, operation) VALUES('asset', ?, 'delete')",
                (f"{collection_id}:{external_id}",),
            )
            return True

    def enqueue_job(
        self,
        job_type: str,
        payload: dict[str, Any],
        *,
        priority: int = 100,
        source_revision: str | None = None,
        asset_version: int | None = None,
        max_attempts: int = 3,
    ) -> str:
        job_id = str(uuid.uuid4())
        con = self.connect()
        try:
            con.execute(
                """INSERT INTO jobs(job_id, job_type, priority, status, payload_json,
                source_revision, asset_version, max_attempts) VALUES(?,?,?,'queued',?,?,?,?)""",
                (job_id, job_type, priority, dumps_json(payload), source_revision, asset_version, max_attempts),
            )
        finally:
            con.close()
        return job_id

    def claim_job(self, lease_seconds: int = 120) -> sqlite3.Row | None:
        now = utcnow()
        lease_until = now + timedelta(seconds=lease_seconds)
        with self.transaction() as con:
            # Recover expired leases before claiming.
            con.execute(
                """UPDATE jobs SET status='queued', lease_expires_at=NULL, heartbeat_at=NULL,
                updated_at=CURRENT_TIMESTAMP
                WHERE status='running' AND lease_expires_at IS NOT NULL AND lease_expires_at < ?
                AND attempts < max_attempts""",
                (iso(now),),
            )
            row = con.execute(
                """SELECT * FROM jobs WHERE status='queued'
                AND (next_run_at IS NULL OR next_run_at <= ?)
                AND attempts < max_attempts
                ORDER BY priority ASC, created_at ASC LIMIT 1""",
                (iso(now),),
            ).fetchone()
            if row is None:
                return None
            con.execute(
                """UPDATE jobs SET status='running', attempts=attempts+1,
                lease_expires_at=?, heartbeat_at=?, updated_at=CURRENT_TIMESTAMP
                WHERE job_id=? AND status='queued'""",
                (iso(lease_until), iso(now), row["job_id"]),
            )
            return con.execute("SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)).fetchone()

    def heartbeat_job(self, job_id: str, lease_seconds: int = 120) -> None:
        now = utcnow()
        con = self.connect()
        try:
            con.execute(
                "UPDATE jobs SET heartbeat_at=?, lease_expires_at=? WHERE job_id=? AND status='running'",
                (iso(now), iso(now + timedelta(seconds=lease_seconds)), job_id),
            )
        finally:
            con.close()

    def finish_job(self, job_id: str, result: dict[str, Any] | None = None) -> None:
        con = self.connect()
        try:
            con.execute(
                """UPDATE jobs SET status='succeeded', result_json=?, error_code=NULL,
                error_message=NULL, lease_expires_at=NULL, updated_at=CURRENT_TIMESTAMP WHERE job_id=?""",
                (dumps_json(result or {}), job_id),
            )
        finally:
            con.close()

    def fail_job(self, job_id: str, error_code: str, message: str, retryable: bool = True) -> None:
        with self.transaction() as con:
            row = con.execute("SELECT attempts, max_attempts FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return
            can_retry = retryable and int(row["attempts"]) < int(row["max_attempts"])
            if can_retry:
                delay = min(300, 2 ** max(0, int(row["attempts"]) - 1) * 5)
                next_run = utcnow() + timedelta(seconds=delay)
                status = "queued"
            else:
                next_run = None
                status = "failed"
            con.execute(
                """UPDATE jobs SET status=?, next_run_at=?, error_code=?, error_message=?,
                lease_expires_at=NULL, updated_at=CURRENT_TIMESTAMP WHERE job_id=?""",
                (status, iso(next_run) if next_run else None, error_code, message[:1000], job_id),
            )

    def get_job(self, job_id: str) -> sqlite3.Row | None:
        con = self.connect()
        try:
            return con.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        finally:
            con.close()

    def save_extraction(
        self,
        *,
        collection_id: str,
        external_id: str,
        expected_asset_version: int,
        extraction_id: str,
        content_sha256: str,
        image_width: int,
        image_height: int,
        status: str,
        diagnostics: dict[str, Any],
        persons: list[dict[str, Any]],
        detector_hash: str | None = None,
        pose_hash: str | None = None,
        preprocess_version: str = "preprocess-v1",
    ) -> bool:
        """Commit extraction only if the asset version still matches (CAS)."""
        with self.transaction() as con:
            asset = con.execute(
                "SELECT asset_version, deleted_at FROM assets WHERE collection_id=? AND external_id=?",
                (collection_id, external_id),
            ).fetchone()
            if asset is None or asset["deleted_at"] is not None or int(asset["asset_version"]) != expected_asset_version:
                return False
            con.execute(
                """INSERT OR REPLACE INTO extractions(extraction_id, content_sha256, detector_hash,
                pose_hash, preprocess_version, keypoint_schema, image_width, image_height, status, diagnostics_json)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (extraction_id, content_sha256, detector_hash, pose_hash, preprocess_version,
                 KEYPOINT_SCHEMA, image_width, image_height, status, dumps_json(diagnostics)),
            )
            con.execute("DELETE FROM persons WHERE extraction_id=?", (extraction_id,))
            for p in persons:
                con.execute(
                    """INSERT INTO persons(person_id, extraction_id, person_index, bbox_blob,
                    keypoints_blob, raw_scores_blob, valid_mask_blob, geometry_blob, quality,
                    feature_version, available_scopes, diagnostics_json)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        p["person_id"], extraction_id, int(p["person_index"]),
                        encode_array(np.asarray(p["bbox"], np.float32)),
                        encode_array(np.asarray(p["keypoints"], np.float32)),
                        encode_array(np.asarray(p["scores"], np.float32)),
                        encode_array(np.asarray(p["valid_mask"], bool)),
                        encode_geometry(p["geometry"]), p["quality"], FEATURE_VERSION,
                        dumps_json(list(p.get("available_scopes", []))),
                        dumps_json(p.get("diagnostics", {})),
                    ),
                )
            con.execute(
                """UPDATE assets SET content_sha256=?, active_extraction_id=?, updated_at=CURRENT_TIMESTAMP
                WHERE collection_id=? AND external_id=?""",
                (content_sha256, extraction_id, collection_id, external_id),
            )
            con.execute(
                "INSERT INTO changes(entity_type, entity_id, operation) VALUES('extraction', ?, 'upsert')",
                (extraction_id,),
            )
            return True

    def load_pose_records(self, collection_id: str | None = None) -> list[PoseRecord]:
        sql = """SELECT a.collection_id,a.external_id,a.character_id,a.album_id,a.work_id,a.content_sha256,
                 p.person_id,p.quality,p.geometry_blob
                 FROM assets a JOIN persons p ON p.extraction_id=a.active_extraction_id
                 WHERE a.deleted_at IS NULL AND p.feature_version=? AND p.quality!='low_quality'"""
        params: list[Any] = [FEATURE_VERSION]
        if collection_id is not None:
            sql += " AND a.collection_id=?"
            params.append(collection_id)
        con = self.connect()
        try:
            rows = con.execute(sql, params).fetchall()
        finally:
            con.close()
        return [
            PoseRecord(
                person_id=r["person_id"], collection_id=r["collection_id"], external_id=r["external_id"],
                geometry=decode_geometry(r["geometry_blob"]), quality=r["quality"],
                character_id=r["character_id"], album_id=r["album_id"], work_id=r["work_id"],
                content_sha256=r["content_sha256"],
            )
            for r in rows
        ]

    def get_asset_pose_records(self, collection_id: str, external_id: str) -> list[PoseRecord]:
        return [r for r in self.load_pose_records(collection_id) if r.external_id == external_id]

    def latest_change_sequence(self) -> int:
        con = self.connect()
        try:
            row = con.execute("SELECT COALESCE(MAX(sequence),0) FROM changes").fetchone()
            return int(row[0])
        finally:
            con.close()

    def stats(self) -> dict[str, Any]:
        con = self.connect()
        try:
            assets = int(con.execute("SELECT COUNT(*) FROM assets WHERE deleted_at IS NULL").fetchone()[0])
            persons = int(con.execute("SELECT COUNT(*) FROM persons").fetchone()[0])
            queued = int(con.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0])
            failed = int(con.execute("SELECT COUNT(*) FROM jobs WHERE status='failed'").fetchone()[0])
            return {"assets": assets, "persons": persons, "queued_jobs": queued, "failed_jobs": failed}
        finally:
            con.close()
