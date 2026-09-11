from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import numpy as np

from .codec import decode_array, decode_geometry, dumps_json, encode_array, encode_geometry
from .constants import FEATURE_VERSION, KEYPOINT_SCHEMA
from .storage import Repository, iso, utcnow
from .types import PoseRecord


class PoseRepository(Repository):
    """Repository extensions for TTL-scoped query analyses and diagnostics."""

    def create_temporary_analysis(
        self, *, analysis_id: str, collection_id: str, upload_path: str, ttl_seconds: int,
    ) -> None:
        expires_at = utcnow() + timedelta(seconds=max(1, int(ttl_seconds)))
        con = self.connect()
        try:
            con.execute(
                """INSERT INTO temporary_analyses(
                analysis_id, collection_id, extraction_id, upload_path, status, expires_at
                ) VALUES(?,?,NULL,?,'queued',?)""",
                (analysis_id, collection_id, upload_path, iso(expires_at)),
            )
        finally:
            con.close()

    def get_temporary_analysis(self, collection_id: str, analysis_id: str):
        con = self.connect()
        try:
            return con.execute(
                """SELECT * FROM temporary_analyses
                WHERE collection_id=? AND analysis_id=? AND expires_at>?""",
                (collection_id, analysis_id, iso()),
            ).fetchone()
        finally:
            con.close()

    def mark_temporary_analysis_running(self, analysis_id: str) -> None:
        con = self.connect()
        try:
            con.execute("UPDATE temporary_analyses SET status='running' WHERE analysis_id=?", (analysis_id,))
        finally:
            con.close()

    def mark_temporary_analysis_failed(self, analysis_id: str) -> None:
        con = self.connect()
        try:
            con.execute("UPDATE temporary_analyses SET status='failed' WHERE analysis_id=?", (analysis_id,))
        finally:
            con.close()

    def save_temporary_extraction(
        self, *, analysis_id: str, collection_id: str, extraction_id: str,
        content_sha256: str, image_width: int, image_height: int, status: str,
        diagnostics: dict[str, Any], persons: list[dict[str, Any]],
        detector_hash: str | None = None, pose_hash: str | None = None,
        preprocess_version: str = "preprocess-v1",
    ) -> bool:
        with self.transaction() as con:
            temp = con.execute(
                """SELECT analysis_id FROM temporary_analyses
                WHERE analysis_id=? AND collection_id=? AND expires_at>?""",
                (analysis_id, collection_id, iso()),
            ).fetchone()
            if temp is None:
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
                """UPDATE temporary_analyses SET extraction_id=?, status=?
                WHERE analysis_id=? AND collection_id=?""",
                (extraction_id, status, analysis_id, collection_id),
            )
            return True

    def get_temporary_pose_records(self, collection_id: str, analysis_id: str) -> list[PoseRecord]:
        con = self.connect()
        try:
            rows = con.execute(
                """SELECT t.collection_id,t.analysis_id,e.content_sha256,
                p.person_id,p.quality,p.geometry_blob
                FROM temporary_analyses t
                JOIN extractions e ON e.extraction_id=t.extraction_id
                JOIN persons p ON p.extraction_id=e.extraction_id
                WHERE t.collection_id=? AND t.analysis_id=? AND t.expires_at>?
                AND p.feature_version=? AND p.quality!='low_quality'
                ORDER BY p.person_index ASC""",
                (collection_id, analysis_id, iso(), FEATURE_VERSION),
            ).fetchall()
        finally:
            con.close()
        return [
            PoseRecord(
                person_id=r["person_id"], collection_id=r["collection_id"],
                external_id=f"analysis:{r['analysis_id']}",
                geometry=decode_geometry(r["geometry_blob"]), quality=r["quality"],
                content_sha256=r["content_sha256"],
            )
            for r in rows
        ]

    def get_person_context(self, person_id: str) -> dict[str, Any] | None:
        con = self.connect()
        try:
            row = con.execute(
                """SELECT p.*, e.image_width, e.image_height, e.diagnostics_json AS extraction_diagnostics,
                a.collection_id AS asset_collection_id, a.external_id, a.root_id, a.relative_path,
                t.collection_id AS temp_collection_id, t.analysis_id, t.expires_at
                FROM persons p
                JOIN extractions e ON e.extraction_id=p.extraction_id
                LEFT JOIN assets a ON a.active_extraction_id=p.extraction_id AND a.deleted_at IS NULL
                LEFT JOIN temporary_analyses t ON t.extraction_id=p.extraction_id AND t.expires_at>?
                WHERE p.person_id=? LIMIT 1""",
                (iso(), person_id),
            ).fetchone()
        finally:
            con.close()
        if row is None:
            return None
        collection_id = row["asset_collection_id"] or row["temp_collection_id"]
        if collection_id is None:
            return None
        return {
            "person_id": row["person_id"], "extraction_id": row["extraction_id"],
            "person_index": int(row["person_index"]), "collection_id": collection_id,
            "external_id": row["external_id"], "analysis_id": row["analysis_id"],
            "root_id": row["root_id"], "relative_path": row["relative_path"],
            "image_width": int(row["image_width"]), "image_height": int(row["image_height"]),
            "bbox": decode_array(row["bbox_blob"]), "keypoints": decode_array(row["keypoints_blob"]),
            "scores": decode_array(row["raw_scores_blob"]),
            "valid_mask": decode_array(row["valid_mask_blob"]).astype(bool),
            "quality": row["quality"], "available_scopes": json.loads(row["available_scopes"]),
            "diagnostics": json.loads(row["diagnostics_json"]),
            "extraction_diagnostics": json.loads(row["extraction_diagnostics"]),
        }

    def cleanup_expired_temporary_analyses(self) -> list[dict[str, Any]]:
        now = iso()
        with self.transaction() as con:
            rows = con.execute(
                "SELECT analysis_id, extraction_id, upload_path FROM temporary_analyses WHERE expires_at<=?",
                (now,),
            ).fetchall()
            cleaned = [dict(r) for r in rows]
            for row in rows:
                extraction_id = row["extraction_id"]
                con.execute("DELETE FROM temporary_analyses WHERE analysis_id=?", (row["analysis_id"],))
                if extraction_id:
                    referenced = con.execute(
                        "SELECT 1 FROM assets WHERE active_extraction_id=? AND deleted_at IS NULL LIMIT 1",
                        (extraction_id,),
                    ).fetchone()
                    if referenced is None:
                        con.execute("DELETE FROM extractions WHERE extraction_id=?", (extraction_id,))
            return cleaned

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        con = self.connect()
        try:
            temporary = int(con.execute(
                "SELECT COUNT(*) FROM temporary_analyses WHERE expires_at>?", (iso(),)
            ).fetchone()[0])
        finally:
            con.close()
        return {**base, "temporary_analyses": temporary}
