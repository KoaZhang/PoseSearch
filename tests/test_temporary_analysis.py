from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from posesearch.api import create_app
from posesearch.config import AuthConfig, Settings, StorageConfig
from posesearch.geometry import build_geometry_from_coco17
from posesearch.search_service import SearchService
from posesearch.temporary_storage import PoseRepository


def sample_person(person_id: str = "p1", offset: float = 0.0):
    k = np.zeros((17, 2), np.float32)
    body = {
        5: (80, 70), 7: (65, 110), 9: (55, 150),
        6: (120, 72), 8: (145, 105), 10: (165, 85),
        11: (88, 155), 13: (82, 220), 15: (75, 285),
        12: (115, 158), 14: (125, 220), 16: (150, 275),
    }
    for idx, xy in body.items():
        k[idx] = np.asarray(xy, np.float32) + offset
    scores = np.ones(17, np.float32) * 0.95
    valid = np.zeros(17, bool)
    valid[list(body)] = True
    return {
        "person_id": person_id, "person_index": 0,
        "bbox": np.array([40, 40, 180, 300], np.float32) + offset,
        "keypoints": k, "scores": scores, "valid_mask": valid,
        "geometry": build_geometry_from_coco17(k, valid),
        "quality": "usable_full", "available_scopes": ["full", "upper", "lower"],
        "diagnostics": {"reasons": []},
    }


def settings(tmp_path: Path) -> Settings:
    return Settings(
        storage=StorageConfig(
            sqlite_path=str(tmp_path / "posesearch.sqlite"), temp_dir=str(tmp_path / "tmp"),
            temporary_analysis_ttl_seconds=3600,
        ),
        auth=AuthConfig(api_keys={"k": ["*"]}),
    )


def test_upload_analysis_is_persistent_job(tmp_path: Path):
    s = settings(tmp_path)
    image = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(image, format="PNG")
    with TestClient(create_app(s)) as client:
        r = client.post(
            "/v1/analyses",
            data={"collection_id": "c", "include_overlay": "true"},
            files={"image": ("query.png", image.getvalue(), "image/png")},
            headers={"X-API-Key": "k"},
        )
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        job = client.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": "k"})
        assert job.status_code == 200
        assert job.json()["job_type"] == "temporary_analysis"
        assert job.json()["status"] == "queued"
        repo = client.app.state.repo
        row = repo.get_job(job_id)
        payload = json.loads(row["payload_json"])
        temp = repo.get_temporary_analysis("c", payload["analysis_id"])
        assert temp is not None
        assert Path(payload["upload_path"]).is_file()


def test_temporary_analysis_can_query_library(tmp_path: Path):
    s = settings(tmp_path)
    repo = PoseRepository(s.storage.sqlite_path)
    repo.migrate()
    version, _ = repo.upsert_asset(
        collection_id="c", external_id="candidate", root_id="library", relative_path="candidate.png",
        source_revision="1", character_id="char-b", album_id="album-b", work_id=None,
        metadata={}, size=1, mtime_ns=1,
    )
    candidate = sample_person("cand")
    assert repo.save_extraction(
        collection_id="c", external_id="candidate", expected_asset_version=version,
        extraction_id="ex-c", content_sha256="abc", image_width=320, image_height=320,
        status="ready", diagnostics={"coordinate_space": "exif_oriented_pixels"}, persons=[candidate],
    )
    repo.create_temporary_analysis(
        analysis_id="a1", collection_id="c", upload_path=str(tmp_path / "gone.upload"), ttl_seconds=3600,
    )
    query = sample_person("query", offset=3.0)
    assert repo.save_temporary_extraction(
        analysis_id="a1", collection_id="c", extraction_id="ex-q", content_sha256="def",
        image_width=320, image_height=320, status="ready",
        diagnostics={"coordinate_space": "exif_oriented_pixels"}, persons=[query],
    )
    service = SearchService(repo, s)
    q, scope, _, hits = service.query_analysis(
        collection_id="c", analysis_id="a1", person_id=None, scope="full",
        mirror="strict", top_k=10,
    )
    assert q.person_id == "query"
    assert scope == "full"
    assert hits and hits[0].external_id == "candidate"


def test_keypoints_and_temporary_overlay_are_authorized(tmp_path: Path):
    s = settings(tmp_path)
    repo = PoseRepository(s.storage.sqlite_path)
    repo.migrate()
    repo.create_temporary_analysis(
        analysis_id="a1", collection_id="c", upload_path=str(tmp_path / "gone.upload"), ttl_seconds=3600,
    )
    person = sample_person("p1")
    assert repo.save_temporary_extraction(
        analysis_id="a1", collection_id="c", extraction_id="ex-q", content_sha256="def",
        image_width=320, image_height=320, status="ready",
        diagnostics={"coordinate_space": "exif_oriented_pixels"}, persons=[person],
    )
    overlay = Path(s.storage.temp_dir) / "overlays" / "a1" / "p1.jpg"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "white").save(overlay, format="JPEG")
    with TestClient(create_app(s)) as client:
        assert client.get("/v1/persons/p1/keypoints").status_code == 401
        keypoints = client.get("/v1/persons/p1/keypoints", headers={"X-API-Key": "k"})
        assert keypoints.status_code == 200
        body = keypoints.json()
        assert body["analysis_id"] == "a1"
        assert len(body["keypoints_coco17_xy"]) == 17
        assert body["coordinate_space"] == "exif_oriented_pixels"
        preview = client.get("/v1/persons/p1/overlay", headers={"X-API-Key": "k"})
        assert preview.status_code == 200
        assert preview.headers["content-type"].startswith("image/jpeg")
