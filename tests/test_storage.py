from __future__ import annotations

import json
from pathlib import Path

from posesearch.storage import Repository


def test_asset_version_and_stale_change(tmp_path: Path):
    repo = Repository(str(tmp_path / "p.sqlite")); repo.migrate()
    v1, changed = repo.upsert_asset(
        collection_id="c", external_id="x", root_id="r", relative_path="a.png",
        source_revision="one", character_id=None, album_id=None, work_id=None,
        metadata={}, size=1, mtime_ns=1,
    )
    assert changed and v1 == 1
    v_same, changed = repo.upsert_asset(
        collection_id="c", external_id="x", root_id="r", relative_path="a.png",
        source_revision="one", character_id=None, album_id=None, work_id=None,
        metadata={}, size=1, mtime_ns=1,
    )
    assert not changed and v_same == 1
    v2, changed = repo.upsert_asset(
        collection_id="c", external_id="x", root_id="r", relative_path="b.png",
        source_revision="two", character_id=None, album_id=None, work_id=None,
        metadata={}, size=1, mtime_ns=2,
    )
    assert changed and v2 == 2


def test_persistent_job_claim_and_finish(tmp_path: Path):
    repo = Repository(str(tmp_path / "p.sqlite")); repo.migrate()
    jid = repo.enqueue_job("asset_extract", {"collection_id":"c"}, asset_version=1)
    job = repo.claim_job()
    assert job is not None and job["job_id"] == jid and job["status"] == "running"
    repo.finish_job(jid, {"ok": True})
    done = repo.get_job(jid)
    assert done["status"] == "succeeded"
    assert json.loads(done["result_json"])["ok"] is True
