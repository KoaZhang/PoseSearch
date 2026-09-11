from pathlib import Path

from fastapi.testclient import TestClient

from posesearch.api import create_app
from posesearch.config import AuthConfig, Settings, StorageConfig


def test_health_and_auth(tmp_path: Path):
    s = Settings(storage=StorageConfig(sqlite_path=str(tmp_path / "x.sqlite"), temp_dir=str(tmp_path / "tmp")), auth=AuthConfig(api_keys={"k":["*"]}))
    with TestClient(create_app(s)) as c:
        assert c.get("/health/live").status_code == 200
        assert c.get("/health/ready").status_code == 200
        assert c.get("/v1/stats").status_code == 401
        assert c.get("/v1/stats", headers={"X-API-Key":"k"}).status_code == 200
