from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(slots=True)
class RuntimeConfig:
    intra_op_threads: int = 2
    inter_op_threads: int = 1
    execution_mode: str = "sequential"
    allow_spinning: bool = False


@dataclass(slots=True)
class PipelineConfig:
    analysis_max_long_edge: int = 1280
    max_persons_per_image: int = 3
    score_threshold: float = 0.25
    max_file_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 24_000_000


@dataclass(slots=True)
class SearchSettings:
    concurrency: int = 1
    block_size: int = 4096
    initial_min_coverage: float = 0.75
    default_top_k: int = 30
    max_top_k: int = 100


@dataclass(slots=True)
class StorageConfig:
    sqlite_path: str = "./data/posesearch.sqlite"
    temp_dir: str = "./data/tmp"
    temporary_analysis_ttl_seconds: int = 3600


@dataclass(slots=True)
class AuthConfig:
    # API key -> collections. ["*"] means all configured collections.
    api_keys: dict[str, list[str]] = field(default_factory=lambda: {"dev-change-me": ["*"]})


@dataclass(slots=True)
class Settings:
    models_dir: str = "./models"
    roots: dict[str, str] = field(default_factory=lambda: {"library": "./media/library"})
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    search: SearchSettings = field(default_factory=SearchSettings)
    storage: StorageConfig = field(default_factory=StorageConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "Settings":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            models_dir=raw.get("models_dir", "./models"),
            roots=dict(raw.get("roots", {"library": "./media/library"})),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
            pipeline=PipelineConfig(**raw.get("pipeline", {})),
            search=SearchSettings(**raw.get("search", {})),
            storage=StorageConfig(**raw.get("storage", {})),
            auth=AuthConfig(**raw.get("auth", {})),
        )

    @classmethod
    def load(cls) -> "Settings":
        cfg = os.getenv("POSESEARCH_CONFIG")
        settings = cls.from_yaml(cfg) if cfg else cls()
        if os.getenv("POSESEARCH_API_KEY"):
            settings.auth.api_keys = {os.environ["POSESEARCH_API_KEY"]: ["*"]}
        if os.getenv("POSESEARCH_SQLITE_PATH"):
            settings.storage.sqlite_path = os.environ["POSESEARCH_SQLITE_PATH"]
        if os.getenv("POSESEARCH_MODELS_DIR"):
            settings.models_dir = os.environ["POSESEARCH_MODELS_DIR"]
        return settings

    def ensure_dirs(self) -> None:
        Path(self.storage.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.storage.temp_dir).mkdir(parents=True, exist_ok=True)
