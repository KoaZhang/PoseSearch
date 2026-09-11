from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import Settings
from .model_lock import ModelLock
from .storage import Repository


def cmd_init_db(args) -> int:
    settings = Settings.load()
    repo = Repository(args.sqlite or settings.storage.sqlite_path)
    repo.migrate()
    print(f"initialized schema v{repo.schema_version()} at {repo.sqlite_path}")
    return 0


def cmd_models_verify(args) -> int:
    settings = Settings.load()
    lock = ModelLock.load(args.lock)
    hashes = lock.verify(args.models_dir or settings.models_dir, require_hashes=not args.allow_unresolved)
    print(json.dumps(hashes, indent=2))
    return 0


def cmd_stats(args) -> int:
    settings = Settings.load()
    repo = Repository(args.sqlite or settings.storage.sqlite_path)
    repo.migrate()
    print(json.dumps(repo.stats(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="posesearch")
    sub = p.add_subparsers(dest="command", required=True)
    p_db = sub.add_parser("init-db")
    p_db.add_argument("--sqlite")
    p_db.set_defaults(func=cmd_init_db)

    p_m = sub.add_parser("models-verify")
    p_m.add_argument("--lock", default=os.getenv("POSESEARCH_MODEL_LOCK", "models.lock.json"))
    p_m.add_argument("--models-dir")
    p_m.add_argument("--allow-unresolved", action="store_true")
    p_m.set_defaults(func=cmd_models_verify)

    p_s = sub.add_parser("stats")
    p_s.add_argument("--sqlite")
    p_s.set_defaults(func=cmd_stats)
    return p


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
