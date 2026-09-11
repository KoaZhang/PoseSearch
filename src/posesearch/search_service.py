from __future__ import annotations

import threading

from .config import Settings
from .quality import choose_auto_scope
from .search import ExactMaskedSearcher, SearchConfig
from .storage import Repository
from .types import PoseRecord, SearchHit


class SearchService:
    def __init__(self, repo: Repository, settings: Settings):
        self.repo = repo
        self.searcher = ExactMaskedSearcher(
            config=SearchConfig(
                min_coverage=settings.search.initial_min_coverage,
                block_size=settings.search.block_size,
                max_top_k=settings.search.max_top_k,
            )
        )
        self.index_revision = -1
        self._search_gate = threading.BoundedSemaphore(max(1, settings.search.concurrency))

    def refresh_if_needed(self) -> None:
        revision = self.repo.latest_change_sequence()
        if revision != self.index_revision:
            self.searcher.replace(self.repo.load_pose_records())
            self.index_revision = revision

    def _search_with_query(
        self,
        query: PoseRecord,
        *,
        scope: str,
        mirror: str,
        top_k: int,
        exclude_character_id: str | None,
        max_per_character: int | None,
        max_per_album: int | None,
    ) -> tuple[PoseRecord, str, str | None, list[SearchHit]]:
        if scope == "auto":
            effective_scope, reason = choose_auto_scope(query.quality, query.geometry.point_mask)
        else:
            effective_scope, reason = scope, None
        hits = self.searcher.search(
            query, scope=effective_scope, mirror=mirror, top_k=top_k,
            exclude_character_id=exclude_character_id,
            max_per_character=max_per_character, max_per_album=max_per_album,
        )
        return query, effective_scope, reason, hits

    def query_asset(
        self,
        *,
        collection_id: str,
        external_id: str,
        person_id: str | None,
        scope: str,
        mirror: str,
        top_k: int,
        exclude_character_id: str | None = None,
        max_per_character: int | None = None,
        max_per_album: int | None = None,
    ) -> tuple[PoseRecord, str, str | None, list[SearchHit]]:
        with self._search_gate:
            self.refresh_if_needed()
            people = self.repo.get_asset_pose_records(collection_id, external_id)
            if person_id is not None:
                people = [p for p in people if p.person_id == person_id]
            if not people:
                raise ValueError("QUERY_POSE_NOT_USABLE")
            return self._search_with_query(
                people[0], scope=scope, mirror=mirror, top_k=top_k,
                exclude_character_id=exclude_character_id,
                max_per_character=max_per_character, max_per_album=max_per_album,
            )

    def query_analysis(
        self,
        *,
        collection_id: str,
        analysis_id: str,
        person_id: str | None,
        scope: str,
        mirror: str,
        top_k: int,
        exclude_character_id: str | None = None,
        max_per_character: int | None = None,
        max_per_album: int | None = None,
    ) -> tuple[PoseRecord, str, str | None, list[SearchHit]]:
        with self._search_gate:
            self.refresh_if_needed()
            analysis = self.repo.get_temporary_analysis(collection_id, analysis_id)
            if analysis is None:
                raise ValueError("ANALYSIS_NOT_FOUND")
            people = self.repo.get_temporary_pose_records(collection_id, analysis_id)
            if person_id is not None:
                people = [p for p in people if p.person_id == person_id]
            if not people:
                raise ValueError("QUERY_POSE_NOT_USABLE")
            return self._search_with_query(
                people[0], scope=scope, mirror=mirror, top_k=top_k,
                exclude_character_id=exclude_character_id,
                max_per_character=max_per_character, max_per_album=max_per_album,
            )
