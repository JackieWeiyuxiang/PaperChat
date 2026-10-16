"""Rerank each candidate list with RRF, then fuse their new rankings."""

from collections.abc import Collection
from dataclasses import replace

from .._ranking import rank_ids, rrf_scores
from ..search_result import SearchResult
from .base import BaseHybridRetriever
from .constants import DEFAULT_RESULT_LIMIT


class HRRFRetriever(BaseHybridRetriever):
    """Keep each route's candidates, rerank each route, then fuse positions."""

    def search(
        self,
        terms: list[str],
        queries: str | set[str],
        score_query: str,
        limit: int = DEFAULT_RESULT_LIMIT,
    ) -> list[SearchResult]:
        """Return final RRF scores, with ties resolved by ascending document ID."""
        query_texts = self._validate_inputs(queries, score_query, limit)
        sparse_results = self.sparse_retriever.term_search(terms, limit=limit)
        sparse_candidates = {result.id: result for result in sparse_results}
        sparse_scores = {result.id: result.score for result in sparse_results}
        dense_candidates, cache = self._dense_candidates(query_texts, limit)
        if not sparse_candidates and not dense_candidates:
            return []

        cache.ensure({score_query: [*dense_candidates, *sparse_candidates]})
        dense_scores = cache.scores[score_query]
        dense_ids = rank_ids(dense_candidates, dense_scores, descending=True)[:limit]
        missing_sparse_ids = [document_id for document_id in dense_ids if document_id not in sparse_scores]
        if missing_sparse_ids:
            sparse_scores.update(self.sparse_retriever.get_score(missing_sparse_ids, terms))

        sparse_ids = _rerank(sparse_candidates, sparse_scores, dense_scores)
        dense_ids = _rerank(dense_ids, sparse_scores, dense_scores)
        candidates = {document_id: dense_candidates[document_id] for document_id in dense_ids}
        candidates.update(sparse_candidates)
        scores = rrf_scores((sparse_ids, dense_ids))
        return [
            replace(candidates[document_id], score=scores[document_id])
            for document_id in rank_ids(scores, scores, descending=True)[:limit]
        ]


def _rerank(
    document_ids: Collection[int],
    sparse_scores: dict[int, float | None],
    dense_scores: dict[int, float | None],
) -> list[int]:
    scores = rrf_scores((
        rank_ids(document_ids, sparse_scores),
        rank_ids(document_ids, dense_scores, descending=True),
    ))
    return rank_ids(scores, scores, descending=True)
