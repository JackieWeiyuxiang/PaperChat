"""Use BM25 candidates for multi-query dense selection and reranking."""

from dataclasses import replace

from .._ranking import rank_ids
from ..search_result import SearchResult
from .base import BaseHybridRetriever, DenseScoreCache
from .constants import DEFAULT_RESULT_LIMIT, SPARSE_CANDIDATE_MULTIPLIER


class SequentialRetriever(BaseHybridRetriever):
    """Search densely within BM25 candidates without a full-index dense search."""

    def search(
        self,
        terms: list[str],
        queries: str | set[str],
        score_query: str,
        limit: int = DEFAULT_RESULT_LIMIT,
    ) -> list[SearchResult]:
        """Recall limit * 60 sparse candidates, select per query, then rerank."""
        query_texts = self._validate_inputs(queries, score_query, limit)
        sparse_results = self.sparse_retriever.term_search(
            terms, limit=limit * SPARSE_CANDIDATE_MULTIPLIER,
        )
        candidates = {result.id: result for result in sparse_results}
        if not candidates:
            return []

        cache = DenseScoreCache(self.dense_retriever)
        cache.ensure({query: list(candidates) for query in query_texts})
        selected_ids: dict[int, None] = {}
        for query in query_texts:
            scores = cache.scores[query]
            scored_ids = [document_id for document_id in candidates if scores[document_id] is not None]
            for document_id in rank_ids(scored_ids, scores, descending=True)[:limit]:
                selected_ids.setdefault(document_id, None)
        if not selected_ids:
            return []

        cache.ensure({score_query: list(selected_ids)})
        dense_scores = cache.scores[score_query]
        return [
            replace(candidates[document_id], score=dense_scores[document_id])
            for document_id in rank_ids(selected_ids, dense_scores, descending=True)[:limit]
        ]
