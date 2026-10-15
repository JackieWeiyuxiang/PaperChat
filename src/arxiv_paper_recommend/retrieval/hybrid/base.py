"""Shared hybrid dependencies and per-search dense score reuse."""

from abc import ABC, abstractmethod

from .._lifecycle import ManagedResource
from .._validation import normalize_queries, validate_integer, validate_text
from ..dense import HNSWRetriever
from ..search_result import SearchResult
from ..sparse import BM25Retriever
from .constants import DEFAULT_RESULT_LIMIT, MAX_RESULT_LIMIT


class DenseScoreCache:
    """Request-local scores; cached None values count as completed work."""

    def __init__(
        self,
        retriever: HNSWRetriever,
        initial: dict[str, dict[int, float | None]] | None = None,
    ) -> None:
        self.retriever = retriever
        self.scores: dict[str, dict[int, float | None]] = {} if initial is None else initial

    def ensure(self, query_ids: dict[str, list[int]]) -> None:
        missing: dict[str, list[int]] = {}
        for query, document_ids in query_ids.items():
            cached = self.scores.setdefault(query, {})
            missing_ids = [
                document_id for document_id in dict.fromkeys(document_ids)
                if document_id not in cached
            ]
            if missing_ids:
                missing[query] = missing_ids
        if missing:
            for query, scores in self.retriever.get_score(missing).items():
                self.scores[query].update(scores)


class BaseHybridRetriever(ManagedResource, ABC):
    """Own default retrievers, borrow injected ones, and validate shared inputs."""

    def __init__(
        self,
        sparse_retriever: BM25Retriever | None = None,
        dense_retriever: HNSWRetriever | None = None,
    ) -> None:
        super().__init__()
        try:
            self.sparse_retriever = BM25Retriever() if sparse_retriever is None else sparse_retriever
            if sparse_retriever is None:
                self._resources.callback(self.sparse_retriever.close)
            self.dense_retriever = HNSWRetriever() if dense_retriever is None else dense_retriever
            if dense_retriever is None:
                self._resources.callback(self.dense_retriever.close)
        except BaseException:
            self.close()
            raise

    @abstractmethod
    def search(
        self,
        terms: list[str],
        queries: str | set[str],
        score_query: str,
        limit: int = DEFAULT_RESULT_LIMIT,
    ) -> list[SearchResult]:
        """Return final results without modifying the underlying result objects."""
        ...

    def _validate_inputs(
        self, queries: str | set[str], score_query: str, limit: int,
    ) -> list[str]:
        self._ensure_open()
        validate_integer(limit, "Result limit", maximum=MAX_RESULT_LIMIT)
        query_texts = normalize_queries(queries, allow_empty=False)
        validate_text(score_query, "Score query")
        return query_texts

    def _dense_candidates(
        self, query_texts: list[str], limit: int,
    ) -> tuple[dict[int, SearchResult], DenseScoreCache]:
        query_results = self.dense_retriever.search(set(query_texts), limit=limit)
        candidates: dict[int, SearchResult] = {}
        scores: dict[str, dict[int, float | None]] = {}
        for query, results in query_results.items():
            scores[query] = {result.id: result.score for result in results}
            for result in results:
                candidates.setdefault(result.id, result)
        return candidates, DenseScoreCache(self.dense_retriever, scores)
