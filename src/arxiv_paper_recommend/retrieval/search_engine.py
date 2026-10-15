"""Generate retrieval inputs and dispatch research requests to retrievers."""

from dataclasses import replace
from typing import Protocol

from ._lifecycle import ManagedResource
from ._ranking import rank_ids
from ._validation import validate_integer, validate_text
from .dense import HNSWRetriever
from .hybrid import CandidateFusionRetriever, SemanticWeightedRetriever, SequentialRetriever
from .hybrid.base import BaseHybridRetriever
from .hybrid.constants import MAX_RESULT_LIMIT
from .query import QueryGenerator
from .search_result import SearchResult
from .sparse import BM25Retriever
from .sparse.bm25_retriever import MAX_KEYWORD_COUNT


class SearchHandler(Protocol):
    """Callable interface for methods registered with SearchEngine."""

    def __call__(
        self,
        request: str,
        limit: int,
        *,
        max_terms: int,
        max_queries: int,
    ) -> list[SearchResult]: ...


class SearchEngine(ManagedResource):
    """Create components on demand; injected components remain caller-owned."""

    def __init__(
        self,
        *,
        query_generator: QueryGenerator | None = None,
        sparse_retriever: BM25Retriever | None = None,
        dense_retriever: HNSWRetriever | None = None,
    ) -> None:
        super().__init__()
        self.query_generator = query_generator
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self._methods: dict[str, SearchHandler] = {}
        self.register_method("BM25", self._search_bm25)
        self.register_method("HNSW", self._search_hnsw)
        self.register_method("HYBRID_CANDIDATE_FUSION", self._search_hybrid_candidate_fusion)
        self.register_method("HYBRID_SEMANTIC_WEIGHTED", self._search_hybrid_semantic_weighted)
        self.register_method("HYBRID_SEQUENTIAL", self._search_hybrid_sequential)

    @property
    def available_methods(self) -> tuple[str, ...]:
        """Return registered names in registration order."""
        self._ensure_open()
        return tuple(self._methods)

    def register_method(self, name: str, handler: SearchHandler) -> None:
        """Register a unique, case-insensitive name and a callable handler."""
        self._ensure_open()
        name = validate_text(name, "Search method name", strip=True).upper()
        if name in self._methods:
            raise ValueError(f"Search method {name!r} is already registered.")
        if not callable(handler):
            raise TypeError("Search method handler must be callable.")
        self._methods[name] = handler

    def search(
        self,
        request: str,
        method: str,
        limit: int = 20,
        *,
        max_terms: int = 12,
        max_queries: int = 4,
    ) -> list[SearchResult]:
        """Validate shared inputs and dispatch; handlers validate their options."""
        self._ensure_open()
        method = validate_text(method, "Search method", strip=True).upper()
        handler = self._methods.get(method)
        if handler is None:
            available = ", ".join(self.available_methods)
            raise ValueError(f"Unsupported search method {method!r}. Available methods: {available}.")
        request = validate_text(request, "Research request", strip=True)
        validate_integer(limit, "Result limit")
        return handler(request, limit, max_terms=max_terms, max_queries=max_queries)

    def _search_bm25(
        self, request: str, limit: int, *, max_terms: int, max_queries: int,
    ) -> list[SearchResult]:
        validate_integer(max_terms, "max_terms", minimum=5, maximum=MAX_KEYWORD_COUNT)
        generator = self._get_query_generator()
        retriever = self._get_sparse_retriever()
        terms = generator.generate_terms(request, max_terms=max_terms)
        return retriever.term_search(terms, limit=limit)

    def _search_hnsw(
        self, request: str, limit: int, *, max_terms: int, max_queries: int,
    ) -> list[SearchResult]:
        validate_integer(max_queries, "max_queries")
        generator = self._get_query_generator()
        retriever = self._get_dense_retriever()
        queries = generator.generate_queries(request, max_queries=max_queries)
        query_results = retriever.search(set(queries), limit=limit)
        candidates: dict[int, SearchResult] = {}
        for results in query_results.values():
            for result in results:
                candidates.setdefault(result.id, result)
        if not candidates:
            return []

        ranking_query = generator.generate_queries(request, max_queries=1)[0]
        scores = retriever.get_score({ranking_query: list(candidates)})[ranking_query]
        return [
            replace(candidates[document_id], score=scores[document_id])
            for document_id in rank_ids(candidates, scores, descending=True)[:limit]
        ]

    def _search_hybrid_candidate_fusion(
        self, request: str, limit: int, *, max_terms: int, max_queries: int,
    ) -> list[SearchResult]:
        return self._run_hybrid(
            request, limit, CandidateFusionRetriever, max_terms=max_terms, max_queries=max_queries,
        )

    def _search_hybrid_semantic_weighted(
        self, request: str, limit: int, *, max_terms: int, max_queries: int,
    ) -> list[SearchResult]:
        return self._run_hybrid(
            request, limit, SemanticWeightedRetriever, max_terms=max_terms, max_queries=max_queries,
        )

    def _search_hybrid_sequential(
        self, request: str, limit: int, *, max_terms: int, max_queries: int,
    ) -> list[SearchResult]:
        return self._run_hybrid(
            request, limit, SequentialRetriever, max_terms=max_terms, max_queries=max_queries,
        )

    def _run_hybrid(
        self,
        request: str,
        limit: int,
        retriever_type: type[BaseHybridRetriever],
        *,
        max_terms: int,
        max_queries: int,
    ) -> list[SearchResult]:
        validate_integer(limit, "Result limit", maximum=MAX_RESULT_LIMIT)
        validate_integer(max_terms, "max_terms", minimum=5, maximum=MAX_KEYWORD_COUNT)
        validate_integer(max_queries, "max_queries")
        generator = self._get_query_generator()
        # These wrappers borrow the engine's retrievers and own no connections.
        with retriever_type(
            sparse_retriever=self._get_sparse_retriever(),
            dense_retriever=self._get_dense_retriever(),
        ) as retriever:
            terms = generator.generate_terms(request, max_terms=max_terms)
            queries = generator.generate_queries(request, max_queries=max_queries)
            score_query = generator.generate_queries(request, max_queries=1)[0]
            return retriever.search(terms, set(queries), score_query, limit=limit)

    def _get_query_generator(self) -> QueryGenerator:
        if self.query_generator is None:
            self.query_generator = QueryGenerator()
            self._resources.callback(self.query_generator.close)
        return self.query_generator

    def _get_sparse_retriever(self) -> BM25Retriever:
        if self.sparse_retriever is None:
            self.sparse_retriever = BM25Retriever()
            self._resources.callback(self.sparse_retriever.close)
        return self.sparse_retriever

    def _get_dense_retriever(self) -> HNSWRetriever:
        if self.dense_retriever is None:
            self.dense_retriever = HNSWRetriever()
            self._resources.callback(self.dense_retriever.close)
        return self.dense_retriever
