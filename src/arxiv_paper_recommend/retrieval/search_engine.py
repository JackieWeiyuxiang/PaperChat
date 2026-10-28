"""Generate retrieval inputs and dispatch research requests to retrievers."""

from functools import partial
from typing import Protocol

from arxiv_paper_recommend.common.lifecycle import ManagedResource
from arxiv_paper_recommend.common.validation import validate_integer, validate_text

from .constants import (
	DEFAULT_MAX_QUERIES,
	DEFAULT_MAX_TERMS,
	DEFAULT_RESULT_LIMIT,
	DEFAULT_TITLE_WEIGHT,
	MAX_TERM_COUNT,
	MIN_GENERATED_TERM_COUNT,
)
from .dense import MultiQueryHNSWRetriever
from .hybrid import (
	HRRFRetriever,
	HybridRetriever,
	SDCascadeRetriever,
	SemanticRRFRetriever,
)
from .hybrid.constants import MAX_RESULT_LIMIT
from .innovation import FacetCoverageRetriever
from .query import QueryGenerator
from .search_result import SearchResult
from .sparse import TitleWeightedBM25Retriever


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
		sparse_retriever: TitleWeightedBM25Retriever | None = None,
		dense_retriever: MultiQueryHNSWRetriever | None = None,
	) -> None:
		super().__init__()
		self.query_generator = query_generator
		self.sparse_retriever = sparse_retriever
		self.dense_retriever = dense_retriever
		self._methods: dict[str, SearchHandler] = {}
		methods: dict[str, SearchHandler] = {
			"BM25": partial(self._search_sparse, title_weight=None),
			"TITLE_WEIGHTED_BM25": partial(
				self._search_sparse, title_weight=DEFAULT_TITLE_WEIGHT
			),
			"HNSW": partial(self._search_dense, multi_query=False),
			"MULTI_QUERY_HNSW": partial(self._search_dense, multi_query=True),
			"HRRF": partial(self._search_hybrid, retriever_type=HRRFRetriever),
			"SEMANTIC_RRF": partial(
				self._search_hybrid, retriever_type=SemanticRRFRetriever
			),
			"SD_CASCADE": partial(
				self._search_hybrid, retriever_type=SDCascadeRetriever
			),
			"FACET_COVERAGE": partial(
				self._search_hybrid, retriever_type=FacetCoverageRetriever
			),
		}
		for name, handler in methods.items():
			self.register_method(name, handler)

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
		limit: int = DEFAULT_RESULT_LIMIT,
		*,
		max_terms: int = DEFAULT_MAX_TERMS,
		max_queries: int = DEFAULT_MAX_QUERIES,
	) -> list[SearchResult]:
		"""Validate shared inputs and dispatch; handlers validate their options."""
		self._ensure_open()
		method = validate_text(method, "Search method", strip=True).upper()
		handler = self._methods.get(method)
		if handler is None:
			available = ", ".join(self.available_methods)
			raise ValueError(
				f"Unsupported search method {method!r}. Available methods: {available}."
			)
		request = validate_text(request, "Research request", strip=True)
		validate_integer(limit, "Result limit")
		return handler(request, limit, max_terms=max_terms, max_queries=max_queries)

	@staticmethod
	def _validate_max_terms(max_terms: int) -> None:
		validate_integer(
			max_terms,
			"max_terms",
			minimum=MIN_GENERATED_TERM_COUNT,
			maximum=MAX_TERM_COUNT,
		)

	def _search_sparse(
		self,
		request: str,
		limit: int,
		*,
		max_terms: int,
		max_queries: int,
		title_weight: float | None,
	) -> list[SearchResult]:
		self._validate_max_terms(max_terms)
		generator = self._get_query_generator()
		retriever = self._get_sparse_retriever()
		terms = generator.generate_terms(request, max_terms=max_terms)
		return retriever.search(terms, limit=limit, title_weight=title_weight)

	def _search_dense(
		self,
		request: str,
		limit: int,
		*,
		max_terms: int,
		max_queries: int,
		multi_query: bool,
	) -> list[SearchResult]:
		if multi_query:
			validate_integer(max_queries, "max_queries")
		generator = self._get_query_generator()
		retriever = self._get_dense_retriever()
		recall_queries = (
			generator.generate_queries(request, max_queries=max_queries)
			if multi_query
			else None
		)
		query = generator.generate_queries(request, max_queries=1)[0]
		return retriever.search(query, limit=limit, recall_queries=recall_queries)

	def _search_hybrid(
		self,
		request: str,
		limit: int,
		*,
		max_terms: int,
		max_queries: int,
		retriever_type: type[HybridRetriever],
	) -> list[SearchResult]:
		validate_integer(limit, "Result limit", maximum=MAX_RESULT_LIMIT)
		self._validate_max_terms(max_terms)
		validate_integer(max_queries, "max_queries")
		generator = self._get_query_generator()
		# These wrappers borrow the engine's retrievers and own no connections.
		with retriever_type(
			sparse_retriever=self._get_sparse_retriever(),
			dense_retriever=self._get_dense_retriever(),
		) as retriever:
			terms = generator.generate_terms(request, max_terms=max_terms)
			recall_queries = generator.generate_queries(
				request, max_queries=max_queries
			)
			query = generator.generate_queries(request, max_queries=1)[0]
			return retriever.search(
				terms, query, limit=limit, recall_queries=recall_queries
			)

	def _get_query_generator(self) -> QueryGenerator:
		if self.query_generator is None:
			self.query_generator = QueryGenerator()
			self._resources.callback(self.query_generator.close)
		return self.query_generator

	def _get_sparse_retriever(self) -> TitleWeightedBM25Retriever:
		if self.sparse_retriever is None:
			self.sparse_retriever = TitleWeightedBM25Retriever()
			self._resources.callback(self.sparse_retriever.close)
		return self.sparse_retriever

	def _get_dense_retriever(self) -> MultiQueryHNSWRetriever:
		if self.dense_retriever is None:
			self.dense_retriever = MultiQueryHNSWRetriever()
			self._resources.callback(self.dense_retriever.close)
		return self.dense_retriever
