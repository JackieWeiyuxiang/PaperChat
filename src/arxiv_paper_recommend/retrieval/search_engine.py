"""Generate retrieval inputs and dispatch research requests to retrievers."""

from functools import partial
from typing import Protocol

from arxiv_paper_recommend.common.cancellation import check_cancelled
from arxiv_paper_recommend.common.lifecycle import ManagedResource
from arxiv_paper_recommend.common.validation import (
	validate_integer,
	validate_text,
	validate_weight,
)

from .constants import (
	DEFAULT_MAX_QUERIES,
	DEFAULT_MAX_TERMS,
	DEFAULT_RESULT_LIMIT,
	DEFAULT_TITLE_WEIGHT,
	MAX_TERM_COUNT,
	MIN_GENERATED_TERM_COUNT,
)
from .dense import HNSWRetriever, MultiQueryHNSWRetriever
from .hybrid import (
	HRRFRetriever,
	HybridRetriever,
	SDCascadeRetriever,
	SemanticRRFRetriever,
)
from .hybrid.constants import MAX_RESULT_LIMIT
from .innovation import FCHNSWRetriever, HPRFRetriever
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
			"FC_HNSW": partial(
				self._search_dense, multi_query=True, facet_coverage=True
			),
			"HRRF": partial(self._search_hybrid, retriever_type=HRRFRetriever),
			"SEMANTIC_RRF": partial(
				self._search_hybrid, retriever_type=SemanticRRFRetriever
			),
			"SD_CASCADE": partial(
				self._search_hybrid, retriever_type=SDCascadeRetriever
			),
			"HPRF": partial(self._search_hybrid, retriever_type=HPRFRetriever),
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
		method: str = "FC_HNSW",
		limit: int = DEFAULT_RESULT_LIMIT,
		*,
		max_terms: int = DEFAULT_MAX_TERMS,
		max_queries: int = DEFAULT_MAX_QUERIES,
		sparse_semantic_weight: float = 0.5,
		dense_semantic_weight: float = 0.5,
		fusion_weight: float = 0.5,
	) -> list[SearchResult]:
		"""Dispatch research requests; fusion weights apply to HRRF and HPRF."""
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
		if method in {"HRRF", "HPRF"}:
			# Bind weights only to the built-in handler; custom handlers keep their API.
			handler = partial(
				handler,
				sparse_semantic_weight=sparse_semantic_weight,
				dense_semantic_weight=dense_semantic_weight,
				fusion_weight=fusion_weight,
			)
		check_cancelled()
		results = handler(request, limit, max_terms=max_terms, max_queries=max_queries)
		check_cancelled()
		return results

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
		check_cancelled()
		return retriever.search(terms, limit=limit, title_weight=title_weight)

	def _search_dense(
		self,
		request: str,
		limit: int,
		*,
		max_terms: int,
		max_queries: int,
		multi_query: bool,
		facet_coverage: bool = False,
	) -> list[SearchResult]:
		if multi_query:
			validate_integer(max_queries, "max_queries")
		generator = self._get_query_generator()
		retriever = self._get_dense_retriever()
		if multi_query:
			inputs = generator.generate(request, max_queries=max_queries)
			query, recall_queries = inputs.query, inputs.recall_queries
		else:
			query = generator.generate_queries(request, max_queries=1)[0]
			recall_queries = None
		check_cancelled()
		if facet_coverage:
			return retriever.search(query, limit=limit, recall_queries=recall_queries)
		# Explicit baseline methods retain their algorithms on the shared FC instance.
		if multi_query:
			return MultiQueryHNSWRetriever.search(
				retriever, query, limit=limit, recall_queries=recall_queries
			)
		return HNSWRetriever.search(retriever, query, limit=limit)

	def _search_hybrid(
		self,
		request: str,
		limit: int,
		*,
		max_terms: int,
		max_queries: int,
		retriever_type: type[HybridRetriever],
		sparse_semantic_weight: float = 0.5,
		dense_semantic_weight: float = 0.5,
		fusion_weight: float = 0.5,
	) -> list[SearchResult]:
		validate_integer(limit, "Result limit", maximum=MAX_RESULT_LIMIT)
		self._validate_max_terms(max_terms)
		validate_integer(max_queries, "max_queries")
		if issubclass(retriever_type, (HRRFRetriever, HPRFRetriever)):
			sparse_semantic_weight = validate_weight(
				sparse_semantic_weight, "Sparse semantic weight"
			)
			dense_semantic_weight = validate_weight(
				dense_semantic_weight, "Dense semantic weight"
			)
			fusion_weight = validate_weight(fusion_weight, "Fusion weight")
		generator = self._get_query_generator()
		# These wrappers borrow the engine's retrievers and own no connections.
		with retriever_type(
			sparse_retriever=self._get_sparse_retriever(),
			dense_retriever=self._get_dense_retriever(),
		) as retriever:
			inputs = generator.generate(
				request, max_terms=max_terms, max_queries=max_queries
			)
			check_cancelled()
			if isinstance(retriever, (HRRFRetriever, HPRFRetriever)):
				return retriever.search(
					inputs.terms,
					inputs.query,
					limit=limit,
					recall_queries=inputs.recall_queries,
					sparse_semantic_weight=sparse_semantic_weight,
					dense_semantic_weight=dense_semantic_weight,
					fusion_weight=fusion_weight,
				)
			return retriever.search(
				inputs.terms,
				inputs.query,
				limit=limit,
				recall_queries=inputs.recall_queries,
			)

	def _get_query_generator(self) -> QueryGenerator:
		check_cancelled()
		if self.query_generator is None:
			self.query_generator = QueryGenerator()
			self._resources.callback(self.query_generator.close)
		check_cancelled()
		return self.query_generator

	def _get_sparse_retriever(self) -> TitleWeightedBM25Retriever:
		check_cancelled()
		if self.sparse_retriever is None:
			self.sparse_retriever = TitleWeightedBM25Retriever()
			self._resources.callback(self.sparse_retriever.close)
		check_cancelled()
		return self.sparse_retriever

	def _get_dense_retriever(self) -> MultiQueryHNSWRetriever:
		check_cancelled()
		if self.dense_retriever is None:
			self.dense_retriever = FCHNSWRetriever()
			self._resources.callback(self.dense_retriever.close)
		check_cancelled()
		return self.dense_retriever
