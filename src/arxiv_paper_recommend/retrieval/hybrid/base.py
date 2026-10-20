"""Shared hybrid dependencies and per-search dense score reuse."""

from abc import ABC, abstractmethod

from .._lifecycle import ManagedResource
from .._validation import normalize_queries, validate_integer, validate_text
from ..dense import DivisionalHNSWRetriever
from ..search_result import SearchResult
from ..sparse import TitleAwareBM25Retriever
from .constants import DEFAULT_RESULT_LIMIT, MAX_RESULT_LIMIT


class DenseScoreCache:
	"""Cache public get_score results per request, including unavailable scores."""

	def __init__(
		self,
		retriever: DivisionalHNSWRetriever,
		initial: dict[str, dict[int, float | None]] | None = None,
	) -> None:
		self.retriever = retriever
		self.scores: dict[str, dict[int, float | None]] = (
			{} if initial is None else initial
		)

	def ensure(self, query_ids: dict[str, list[int]]) -> None:
		for query, document_ids in query_ids.items():
			cached = self.scores.setdefault(query, {})
			missing_ids = [
				document_id
				for document_id in dict.fromkeys(document_ids)
				if document_id not in cached
			]
			if missing_ids:
				cached.update(self.retriever.get_score(missing_ids, query))


class HybridRetriever(ManagedResource, ABC):
	"""Define hybrid search, own defaults, and borrow injected retrievers.

	Hybrid strategies have no standalone candidate-scoring interface.
	Dense components are used only through public search and get_score methods.
	Components handle None options by falling back to their base implementations.
	"""

	def __init__(
		self,
		sparse_retriever: TitleAwareBM25Retriever | None = None,
		dense_retriever: DivisionalHNSWRetriever | None = None,
	) -> None:
		super().__init__()
		try:
			self.sparse_retriever = (
				TitleAwareBM25Retriever()
				if sparse_retriever is None
				else sparse_retriever
			)
			if sparse_retriever is None:
				self._resources.callback(self.sparse_retriever.close)
			self.dense_retriever = (
				DivisionalHNSWRetriever()
				if dense_retriever is None
				else dense_retriever
			)
			if dense_retriever is None:
				self._resources.callback(self.dense_retriever.close)
		except BaseException:
			self.close()
			raise

	@abstractmethod
	def search(
		self,
		terms: list[str],
		query: str,
		limit: int = DEFAULT_RESULT_LIMIT,
		*,
		title_weight: float | None = None,
		divisional_queries: str | set[str] | None = None,
	) -> list[SearchResult]:
		"""Return results with optional title weighting and divisional recall.

		Without divisions, query also supplies the recall/selection query.
		The query still determines final semantic ranking.
		"""
		...

	def _validate_inputs(
		self,
		query: str,
		limit: int,
		divisional_queries: str | set[str] | None = None,
	) -> list[str]:
		self._ensure_open()
		validate_integer(limit, "Result limit", maximum=MAX_RESULT_LIMIT)
		validate_text(query, "Query")
		if divisional_queries is None:
			return [query]
		return normalize_queries(divisional_queries)

	def _dense_candidates(
		self,
		query: str,
		limit: int,
		*,
		divisional_queries: str | set[str] | None = None,
	) -> tuple[dict[int, SearchResult], DenseScoreCache]:
		results = self.dense_retriever.search(
			query, limit=limit, divisional_queries=divisional_queries
		)
		candidates = {result.id: result for result in results}
		scores = {query: {result.id: result.score for result in results}}
		return candidates, DenseScoreCache(self.dense_retriever, scores)
