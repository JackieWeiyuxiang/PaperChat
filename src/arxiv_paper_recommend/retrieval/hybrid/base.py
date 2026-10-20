"""Shared hybrid dependencies and per-search dense score reuse."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from arxiv_paper_recommend.common.lifecycle import ManagedResource
from arxiv_paper_recommend.common.validation import (
	normalize_queries,
	validate_integer,
	validate_text,
)

from ..constants import DEFAULT_RESULT_LIMIT
from ..dense import MultiQueryHNSWRetriever
from ..search_result import SearchResult
from ..sparse import TitleWeightedBM25Retriever
from .constants import MAX_RESULT_LIMIT


class DenseScoreCache:
	"""Cache public score results per request, including unavailable scores."""

	def __init__(
		self,
		retriever: MultiQueryHNSWRetriever,
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
				cached.update(self.retriever.score(query, missing_ids))


class HybridRetriever(ManagedResource, ABC):
	"""Define hybrid search, own defaults, and borrow injected retrievers.

	Hybrid strategies have no standalone candidate-scoring interface.
	Dense components are used only through public search and score methods.
	Components handle None options by falling back to their base implementations.
	"""

	def __init__(
		self,
		sparse_retriever: TitleWeightedBM25Retriever | None = None,
		dense_retriever: MultiQueryHNSWRetriever | None = None,
	) -> None:
		super().__init__()
		try:
			self.sparse_retriever = (
				TitleWeightedBM25Retriever()
				if sparse_retriever is None
				else sparse_retriever
			)
			if sparse_retriever is None:
				self._resources.callback(self.sparse_retriever.close)
			self.dense_retriever = (
				MultiQueryHNSWRetriever()
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
		recall_queries: Sequence[str] | None = None,
	) -> list[SearchResult]:
		"""Return results with optional title weighting and multi-query recall.

		Without divisions, query also supplies the recall/selection query.
		The query still determines final semantic ranking.
		"""
		...

	def _validate_inputs(
		self,
		query: str,
		limit: int,
		recall_queries: Sequence[str] | None = None,
	) -> list[str]:
		self._ensure_open()
		validate_integer(limit, "Result limit", maximum=MAX_RESULT_LIMIT)
		validate_text(query, "Query")
		if recall_queries is None:
			return [query]
		return normalize_queries(recall_queries)

	def _dense_candidates(
		self,
		query: str,
		limit: int,
		*,
		recall_queries: Sequence[str] | None = None,
	) -> tuple[dict[int, SearchResult], DenseScoreCache]:
		results = self.dense_retriever.search(
			query, limit=limit, recall_queries=recall_queries
		)
		candidates = {result.document_id: result for result in results}
		scores = {query: {result.document_id: result.score for result in results}}
		return candidates, DenseScoreCache(self.dense_retriever, scores)
