"""Common search and candidate-scoring interface for sparse and dense retrieval."""

from abc import ABC, abstractmethod

from arxiv_paper_recommend.common.lifecycle import ManagedResource

from .constants import DEFAULT_RESULT_LIMIT
from .search_result import SearchResult


class Retriever[QueryT](ManagedResource, ABC):
	"""Share result shapes and resource management with algorithm-specific queries.

	Scores retain each algorithm's native scale and ordering direction.
	"""

	@abstractmethod
	def search(
		self,
		query: QueryT,
		limit: int = DEFAULT_RESULT_LIMIT,
	) -> list[SearchResult]:
		"""Return at most limit results in the algorithm's relevance order."""
		...

	@abstractmethod
	def score(self, query: QueryT, document_ids: list[int]) -> dict[int, float | None]:
		"""Score unique IDs in first-occurrence order; unavailable scores are None."""
		...
