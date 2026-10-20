"""Common search and candidate-scoring interface for sparse and dense retrieval."""

from abc import ABC, abstractmethod

from ._lifecycle import ManagedResource
from .search_result import SearchResult

DEFAULT_RESULT_LIMIT = 20


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
	def get_score(self, ids: list[int], query: QueryT) -> dict[int, float | None]:
		"""Score unique IDs in first-occurrence order; unavailable scores are None."""
		...
