"""BM25 retrieval with a per-call title weight and unit abstract weight."""

from math import isfinite

from ..base import DEFAULT_RESULT_LIMIT
from ..search_result import SearchResult
from .bm25 import BM25Retriever


class TitleAwareBM25Retriever(BM25Retriever):
	"""Use the same weighted FTS5 BM25 for search and candidate scoring."""

	def search(
		self,
		query: list[str],
		limit: int = DEFAULT_RESULT_LIMIT,
		*,
		title_weight: float | None = None,
	) -> list[SearchResult]:
		"""Apply a title weight for this search, or use ordinary BM25 when omitted."""
		if title_weight is None:
			return super().search(query, limit=limit)
		self._ensure_open()
		title_weight = _validate_title_weight(title_weight)
		return self._search(query, limit=limit, title_weight=title_weight)

	def get_score(
		self,
		ids: list[int],
		query: list[str],
		*,
		title_weight: float | None = None,
	) -> dict[int, float | None]:
		"""Apply a title weight for this scoring call, independent of prior calls."""
		if title_weight is None:
			return super().get_score(ids, query)
		self._ensure_open()
		title_weight = _validate_title_weight(title_weight)
		return self._get_score(ids, query, title_weight=title_weight)


def _validate_title_weight(title_weight: float) -> float:
	"""Accept finite, non-negative numeric weights, excluding booleans."""
	if isinstance(title_weight, bool) or not isinstance(title_weight, (int, float)):
		raise TypeError("Title weight must be an integer or a float.")
	try:
		title_weight = float(title_weight)
	except OverflowError as error:
		raise ValueError("Title weight must be finite and non-negative.") from error
	if not isfinite(title_weight) or title_weight < 0:
		raise ValueError("Title weight must be finite and non-negative.")
	return title_weight
