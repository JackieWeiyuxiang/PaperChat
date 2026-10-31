"""Sum independent field BM25 scores with a per-call title multiplier."""

from math import isfinite

from ..constants import DEFAULT_RESULT_LIMIT
from ..search_result import SearchResult
from .bm25 import BM25Retriever


class TitleWeightedBM25Retriever(BM25Retriever):
	"""Weight the final title BM25 score equally in search and candidate scoring."""

	def search(
		self,
		query: list[str],
		limit: int = DEFAULT_RESULT_LIMIT,
		*,
		title_weight: float | None = None,
	) -> list[SearchResult]:
		"""Scale the final title score; None uses unit weights for both fields."""
		if title_weight is None:
			return super().search(query, limit=limit)
		self._ensure_open()
		title_weight = _validate_title_weight(title_weight)
		return self._search(query, limit=limit, title_weight=title_weight)

	def score(
		self,
		query: list[str],
		document_ids: list[int],
		*,
		title_weight: float | None = None,
	) -> dict[int, float | None]:
		"""Apply a title weight for this scoring call, independent of prior calls."""
		if title_weight is None:
			return super().score(query, document_ids)
		self._ensure_open()
		title_weight = _validate_title_weight(title_weight)
		return self._score(query, document_ids, title_weight=title_weight)


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
