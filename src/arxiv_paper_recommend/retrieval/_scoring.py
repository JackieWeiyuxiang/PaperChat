"""Per-search reuse of raw dense scores, independent of ranking algorithms."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .dense import HNSWRetriever


class DenseScoreCache:
	"""Cache raw_score() results per request, including unavailable scores."""

	def __init__(
		self,
		retriever: HNSWRetriever,
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
				cached.update(self.retriever.raw_score(query, missing_ids))
