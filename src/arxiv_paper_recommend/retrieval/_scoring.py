"""Per-search reuse of raw dense scores, independent of ranking algorithms."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .dense import HNSWRetriever


class DenseScoreCache:
	"""Batch and cache raw inner products per request, including missing scores."""

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
		missing: dict[str, list[int]] = {}
		for query, document_ids in query_ids.items():
			cached = self.scores.setdefault(query, {})
			missing_ids = [
				document_id
				for document_id in dict.fromkeys(document_ids)
				if document_id not in cached
			]
			if missing_ids:
				missing[query] = missing_ids
		if not missing:
			return
		document_ids = list(
			dict.fromkeys(
				document_id for ids in missing.values() for document_id in ids
			)
		)
		batch_scores = self.retriever.raw_score_many(list(missing), document_ids)
		for query, ids in missing.items():
			self.scores[query].update(
				(document_id, batch_scores[query][document_id]) for document_id in ids
			)
