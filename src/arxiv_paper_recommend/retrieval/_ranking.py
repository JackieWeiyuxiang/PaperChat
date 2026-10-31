"""Deterministic score ordering, rank percentiles, and reciprocal rank fusion."""

from collections.abc import Collection, Iterable, Mapping

RRF_RANK_CONSTANT = 60


def rank_ids(
	document_ids: Iterable[int],
	scores: Mapping[int, float | None],
	*,
	descending: bool = False,
) -> list[int]:
	"""Put unavailable scores last and resolve equal scores by ascending ID."""

	def sort_key(document_id: int) -> tuple[bool, float, int]:
		score = scores[document_id]
		return (
			score is None,
			0.0 if score is None else (-score if descending else score),
			document_id,
		)

	return sorted(document_ids, key=sort_key)


def rank_percentiles(
	document_ids: Collection[int],
	scores: Mapping[int, float | None],
	*,
	descending: bool = False,
) -> dict[int, float]:
	"""Use average ranks for ties and the full pool size; missing scores get zero."""
	candidate_count = len(document_ids)
	scored_ids = [
		document_id
		for document_id in document_ids
		if scores.get(document_id) is not None
	]
	ranked_ids = rank_ids(scored_ids, scores, descending=descending)
	percentiles: dict[int, float] = {}
	start = 0
	while start < len(ranked_ids):
		end = start + 1
		while (
			end < len(ranked_ids)
			and scores[ranked_ids[end]] == scores[ranked_ids[start]]
		):
			end += 1
		average_rank = (start + 1 + end) / 2
		percentile = (candidate_count - average_rank + 1) / candidate_count
		for document_id in ranked_ids[start:end]:
			percentiles[document_id] = percentile
		start = end
	return {
		document_id: percentiles.get(document_id, 0.0) for document_id in document_ids
	}


def rrf_scores(
	rankings: Iterable[Iterable[int]],
	*,
	weights: Iterable[float] | None = None,
) -> dict[int, float]:
	"""Use one-based positions and optional list weights; absence contributes zero."""
	weighted_rankings = (
		((ranking, 1.0) for ranking in rankings)
		if weights is None
		else zip(rankings, weights, strict=True)
	)
	scores: dict[int, float] = {}
	for ranking, weight in weighted_rankings:
		for rank, document_id in enumerate(ranking, start=1):
			scores[document_id] = scores.get(document_id, 0.0) + weight / (
				RRF_RANK_CONSTANT + rank
			)
	return scores
