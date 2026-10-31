"""Pool-independent raw facet coverage scoring for FC-HNSW."""

from math import exp, fsum, log

from .._scoring import DenseScoreCache

DEFAULT_CANDIDATE_LIMIT = 50
DEFAULT_COVERAGE_WEIGHT = 0.5


def raw_coverage_scores(
	document_ids: list[int],
	query: str,
	facets: list[str],
	coverage_weight: float,
	cache: DenseScoreCache,
) -> dict[int, float | None]:
	"""Blend raw query similarity with geometric mean nonnegative facet scores.

	Scores do not depend on the candidate pool. Missing or nonpositive facets
	give zero coverage; an entirely unavailable weighted signal returns None.
	"""
	document_ids = list(dict.fromkeys(document_ids))
	if not facets or coverage_weight == 0.0:
		cache.ensure({query: document_ids})
		return {
			document_id: cache.scores[query].get(document_id)
			for document_id in document_ids
		}

	query_weight = 1.0 - coverage_weight
	scoring_queries = list(dict.fromkeys([*([query] if query_weight else []), *facets]))
	cache.ensure({text: document_ids for text in scoring_queries})
	query_scores = cache.scores[query] if query_weight else {}
	scores: dict[int, float | None] = {}
	for document_id in document_ids:
		query_score = query_scores.get(document_id)
		facet_values = [cache.scores[facet].get(document_id) for facet in facets]
		if query_score is None and all(value is None for value in facet_values):
			scores[document_id] = None
			continue
		coverage = (
			exp(fsum(log(value) for value in facet_values) / len(facet_values))
			if all(value is not None and value > 0.0 for value in facet_values)
			else 0.0
		)
		scores[document_id] = (
			query_weight * (0.0 if query_score is None else query_score)
			+ coverage_weight * coverage
		)
	return scores
