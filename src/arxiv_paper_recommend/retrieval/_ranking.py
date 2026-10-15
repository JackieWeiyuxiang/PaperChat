"""Deterministic score ordering and reciprocal rank fusion."""

from collections.abc import Iterable, Mapping


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


def rrf_scores(rankings: Iterable[Iterable[int]]) -> dict[int, float]:
    """Use one-based positions; absence from a list contributes zero."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (RRF_RANK_CONSTANT + rank)
    return scores
