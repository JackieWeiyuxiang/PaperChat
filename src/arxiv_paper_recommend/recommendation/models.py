"""Recommendation records independent of LLM client dependencies."""

from dataclasses import dataclass

from arxiv_paper_recommend.retrieval import SearchResult


@dataclass(frozen=True)
class PaperRecommendation:
	"""A retrieved paper paired with its recommendation rank and reason."""

	rank: int
	paper: SearchResult
	reason: str
