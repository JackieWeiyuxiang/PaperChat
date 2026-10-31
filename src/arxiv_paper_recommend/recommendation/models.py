"""Recommendation records independent of LLM client dependencies."""

from dataclasses import dataclass

from arxiv_paper_recommend.retrieval import SearchResult


@dataclass(frozen=True)
class PaperRecommendation:
	"""A ranked paper with its screened reason and prompt-requested citations."""

	rank: int
	paper: SearchResult
	reason: str
