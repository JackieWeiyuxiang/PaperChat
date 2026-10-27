"""Result models for evidence-grounded paper recommendations."""

from dataclasses import dataclass

from arxiv_paper_recommend.retrieval import SearchResult


@dataclass(frozen=True)
class EvidenceGroundedRecommendation:
	"""A recommended paper with verifiable supporting evidence."""

	rank: int
	paper: SearchResult
	evidence: str
	reason: str
