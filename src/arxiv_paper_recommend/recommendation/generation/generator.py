"""Generate short, evidence-grounded reasons for retrieved paper suggestions."""

import json

from pydantic import BaseModel, ConfigDict, Field

from arxiv_paper_recommend.common.validation import validate_integer, validate_text
from arxiv_paper_recommend.llm import StructuredGenerator
from arxiv_paper_recommend.retrieval import SearchResult

from ..models import PaperRecommendation

MAX_RECOMMENDATIONS = 10
MAX_GENERATION_CANDIDATES = 30


class _GeneratedRecommendation(BaseModel):
	"""Structured model output for one retrieved paper."""

	model_config = ConfigDict(extra="forbid")
	doc_id: str = Field(min_length=1)
	reason: str = Field(min_length=1)


class _GenerationResponse(BaseModel):
	"""Structured list of recommended retrieved paper identifiers and reasons."""

	model_config = ConfigDict(extra="forbid")
	recommendations: list[_GeneratedRecommendation] = Field(
		max_length=MAX_RECOMMENDATIONS,
	)


class RecommendationGenerator(StructuredGenerator):
	"""Create a ranked paper shortlist with concise reasons grounded in abstracts."""

	instructions = """
You recommend academic papers for a user's research request using only the
retrieved papers supplied in the input. Return up to the requested number of
papers, ordered by usefulness to the request. Select only supplied doc_id values
and never invent papers, findings, methods, or claims. Each reason must be one
or two concise sentences supported by that paper's title or abstract. Explain
the connection to the request, and omit a paper when the supplied evidence is
too weak to justify a recommendation. Do not use outside knowledge.
"""

	def generate(
		self,
		request: str,
		papers: list[SearchResult],
		limit: int = MAX_RECOMMENDATIONS,
	) -> list[PaperRecommendation]:
		"""Generate up to limit recommendations from the retrieved candidate list."""
		self._ensure_open()
		request = validate_text(request, "Research request", strip=True)
		validate_integer(
			limit, "Recommendation limit", minimum=1, maximum=MAX_RECOMMENDATIONS
		)
		if not isinstance(papers, list):
			raise TypeError("Papers must be a list of SearchResult objects.")
		for index, paper in enumerate(papers):
			if not isinstance(paper, SearchResult):
				raise TypeError(f"Paper at index {index} must be a SearchResult.")
		if not papers:
			return []

		candidates = papers[:MAX_GENERATION_CANDIDATES]
		papers_by_doc_id: dict[str, SearchResult] = {}
		for paper in candidates:
			papers_by_doc_id.setdefault(paper.arxiv_id, paper)
		payload = {
			"request": request,
			"max_recommendations": min(limit, len(papers_by_doc_id)),
			"retrieved_papers": [
				{
					"doc_id": paper.arxiv_id,
					"title": paper.title,
					"abstract": paper.abstract or "",
				}
				for paper in papers_by_doc_id.values()
			],
		}
		parsed = self._generate(
			json.dumps(payload, ensure_ascii=False),
			self.instructions,
			_GenerationResponse,
		)

		recommendations: list[PaperRecommendation] = []
		seen_doc_ids: set[str] = set()
		for item in parsed.recommendations:
			if item.doc_id not in papers_by_doc_id:
				raise RuntimeError(
					f"Generation returned a paper outside the retrieved candidates: {item.doc_id!r}."
				)
			if item.doc_id in seen_doc_ids:
				raise RuntimeError(
					f"Generation returned duplicate paper {item.doc_id!r}."
				)
			reason = item.reason.strip()
			if not reason:
				raise RuntimeError(
					f"Generation returned an empty reason for paper {item.doc_id!r}."
				)
			seen_doc_ids.add(item.doc_id)
			recommendations.append(
				PaperRecommendation(
					rank=len(recommendations) + 1,
					paper=papers_by_doc_id[item.doc_id],
					reason=reason,
				)
			)
		return recommendations[:limit]
