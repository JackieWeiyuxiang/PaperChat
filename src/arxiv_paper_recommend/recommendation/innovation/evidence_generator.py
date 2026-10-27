"""Generate paper recommendations with verifiable title or abstract evidence."""

import json

from pydantic import BaseModel, ConfigDict, Field

from arxiv_paper_recommend.common.validation import validate_integer, validate_text
from arxiv_paper_recommend.llm import StructuredGenerator
from arxiv_paper_recommend.retrieval import SearchResult

from .models import EvidenceGroundedRecommendation

MAX_RECOMMENDATIONS = 10
MAX_GENERATION_CANDIDATES = 30


class _GeneratedEvidenceRecommendation(BaseModel):
	"""Structured model output for one evidence-grounded recommendation."""

	model_config = ConfigDict(extra="forbid")
	doc_id: str = Field(min_length=1)
	evidence: str = Field(min_length=20, max_length=600)
	reason: str = Field(min_length=1)


class _EvidenceGenerationResponse(BaseModel):
	"""Structured list of evidence-grounded recommendations."""

	model_config = ConfigDict(extra="forbid")
	recommendations: list[_GeneratedEvidenceRecommendation] = Field(
		max_length=MAX_RECOMMENDATIONS,
	)

def _validate_evidence(
	evidence: str,
	paper: SearchResult,
) -> str:
	"""Require one exact, continuous evidence span from the title or abstract."""
	cleaned_evidence = evidence.strip()
	if not cleaned_evidence:
		raise RuntimeError(
			f"Generation returned empty evidence for paper {paper.arxiv_id!r}."
		)

	title = paper.title or ""
	abstract = paper.abstract or ""
	if cleaned_evidence not in title and cleaned_evidence not in abstract:
		raise RuntimeError(
			"Generation returned evidence that is not an exact span from "
			f"paper {paper.arxiv_id!r}."
		)

	return cleaned_evidence

class EvidenceGroundedRecommendationGenerator(StructuredGenerator):
	"""Create ranked paper recommendations with verifiable evidence spans."""

	instructions = """
You recommend academic papers for a user's research request using only the
retrieved papers supplied in the input. Return up to the requested number of
papers, ordered by usefulness to the request. Select only supplied doc_id values
and never invent papers, findings, methods, or claims.

For every selected paper, copy one short, informative, continuous evidence span
verbatim from that paper's title or abstract. The evidence must match the source
characters exactly. Do not paraphrase the evidence, combine non-contiguous
spans, add quotation marks, or insert ellipses unless those characters already
appear in the source.

Write one concise sentence explaining how the evidence connects the paper to
the user's request. The reason may explain relevance to the request, but it
must not introduce any paper-specific method, result, capability, or detail
that is not explicitly present in the selected evidence span. Do not rely on
other parts of the title or abstract that are outside the selected evidence.
If a stronger explanation requires more information, select a longer single
continuous evidence span that contains that information. Omit a paper when no
suitable evidence span supports a recommendation. Do not use outside knowledge.
"""
	def generate(
		self,
		request: str,
		papers: list[SearchResult],
		limit: int = MAX_RECOMMENDATIONS,
	) -> list[EvidenceGroundedRecommendation]:
		"""Generate recommendations backed by exact title or abstract evidence."""
		self._ensure_open()
		request = validate_text(request, "Research request", strip=True)
		validate_integer(
			limit,
			"Recommendation limit",
			minimum=1,
			maximum=MAX_RECOMMENDATIONS,
		)

		if not isinstance(papers, list):
			raise TypeError("Papers must be a list of SearchResult objects.")
		for index, paper in enumerate(papers):
			if not isinstance(paper, SearchResult):
				raise TypeError(
					f"Paper at index {index} must be a SearchResult."
				)
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
			_EvidenceGenerationResponse,
		)

		recommendations: list[EvidenceGroundedRecommendation] = []
		seen_doc_ids: set[str] = set()

		for item in parsed.recommendations:
			if item.doc_id not in papers_by_doc_id:
				raise RuntimeError(
					"Generation returned a paper outside the retrieved "
					f"candidates: {item.doc_id!r}."
				)
			if item.doc_id in seen_doc_ids:
				raise RuntimeError(
					f"Generation returned duplicate paper {item.doc_id!r}."
				)

			paper = papers_by_doc_id[item.doc_id]
			evidence = _validate_evidence(item.evidence, paper)
			reason = item.reason.strip()
			if not reason:
				raise RuntimeError(
					"Generation returned an empty reason for paper "
					f"{item.doc_id!r}."
				)

			seen_doc_ids.add(item.doc_id)
			recommendations.append(
				EvidenceGroundedRecommendation(
					rank=len(recommendations) + 1,
					paper=paper,
					evidence=evidence,
					reason=reason,
				)
			)

		return recommendations[:limit]
