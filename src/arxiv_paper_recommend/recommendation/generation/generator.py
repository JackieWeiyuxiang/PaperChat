"""Assess retrieved papers using their full titles and abstracts."""

from ..base import BaseRecommendationGenerator


class RecommendationGenerator(BaseRecommendationGenerator):
	"""Create a ranked shortlist with reasons and marked verbatim citations."""

	instructions = """
        Explain the paper's relevance using the full title and abstract.

        - **reason**: connect the paper to the request, staying faithful to its stated scope, methods, and findings.
        - Choose informative citations that help the reader verify the assessment.

    """
