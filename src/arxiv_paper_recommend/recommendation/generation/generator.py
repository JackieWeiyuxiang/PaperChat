"""Assess retrieved papers using their full titles and abstracts."""

from ..base import BaseRecommendationGenerator


class RecommendationGenerator(BaseRecommendationGenerator):
	"""Create a ranked shortlist with reasons based on titles and abstracts."""

	instructions = """
        Explain the paper's relevance using the full title and abstract.

        - **reason**: connect the paper to the request, staying faithful to its stated scope, methods, and findings.

    """
