"""Assess retrieved papers using their full titles and abstracts."""

from ..base import BaseRecommendationGenerator


class RecommendationGenerator(BaseRecommendationGenerator):
	"""Create a ranked shortlist with reasons and marked verbatim citations."""

	instructions = """
Explain the paper's connection to the research request using the supplied title
and abstract as a whole. Cite one or two informative source spans that help the
reader check your assessment. Keep the explanation concise and faithful to the
paper's stated scope, methods, and findings.
"""
