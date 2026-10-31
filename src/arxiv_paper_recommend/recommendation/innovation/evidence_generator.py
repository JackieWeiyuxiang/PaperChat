"""Assess retrieved papers with explanations grounded in marked source spans."""

from ..base import BaseRecommendationGenerator


class EvidenceGroundedRecommendationGenerator(BaseRecommendationGenerator):
	"""Require paper-specific claims to be supported by the marked citations."""

	instructions = """
First identify one or two informative, continuous source spans and place them in
the required citation markers. Base your explanation on these marked spans alone.
Every paper-specific method, result, capability, or detail in reason must be
explicitly supported by the cited text. Do not rely on uncited parts of the title
or abstract to add paper-specific claims. Choose longer spans when needed to
support the explanation. If the source cannot support a recommendation, return
recommended=false, cite the available scope, and explain the missing support
without claiming unmentioned methods or capabilities are absent from the paper.
"""
