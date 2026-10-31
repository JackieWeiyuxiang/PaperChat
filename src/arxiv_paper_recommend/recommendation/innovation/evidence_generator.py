"""Assess retrieved papers with explanations grounded in marked source spans."""

from ..base import BaseRecommendationGenerator


class EvidenceGroundedRecommendationGenerator(BaseRecommendationGenerator):
	"""Require paper-specific claims to be supported by the marked citations."""

	instructions = """
        Explain the assessment using marked source spans alone.

        - Select 1 to 2 informative spans before writing reason; use longer spans when needed.
        - **reason**: every paper-specific method, result, capability, or detail must be explicitly supported by the cited text, without claims from uncited content.
        - If the source cannot support a recommendation, return recommended=false, cite the available scope, and explain the missing support.

    """
