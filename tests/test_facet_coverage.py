"""Unit tests for facet-coverage ranking and registration."""

import unittest

from arxiv_paper_recommend.retrieval import SearchEngine
from arxiv_paper_recommend.retrieval.innovation.facet_coverage import (
	_facet_coverage_scores,
	_validate_coverage_weight,
)


class FacetCoverageScoreTests(unittest.TestCase):
	def test_weakest_facet_can_promote_a_balanced_candidate(self) -> None:
		document_ids = [1, 2, 3]
		facet_scores = {
			"topic": {1: 0.9, 2: 0.8, 3: 0.7},
			"constraint": {1: 0.2, 2: 0.9, 3: 0.8},
		}

		scores = _facet_coverage_scores(document_ids, facet_scores, 0.75)

		self.assertGreater(scores[2], scores[1])
		self.assertGreater(scores[1], scores[3])

	def test_missing_facet_score_has_zero_coverage(self) -> None:
		scores = _facet_coverage_scores(
			[1, 2],
			{"facet": {1: 0.5, 2: None}},
			1.0,
		)

		self.assertEqual(scores[1], 1.0)
		self.assertEqual(scores[2], 0.0)

	def test_coverage_weight_validation(self) -> None:
		self.assertEqual(_validate_coverage_weight(0.3), 0.3)
		with self.assertRaises(TypeError):
			_validate_coverage_weight(True)
		with self.assertRaises(ValueError):
			_validate_coverage_weight(1.1)


class SearchEngineRegistrationTests(unittest.TestCase):
	def test_facet_coverage_is_registered(self) -> None:
		with SearchEngine() as engine:
			self.assertIn("FACET_COVERAGE", engine.available_methods)


if __name__ == "__main__":
	unittest.main()
