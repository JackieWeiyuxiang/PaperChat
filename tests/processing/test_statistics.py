"""Offline statistics checks: counting semantics, coverage, and edge cases."""

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from arxiv_paper_recommend.corpus.processing.statistics import compute_statistics


def document(index=1, **overrides):
    row = {"doc_id": f"arxiv:{index}v1", "paper_id": str(index), "version": 1,
           "title": "A short title", "abstract": "A short abstract.",
           "text": "A short title\n\nA short abstract.", "authors": ["A. Author"],
           "categories": ["cs.CL"], "primary_category": "cs.CL",
           "published": "2025-06-01T00:00:00Z", "updated": "2025-06-02T00:00:00Z",
           "author_keywords": None, "author_keywords_status": "pending", "quality_flags": []}
    row.update(overrides)
    return row


class StatisticsTests(unittest.TestCase):
    def test_passages_do_not_double_corpus_words(self):
        row = document()
        result = compute_statistics([row], [{"chunk_id": "1", "doc_id": row["doc_id"], "text": row["text"]}])
        self.assertEqual(result["counts"]["corpus_words"], 6)
        self.assertEqual(result["word_statistics"]["passages"]["total_words"], 6)
        self.assertEqual(result["word_statistics"]["combined"]["total_words"], 6)
        self.assertEqual(result["lexical_statistics"]["combined"], {"tokens": 6, "types": 4})

    def test_lexical_types_have_explicit_case_and_punctuation_policy(self):
        row = document(text="RAG rag retrieval-augmented author's AUTHOR’S 123 中文")
        result = compute_statistics([row], [])
        self.assertEqual(result["counts"]["corpus_words"], 7)
        self.assertEqual(result["lexical_statistics"]["combined"], {"tokens": 5, "types": 3})

    def test_quantiles_are_interpolated_and_singletons_work(self):
        rows = [document(index, text="x " * length) for index, length in enumerate((0, 10))]
        result = compute_statistics(rows, [])["word_statistics"]["combined"]
        self.assertEqual(result["mean"], 5)
        self.assertEqual(result["median"], 5)
        self.assertEqual(result["p05"], .5)
        self.assertEqual(result["p25"], 2.5)
        self.assertEqual(result["p75"], 7.5)
        self.assertEqual(result["p95"], 9.5)
        singleton = compute_statistics([document(text="one two")], [])["word_statistics"]["combined"]
        self.assertEqual(singleton["p95"], 2)

    def test_empty_input_is_json_safe_without_nan(self):
        result = compute_statistics([], [])
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["counts"]["corpus_words"], 0)
        self.assertIsNone(result["word_statistics"]["combined"]["median"])
        self.assertEqual(result["lexical_statistics"]["combined"]["types"], 0)
        self.assertIsNone(result["author_keywords"]["coverage_fraction"])

    def test_pending_and_failed_are_not_treated_as_absent_keywords(self):
        rows = [document(1, author_keywords_status="found", author_keywords=["RAG"]),
                document(2), document(3, author_keywords_status="fetch_failed"),
                document(4, author_keywords_status="not_found_in_source"),
                document(5, author_keywords_status="found", author_keywords=[]),
                document(6, author_keywords_status="pending", author_keywords=["stray"])]
        stats = compute_statistics(rows, [])["author_keywords"]
        self.assertEqual(stats["documents_with_keywords"], 1)
        self.assertEqual(stats["coverage_fraction"], 1 / 6)
        self.assertEqual(stats["status_counts"]["pending"], 2)
        self.assertEqual(stats["status_counts"]["not_found_in_source"], 1)
        self.assertEqual(stats["inconsistent_record_indices"], [4, 5])

    def test_multilabel_categories_and_flags_count_documents_once(self):
        rows = [document(categories=["cs.CL", "cs.IR", "cs.IR"], quality_flags=["short", "short"]),
                document(2, categories=["cs.IR"], primary_category="cs.IR", quality_flags=["short", "check"])]
        result = compute_statistics(rows, [])
        self.assertEqual(result["distributions"]["all_category_document_counts"], {"cs.CL": 1, "cs.IR": 2})
        self.assertEqual(result["quality_flags"], {"check": 1, "short": 2})

    def test_missing_fields_and_invalid_dates_are_visible(self):
        rows = [document(title="  ", abstract=None, authors=[], published="nonsense", updated=None), {}]
        result = compute_statistics(rows, [])
        self.assertEqual(result["missing_fields"]["documents"]["title"], 2)
        self.assertEqual(result["missing_fields"]["documents"]["authors"], 2)
        self.assertEqual(result["distributions"]["published_year"], {"invalid": 1, "missing": 1})
        self.assertEqual(result["word_statistics"]["title"]["total_words"], 0)

    def test_duplicate_extras_and_blank_exclusion(self):
        rows = [document(1), document(1), document(1), document(2, text=""), document(3, text="")]
        result = compute_statistics(rows, [])["duplicates"]
        self.assertEqual(result["doc_id"]["groups"], 1)
        self.assertEqual(result["doc_id"]["duplicate_records"], 2)
        self.assertEqual(result["document_text"]["records_in_duplicate_groups"], 3)
        self.assertEqual(result["document_text"]["details"], [{"record_indices": [0, 1, 2]}])

    def test_length_thresholds_do_not_discard_boundary_records(self):
        rows = [document(i, abstract="x " * length) for i, length in enumerate((49, 50, 500, 501))]
        original = copy.deepcopy(rows)
        result = compute_statistics(rows, [])
        warning = result["length_warnings"]["abstract"]
        self.assertEqual(warning["short_record_indices"], [0])
        self.assertEqual(warning["long_record_indices"], [3])
        self.assertEqual(result["counts"]["documents"], 4)
        self.assertEqual(rows, original)

    def test_missing_and_unknown_chunk_parents_are_orphans(self):
        rows = [document(1), document(2)]
        chunks = [{"chunk_id": "1", "doc_id": "arxiv:1v1", "text": "first"},
                  {"chunk_id": "2", "doc_id": "unknown", "text": "second"},
                  {"chunk_id": "3", "text": "third"}]
        self.assertEqual(compute_statistics(rows, chunks)["chunk_integrity"],
                         {"orphan_passages": 2, "documents_without_passages": 1})


if __name__ == "__main__":
    unittest.main()
