"""Behavior and hand-calculated scoring tests, using only synthetic passages."""

from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from arxiv_paper_recommend.retrieval.bm25.baseline import BM25Index, STOPWORDS, tokenize


def passage(chunk_id: str, text: str, **extra) -> dict:
    return {"chunk_id": chunk_id, "doc_id": f"doc:{chunk_id}", "text": text, **extra}


class TokenizeTests(unittest.TestCase):
    def test_normalization_punctuation_hyphen_and_numbers(self):
        self.assertEqual(
            tokenize("Ｔｈｅ RAG-based Retrieval, 2025! Straße"),
            ["rag", "based", "retrieval", "2025", "strasse"],
        )

    def test_no_stemming_or_synonyms(self):
        self.assertEqual(tokenize("models model retrieval search"),
                         ["models", "model", "retrieval", "search"])

    def test_stopwords_and_empty_input(self):
        self.assertEqual(tokenize(" "), [])
        self.assertEqual(tokenize(" ".join(sorted(STOPWORDS))), [])
        self.assertEqual(tokenize("中文"), [])

    def test_non_string_rejected(self):
        with self.assertRaises(TypeError):
            tokenize(None)


class BM25Tests(unittest.TestCase):
    def test_hand_calculated_score_and_length_normalization(self):
        index = BM25Index([
            passage("a", "cat cat dog"),
            passage("b", "cat"),
            passage("c", "bird bird"),
        ])
        results = {row["chunk_id"]: row for row in index.search("cat")}
        idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
        score_a = idf * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 3 / 2))
        score_b = idf * 1 * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 1 / 2))
        self.assertAlmostEqual(results["a"]["score"], score_a, places=14)
        self.assertAlmostEqual(results["b"]["score"], score_b, places=14)
        self.assertEqual(results["a"]["matched_terms"], ["cat"])
        self.assertNotIn("c", results)
        self.assertEqual(index.size, 3)
        self.assertEqual(index.avgdl, 2)

    def test_same_term_frequency_prefers_shorter_passage(self):
        index = BM25Index([passage("long", "cat dog bird"), passage("short", "cat")])
        self.assertEqual([row["chunk_id"] for row in index.search("cat")], ["short", "long"])

    def test_b_zero_disables_length_normalization(self):
        index = BM25Index([passage("a", "cat dog bird"), passage("b", "cat")], b=0)
        results = index.search("cat")
        self.assertEqual(results[0]["score"], results[1]["score"])

    def test_k1_zero_removes_frequency_weight(self):
        index = BM25Index([passage("a", "cat cat cat"), passage("b", "cat")], k1=0)
        results = index.search("cat")
        self.assertEqual(results[0]["score"], results[1]["score"])
        self.assertAlmostEqual(results[0]["score"], math.log(1 + 0.5 / 2.5))

    def test_repeated_query_tokens_do_not_reweight(self):
        index = BM25Index([passage("a", "cat dog"), passage("b", "dog")])
        self.assertEqual(index.rank_all("cat dog"), index.rank_all("dog CAT cat cat dog"))
        self.assertEqual(index.rank_all("cat dog")[0]["matched_terms"], ["cat", "dog"])

    def test_identical_analysis_for_query_and_document(self):
        index = BM25Index([passage("a", "THE ＲＡＧ-based System")])
        self.assertEqual(index.search("rag based"), index.search("The ＲＡＧ-BASED!"))
        self.assertEqual(index.avgdl, 3)

    def test_empty_oov_stopwords_and_zero_padding(self):
        index = BM25Index([passage("a", "cat"), passage("b", "dog")])
        for query in ("", " ", "the and", "bird", "中文"):
            with self.subTest(query=query):
                self.assertEqual(index.search(query), [])
        self.assertEqual(len(index.search("cat", k=20)), 1)
        self.assertEqual(index.search("cat", k=0), [])

    def test_empty_index_and_all_empty_passages(self):
        self.assertEqual(BM25Index([]).search("cat"), [])
        index = BM25Index([passage("a", ""), passage("b", "the")])
        self.assertEqual(index.avgdl, 0)
        self.assertEqual(index.search("cat"), [])

    def test_empty_passage_is_in_document_count(self):
        index = BM25Index([passage("a", "cat"), passage("b", "")], b=0)
        self.assertAlmostEqual(index.search("cat")[0]["score"], math.log(2))

    def test_ties_use_chunk_id_and_are_independent_of_input_order(self):
        rows = [passage("z", "cat"), passage("a", "cat"), passage("m", "cat")]
        result = BM25Index(rows).rank_all("cat")
        self.assertEqual([row["chunk_id"] for row in result], ["a", "m", "z"])
        self.assertEqual(result, BM25Index(list(reversed(rows))).rank_all("cat"))
        self.assertEqual(BM25Index(rows).search("cat", 2), result[:2])

    def test_source_and_index_are_isolated_from_caller_mutations(self):
        rows = [passage("a", "cat", metadata={"author_keywords": ["dog"]})]
        original = deepcopy(rows)
        index = BM25Index(rows)
        index.search("cat")
        self.assertEqual(rows, original)
        rows[0]["text"] = "dog"
        rows[0]["chunk_id"] = "renamed"
        results = index.search("cat")
        self.assertEqual(results[0]["chunk_id"], "a")
        results[0]["matched_terms"].append("fake")
        self.assertEqual(index.search("cat")[0]["matched_terms"], ["cat"])

    def test_only_text_is_indexed(self):
        index = BM25Index([passage("dog", "cat", author_keywords=["bird"], title="fish")])
        for query in ("dog", "bird", "fish"):
            self.assertEqual(index.search(query), [])

    def test_duplicate_chunks_rejected_but_shared_documents_allowed(self):
        with self.assertRaisesRegex(ValueError, "duplicate chunk_id"):
            BM25Index([passage("a", "cat"), passage("a", "dog")])
        index = BM25Index([passage("a", "cat", doc_id="shared"),
                           passage("b", "cat", doc_id="shared")])
        self.assertEqual(len(index.search("cat")), 2)

    def test_invalid_parameters(self):
        for value in (-1, float("inf"), float("nan"), True, "1.2", None):
            with self.subTest(k1=value), self.assertRaises(ValueError):
                BM25Index([], k1=value)
        for value in (-0.1, 1.1, float("inf"), float("nan"), True, "0.75", None):
            with self.subTest(b=value), self.assertRaises(ValueError):
                BM25Index([], b=value)

    def test_invalid_record_fields(self):
        invalid = [None, {}, passage("", "cat"), passage("a", None),
                   passage("a", "cat", doc_id=" "), {**passage("a", "cat"), "chunk_id": 1}]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                BM25Index([row])

    def test_invalid_k(self):
        index = BM25Index([passage("a", "cat")])
        for value in (-1, 1.5, True, "10", None):
            with self.subTest(k=value), self.assertRaises(ValueError):
                index.search("cat", k=value)


if __name__ == "__main__":
    unittest.main()
