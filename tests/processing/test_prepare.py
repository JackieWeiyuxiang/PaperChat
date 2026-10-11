"""Integration checks for immutable inputs, stable passages, and storage."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from arxiv_paper_recommend.corpus.processing import prepare


def paper(index=1, *, version=1, title=None, abstract=None):
    paper_id = f"2501.{index:05d}"
    title = title if title is not None else f"RAG retrieval experiment {index}"
    abstract = abstract if abstract is not None else f"We investigate retrieval and compare evidence sources for study {index}."
    return {"paper_id": paper_id, "version": version, "title": title, "abstract": abstract,
            "text": title + "\n\n" + abstract, "authors": ["An Author"],
            "categories": ["cs.CL", "cs.IR"], "primary_category": "cs.CL",
            "published": "2025-01-02T10:00:00Z", "updated": "2025-02-03T10:00:00Z",
            "url": f"https://arxiv.org/abs/{paper_id}", "pdf_url": f"https://arxiv.org/pdf/{paper_id}v{version}",
            "doi": None, "source": "arxiv_api", "fetched_at": "2026-10-02T00:00:00+00:00"}


def supplements(rows, *, found=False):
    return [{"paper_id": row["paper_id"], "version": row["version"],
             "author_keywords": ["retrieval-augmented generation"] if found else None,
             "author_keywords_status": "found" if found else "pending",
             "author_keywords_provenance": {"url": row["pdf_url"], "locator": "pdf:page=1"} if found else None}
            for row in rows]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class PrepareRecordsTests(unittest.TestCase):
    def test_unicode_and_whitespace_cleaning_preserves_math_case_and_punctuation(self):
        row = paper(title="\ufeffCafe\u0301\tRAG\u00a0Study!", abstract="\u200bWe\n use  $X_i \\ne x_i$; μ ≥ 0.\x00 Next.")
        original = copy.deepcopy(row)
        docs, chunks, removals = prepare.prepare_records([row], supplements([row]))
        self.assertEqual(docs[0]["title"], "Café RAG Study!")
        self.assertEqual(docs[0]["abstract"], "We use $X_i \\ne x_i$; μ ≥ 0. Next.")
        self.assertEqual(chunks[0]["text"], docs[0]["text"])
        self.assertEqual(row, original)
        self.assertEqual(removals, [])

    def test_long_abstract_is_preserved_whole_with_stable_versioned_ids(self):
        abstract = " ".join(f"word{number}" for number in range(650)) + "."
        row = paper(version=2, abstract=abstract)
        docs, chunks, _ = prepare.prepare_records([row], supplements([row]))
        self.assertEqual(docs[0]["doc_id"], "arxiv:2501.00001v2")
        self.assertEqual(chunks[0]["chunk_id"], "arxiv:2501.00001v2:abstract:0000")
        self.assertEqual(docs[0]["abstract"], abstract)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["char_start"], 0)
        self.assertEqual(chunks[0]["char_end"], len(docs[0]["text"]))
        self.assertIn("abstract_above_500_words", docs[0]["quality_flags"])

    def test_same_identity_conflict_and_multiple_versions_fail(self):
        examples = [[paper(), paper(abstract="This conflicts with the first record.")],
                    [paper(version=1), paper(version=2)]]
        for rows in examples:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                prepare.prepare_records(rows, supplements(rows))

    def test_invalid_ids_versions_and_inconsistent_source_text_fail(self):
        cases = [{"paper_id": "not-an-arxiv-id"}, {"version": True}, {"version": 0},
                 {"text": "This was altered independently of the abstract."}]
        for changes in cases:
            row = {**paper(), **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                prepare.prepare_records([row], supplements([row]))

    def test_duplicate_records_and_normalized_text_have_distinct_audit_reasons(self):
        row = paper()
        duplicate_text = paper(2, title="  " + row["title"], abstract=row["abstract"] + "\n")
        rows = [row, copy.deepcopy(row), duplicate_text]
        docs, chunks, removals = prepare.prepare_records(rows, supplements(rows))
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual([item["reason"] for item in removals],
                         ["identical_record", "identical_normalized_title_and_abstract"])
        self.assertEqual([item["kept_doc_id"] for item in removals], [docs[0]["doc_id"]] * 2)
        self.assertEqual(removals[1]["removed_doc_id"], "arxiv:2501.00002v1")

    def test_deduplication_preserves_mathematical_case_distinctions(self):
        rows = [paper(1, title="A variable study", abstract="We study $x$ in retrieval."),
                paper(2, title="A variable study", abstract="We study $X$ in retrieval.")]
        docs, chunks, removals = prepare.prepare_records(rows, supplements(rows))
        self.assertEqual(len(docs), 2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(removals, [])

    def test_mismatched_keyword_id_and_inconsistent_found_status_fail(self):
        rows = [paper()]
        wrong_id = supplements(rows)
        wrong_id[0]["paper_id"] = "2501.99999"
        empty_found = supplements(rows, found=True)
        empty_found[0]["author_keywords"] = []
        unconfirmed = supplements(rows)
        unconfirmed[0]["author_keywords"] = ["unconfirmed"]
        for keywords in (wrong_id, empty_found, unconfirmed):
            with self.subTest(keywords=keywords), self.assertRaises(ValueError):
                prepare.prepare_records(rows, keywords)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "raw_papers.jsonl"
        self.output = self.root / "processed"
        self.report = self.root / "statistics.md"
        rows = [paper(1), paper(2, abstract="Another study of evidence retrieval.")]
        self.rows = [{**row, **kw} for row, kw in zip(rows, supplements(rows, found=True))]
        self.original = prepare.jsonl(self.rows)
        self.input.write_bytes(self.original)

    def test_one_searchable_output_and_report_without_input_changes(self):
        from arxiv_paper_recommend.retrieval.bm25.baseline import BM25Index
        result = prepare.build(self.input, self.output, self.report)
        self.assertEqual(self.input.read_bytes(), self.original)
        self.assertEqual({p.name for p in self.output.iterdir()}, {"papers.jsonl"})
        records = read_jsonl(self.output / "papers.jsonl")
        self.assertEqual(len(records), 2)
        self.assertEqual(result["keywords_found"], 2)
        self.assertTrue(self.report.is_file())
        self.assertTrue(BM25Index(records).search("retrieval", 2))
        self.assertNotIn("quality_flags", records[0])
        self.assertEqual(records[0]["author_keywords"], self.rows[0]["author_keywords"])
        before = (self.output / "papers.jsonl").read_bytes(), self.report.read_bytes()
        prepare.build(self.input, self.output, self.report)
        self.assertEqual(before, ((self.output / "papers.jsonl").read_bytes(), self.report.read_bytes()))

    def test_deduplication_and_cleaning_in_full_pipeline(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        self.rows[1]["title"] = "  Second\tstudy "
        self.rows[1]["text"] = self.rows[1]["title"] + "\n\n" + self.rows[1]["abstract"]
        self.input.write_bytes(prepare.jsonl(self.rows))
        result = prepare.build(self.input, self.output, self.report)
        self.assertEqual(result["duplicates_removed"], 1)
        self.assertEqual(read_jsonl(self.output / "papers.jsonl")[1]["title"], "Second study")

    def test_input_and_output_collisions_fail(self):
        with self.assertRaises(ValueError):
            prepare.build(self.input, self.input)
        with self.assertRaises(ValueError):
            prepare.build(self.input, self.output, self.input)
        with self.assertRaises(ValueError):
            prepare.build(self.input, self.output, self.output / "papers.jsonl")
        self.assertEqual(self.input.read_bytes(), self.original)

    def test_incomplete_keyword_metadata_fails(self):
        del self.rows[0]["author_keywords_status"]
        self.input.write_bytes(prepare.jsonl(self.rows))
        with self.assertRaises(ValueError):
            prepare.build(self.input, self.output, self.report)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
