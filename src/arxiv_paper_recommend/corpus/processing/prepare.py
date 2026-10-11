"""Clean frozen paper metadata, make one abstract passage per paper, and store it.

This module performs no network access, model inference, or retrieval indexing.
The output is one JSONL record per paper, ready for retrieval.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from .statistics import compute_statistics

ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "abstract-corpus-v1"
KEYWORD_FIELDS = {"author_keywords", "author_keywords_status", "author_keywords_provenance"}
KEYWORD_STATUSES = {"found", "not_found_in_source", "needs_review", "fetch_failed", "pending"}


def sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def encoded(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def write_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(body)
    temporary.replace(path)


def parse_rows(body: bytes) -> list[dict]:
    rows = [json.loads(line) for line in body.decode("utf-8-sig").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Every JSONL record must be an object")
    return rows


def doc_id(row: dict) -> str:
    paper_id, version = row.get("paper_id"), row.get("version")
    if not isinstance(paper_id, str) or not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[A-Za-z.-]+/\d{7})", paper_id):
        raise ValueError(f"Invalid paper_id: {paper_id!r}")
    if type(version) is not int or version < 1:
        raise ValueError(f"Invalid version: {paper_id}")
    return f"arxiv:{paper_id}v{version}"


def clean_text(value: str) -> str:
    """NFC + whitespace normalization; preserve case, punctuation and LaTeX.

    Remove BOM/zero-width space; replace other control characters with a space.
    Do not use compatibility normalization, lowercase, stem, or remove stopwords.
    """
    value = unicodedata.normalize("NFC", value)
    value = "".join("" if char in "\ufeff\u200b" else
                    " " if unicodedata.category(char) == "Cc" else char for char in value)
    return " ".join(value.split())


def quality_flags(row: dict) -> list[str]:
    flags = []
    title, abstract = row["title"], row["abstract"]
    for name, text, lower, upper in (("title", title, 3, 40), ("abstract", abstract, 50, 500)):
        count = len(text.split())
        if count < lower:
            flags.append(f"{name}_below_{lower}_words")
        if count > upper:
            flags.append(f"{name}_above_{upper}_words")
    if "\ufffd" in row["text"]:
        flags.append("unicode_replacement_character")
    if re.search(r"\\[A-Za-z]+(?:\{|\[|\b)", row["text"]):
        flags.append("latex_markup_present")
    if not re.search(r"[.!?。！？][\"'”’)}\]]*$", abstract):
        flags.append("abstract_ending_needs_check")
    if datetime.fromisoformat(row["updated"].replace("Z", "+00:00")) < datetime.fromisoformat(row["published"].replace("Z", "+00:00")):
        flags.append("updated_before_published")
    return flags


def validate_keywords(rows: list[dict], base: list[dict]) -> None:
    if len(rows) != len(base):
        raise ValueError("Keyword snapshot row count differs from input")
    for original, item in zip(base, rows):
        if doc_id(original) != doc_id(item):
            raise ValueError("Keyword snapshot ID/version differs from input")
        status, keywords = item.get("author_keywords_status"), item.get("author_keywords")
        if status not in KEYWORD_STATUSES:
            raise ValueError(f"Unknown author-keyword status: {status}")
        if status == "found":
            if not isinstance(keywords, list) or not keywords or any(not isinstance(k, str) or not k.strip() for k in keywords):
                raise ValueError("Found author keywords must be a nonempty list of strings")
        elif keywords is not None:
            raise ValueError("Unconfirmed author keywords must be null")


def prepare_records(base: list[dict], keywords: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return documents, passages and an auditable exact-duplicate removal log."""
    validate_keywords(keywords, base)
    documents, passages, removals = [], [], []
    seen_ids, seen_text, paper_versions = {}, {}, {}
    for original, supplement in zip(base, keywords):
        identity = doc_id(original)
        for field in ("title", "abstract", "published", "updated", "url", "pdf_url"):
            if not isinstance(original.get(field), str) or not original[field].strip():
                raise ValueError(f"Missing {field}: {identity}")
        if not original.get("authors") or any(not isinstance(a, str) or not a.strip() for a in original["authors"]):
            raise ValueError(f"Missing/invalid authors: {identity}")
        if not isinstance(original.get("categories"), list) or not original["categories"]:
            raise ValueError(f"Missing categories: {identity}")
        for field in ("published", "updated"):
            date = datetime.fromisoformat(original[field].replace("Z", "+00:00"))
            if date.tzinfo is None:
                raise ValueError(f"Timezone missing for {field}: {identity}")
        title, abstract = clean_text(original["title"]), clean_text(original["abstract"])
        if not title or not abstract:
            raise ValueError(f"Empty text after normalization: {identity}")
        text = title + "\n\n" + abstract
        source_text = original["title"] + "\n\n" + original["abstract"]
        if original.get("text") != source_text:
            raise ValueError(f"Frozen text differs from title + abstract: {identity}")
        previous_version = paper_versions.get(original["paper_id"])
        if previous_version is not None and previous_version != original["version"]:
            raise ValueError(f"Multiple versions in frozen input: {original['paper_id']}")
        paper_versions[original["paper_id"]] = original["version"]
        record_hash = sha256(json.dumps(original, ensure_ascii=False, sort_keys=True).encode())
        if identity in seen_ids:
            if seen_ids[identity] != record_hash:
                raise ValueError(f"Conflicting records with the same ID/version: {identity}")
            removals.append({"removed_doc_id": identity, "kept_doc_id": identity, "reason": "identical_record"})
            continue
        seen_ids[identity] = record_hash
        # Remove only identical normalized title + abstract, never near matches.
        duplicate_hash = sha256(text.encode("utf-8"))
        if duplicate_hash in seen_text:
            removals.append({"removed_doc_id": identity, "kept_doc_id": seen_text[duplicate_hash],
                             "reason": "identical_normalized_title_and_abstract"})
            continue
        seen_text[duplicate_hash] = identity
        document = {**original, **{field: supplement[field] for field in KEYWORD_FIELDS},
                    "doc_id": identity, "title": title, "abstract": abstract, "text": text,
                    "authors": [clean_text(name) for name in original["authors"]],
                    "source_text_sha256": sha256(source_text.encode("utf-8")),
                    "text_sha256": sha256(text.encode("utf-8")), "source_record_sha256": record_hash,
                    "cleaning_changed_fields": [field for field, value in (("title", title), ("abstract", abstract))
                                                if original[field] != value]}
        if document["authors"] != original["authors"]:
            document["cleaning_changed_fields"].append("authors")
        document["quality_flags"] = quality_flags(document)
        documents.append(document)
        passages.append({"chunk_id": identity + ":abstract:0000", "doc_id": identity,
                         "paper_id": original["paper_id"], "version": original["version"],
                         "chunk_index": 0, "section": "title_and_abstract", "text": text,
                         "char_start": 0, "char_end": len(text), "char_count": len(text),
                         "word_count_whitespace": len(text.split()), "text_sha256": document["text_sha256"]})
    return documents, passages, removals


def build(input_path: Path, output: Path, report_path: Path | None = None) -> dict:
    from .report import render_report

    input_path, output = input_path.resolve(), output.resolve()
    if input_path == output or input_path == output / "papers.jsonl":
        raise ValueError("Processed text must not overwrite the original input")
    if report_path is not None:
        report_path = report_path.resolve()
        if report_path in (input_path, output / "papers.jsonl"):
            raise ValueError("Report must not overwrite input or processed text")
    body = input_path.read_bytes()
    base = parse_rows(body)
    if not base:
        raise ValueError("Input corpus is empty")
    keywords = []
    for row in base:
        present = KEYWORD_FIELDS.intersection(row)
        if present and present != KEYWORD_FIELDS:
            raise ValueError("Incomplete author-keyword metadata")
        fields = {key: row[key] for key in KEYWORD_FIELDS} if present else {
            "author_keywords": None, "author_keywords_status": "pending",
            "author_keywords_provenance": None,
        }
        keywords.append({"paper_id": row["paper_id"], "version": row["version"], **fields})
    documents, passages, removed = prepare_records(base, keywords)
    stats = compute_statistics(documents, passages)
    excluded = {"quality_flags", "source_text_sha256", "text_sha256",
                "source_record_sha256", "cleaning_changed_fields"}
    records = [{**{key: value for key, value in doc.items() if key not in excluded},
                "chunk_id": passage["chunk_id"]}
               for doc, passage in zip(documents, passages)]
    if input_path.read_bytes() != body:
        raise ValueError("Original input changed during processing")
    write_bytes(output / "papers.jsonl", jsonl(records))
    if report_path is not None:
        write_bytes(report_path, render_report(documents, stats, len(base), len(removed)).encode("utf-8"))
    return {"input_records": len(base), "documents": len(records),
            "duplicates_removed": len(removed), "corpus_words": stats["counts"]["corpus_words"],
            "keywords_found": stats["author_keywords"]["documents_with_keywords"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw_papers.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--report", type=Path, default=ROOT / "data/statistics.md")
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output, args.report), ensure_ascii=False))


if __name__ == "__main__":
    main()
