"""Deterministic descriptive statistics for a cleaned abstract corpus.

``words`` always means Python ``str.split()`` whitespace units, not a model's
tokens. Lexical types are counted separately with the ASCII English-word
pattern below. Document text and passage text describe the same knowledge;
their totals are deliberately never added together. Length thresholds are
review hints, not rejection or truncation rules.
"""

from collections import Counter, defaultdict
from datetime import datetime
import re
from statistics import mean, median


LEXICAL_PATTERN = r"[A-Za-z]+(?:[-'’][A-Za-z]+)*"
LEXICAL_RE = re.compile(LEXICAL_PATTERN)
LENGTH_THRESHOLDS = {
    "title": {"shorter_than": 3, "longer_than": 40},
    "abstract": {"shorter_than": 50, "longer_than": 500},
    "combined": {"shorter_than": 60, "longer_than": 550},
    "passages": {"shorter_than": 60, "longer_than": 550},
}
DOCUMENT_FIELDS = (
    "doc_id", "paper_id", "version", "title", "abstract", "text", "authors",
    "categories", "primary_category", "published", "updated", "author_keywords",
    "author_keywords_status", "quality_flags",
)
CHUNK_FIELDS = ("chunk_id", "doc_id", "text")


def _text(value):
    return value if isinstance(value, str) else ""


def _missing(value):
    return value is None or (isinstance(value, str) and not value.strip()) or value == []


def _sorted_counts(values):
    return dict(sorted(Counter(values).items()))


def _quantile(sorted_values, probability):
    """Linear interpolation at (n - 1) * probability; safe for one value."""
    location = (len(sorted_values) - 1) * probability
    low = int(location)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (location - low)


def _length_statistics(texts):
    lengths = sorted(len(text.split()) for text in texts)
    result = {"records": len(lengths), "total_words": sum(lengths)}
    result.update({name: None for name in ("min", "mean", "median", "p05", "p25", "p75", "p95", "max")})
    if lengths:
        result.update(min=lengths[0], mean=mean(lengths), median=median(lengths), max=lengths[-1])
        result.update({name: _quantile(lengths, probability) for name, probability in
                       (("p05", .05), ("p25", .25), ("p75", .75), ("p95", .95))})
    return result


def _lexical_statistics(texts):
    counts = Counter(token.casefold().replace("’", "'")
                     for text in texts for token in LEXICAL_RE.findall(text))
    return {"tokens": sum(counts.values()), "types": len(counts)}


def _year(value):
    if _missing(value):
        return "missing"
    try:
        return str(datetime.fromisoformat(value.replace("Z", "+00:00")).year)
    except (ValueError, TypeError, AttributeError):
        return "invalid"


def _duplicates(rows, field):
    """Count exact nonblank strings; duplicate_records excludes each first row."""
    positions = defaultdict(list)
    for index, row in enumerate(rows):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            positions[value].append(index)
    groups = [{"record_indices": indices} for indices in positions.values() if len(indices) > 1]
    return {"groups": len(groups),
            "duplicate_records": sum(len(group["record_indices"]) - 1 for group in groups),
            "records_in_duplicate_groups": sum(len(group["record_indices"]) for group in groups),
            "details": groups}


def _missing_fields(rows, fields):
    return {field: sum(_missing(row.get(field)) for row in rows) for field in fields}


def compute_statistics(documents: list[dict], chunks: list[dict]) -> dict:
    """Return JSON-safe statistics without modifying the input rows.

    Combined text is ``documents[*].text``; no title is prepended again. Blank
    or absent text contributes zero words. Quantiles use linear interpolation.
    Category and quality-flag frequencies count each value once per document.
    A keyword-bearing document must have status ``found`` and at least one
    nonblank string keyword. Pending/failed/not-found rows are kept separate.
    Exact duplicate comparisons do not normalize case or whitespace and omit
    missing/blank fields. Duplicate details use zero-based input row indices.
    """
    texts = {
        "title": [_text(row.get("title")) for row in documents],
        "abstract": [_text(row.get("abstract")) for row in documents],
        "combined": [_text(row.get("text")) for row in documents],
        "passages": [_text(row.get("text")) for row in chunks],
    }
    doc_ids = {row["doc_id"] for row in documents if isinstance(row.get("doc_id"), str) and row["doc_id"].strip()}
    chunk_doc_ids = {row["doc_id"] for row in chunks if isinstance(row.get("doc_id"), str) and row["doc_id"].strip()}
    keyword_statuses = _sorted_counts(
        _text(row.get("author_keywords_status")).strip() or "missing_status" for row in documents
    )
    keyword_rows = []
    keyword_inconsistencies = []
    for index, row in enumerate(documents):
        keywords = row.get("author_keywords")
        valid_keywords = [word for word in keywords if isinstance(word, str) and word.strip()] if isinstance(keywords, list) else []
        found = row.get("author_keywords_status") == "found"
        if found and valid_keywords:
            keyword_rows.append(valid_keywords)
        if (found and not valid_keywords) or (not found and valid_keywords):
            keyword_inconsistencies.append(index)

    warnings = {}
    for field, values in texts.items():
        lengths = [len(value.split()) for value in values]
        threshold = LENGTH_THRESHOLDS[field]
        short_indices = [index for index, length in enumerate(lengths) if length < threshold["shorter_than"]]
        long_indices = [index for index, length in enumerate(lengths) if length > threshold["longer_than"]]
        warnings[field] = {**threshold, "short_count": len(short_indices), "long_count": len(long_indices),
                           "short_record_indices": short_indices, "long_record_indices": long_indices}

    return {
        "schema_version": "1.0",
        "definitions": {
            "words": "Whitespace units: len(text.split()); not embedding/model tokens.",
            "word_lengths": "Per-record whitespace words, including zero for blank/missing text.",
            "percentiles": "Linear interpolation at sorted index (n - 1) * p; empty input yields null.",
            "combined": "documents.text; corpus word total excludes passages to avoid double counting.",
            "lexical_token_regex": LEXICAL_PATTERN,
            "lexical_normalization": "ASCII English-letter words with internal hyphen/apostrophe; casefold; curly apostrophe normalized to straight apostrophe. Numbers and other scripts excluded.",
            "lexical_types": "Number of distinct normalized lexical tokens across the specified text field.",
            "category_counts": "One count per distinct category per document; multi-category totals may exceed documents.",
            "quality_flag_counts": "One count per distinct quality flag per document.",
            "keyword_coverage": "Documents with status found and a nonempty string keyword list / all documents; pending, failed and not-found are separate statuses.",
            "missing_fields": "Absent, null, blank string or empty list. Empty author_keywords/quality_flags can be valid and are not automatically errors.",
            "duplicates": "Exact case-sensitive, whitespace-sensitive nonblank string equality; extra records beyond the first are counted as duplicate_records.",
            "record_indices": "Zero-based positions in the corresponding documents or chunks input list.",
            "length_warnings": "Strictly below/above the stated whitespace-word thresholds, including blank text; review hints only; no rows deleted.",
        },
        "counts": {
            "documents": len(documents), "passages": len(chunks),
            "unique_doc_ids": len(doc_ids),
            "unique_paper_ids": len({row["paper_id"] for row in documents if isinstance(row.get("paper_id"), str) and row["paper_id"].strip()}),
            "unique_chunk_ids": len({row["chunk_id"] for row in chunks if isinstance(row.get("chunk_id"), str) and row["chunk_id"].strip()}),
            "corpus_words": sum(len(value.split()) for value in texts["combined"]),
        },
        "word_statistics": {field: _length_statistics(values) for field, values in texts.items()},
        "lexical_statistics": {field: _lexical_statistics(values) for field, values in texts.items()},
        "distributions": {
            "published_year": _sorted_counts(_year(row.get("published")) for row in documents),
            "updated_year": _sorted_counts(_year(row.get("updated")) for row in documents),
            "primary_category": _sorted_counts(_text(row.get("primary_category")).strip() or "missing" for row in documents),
            "all_category_document_counts": _sorted_counts(category for row in documents
                for category in sorted({value for value in (row.get("categories") or []) if isinstance(value, str) and value.strip()})),
        },
        "author_keywords": {
            "status_counts": keyword_statuses,
            "documents_with_keywords": len(keyword_rows),
            "coverage_fraction": len(keyword_rows) / len(documents) if documents else None,
            "keyword_entries": sum(len(words) for words in keyword_rows),
            "inconsistent_status_or_keywords_count": len(keyword_inconsistencies),
            "inconsistent_record_indices": keyword_inconsistencies,
        },
        "missing_fields": {
            "documents": _missing_fields(documents, DOCUMENT_FIELDS),
            "passages": _missing_fields(chunks, CHUNK_FIELDS),
        },
        "quality_flags": _sorted_counts(flag for row in documents
            for flag in sorted({value for value in (row.get("quality_flags") or []) if isinstance(value, str) and value.strip()})),
        "length_warnings": warnings,
        "duplicates": {
            "doc_id": _duplicates(documents, "doc_id"),
            "paper_id": _duplicates(documents, "paper_id"),
            "document_text": _duplicates(documents, "text"),
            "chunk_id": _duplicates(chunks, "chunk_id"),
            "passage_text": _duplicates(chunks, "text"),
        },
        "chunk_integrity": {
            "orphan_passages": sum(row.get("doc_id") not in doc_ids for row in chunks),
            "documents_without_passages": len(doc_ids - chunk_doc_ids),
        },
    }
