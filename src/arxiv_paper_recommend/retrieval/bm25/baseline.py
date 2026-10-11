"""Small, deterministic BM25 baseline for offline development checks.

Only passage ``text`` is indexed. Author keywords, labels, and other metadata
are deliberately ignored. This is not an implementation of Lucene's analyzer:
we normalize with NFKC/casefold, split on ``[a-z0-9]+``, and remove the fixed
English function words below for both documents and queries. Hyphens split
terms, numbers remain, and no stemming or synonym expansion is performed.

The BM25 formula and default k1=1.2, b=0.75 follow BM25Similarity:
https://lucene.apache.org/core/9_12_1/core/org/apache/lucene/search/similarities/BM25Similarity.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import math
import re
import unicodedata


STOPWORDS = frozenset(
    "a an and are as at be been being by for from had has have in is it its "
    "of on or that the their these this those to was were will with".split()
)
TOKEN_PATTERN = r"[a-z0-9]+"
TOKENIZER_VERSION = "nfkc-casefold-ascii-alnum-function-words-v1"
ALGORITHM_VERSION = "bm25-development-v1"
_TOKEN_RE = re.compile(TOKEN_PATTERN)


def tokenize(text: str) -> list[str]:
    """Return normalized English lexical tokens, excluding fixed stopwords."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [term for term in _TOKEN_RE.findall(normalized) if term not in STOPWORDS]


class BM25Index:
    """In-memory BM25 index; input records and text are never modified.

    ``chunk_id`` and ``doc_id`` must be nonempty strings, and ``text`` must be
    a string. Chunk IDs must be unique; multiple chunks may share one doc ID.
    Empty passages are allowed and count toward N, but produce no postings.
    Document length is the number of tokens left after stopword removal.
    k1 must be finite and nonnegative; b must be finite and between 0 and 1.
    """

    def __init__(self, passages: list[dict], k1: float = 1.2, b: float = 0.75):
        for name, value in (("k1", k1), ("b", b)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if k1 < 0:
            raise ValueError("k1 must be nonnegative")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.k1 = float(k1)
        self.b = float(b)
        self._records: list[tuple[str, str]] = []
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        seen: set[str] = set()
        for row in passages:
            if not isinstance(row, dict):
                raise ValueError("each passage must be a dictionary")
            for field in ("chunk_id", "doc_id"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"passage {field} must be a nonempty string")
            if not isinstance(row.get("text"), str):
                raise ValueError("passage text must be a string")
            chunk_id = row["chunk_id"]
            if chunk_id in seen:
                raise ValueError(f"duplicate chunk_id: {chunk_id}")
            seen.add(chunk_id)
            position = len(self._records)
            self._records.append((chunk_id, row["doc_id"]))
            counts = Counter(tokenize(row["text"]))
            self._lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self._postings[term].append((position, frequency))

        self.size = len(self._records)
        self.avgdl = sum(self._lengths) / self.size if self.size else 0.0
        self._idf = {
            term: math.log1p((self.size - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }

    def rank_all(self, query: str) -> list[dict]:
        """Return every positive-score match, with deterministic ID tie breaks.

        Repeated query terms count once. Results contain only IDs, score, and
        sorted matched terms. Empty/OOV/stopword-only queries yield no results.
        Zero-score documents are never added to fill a requested result count.
        """
        terms = sorted(set(tokenize(query)))
        if not terms or self.avgdl == 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        matches: dict[int, list[str]] = defaultdict(list)
        for term in terms:
            for position, frequency in self._postings.get(term, ()):
                length_ratio = self._lengths[position] / self.avgdl
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                scores[position] += (
                    self._idf[term] * frequency * (self.k1 + 1) / denominator
                )
                matches[position].append(term)
        results = [
            {
                "chunk_id": self._records[position][0],
                "doc_id": self._records[position][1],
                "score": score,
                "matched_terms": matches[position],
            }
            for position, score in scores.items()
            if score > 0
        ]
        return sorted(results, key=lambda result: (-result["score"], result["chunk_id"]))

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Return at most k positive-score matches; k must be a nonnegative int."""
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k must be a nonnegative integer")
        if k == 0:
            return []
        return self.rank_all(query)[:k]


ROOT = Path(__file__).resolve().parents[4]


def search(corpus: Path, query: str, k: int) -> list[dict]:
    records = [json.loads(line) for line in (corpus / "papers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    documents = {row["doc_id"]: row for row in records}
    if len(documents) != len(records):
        raise ValueError("Duplicate document IDs")
    passages = records
    matches = BM25Index(passages).search(query, k)
    return [{"rank": rank, **match,
             **{field: documents[match["doc_id"]][field] for field in ("title", "abstract", "url")}}
            for rank, match in enumerate(matches, 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="English search query")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 0:
        parser.error("--top-k must be nonnegative")
    print(json.dumps({"query": args.query, "results": search(args.corpus, args.query, args.top_k)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
