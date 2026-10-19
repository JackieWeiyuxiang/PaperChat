"""Run frozen-input retrieval comparisons and score judged candidate pools."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from dataclasses import dataclass
from math import log2
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from arxiv_paper_recommend.retrieval._ranking import rank_ids
from arxiv_paper_recommend.retrieval.dense import HNSWRetriever
from arxiv_paper_recommend.retrieval.hybrid import (
	HRRFRetriever,
	SDCascadeRetriever,
	SemanticRRFRetriever,
)
from arxiv_paper_recommend.retrieval.innovation import TitleAwareBM25Retriever
from arxiv_paper_recommend.retrieval.search_result import SearchResult
from arxiv_paper_recommend.retrieval.sparse import BM25Retriever

DEFAULT_METHODS = (
	"BM25",
	"TITLE_BM25",
	"HNSW",
	"HRRF",
	"SEMANTIC_RRF",
	"SD_CASCADE",
)
HYBRID_METHODS = {
	"HRRF": HRRFRetriever,
	"SEMANTIC_RRF": SemanticRRFRetriever,
	"SD_CASCADE": SDCascadeRetriever,
}


@dataclass(frozen=True)
class FrozenQuery:
	"""LLM-independent inputs shared by all methods for one query."""

	query_id: str
	request: str
	terms: list[str]
	queries: list[str]
	score_query: str

	@classmethod
	def from_record(cls, record: dict[str, Any], line_number: int) -> FrozenQuery:
		"""Validate one JSONL record and report errors with its line number."""
		required = ("query_id", "request", "terms", "queries", "score_query")
		missing = [field for field in required if field not in record]
		if missing:
			raise ValueError(f"Input line {line_number} is missing fields: {missing}.")
		for field in ("query_id", "request", "score_query"):
			if not isinstance(record[field], str) or not record[field].strip():
				raise ValueError(
					f"Input line {line_number}: {field} must be non-empty text."
				)
		for field in ("terms", "queries"):
			value = record[field]
			if (
				not isinstance(value, list)
				or not value
				or any(not isinstance(item, str) or not item.strip() for item in value)
			):
				raise ValueError(
					f"Input line {line_number}: {field} must be a non-empty string list."
				)
		return cls(
			query_id=record["query_id"].strip(),
			request=record["request"].strip(),
			terms=record["terms"],
			queries=record["queries"],
			score_query=record["score_query"].strip(),
		)


def run_experiment(
	inputs_path: str | Path,
	runs_path: str | Path,
	pool_path: str | Path,
	*,
	methods: tuple[str, ...] = DEFAULT_METHODS,
	limit: int = 20,
	repeats: int = 3,
) -> dict[str, int]:
	"""Run all selected methods on identical frozen query inputs and save runs."""
	if type(limit) is not int or limit <= 0:
		raise ValueError("Result limit must be a positive integer.")
	if type(repeats) is not int or repeats <= 0:
		raise ValueError("Repeats must be a positive integer.")
	if not methods or any(not isinstance(method, str) for method in methods):
		raise ValueError("Methods must be a non-empty list of names.")
	methods = tuple(method.upper() for method in methods)
	if len(set(methods)) != len(methods):
		raise ValueError("Methods must not contain duplicates.")
	unknown = set(methods) - set(DEFAULT_METHODS)
	if unknown:
		raise ValueError(f"Unsupported methods: {sorted(unknown)}.")
	if any(method in HYBRID_METHODS for method in methods) and limit > 100:
		raise ValueError("Hybrid retrieval supports result limits up to 100.")

	queries = _read_frozen_queries(inputs_path)
	runs: list[dict[str, Any]] = []
	pool: dict[tuple[str, str], dict[str, Any]] = {}
	with ExitStack() as resources:
		sparse = None
		if "BM25" in methods or any(method in HYBRID_METHODS for method in methods):
			sparse = resources.enter_context(BM25Retriever())
		title_sparse = None
		if "TITLE_BM25" in methods:
			title_sparse = resources.enter_context(TitleAwareBM25Retriever())
		dense = None
		if any(method in {"HNSW", *HYBRID_METHODS} for method in methods):
			dense = resources.enter_context(HNSWRetriever())
			dense.search(queries[0].queries[0], limit=1)
		hybrids = {
			method: resources.enter_context(
				HYBRID_METHODS[method](
					sparse_retriever=sparse,
					dense_retriever=dense,
				)
			)
			for method in methods
			if method in HYBRID_METHODS
		}

		for query in queries:
			for method in methods:
				runtimes: list[float] = []
				for _ in range(repeats):
					started = perf_counter()
					if method == "BM25":
						assert sparse is not None
						results = sparse.term_search(query.terms, limit=limit)
					elif method == "TITLE_BM25":
						assert title_sparse is not None
						results = title_sparse.term_search(query.terms, limit=limit)
					elif method == "HNSW":
						assert dense is not None
						results = _search_hnsw(dense, query, limit)
					else:
						results = hybrids[method].search(
							query.terms,
							set(query.queries),
							query.score_query,
							limit=limit,
						)
					runtimes.append((perf_counter() - started) * 1000)
				run = {
					"query_id": query.query_id,
					"method": method,
					"runtime_ms": runtimes,
					"mean_runtime_ms": mean(runtimes),
					"median_runtime_ms": median(runtimes),
					"results": [
						_result_record(result, rank)
						for rank, result in enumerate(results, 1)
					],
				}
				runs.append(run)
				for result in results:
					key = (query.query_id, result.doc_id)
					candidate = pool.setdefault(
						key,
						{
							"query_id": query.query_id,
							"doc_id": result.doc_id,
							"title": result.title,
							"abstract": result.abstract,
							"methods": [],
						},
					)
					if method not in candidate["methods"]:
						candidate["methods"].append(method)

	_write_jsonl(runs_path, runs)
	_write_jsonl(pool_path, pool.values())
	return {"queries": len(queries), "runs": len(runs), "candidates": len(pool)}


def evaluate_runs(
	qrels_path: str | Path,
	runs_path: str | Path,
	output_path: str | Path,
	*,
	k_values: tuple[int, ...] = (5, 10, 20),
) -> dict[str, Any]:
	"""Evaluate a fully judged common candidate pool with P/R, MRR and nDCG."""
	if not k_values or any(type(k) is not int or k <= 0 for k in k_values):
		raise ValueError("K values must be positive integers.")
	qrels = _read_qrels(qrels_path)
	runs = _read_runs(runs_path)
	methods = sorted({run["method"] for run in runs})
	query_ids = sorted({run["query_id"] for run in runs})
	all_results: dict[str, dict[str, list[str]]] = {}
	for run in runs:
		all_results.setdefault(run["query_id"], {})[run["method"]] = [
			item["doc_id"] for item in run["results"]
		]
	fully_judged = {
		query_id
		for query_id in query_ids
		if query_id in qrels
		and set(all_results[query_id]) == set(methods)
		and all(
			doc_id in qrels[query_id]
			for method_results in all_results[query_id].values()
			for doc_id in method_results
		)
	}
	metrics: dict[str, dict[str, Any]] = {}
	for method in methods:
		method_runs = {
			run["query_id"]: run["results"]
			for run in runs
			if run["method"] == method and run["query_id"] in fully_judged
		}
		method_metrics: dict[str, Any] = {
			"evaluated_queries": len(method_runs),
			"skipped_unjudged_queries": len(query_ids) - len(fully_judged),
			"mean_runtime_ms": _mean(
				run["mean_runtime_ms"] for run in runs if run["method"] == method
			),
			"median_runtime_ms": _mean(
				run["median_runtime_ms"] for run in runs if run["method"] == method
			),
		}
		for k in k_values:
			per_query = [
				_query_metrics(
					[entry["doc_id"] for entry in method_runs[query_id]],
					qrels[query_id],
					k,
				)
				for query_id in method_runs
			]
			method_metrics[f"@{k}"] = {
				name: _mean(item[name] for item in per_query)
				for name in ("precision", "recall", "mrr", "ndcg")
			}
		metrics[method] = method_metrics
	report = {
		"qrels": str(Path(qrels_path)),
		"runs": str(Path(runs_path)),
		"judged_query_count": len(fully_judged),
		"unjudged_or_incomplete_query_count": len(query_ids) - len(fully_judged),
		"relevance_threshold": 1,
		"ndcg_gain": "2^relevance - 1",
		"metrics": metrics,
	}
	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	Path(output_path).write_text(
		json.dumps(report, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)
	return report


def _search_hnsw(
	retriever: HNSWRetriever,
	query: FrozenQuery,
	limit: int,
) -> list[SearchResult]:
	"""Mirror SearchEngine's multi-query union and integrated-query reranking."""
	query_results = retriever.search(set(query.queries), limit=limit)
	candidates: dict[int, SearchResult] = {}
	for results in query_results.values():
		for result in results:
			candidates.setdefault(result.id, result)
	if not candidates:
		return []
	scores = retriever.get_score({query.score_query: list(candidates)})[
		query.score_query
	]
	return [
		SearchResult(
			id=candidates[document_id].id,
			doc_id=candidates[document_id].doc_id,
			title=candidates[document_id].title,
			abstract=candidates[document_id].abstract,
			score=scores[document_id],
		)
		for document_id in rank_ids(candidates, scores, descending=True)[:limit]
	]


def _result_record(result: SearchResult, rank: int) -> dict[str, Any]:
	return {
		"rank": rank,
		"id": result.id,
		"doc_id": result.doc_id,
		"title": result.title,
		"abstract": result.abstract,
		"score": result.score,
	}


def _read_frozen_queries(path: str | Path) -> list[FrozenQuery]:
	queries: list[FrozenQuery] = []
	seen: set[str] = set()
	for line_number, line in enumerate(
		Path(path).read_text(encoding="utf-8").splitlines(), 1
	):
		if not line.strip():
			continue
		try:
			record = json.loads(line)
			if not isinstance(record, dict):
				raise TypeError("Expected a JSON object.")
			query = FrozenQuery.from_record(record, line_number)
		except (json.JSONDecodeError, TypeError, ValueError) as error:
			raise ValueError(f"{path}:{line_number}: {error}") from error
		if query.query_id in seen:
			raise ValueError(f"Duplicate query_id {query.query_id!r} in {path}.")
		seen.add(query.query_id)
		queries.append(query)
	if not queries:
		raise ValueError(f"No frozen queries found in {path}.")
	return queries


def _read_qrels(path: str | Path) -> dict[str, dict[str, int]]:
	qrels: dict[str, dict[str, int]] = {}
	for line_number, line in enumerate(
		Path(path).read_text(encoding="utf-8").splitlines(), 1
	):
		if not line.strip():
			continue
		try:
			row = json.loads(line)
			query_id, doc_id, relevance = (
				row["query_id"],
				row["doc_id"],
				row["relevance"],
			)
			if not isinstance(query_id, str) or not isinstance(doc_id, str):
				raise TypeError("query_id and doc_id must be strings.")
			if type(relevance) is not int or not 0 <= relevance <= 2:
				raise ValueError("relevance must be an integer from 0 to 2.")
			documents = qrels.setdefault(query_id, {})
			if doc_id in documents:
				raise ValueError(f"Duplicate judgment for ({query_id}, {doc_id}).")
			documents[doc_id] = relevance
		except (KeyError, json.JSONDecodeError, TypeError, ValueError) as error:
			raise ValueError(f"{path}:{line_number}: {error}") from error
	if not qrels:
		raise ValueError(f"No judgments found in {path}.")
	return qrels


def _read_runs(path: str | Path) -> list[dict[str, Any]]:
	runs: list[dict[str, Any]] = []
	seen: set[tuple[str, str]] = set()
	for line_number, line in enumerate(
		Path(path).read_text(encoding="utf-8").splitlines(), 1
	):
		if not line.strip():
			continue
		try:
			run = json.loads(line)
			key = (run["query_id"], run["method"])
			if key in seen:
				raise ValueError(f"Duplicate run for {key}.")
			if not isinstance(run["results"], list):
				raise TypeError("results must be a list.")
			seen.add(key)
			runs.append(run)
		except (KeyError, json.JSONDecodeError, TypeError, ValueError) as error:
			raise ValueError(f"{path}:{line_number}: {error}") from error
	if not runs:
		raise ValueError(f"No runs found in {path}.")
	return runs


def _query_metrics(
	ranked_doc_ids: list[str],
	judgments: dict[str, int],
	k: int,
) -> dict[str, float]:
	relevant = {doc_id for doc_id, relevance in judgments.items() if relevance >= 1}
	top_k = ranked_doc_ids[:k]
	precision = sum(doc_id in relevant for doc_id in top_k) / k
	recall = (
		sum(doc_id in relevant for doc_id in top_k) / len(relevant) if relevant else 0.0
	)
	mrr = next(
		(1.0 / rank for rank, doc_id in enumerate(top_k, 1) if doc_id in relevant),
		0.0,
	)

	def gain(doc_id: str) -> float:
		return 2 ** judgments[doc_id] - 1

	dcg = sum(gain(doc_id) / log2(rank + 1) for rank, doc_id in enumerate(top_k, 1))
	ideal = sorted(judgments.values(), reverse=True)[:k]
	idcg = sum(
		(2**relevance - 1) / log2(rank + 1) for rank, relevance in enumerate(ideal, 1)
	)
	return {
		"precision": precision,
		"recall": recall,
		"mrr": mrr,
		"ndcg": dcg / idcg if idcg else 0.0,
	}


def _mean(values: Any) -> float | None:
	items = list(values)
	return sum(items) / len(items) if items else None


def _write_jsonl(path: str | Path, records: Any) -> None:
	output = Path(path)
	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text(
		"".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
		encoding="utf-8",
	)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	commands = parser.add_subparsers(dest="command", required=True)
	run_parser = commands.add_parser(
		"run", help="Run retrieval methods on frozen inputs."
	)
	run_parser.add_argument("--inputs", type=Path, required=True)
	run_parser.add_argument("--runs", type=Path, required=True)
	run_parser.add_argument("--pool", type=Path, required=True)
	run_parser.add_argument("--limit", type=int, default=20)
	run_parser.add_argument("--repeats", type=int, default=3)
	run_parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
	eval_parser = commands.add_parser(
		"evaluate", help="Score complete human-judged runs."
	)
	eval_parser.add_argument("--qrels", type=Path, required=True)
	eval_parser.add_argument("--runs", type=Path, required=True)
	eval_parser.add_argument("--output", type=Path, required=True)
	eval_parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
	args = parser.parse_args()
	if args.command == "run":
		method_names = tuple(
			name.strip() for name in args.methods.split(",") if name.strip()
		)
		summary = run_experiment(
			args.inputs,
			args.runs,
			args.pool,
			methods=method_names,
			limit=args.limit,
			repeats=args.repeats,
		)
	else:
		summary = evaluate_runs(
			args.qrels, args.runs, args.output, k_values=tuple(args.k)
		)
	print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
