"""Run the four-method facet-coverage retrieval experiment.

The bundled example queries have topic labels but no human relevance judgments.
This experiment therefore treats a paper's collection_topic as binary proxy
relevance. The generated report labels every quality result accordingly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import dotenv_values

from arxiv_paper_recommend.retrieval.dense import MultiQueryHNSWRetriever
from arxiv_paper_recommend.retrieval.hybrid import HRRFRetriever
from arxiv_paper_recommend.retrieval.innovation import FacetCoverageRetriever
from arxiv_paper_recommend.retrieval.query import QueryGenerator
from arxiv_paper_recommend.retrieval.search_result import SearchResult
from arxiv_paper_recommend.retrieval.sparse import TitleWeightedBM25Retriever
from arxiv_paper_recommend.vectorization import Qwen3Vectorizer

METHODS = ("BM25", "HNSW", "HRRF", "FC_HRRF")
WEIGHT_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)
ARCHIVE_QUERY_INSTRUCTION = (
	"Given a research question, retrieve scientific papers whose core research "
	"addresses the question."
)


class PrecomputedVectorizer:
	"""Serve fixed query vectors through the HNSW vectorizer interface."""

	def __init__(self, vectors: dict[str, np.ndarray]) -> None:
		self.vectors = vectors

	def encode_queries(self, texts: str | list[str]) -> np.ndarray:
		single = isinstance(texts, str)
		items = [texts] if single else texts
		vectors = np.stack([self.vectors[text] for text in items]).astype(
			np.float32, copy=True
		)
		return vectors[0] if single else vectors


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--queries", type=Path, default=Path("data/example_queries.json")
	)
	parser.add_argument("--corpus", type=Path, default=Path("data/raw/papers.jsonl"))
	parser.add_argument("--output-dir", type=Path, default=Path("experiments/results"))
	parser.add_argument("--env-file", type=Path, default=Path(".env"))
	parser.add_argument("--candidate-depth", type=int, default=50)
	parser.add_argument("--pool-depth", type=int, default=20)
	parser.add_argument("--metric-k", type=int, default=10)
	parser.add_argument("--max-terms", type=int, default=12)
	parser.add_argument("--max-facets", type=int, default=4)
	parser.add_argument("--timing-repeats", type=int, default=3)
	parser.add_argument("--bootstrap-samples", type=int, default=10_000)
	parser.add_argument("--device", default=None)
	parser.add_argument("--regenerate-inputs", action="store_true")
	parser.add_argument("--regenerate-embeddings", action="store_true")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	_validate_args(args)
	args.output_dir.mkdir(parents=True, exist_ok=True)
	_load_environment(args.env_file)
	queries = _load_queries(args.queries)
	query_inputs = _prepare_query_inputs(queries, args)
	vectors, encoding_seconds = _prepare_query_vectors(query_inputs, args)
	topic_by_paper, topic_sizes = _load_topics(args.corpus)
	dev_ids, test_ids = _split_queries(queries)

	vectorizer = PrecomputedVectorizer(vectors)
	with (
		TitleWeightedBM25Retriever() as sparse,
		MultiQueryHNSWRetriever(vectorizer=vectorizer) as dense,
		HRRFRetriever(sparse_retriever=sparse, dense_retriever=dense) as hybrid,
		FacetCoverageRetriever(
			sparse_retriever=sparse, dense_retriever=dense
		) as innovation,
	):
		baseline_runs = _run_baselines(
			queries, query_inputs, sparse, dense, hybrid, args.candidate_depth
		)
		weight_scores: dict[float, float] = {}
		dev_innovation_runs: dict[float, dict[str, list[SearchResult]]] = {}
		for weight in WEIGHT_GRID:
			runs = _run_innovation(
				queries,
				query_inputs,
				innovation,
				weight,
				args.pool_depth,
				args.candidate_depth,
				query_ids=dev_ids,
			)
			dev_innovation_runs[weight] = runs
			weight_scores[weight] = statistics.fmean(
				_metrics_for_results(
					runs[query_id],
					_query_by_id(queries)[query_id]["topic"],
					topic_by_paper,
					topic_sizes,
					args.metric_k,
				)["ndcg"]
				for query_id in dev_ids
			)
		selected_weight = max(
			WEIGHT_GRID, key=lambda weight: (weight_scores[weight], -weight)
		)
		innovation_runs = _run_innovation(
			queries,
			query_inputs,
			innovation,
			selected_weight,
			args.pool_depth,
			args.candidate_depth,
		)
		timings = _measure_timings(
			queries,
			query_inputs,
			sparse,
			dense,
			hybrid,
			innovation,
			selected_weight,
			args,
		)

	runs = {
		"BM25": {
			query_id: results[: args.pool_depth]
			for query_id, results in baseline_runs["BM25"].items()
		},
		"HNSW": {
			query_id: results[: args.pool_depth]
			for query_id, results in baseline_runs["HNSW"].items()
		},
		"HRRF": {
			query_id: results[: args.pool_depth]
			for query_id, results in baseline_runs["HRRF"].items()
		},
		"FC_HRRF": innovation_runs,
	}
	query_metrics = _calculate_metrics(
		queries, runs, topic_by_paper, topic_sizes, args.metric_k
	)
	summary = _summarize(query_metrics, test_ids)
	confidence_intervals = _bootstrap_differences(
		query_metrics, test_ids, args.bootstrap_samples
	)
	_write_outputs(
		args,
		queries,
		query_inputs,
		runs,
		query_metrics,
		summary,
		timings,
		weight_scores,
		selected_weight,
		dev_ids,
		test_ids,
		confidence_intervals,
		encoding_seconds,
	)
	print(f"Selected coverage weight: {selected_weight:.1f}")
	print(f"Report: {(args.output_dir / 'report.md').resolve()}")


def _validate_args(args: argparse.Namespace) -> None:
	if args.candidate_depth < args.pool_depth:
		raise ValueError("candidate-depth must be at least pool-depth.")
	if args.pool_depth < args.metric_k:
		raise ValueError("pool-depth must be at least metric-k.")
	if args.metric_k < 5:
		raise ValueError("metric-k must be at least 5 to report Precision@5.")
	for name in (
		"candidate_depth",
		"pool_depth",
		"metric_k",
		"max_terms",
		"max_facets",
		"timing_repeats",
		"bootstrap_samples",
	):
		if getattr(args, name) <= 0:
			raise ValueError(f"{name.replace('_', '-')} must be positive.")
	if args.candidate_depth > 100:
		raise ValueError("candidate-depth must be at most 100.")


def _load_environment(path: Path) -> None:
	if not path.exists():
		return
	for key, value in dotenv_values(path).items():
		if value is not None:
			os.environ.setdefault(key, value)


def _load_queries(path: Path) -> list[dict[str, str]]:
	payload = json.loads(path.read_text(encoding="utf-8"))
	queries = payload.get("questions")
	if not isinstance(queries, list) or not queries:
		raise ValueError("Query file must contain a non-empty 'questions' list.")
	return queries


def _prepare_query_inputs(
	queries: list[dict[str, str]], args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
	path = args.output_dir / "query_inputs.json"
	inputs: dict[str, dict[str, Any]] = {}
	if path.exists() and not args.regenerate_inputs:
		inputs = json.loads(path.read_text(encoding="utf-8"))
	missing = [query for query in queries if query["id"] not in inputs]
	if missing:
		with QueryGenerator() as generator:
			for query in missing:
				request = query["query_en"]
				inputs[query["id"]] = {
					"request": request,
					"terms": generator.generate_terms(
						request, max_terms=args.max_terms
					),
					"query": request,
					"facets": generator.generate_queries(
						request, max_queries=args.max_facets
					),
				}
				_atomic_json(path, inputs)
	return inputs


def _prepare_query_vectors(
	query_inputs: dict[str, dict[str, Any]], args: argparse.Namespace
) -> tuple[dict[str, np.ndarray], float]:
	path = args.output_dir / "query_embeddings.npz"
	metadata_path = args.output_dir / "query_embeddings_metadata.json"
	texts = list(
		dict.fromkeys(
			text
			for inputs in query_inputs.values()
			for text in [inputs["query"], *inputs["facets"]]
		)
	)
	if path.exists() and not args.regenerate_embeddings:
		with np.load(path, allow_pickle=False) as archive:
			stored_texts = archive["texts"].tolist()
			stored_vectors = archive["vectors"]
		if stored_texts == texts:
			metadata = (
				json.loads(metadata_path.read_text(encoding="utf-8"))
				if metadata_path.exists()
				else {}
			)
			return dict(zip(stored_texts, stored_vectors)), float(
				metadata.get("encoding_seconds", 0.0)
			)
	start = time.perf_counter()
	vectorizer = Qwen3Vectorizer(
		device=args.device,
		query_instruction=ARCHIVE_QUERY_INSTRUCTION,
	)
	vectors = vectorizer.encode_queries(texts)
	elapsed = time.perf_counter() - start
	np.savez_compressed(path, texts=np.asarray(texts), vectors=vectors)
	_atomic_json(
		metadata_path,
		{
			"query_count": len(texts),
			"encoding_seconds": elapsed,
			"query_instruction": ARCHIVE_QUERY_INSTRUCTION,
			"device": str(vectorizer.device),
		},
	)
	return dict(zip(texts, vectors)), elapsed


def _load_topics(path: Path) -> tuple[dict[str, str], dict[str, int]]:
	topic_by_paper: dict[str, str] = {}
	topic_sizes: dict[str, int] = defaultdict(int)
	with path.open(encoding="utf-8") as corpus:
		for line in corpus:
			record = json.loads(line)
			paper_id = record["doc_id"].removeprefix("arxiv:")
			topic = record["collection_topic"]
			topic_by_paper[paper_id] = topic
			topic_sizes[topic] += 1
	return topic_by_paper, dict(topic_sizes)


def _split_queries(queries: list[dict[str, str]]) -> tuple[list[str], list[str]]:
	dev_ids: list[str] = []
	test_ids: list[str] = []
	seen_topics: set[str] = set()
	for query in queries:
		if query["topic"] not in seen_topics:
			dev_ids.append(query["id"])
			seen_topics.add(query["topic"])
		else:
			test_ids.append(query["id"])
	return dev_ids, test_ids


def _run_baselines(
	queries: list[dict[str, str]],
	query_inputs: dict[str, dict[str, Any]],
	sparse: TitleWeightedBM25Retriever,
	dense: MultiQueryHNSWRetriever,
	hybrid: HRRFRetriever,
	candidate_depth: int,
) -> dict[str, dict[str, list[SearchResult]]]:
	runs = {method: {} for method in METHODS[:3]}
	for query in queries:
		query_id = query["id"]
		inputs = query_inputs[query_id]
		runs["BM25"][query_id] = sparse.search(inputs["terms"], limit=candidate_depth)
		runs["HNSW"][query_id] = dense.search(inputs["query"], limit=candidate_depth)
		runs["HRRF"][query_id] = hybrid.search(
			inputs["terms"],
			inputs["query"],
			limit=candidate_depth,
			recall_queries=inputs["facets"],
		)
	return runs


def _run_innovation(
	queries: list[dict[str, str]],
	query_inputs: dict[str, dict[str, Any]],
	retriever: FacetCoverageRetriever,
	coverage_weight: float,
	limit: int,
	candidate_depth: int,
	*,
	query_ids: Iterable[str] | None = None,
) -> dict[str, list[SearchResult]]:
	selected_ids = None if query_ids is None else set(query_ids)
	runs: dict[str, list[SearchResult]] = {}
	for query in queries:
		query_id = query["id"]
		if selected_ids is not None and query_id not in selected_ids:
			continue
		inputs = query_inputs[query_id]
		runs[query_id] = retriever.search(
			inputs["terms"],
			inputs["query"],
			limit=limit,
			recall_queries=inputs["facets"],
			candidate_limit=candidate_depth,
			coverage_weight=coverage_weight,
		)
	return runs


def _measure_timings(
	queries: list[dict[str, str]],
	query_inputs: dict[str, dict[str, Any]],
	sparse: TitleWeightedBM25Retriever,
	dense: MultiQueryHNSWRetriever,
	hybrid: HRRFRetriever,
	innovation: FacetCoverageRetriever,
	coverage_weight: float,
	args: argparse.Namespace,
) -> dict[str, list[float]]:
	timings = {method: [] for method in METHODS}
	for query in queries:
		inputs = query_inputs[query["id"]]
		calls: dict[str, Callable[[], list[SearchResult]]] = {
			"BM25": partial(sparse.search, inputs["terms"], limit=args.candidate_depth),
			"HNSW": partial(dense.search, inputs["query"], limit=args.candidate_depth),
			"HRRF": partial(
				hybrid.search,
				inputs["terms"],
				inputs["query"],
				limit=args.candidate_depth,
				recall_queries=inputs["facets"],
			),
			"FC_HRRF": partial(
				innovation.search,
				inputs["terms"],
				inputs["query"],
				limit=args.pool_depth,
				recall_queries=inputs["facets"],
				candidate_limit=args.candidate_depth,
				coverage_weight=coverage_weight,
			),
		}
		for method, call in calls.items():
			call()
			for _ in range(args.timing_repeats):
				start = time.perf_counter()
				call()
				timings[method].append((time.perf_counter() - start) * 1000.0)
	return timings


def _calculate_metrics(
	queries: list[dict[str, str]],
	runs: dict[str, dict[str, list[SearchResult]]],
	topic_by_paper: dict[str, str],
	topic_sizes: dict[str, int],
	k: int,
) -> list[dict[str, Any]]:
	rows = []
	for query in queries:
		for method in METHODS:
			metrics = _metrics_for_results(
				runs[method][query["id"]],
				query["topic"],
				topic_by_paper,
				topic_sizes,
				k,
			)
			rows.append(
				{
					"query_id": query["id"],
					"topic": query["topic"],
					"query_type": query["query_type"],
					"method": method,
					**metrics,
				}
			)
	return rows


def _metrics_for_results(
	results: list[SearchResult],
	topic: str,
	topic_by_paper: dict[str, str],
	topic_sizes: dict[str, int],
	k: int,
) -> dict[str, float]:
	relevance = [
		1.0 if topic_by_paper.get(result.arxiv_id) == topic else 0.0
		for result in results[:k]
	]
	relevance.extend([0.0] * (k - len(relevance)))
	relevant_count = sum(relevance)
	dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(relevance, 1))
	ideal_count = min(k, topic_sizes[topic])
	idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
	first_relevant = next(
		(rank for rank, value in enumerate(relevance, start=1) if value), None
	)
	return {
		"precision_at_5": sum(relevance[:5]) / 5,
		"precision": relevant_count / k,
		"recall": relevant_count / topic_sizes[topic],
		"mrr": 0.0 if first_relevant is None else 1.0 / first_relevant,
		"ndcg": 0.0 if idcg == 0.0 else dcg / idcg,
	}


def _summarize(
	query_metrics: list[dict[str, Any]], test_ids: list[str]
) -> dict[str, dict[str, float]]:
	test_set = set(test_ids)
	return {
		method: {
			metric: statistics.fmean(
				row[metric]
				for row in query_metrics
				if row["query_id"] in test_set and row["method"] == method
			)
			for metric in ("precision_at_5", "precision", "recall", "mrr", "ndcg")
		}
		for method in METHODS
	}


def _bootstrap_differences(
	query_metrics: list[dict[str, Any]],
	test_ids: list[str],
	samples: int,
) -> dict[str, tuple[float, float, float]]:
	values = {(row["query_id"], row["method"]): row["ndcg"] for row in query_metrics}
	rng = random.Random(6127)
	intervals = {}
	for baseline in METHODS[:3]:
		differences = [
			values[(query_id, "FC_HRRF")] - values[(query_id, baseline)]
			for query_id in test_ids
		]
		bootstrap = sorted(
			statistics.fmean(rng.choices(differences, k=len(differences)))
			for _ in range(samples)
		)
		intervals[baseline] = (
			statistics.fmean(differences),
			_percentile(bootstrap, 2.5),
			_percentile(bootstrap, 97.5),
		)
	return intervals


def _write_outputs(
	args: argparse.Namespace,
	queries: list[dict[str, str]],
	query_inputs: dict[str, dict[str, Any]],
	runs: dict[str, dict[str, list[SearchResult]]],
	query_metrics: list[dict[str, Any]],
	summary: dict[str, dict[str, float]],
	timings: dict[str, list[float]],
	weight_scores: dict[float, float],
	selected_weight: float,
	dev_ids: list[str],
	test_ids: list[str],
	confidence_intervals: dict[str, tuple[float, float, float]],
	encoding_seconds: float,
) -> None:
	query_lookup = _query_by_id(queries)
	with (args.output_dir / "runs.jsonl").open("w", encoding="utf-8") as output:
		for method in METHODS:
			for query_id, results in runs[method].items():
				for rank, result in enumerate(results, start=1):
					output.write(
						json.dumps(
							{
								"query_id": query_id,
								"topic": query_lookup[query_id]["topic"],
								"method": method,
								"rank": rank,
								"document_id": result.document_id,
								"arxiv_id": result.arxiv_id,
								"title": result.title,
								"score": result.score,
							},
							ensure_ascii=False,
						)
						+ "\n"
					)
	_write_csv(args.output_dir / "query_metrics.csv", query_metrics)
	summary_rows = [
		{
			"method": method,
			**summary[method],
			"latency_p50_ms": _percentile(timings[method], 50),
			"latency_p95_ms": _percentile(timings[method], 95),
		}
		for method in METHODS
	]
	_write_csv(args.output_dir / "summary.csv", summary_rows)
	metadata = {
		"quality_label": "binary collection_topic proxy",
		"query_count": len(queries),
		"dev_query_ids": dev_ids,
		"test_query_ids": test_ids,
		"candidate_depth": args.candidate_depth,
		"pool_depth": args.pool_depth,
		"metric_k": args.metric_k,
		"weight_grid": list(WEIGHT_GRID),
		"development_ndcg": {str(key): value for key, value in weight_scores.items()},
		"selected_coverage_weight": selected_weight,
		"query_instruction": ARCHIVE_QUERY_INSTRUCTION,
		"query_encoding_seconds": encoding_seconds,
		"timing_repeats": args.timing_repeats,
	}
	_atomic_json(args.output_dir / "metadata.json", metadata)
	lines = [
		"# FC-HRRF 四方法代理实验",
		"",
		(
			"> 本实验没有人工相关性标签。质量指标把论文的 `collection_topic` "
			"与查询 `topic` 相同视为相关，因此只反映主题级检索能力，不能替代人工0/1/2相关性评估。"
		),
		"",
		f"- 查询：{len(queries)}条；开发集{len(dev_ids)}条，测试集{len(test_ids)}条。",
		f"- 内部候选深度：{args.candidate_depth}；保存深度：{args.pool_depth}；指标：@{args.metric_k}。",
		f"- 开发集选择的覆盖权重：{selected_weight:.1f}。",
		f"- 查询向量预计算耗时：{encoding_seconds:.3f}秒；检索延迟不含LLM和查询编码。",
		"",
		f"## 测试集结果（K={args.metric_k}）",
		"",
		(
			f"| 方法 | P@5 | P@{args.metric_k} | Recall@{args.metric_k} | "
			f"MRR@{args.metric_k} | NDCG@{args.metric_k} | P50 ms | P95 ms |"
		),
		"|---|---:|---:|---:|---:|---:|---:|---:|",
	]
	for row in summary_rows:
		lines.append(
			f"| {row['method']} | {row['precision_at_5']:.4f} | "
			f"{row['precision']:.4f} | {row['recall']:.5f} | "
			f"{row['mrr']:.4f} | {row['ndcg']:.4f} | "
			f"{row['latency_p50_ms']:.3f} | {row['latency_p95_ms']:.3f} |"
		)
	lines.extend(
		[
			"",
			"## FC-HRRF 的 NDCG 差值",
			"",
			"| 对照方法 | 平均差值 | 95% bootstrap CI |",
			"|---|---:|---:|",
		]
	)
	for baseline, (difference, lower, upper) in confidence_intervals.items():
		lines.append(
			f"| {baseline} | {difference:+.4f} | [{lower:+.4f}, {upper:+.4f}] |"
		)
	test_set = set(test_ids)
	metric_lookup = {
		(row["query_id"], row["method"]): row
		for row in query_metrics
		if row["query_id"] in test_set
	}
	deltas = {
		query_id: metric_lookup[(query_id, "FC_HRRF")]["ndcg"]
		- metric_lookup[(query_id, "HRRF")]["ndcg"]
		for query_id in test_ids
	}
	wins = sum(delta > 1e-12 for delta in deltas.values())
	losses = sum(delta < -1e-12 for delta in deltas.values())
	ties = len(deltas) - wins - losses
	best_query_id = max(deltas, key=deltas.get)
	worst_query_id = min(deltas, key=deltas.get)
	lines.extend(
		[
			"",
			"## 与 HRRF 的逐查询比较",
			"",
			f"FC-HRRF 在 {wins} 条测试查询上更高、{ties} 条持平、{losses} 条更低。",
			"",
			"| 案例 | 查询 | HRRF NDCG | FC-HRRF NDCG | 差值 |",
			"|---|---|---:|---:|---:|",
			(
				f"| 最大提升（{best_query_id}） | "
				f"{query_lookup[best_query_id]['query_en']} | "
				f"{metric_lookup[(best_query_id, 'HRRF')]['ndcg']:.4f} | "
				f"{metric_lookup[(best_query_id, 'FC_HRRF')]['ndcg']:.4f} | "
				f"{deltas[best_query_id]:+.4f} |"
			),
			(
				f"| 最大下降（{worst_query_id}） | "
				f"{query_lookup[worst_query_id]['query_en']} | "
				f"{metric_lookup[(worst_query_id, 'HRRF')]['ndcg']:.4f} | "
				f"{metric_lookup[(worst_query_id, 'FC_HRRF')]['ndcg']:.4f} | "
				f"{deltas[worst_query_id]:+.4f} |"
			),
		]
	)
	lines.extend(
		[
			"",
			"## 开发集权重选择",
			"",
			"| coverage weight | NDCG |",
			"|---:|---:|",
		]
	)
	for weight, ndcg in weight_scores.items():
		lines.append(f"| {weight:.1f} | {ndcg:.4f} |")
	query_types = sorted(
		{row["query_type"] for row in query_metrics if row["query_id"] in test_set}
	)
	lines.extend(
		[
			"",
			f"## 按查询类型的 NDCG@{args.metric_k}",
			"",
			"| 查询类型 | " + " | ".join(METHODS) + " |",
			"|---|" + "---:|" * len(METHODS),
		]
	)
	for query_type in query_types:
		values = []
		for method in METHODS:
			values.append(
				statistics.fmean(
					row["ndcg"]
					for row in query_metrics
					if row["query_id"] in test_set
					and row["query_type"] == query_type
					and row["method"] == method
				)
			)
		lines.append(
			f"| {query_type} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
		)
	lines.extend(
		[
			"",
			"## 限制",
			"",
			"- `collection_topic` 是构建语料时的收录类别，不代表论文满足查询的具体方法、场景或约束。",
			"- Recall 的分母是该主题的全部论文数；它不是人工判断的完整相关论文集合。",
			"- 正式课程结论仍需使用盲法双人标注的 query–paper 标签重新运行同一评估流程。",
		]
	)
	(args.output_dir / "report.md").write_text(
		"\n".join(lines) + "\n", encoding="utf-8"
	)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
	with path.open("w", newline="", encoding="utf-8-sig") as output:
		writer = csv.DictWriter(output, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


def _atomic_json(path: Path, value: Any) -> None:
	temporary = path.with_suffix(path.suffix + ".tmp")
	temporary.write_text(
		json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
	)
	temporary.replace(path)


def _query_by_id(queries: list[dict[str, str]]) -> dict[str, dict[str, str]]:
	return {query["id"]: query for query in queries}


def _percentile(values: list[float], percentile: float) -> float:
	ordered = sorted(values)
	if len(ordered) == 1:
		return ordered[0]
	position = (len(ordered) - 1) * percentile / 100.0
	lower = math.floor(position)
	upper = math.ceil(position)
	if lower == upper:
		return ordered[lower]
	return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


if __name__ == "__main__":
	main()
