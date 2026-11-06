"""Calculate automatic metrics for the generation comparison."""

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_INPUT_PATH = Path(
	"experiment/generation/benchmark_comparison_bm25_10.jsonl"
)
DEFAULT_OUTPUT_PATH = Path(
	"experiment/generation/benchmark_metrics_bm25_10.json"
)
CITATION_PATTERN = re.compile(
	r"\[\[cite\]\](.*?)\[\[/cite\]\]",
	re.DOTALL,
)


def _parse_args() -> argparse.Namespace:
	"""Parse evaluation file paths."""
	parser = argparse.ArgumentParser(
		description="Evaluate generation comparison results."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=DEFAULT_INPUT_PATH,
		help="Comparison JSONL created by run_comparison.py.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT_PATH,
		help="JSON file used to store automatic metrics.",
	)
	return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
	"""Load comparison records from JSONL."""
	records: list[dict[str, Any]] = []

	with path.open(encoding="utf-8") as input_file:
		for line_number, line in enumerate(input_file, start=1):
			stripped = line.strip()
			if not stripped:
				continue

			record = json.loads(stripped)
			if not isinstance(record, dict):
				raise ValueError(
					f"Line {line_number} must contain one JSON object."
				)
			records.append(record)

	if not records:
		raise ValueError(f"No comparison records found in {path}.")

	return records


def _average(values: list[float]) -> float | None:
	"""Return the arithmetic mean, or None for an empty list."""
	if not values:
		return None
	return mean(values)


def _extract_citations(reason: str) -> list[str]:
	"""Extract citation spans from one generated reason."""
	return [
		match.strip()
		for match in CITATION_PATTERN.findall(reason)
		if match.strip()
	]


def main() -> None:
	"""Calculate aggregate and per-query comparison metrics."""
	args = _parse_args()
	records = _load_jsonl(args.input)

	per_query: list[dict[str, Any]] = []

	baseline_counts: list[float] = []
	innovation_counts: list[float] = []
	baseline_times: list[float] = []
	innovation_times: list[float] = []
	jaccard_scores: list[float] = []
	overlap_counts: list[float] = []
	rank_changes: list[float] = []

	baseline_total_recommendations = 0
	baseline_recommendations_with_citations = 0
	innovation_total_recommendations = 0
	innovation_recommendations_with_citations = 0
	innovation_total_citation_spans = 0
	innovation_exact_citation_spans = 0
	successful_queries = 0

	for record in records:
		generation = record["generation"]
		baseline = generation["citation_free_baseline"]
		innovation = generation["evidence_grounded"]

		query_result: dict[str, Any] = {
			"query_id": record.get("query_id"),
			"topic": record.get("topic"),
			"baseline_status": baseline["status"],
			"innovation_status": innovation["status"],
		}

		if baseline["status"] != "ok" or innovation["status"] != "ok":
			per_query.append(query_result)
			continue

		successful_queries += 1

		baseline_recommendations = baseline["recommendations"]
		innovation_recommendations = innovation["recommendations"]

		baseline_count = len(baseline_recommendations)
		innovation_count = len(innovation_recommendations)

		baseline_counts.append(float(baseline_count))
		innovation_counts.append(float(innovation_count))
		baseline_times.append(float(baseline["elapsed_seconds"]))
		innovation_times.append(float(innovation["elapsed_seconds"]))

		baseline_ids = [
			item["arxiv_id"]
			for item in baseline_recommendations
		]
		innovation_ids = [
			item["arxiv_id"]
			for item in innovation_recommendations
		]

		baseline_set = set(baseline_ids)
		innovation_set = set(innovation_ids)
		common_ids = baseline_set & innovation_set
		union_ids = baseline_set | innovation_set

		overlap_count = len(common_ids)
		jaccard = (
			overlap_count / len(union_ids)
			if union_ids
			else 1.0
		)

		overlap_counts.append(float(overlap_count))
		jaccard_scores.append(jaccard)

		baseline_ranks = {
			item["arxiv_id"]: int(item["rank"])
			for item in baseline_recommendations
		}
		innovation_ranks = {
			item["arxiv_id"]: int(item["rank"])
			for item in innovation_recommendations
		}

		query_rank_changes = [
			abs(
				baseline_ranks[arxiv_id]
				- innovation_ranks[arxiv_id]
			)
			for arxiv_id in common_ids
		]
		rank_changes.extend(
			float(value)
			for value in query_rank_changes
		)

		baseline_total_recommendations += baseline_count
		for item in baseline_recommendations:
			citations = _extract_citations(item["reason"])
			if citations:
				baseline_recommendations_with_citations += 1

		innovation_total_recommendations += innovation_count
		query_citation_count = 0
		query_exact_citation_count = 0

		for item in innovation_recommendations:
			citations = _extract_citations(item["reason"])
			if citations:
				innovation_recommendations_with_citations += 1

			source_text = (
				(item.get("title") or "")
				+ "\n"
				+ (item.get("abstract") or "")
			)

			for citation in citations:
				innovation_total_citation_spans += 1
				query_citation_count += 1

				if citation in source_text:
					innovation_exact_citation_spans += 1
					query_exact_citation_count += 1

		query_result.update(
			{
				"baseline_returned_count": baseline_count,
				"innovation_returned_count": innovation_count,
				"overlap_count": overlap_count,
				"union_count": len(union_ids),
				"jaccard": jaccard,
				"mean_absolute_rank_change": (
					_average(
						[
							float(value)
							for value in query_rank_changes
						]
					)
				),
				"baseline_elapsed_seconds": baseline[
					"elapsed_seconds"
				],
				"innovation_elapsed_seconds": innovation[
					"elapsed_seconds"
				],
				"innovation_citation_spans": (
					query_citation_count
				),
				"innovation_exact_citation_spans": (
					query_exact_citation_count
				),
			}
		)
		per_query.append(query_result)

	baseline_citation_free_rate = (
		1
		- (
			baseline_recommendations_with_citations
			/ baseline_total_recommendations
		)
		if baseline_total_recommendations
		else None
	)
	innovation_citation_coverage = (
		innovation_recommendations_with_citations
		/ innovation_total_recommendations
		if innovation_total_recommendations
		else None
	)
	innovation_exact_citation_rate = (
		innovation_exact_citation_spans
		/ innovation_total_citation_spans
		if innovation_total_citation_spans
		else None
	)

	summary = {
		"query_count": len(records),
		"successful_query_count": successful_queries,
		"retrieval_methods": sorted(
			{
				record.get("retrieval", {}).get("method")
				for record in records
				if record.get("retrieval", {}).get("method")
			}
		),
		"average_candidate_count": _average(
			[
				float(record.get("candidate_count", 0))
				for record in records
			]
		),
		"baseline_average_returned_count": _average(
			baseline_counts
		),
		"innovation_average_returned_count": _average(
			innovation_counts
		),
		"average_overlap_count": _average(overlap_counts),
		"average_jaccard": _average(jaccard_scores),
		"average_absolute_rank_change": _average(
			rank_changes
		),
		"baseline_average_elapsed_seconds": _average(
			baseline_times
		),
		"innovation_average_elapsed_seconds": _average(
			innovation_times
		),
		"baseline_total_recommendations": (
			baseline_total_recommendations
		),
		"innovation_total_recommendations": (
			innovation_total_recommendations
		),
		"baseline_citation_free_rate": (
			baseline_citation_free_rate
		),
		"innovation_citation_coverage": (
			innovation_citation_coverage
		),
		"innovation_total_citation_spans": (
			innovation_total_citation_spans
		),
		"innovation_exact_citation_spans": (
			innovation_exact_citation_spans
		),
		"innovation_exact_citation_rate": (
			innovation_exact_citation_rate
		),
	}

	result = {
		"summary": summary,
		"per_query": per_query,
	}

	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(
		json.dumps(result, ensure_ascii=False, indent=2) + "\n",
		encoding="utf-8",
	)

	print(json.dumps(summary, ensure_ascii=False, indent=2))
	print(f"\nSaved metrics to: {args.output}")


if __name__ == "__main__":
	main()
