"""Compare citation-free and evidence-grounded generation on fixed candidates."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from arxiv_paper_recommend.recommendation.innovation import (
	EvidenceGroundedRecommendationGenerator,
)
from arxiv_paper_recommend.retrieval import SearchResult

from citation_free_generator import CitationFreeRecommendationGenerator


DEFAULT_CANDIDATES_PATH = Path("experiment/generation/candidates.jsonl")
DEFAULT_OUTPUT_PATH = Path("experiment/generation/comparison.jsonl")
DEFAULT_RECOMMENDATION_LIMIT = 10


def _parse_args() -> argparse.Namespace:
	"""Parse generation comparison settings."""
	parser = argparse.ArgumentParser(
		description=(
			"Compare citation-free and evidence-grounded generation "
			"on identical retrieval candidates."
		)
	)
	parser.add_argument(
		"--candidates",
		type=Path,
		default=DEFAULT_CANDIDATES_PATH,
		help="JSONL file created by prepare_candidates.py.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT_PATH,
		help="JSONL file used to store comparison results.",
	)
	parser.add_argument(
		"--model",
		default=None,
		help="Optional LLM model override; otherwise use LLM_MODEL from .env.",
	)
	parser.add_argument(
		"--recommendation-limit",
		type=int,
		default=DEFAULT_RECOMMENDATION_LIMIT,
		help="Maximum number of recommendations from each generator.",
	)
	parser.add_argument(
		"--query-limit",
		type=int,
		default=None,
		help="Optional number of candidate records used for a pilot run.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Replace the output file if it already exists.",
	)
	return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
	"""Load non-empty JSON objects from one JSONL file."""
	records: list[dict[str, Any]] = []

	with path.open(encoding="utf-8") as input_file:
		for line_number, line in enumerate(input_file, start=1):
			stripped = line.strip()
			if not stripped:
				continue

			record = json.loads(stripped)
			if not isinstance(record, dict):
				raise ValueError(
					f"JSONL line {line_number} must contain one object."
				)
			records.append(record)

	if not records:
		raise ValueError(f"No candidate records found in {path}.")

	return records


def _deserialize_candidates(
	record: dict[str, Any],
) -> tuple[list[SearchResult], dict[str, int]]:
	"""Rebuild SearchResult objects and preserve their retrieval ranks."""
	raw_candidates = record.get("candidates")
	if not isinstance(raw_candidates, list):
		raise ValueError("Candidate record must contain a candidates list.")

	papers: list[SearchResult] = []
	candidate_ranks: dict[str, int] = {}

	for index, candidate in enumerate(raw_candidates, start=1):
		if not isinstance(candidate, dict):
			raise ValueError(f"Candidate at index {index} must be an object.")

		arxiv_id = candidate.get("arxiv_id")
		title = candidate.get("title")
		abstract = candidate.get("abstract")
		score = candidate.get("score")

		if not isinstance(arxiv_id, str) or not arxiv_id.strip():
			raise ValueError(
				f"Candidate at index {index} has an invalid arxiv_id."
			)
		if not isinstance(title, str) or not title.strip():
			raise ValueError(
				f"Candidate at index {index} has an invalid title."
			)
		if abstract is not None and not isinstance(abstract, str):
			raise ValueError(
				f"Candidate at index {index} has an invalid abstract."
			)

		paper = SearchResult(
			document_id=int(candidate["document_id"]),
			arxiv_id=arxiv_id,
			title=title,
			abstract=abstract,
			score=None if score is None else float(score),
		)
		papers.append(paper)
		candidate_ranks.setdefault(
			arxiv_id,
			int(candidate.get("rank", index)),
		)

	return papers, candidate_ranks


def _serialize_recommendation(
	item: Any,
	candidate_ranks: dict[str, int],
) -> dict[str, Any]:
	"""Convert one generated recommendation into JSON-safe experiment data."""
	return {
		"rank": item.rank,
		"arxiv_id": item.paper.arxiv_id,
		"title": item.paper.title,
		"abstract": item.paper.abstract,
		"candidate_rank": candidate_ranks.get(item.paper.arxiv_id),
		"retrieval_score": item.paper.score,
		"reason": item.reason,
	}


def _run_generator(
	generator: Any,
	generator_name: str,
	request: str,
	papers: list[SearchResult],
	candidate_ranks: dict[str, int],
	limit: int,
) -> dict[str, Any]:
	"""Run one generator and record its output, latency, and failures."""
	started_at = perf_counter()

	try:
		recommendations = generator.generate(
			request,
			papers,
			limit=limit,
		)
	except Exception as error:
		return {
			"generator": generator_name,
			"status": "error",
			"elapsed_seconds": perf_counter() - started_at,
			"error_type": type(error).__name__,
			"error_message": str(error),
			"returned_count": 0,
			"recommendations": [],
		}

	return {
		"generator": generator_name,
		"status": "ok",
		"elapsed_seconds": perf_counter() - started_at,
		"returned_count": len(recommendations),
		"recommendations": [
			_serialize_recommendation(item, candidate_ranks)
			for item in recommendations
		],
	}


def main() -> None:
	"""Run both generators on exactly the same saved candidates."""
	load_dotenv()
	args = _parse_args()

	if not 1 <= args.recommendation_limit <= 10:
		raise ValueError(
			"--recommendation-limit must be between 1 and 10."
		)
	if args.query_limit is not None and args.query_limit < 1:
		raise ValueError(
			"--query-limit must be at least 1 when provided."
		)

	records = _load_jsonl(args.candidates)
	if args.query_limit is not None:
		records = records[: args.query_limit]

	args.output.parent.mkdir(parents=True, exist_ok=True)
	output_mode = "w" if args.overwrite else "x"
	model = args.model or os.getenv("LLM_MODEL")
	created_at = datetime.now(UTC).isoformat()

	with args.output.open(output_mode, encoding="utf-8") as output:
		with CitationFreeRecommendationGenerator(
			model=args.model
		) as baseline:
			with EvidenceGroundedRecommendationGenerator(
				model=args.model
			) as innovation:
				for position, record in enumerate(records, start=1):
					request = record.get("request")
					if not isinstance(request, str) or not request.strip():
						raise ValueError(
							f"Record {position} must contain "
							"a non-empty request."
						)

					papers, candidate_ranks = _deserialize_candidates(
						record
					)

					baseline_result = _run_generator(
						baseline,
						"CitationFreeRecommendationGenerator",
						request,
						papers,
						candidate_ranks,
						args.recommendation_limit,
					)
					innovation_result = _run_generator(
						innovation,
						"EvidenceGroundedRecommendationGenerator",
						request,
						papers,
						candidate_ranks,
						args.recommendation_limit,
					)

					comparison = {
						"query_id": record.get("query_id"),
						"request": request,
						"question": record.get("question"),
						"scope": record.get("scope"),
						"topic": record.get("topic"),
						"query_type": record.get("query_type"),
						"retrieval": record.get("retrieval"),
						"candidate_count": len(papers),
						"generation": {
							"model": model,
							"recommendation_limit": (
								args.recommendation_limit
							),
							"created_at": created_at,
							"citation_free_baseline": baseline_result,
							"evidence_grounded": innovation_result,
						},
					}

					output.write(
						json.dumps(
							comparison,
							ensure_ascii=False,
						)
						+ "\n"
					)
					output.flush()

					print(
						f"[{position}/{len(records)}] "
						f"{record.get('query_id')}: "
						f"baseline={baseline_result['status']}, "
						f"innovation={innovation_result['status']}"
					)


if __name__ == "__main__":
	main()
