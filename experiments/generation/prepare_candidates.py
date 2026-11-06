"""Prepare fixed retrieval candidates for generation comparison."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from arxiv_paper_recommend.retrieval import SearchEngine, SearchResult


DEFAULT_QUERIES_PATH = Path("data/example_queries.json")
DEFAULT_OUTPUT_PATH = Path("experiment/generation/candidates.jsonl")
DEFAULT_CANDIDATE_LIMIT = 30
DEFAULT_RETRIEVAL_METHOD = "BM25"


def _parse_args() -> argparse.Namespace:
	"""Parse options using BM25 by default for generation comparison."""
	parser = argparse.ArgumentParser(
		description="Retrieve and save one fixed candidate set per research request."
	)
	parser.add_argument(
		"--queries",
		type=Path,
		default=DEFAULT_QUERIES_PATH,
		help="JSON file containing the shared questions list.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT_PATH,
		help="JSONL file used to store fixed retrieval candidates.",
	)
	parser.add_argument(
		"--method",
		default=DEFAULT_RETRIEVAL_METHOD,
		help=(
			"SearchEngine method. Defaults to BM25 for the controlled "
			"generation comparison."
		),
	)
	parser.add_argument(
		"--candidate-limit",
		type=int,
		default=DEFAULT_CANDIDATE_LIMIT,
		help="Maximum number of retrieved candidates per request.",
	)
	parser.add_argument(
		"--query-limit",
		type=int,
		default=None,
		help="Optional number of queries used for a pilot run.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Replace the output file if it already exists.",
	)
	return parser.parse_args()


def _load_questions(path: Path) -> list[dict[str, Any]]:
	"""Load and minimally validate the shared query collection."""
	payload = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(payload, dict):
		raise ValueError("Query file must contain one JSON object.")

	questions = payload.get("questions")
	if not isinstance(questions, list):
		raise ValueError("Query file must contain a questions list.")

	for index, question in enumerate(questions):
		if not isinstance(question, dict):
			raise ValueError(
				f"Question at index {index} must be an object."
			)

		request = question.get("query_en")
		if not isinstance(request, str) or not request.strip():
			raise ValueError(
				f"Question at index {index} must contain "
				"a non-empty query_en."
			)

	return questions


def _serialize_candidate(
	rank: int,
	paper: SearchResult,
) -> dict[str, Any]:
	"""Convert one SearchResult into JSON-safe experiment data."""
	return {
		"rank": rank,
		"document_id": int(paper.document_id),
		"arxiv_id": paper.arxiv_id,
		"title": paper.title,
		"abstract": paper.abstract,
		"score": (
			None
			if paper.score is None
			else float(paper.score)
		),
	}


def main() -> None:
	"""Retrieve each query once and save candidates immediately."""
	load_dotenv()
	args = _parse_args()

	if args.candidate_limit < 1:
		raise ValueError(
			"--candidate-limit must be at least 1."
		)
	if args.query_limit is not None and args.query_limit < 1:
		raise ValueError(
			"--query-limit must be at least 1 when provided."
		)

	questions = _load_questions(args.queries)
	if args.query_limit is not None:
		questions = questions[: args.query_limit]

	method = args.method.strip().upper()
	args.output.parent.mkdir(parents=True, exist_ok=True)
	output_mode = "w" if args.overwrite else "x"
	created_at = datetime.now(UTC).isoformat()

	with args.output.open(
		output_mode,
		encoding="utf-8",
	) as output:
		with SearchEngine() as engine:
			if method not in engine.available_methods:
				available = ", ".join(
					engine.available_methods
				)
				raise ValueError(
					f"Unsupported method {method!r}. "
					f"Available methods: {available}."
				)

			for position, question in enumerate(
				questions,
				start=1,
			):
				request = question["query_en"].strip()
				started_at = perf_counter()

				papers = engine.search(
					request,
					method=method,
					limit=args.candidate_limit,
				)
				elapsed_seconds = (
					perf_counter() - started_at
				)

				record = {
					"query_id": question.get("id"),
					"request": request,
					"question": question.get("question"),
					"scope": question.get("scope"),
					"topic": question.get("topic"),
					"query_type": question.get(
						"query_type"
					),
					"retrieval": {
						"method": method,
						"candidate_limit": (
							args.candidate_limit
						),
						"returned_count": len(papers),
						"elapsed_seconds": (
							elapsed_seconds
						),
						"llm_model": os.getenv(
							"LLM_MODEL"
						),
						"created_at": created_at,
					},
					"candidates": [
						_serialize_candidate(
							rank,
							paper,
						)
						for rank, paper in enumerate(
							papers,
							start=1,
						)
					],
				}

				output.write(
					json.dumps(
						record,
						ensure_ascii=False,
					)
					+ "\n"
				)
				output.flush()

				print(
					f"[{position}/{len(questions)}] "
					f"{question.get('id')}: "
					f"{len(papers)} candidates "
					f"in {elapsed_seconds:.2f}s"
				)


if __name__ == "__main__":
	main()
