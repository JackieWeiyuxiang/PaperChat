"""Input validation shared by retrieval and generation."""

from collections.abc import Sequence
from math import isfinite


def validate_integer(
	value: int,
	name: str,
	*,
	minimum: int = 1,
	maximum: int | None = None,
) -> None:
	"""Require an integer in the specified range, excluding booleans."""
	if (
		isinstance(value, bool)
		or not isinstance(value, int)
		or value < minimum
		or (maximum is not None and value > maximum)
	):
		bounds = (
			f"at least {minimum}"
			if maximum is None
			else f"between {minimum} and {maximum}"
		)
		raise ValueError(f"{name} must be an integer {bounds}.")


def validate_text(value: str, name: str, *, strip: bool = False) -> str:
	"""Reject blank strings and optionally strip surrounding whitespace."""
	if not isinstance(value, str):
		raise TypeError(f"{name} must be a string.")
	if not value.strip():
		raise ValueError(f"{name} must not be empty or whitespace.")
	return value.strip() if strip else value


def validate_weight(weight: float, name: str) -> float:
	"""Accept finite numeric weights from zero through one, excluding booleans."""
	if isinstance(weight, bool) or not isinstance(weight, (int, float)):
		raise TypeError(f"{name} must be an integer or a float.")
	try:
		weight = float(weight)
	except OverflowError as error:
		raise ValueError(f"{name} must be finite and between 0 and 1.") from error
	if not isfinite(weight) or not 0.0 <= weight <= 1.0:
		raise ValueError(f"{name} must be finite and between 0 and 1.")
	return weight


def normalize_queries(queries: Sequence[str]) -> list[str]:
	"""Validate a sequence and deduplicate queries in first-occurrence order."""
	if isinstance(queries, str) or not isinstance(queries, Sequence):
		raise TypeError(
			"Recall queries must be a sequence of strings, not a string or set."
		)
	for query in queries:
		validate_text(query, "Recall query")
	return list(dict.fromkeys(queries))
