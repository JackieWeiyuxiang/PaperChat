"""Input validation shared by retrieval and generation."""

from collections.abc import Sequence


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


def normalize_queries(queries: Sequence[str]) -> list[str]:
	"""Validate a sequence and deduplicate queries in first-occurrence order."""
	if isinstance(queries, str) or not isinstance(queries, Sequence):
		raise TypeError(
			"Recall queries must be a sequence of strings, not a string or set."
		)
	for query in queries:
		validate_text(query, "Recall query")
	return list(dict.fromkeys(queries))
