"""Input validation shared by retrieval and query generation."""


def validate_integer(
    value: int, name: str, *, minimum: int = 1, maximum: int | None = None,
) -> None:
    """Require an integer in the specified range, excluding booleans."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        bounds = f"at least {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise ValueError(f"{name} must be an integer {bounds}.")


def validate_text(value: str, name: str, *, strip: bool = False) -> str:
    """Validate before resource creation; preserve query keys unless requested."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty or whitespace.")
    return value.strip() if strip else value


def normalize_queries(queries: str | set[str], *, allow_empty: bool = True) -> list[str]:
    """Preserve a single query; sort sets deterministically and reject blanks."""
    if isinstance(queries, str):
        texts = [queries]
    elif isinstance(queries, set):
        if any(not isinstance(query, str) for query in queries):
            raise TypeError("Queries must contain only strings.")
        texts = sorted(queries)
    else:
        raise TypeError("Queries must be a string or a set of strings.")
    if not texts and not allow_empty:
        raise ValueError("At least one query is required.")
    for text in texts:
        validate_text(text, "Query")
    return texts
