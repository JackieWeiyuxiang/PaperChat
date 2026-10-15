"""Common ownership and context management for retrieval resources."""

from contextlib import ExitStack
from types import TracebackType
from typing import Self


class ManagedResource:
    """Close registered, owned resources once; borrowed resources stay open."""

    def __init__(self) -> None:
        self._resources = ExitStack()
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed.")

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release owned resources; a closed instance cannot be reopened."""
        if not self._closed:
            self._closed = True
            self._resources.close()
