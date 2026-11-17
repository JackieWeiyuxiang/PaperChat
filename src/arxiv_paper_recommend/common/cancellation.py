"""Cooperative cancellation scoped to one search and its assessment workers."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event


class SearchCancelled(Exception):
	"""Signal intentional cancellation without treating it as a search failure."""


class CancellationToken:
	"""Share a cancellation signal between request and worker threads."""

	def __init__(self) -> None:
		self._event = Event()

	@property
	def cancelled(self) -> bool:
		return self._event.is_set()

	def cancel(self) -> None:
		self._event.set()

	def check(self) -> None:
		if self.cancelled:
			raise SearchCancelled("Search cancelled.")

	def wait(self, seconds: float) -> None:
		if self._event.wait(seconds):
			raise SearchCancelled("Search cancelled.")


_current_token: ContextVar[CancellationToken | None] = ContextVar(
	"search_cancellation", default=None
)


@contextmanager
def cancellation_scope(token: CancellationToken) -> Iterator[None]:
	"""Install a request token temporarily, leaving resource ownership unchanged."""
	previous = _current_token.set(token)
	try:
		yield
	finally:
		_current_token.reset(previous)


def check_cancelled() -> None:
	"""Stop at a safe boundary, or do nothing outside a cancellable search."""
	token = _current_token.get()
	if token is not None:
		token.check()


def cancellable_wait(seconds: float) -> None:
	"""Allow retry backoff to end immediately when this search is cancelled."""
	token = _current_token.get()
	if token is None:
		time.sleep(seconds)
	else:
		token.wait(seconds)
