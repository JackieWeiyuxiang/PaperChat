"""Serve the local paper search UI: python -m arxiv_paper_recommend.web."""

import argparse
import json
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4

from arxiv_paper_recommend.common.cancellation import (
	CancellationToken,
	SearchCancelled,
	cancellation_scope,
	check_cancelled,
)

logger = logging.getLogger(__name__)


class SearchBusy(Exception):
	"""Reject new work without queueing it behind another search."""


@dataclass
class SearchTask:
	"""Keep a search identity and its cancellation signal together."""

	search_id: str
	token: CancellationToken = field(default_factory=CancellationToken)


class RecommendationService:
	"""Keep the pipeline and its SQLite connections on one worker thread."""

	def __init__(self):
		self.executor = ThreadPoolExecutor(max_workers=1)
		self.state_lock = Lock()
		self.active: SearchTask | None = None
		self.recent: OrderedDict[str, str] = OrderedDict()
		self.closed = False
		self.pipeline = None

	def recommend(self, request, search_id=None):
		if search_id is None:
			search_id = str(uuid4())
		with self.state_lock:
			if self.closed:
				raise SearchBusy("The recommendation service is shutting down.")
			if self.recent.get(search_id) == "cancelled":
				raise SearchCancelled("Search cancelled.")
			if search_id in self.recent:
				raise SearchBusy(
					"This search ID has already been used. Submit a new search."
				)
			if self.active is not None:
				message = (
					"A search is being cancelled. Please try again shortly."
					if self.active.token.cancelled
					else "A search is already running. Please try again when it finishes."
				)
				raise SearchBusy(message)
			task = SearchTask(search_id)
			self.active = task
			try:
				future = self.executor.submit(self._run, request, task)
			except BaseException:
				self.active = None
				raise
		return future.result()

	def cancel(self, search_id):
		with self.state_lock:
			if self.active is not None and self.active.search_id == search_id:
				self.active.token.cancel()
			elif self.recent.get(search_id) != "finished":
				# Remember cancellations that arrive before the search POST is registered.
				self._remember(search_id, "cancelled")

	def _remember(self, search_id, state):
		"""Update bounded terminal history while holding state_lock."""
		self.recent[search_id] = state
		self.recent.move_to_end(search_id)
		while len(self.recent) > 256:
			self.recent.popitem(last=False)

	def _run(self, request, task):
		try:
			with cancellation_scope(task.token):
				check_cancelled()
				pipeline = self._get_pipeline()
				check_cancelled()
				return pipeline.recommend(request)
		finally:
			# Completion and cancellation have one ordering, including exceptional exits.
			with self.state_lock:
				self._remember(
					task.search_id, "cancelled" if task.token.cancelled else "finished"
				)
				self.active = None
				task.token.check()

	def _get_pipeline(self):
		if self.pipeline is None:
			from arxiv_paper_recommend.config import LLMSettings

			settings = LLMSettings.from_env()
			missing = []
			if not settings.api_key:
				missing.append("LLM_API_KEY")
			if not settings.model:
				missing.append("LLM_MODEL")
			if missing:
				raise ValueError(
					"Configure "
					+ " and ".join(missing)
					+ " in the server's .env file first."
				)
			from arxiv_paper_recommend.pipeline import PaperRecommender

			self.pipeline = PaperRecommender()
		return self.pipeline

	def close(self):
		with self.state_lock:
			if self.closed:
				return
			self.closed = True
			if self.active is not None:
				self.active.token.cancel()
			cleanup = self.executor.submit(self._close_pipeline)
		try:
			cleanup.result()
		finally:
			self.executor.shutdown(wait=True)

	def _close_pipeline(self):
		if self.pipeline is not None:
			self.pipeline.close()


class UIHandler(BaseHTTPRequestHandler):
	"""Expose the page, recommendations and request-scoped cancellation."""

	def respond(self, status, payload, content_type="application/json; charset=utf-8"):
		body = (
			payload
			if isinstance(payload, bytes)
			else json.dumps(payload, ensure_ascii=False).encode("utf-8")
		)
		self.send_response(status)
		self.send_header("Content-Type", content_type)
		self.send_header("Content-Length", str(len(body)))
		self.send_header("Cache-Control", "no-store")
		self.send_header("X-Content-Type-Options", "nosniff")
		self.end_headers()
		try:
			self.wfile.write(body)
		except BrokenPipeError, ConnectionResetError:
			pass

	def do_GET(self):
		if self.path == "/":
			self.respond(
				200,
				Path(__file__).with_name("index.html").read_bytes(),
				"text/html; charset=utf-8",
			)
		else:
			self.respond(404, {"error": "Not found."})

	def do_POST(self):
		if self.path not in {"/api/recommend", "/api/recommend/cancel"}:
			self.respond(404, {"error": "Not found."})
			return
		# Browser requests must come from this UI, not a different website.
		origin = self.headers.get("Origin")
		if origin and origin != f"http://{self.headers.get('Host')}":
			self.respond(403, {"error": "Origin not allowed."})
			return
		try:
			length = int(self.headers.get("Content-Length", "0"))
			if not 0 < length <= 65536:
				raise ValueError("Invalid request size.")
			payload = json.loads(self.rfile.read(length))
			if not isinstance(payload, dict):
				raise TypeError("Expected a JSON object.")
			cancelling = self.path == "/api/recommend/cancel"
			if "search_id" in payload:
				search_id = payload["search_id"]
				if not isinstance(search_id, str):
					raise TypeError("search_id must be a UUID string.")
				try:
					search_id = str(UUID(search_id))
				except ValueError:
					raise ValueError("search_id must be a UUID string.") from None
			elif cancelling:
				raise ValueError("Provide the search_id to cancel.")
			else:
				search_id = str(uuid4())
			if not cancelling:
				request = payload.get("request")
				if not isinstance(request, str) or not request.strip():
					raise ValueError("Enter a research request.")
				if len(request) > 4000:
					raise ValueError("Keep your request under 4,000 characters.")
		except (TypeError, ValueError) as exc:
			self.respond(400, {"error": str(exc)})
			return
		service = self.server.service
		if cancelling:
			service.cancel(search_id)
			self.respond(202, {"status": "cancelling"})
			return
		try:
			result = service.recommend(request.strip(), search_id)
			self.respond(200, result)
		except SearchCancelled:
			self.respond(409, {"error": "Search cancelled.", "cancelled": True})
		except SearchBusy as exc:
			self.respond(409, {"error": str(exc)})
		except ModuleNotFoundError:
			self.respond(
				503,
				{
					"error": "Backend dependencies are missing. Run uv sync --extra vectorization before searching."
				},
			)
		except ValueError as exc:
			if str(exc).startswith("Configure LLM_"):
				self.respond(503, {"error": str(exc)})
			else:
				logger.exception("Recommendation failed")
				self.respond(
					502,
					{
						"error": "The recommendation service could not complete this search. Please try again."
					},
				)
		except Exception:
			logger.exception("Recommendation failed")
			self.respond(
				502,
				{
					"error": "The recommendation service could not complete this search. Check the server configuration or try again."
				},
			)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--port", type=int, default=8000)
	args = parser.parse_args()
	service = RecommendationService()
	server = ThreadingHTTPServer(("127.0.0.1", args.port), UIHandler)
	server.service = service
	print(f"Paper search: http://127.0.0.1:{args.port}", flush=True)
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		pass
	finally:
		try:
			service.close()
		finally:
			server.server_close()


if __name__ == "__main__":
	main()
