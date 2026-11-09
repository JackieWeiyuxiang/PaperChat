"""Serve the local paper search UI: python -m arxiv_paper_recommend.web."""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


class RecommendationService:
	"""Keep the pipeline and its SQLite connections on one worker thread."""

	def __init__(self):
		self.executor = ThreadPoolExecutor(max_workers=1)
		self.busy = Lock()
		self.pipeline = None

	def recommend(self, request):
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
		return self.pipeline.recommend(request)

	def close(self):
		if self.pipeline is not None:
			self.executor.submit(self.pipeline.close).result()
		self.executor.shutdown(wait=True)


class UIHandler(BaseHTTPRequestHandler):
	"""Expose only the page and the recommendation endpoint."""

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
		if self.path != "/api/recommend":
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
			request = payload.get("request") if isinstance(payload, dict) else None
			if not isinstance(request, str) or not request.strip():
				raise ValueError("Enter a research request.")
			if len(request) > 4000:
				raise ValueError("Keep your request under 4,000 characters.")
		except (ValueError, UnicodeDecodeError) as exc:
			self.respond(400, {"error": str(exc)})
			return
		service = self.server.service
		if not service.busy.acquire(blocking=False):
			self.respond(
				409,
				{
					"error": "A search is already running. Please try again when it finishes."
				},
			)
			return
		try:
			result = service.executor.submit(
				service.recommend, request.strip()
			).result()
			self.respond(200, result)
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
		finally:
			service.busy.release()


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
		server.server_close()
		service.close()


if __name__ == "__main__":
	main()
