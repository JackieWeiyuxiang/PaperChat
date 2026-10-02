"""Small, paced arXiv metadata harvests; no third-party Python dependencies."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

API = "https://export.arxiv.org/api/query"
USER_AGENT = "AI6127-ArxivCorpus/0.1 (+https://github.com/Esxyko/arxiv_paper_recommend)"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom",
      "op": "http://a9.com/-/spec/opensearch/1.1/"}
MAX_BYTES = 10_000_000
PROJECT_ROOT = Path(__file__).resolve().parents[4]


class FetchError(RuntimeError):
    """An unavailable source, invalid response, or exhausted retry budget."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


class HttpClient:
    """One connection at a time, at least 3 s apart, with bounded retries.

    Windows uses native curl/Schannel by default to use the system certificate
    store. TLS certificate verification remains enabled for both transports.
    """

    def __init__(self, delay: float = 3.2, timeout: float = 30, retries: int = 2,
                 transport: str = "auto"):
        if delay < 3 or timeout <= 0 or retries < 0:
            raise ValueError("delay must be >= 3, timeout > 0, retries >= 0")
        if transport == "auto":
            transport = "curl" if sys.platform == "win32" and shutil.which("curl.exe") else "urllib"
        self.transport = transport
        self.delay, self.timeout, self.retries = delay, timeout, retries
        self._next_request = 0.0
        self.events: list[dict] = []

    def _request(self, url: str) -> tuple[int, dict, bytes]:
        if self.transport == "curl":
            curl = shutil.which("curl.exe") or shutil.which("curl")
            if not curl:
                raise FetchError("curl is unavailable; try --transport urllib")
            with tempfile.TemporaryDirectory(prefix="arxiv-http-") as temp:
                headers_path, body_path = Path(temp) / "headers", Path(temp) / "body"
                command = [curl, "--silent", "--show-error", "--location", "--max-redirs", "3",
                           "--proto", "=https", "--proto-redir", "=https",
                           "--connect-timeout", "10", "--max-time", str(self.timeout),
                           "--max-filesize", str(MAX_BYTES), "--user-agent", USER_AGENT,
                           "--header", "Accept: application/atom+xml, application/xml",
                           "--dump-header", str(headers_path), "--output", str(body_path),
                           "--write-out", "%{http_code}", url]
                result = subprocess.run(command, capture_output=True, text=True,
                                        timeout=self.timeout + 5)
                if result.returncode:
                    raise FetchError(result.stderr.strip() or f"curl exit {result.returncode}")
                headers = {}
                for block in headers_path.read_text(encoding="iso-8859-1").split("\n\n"):
                    if block.startswith("HTTP/"):
                        headers = dict(line.split(":", 1) for line in block.splitlines()[1:] if ":" in line)
                return int(result.stdout.strip()), {k.lower(): v.strip() for k, v in headers.items()}, body_path.read_bytes()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                        "Accept": "application/atom+xml, application/xml"})
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            body = response.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES:
                raise FetchError("Response exceeded 10 MB; reduce the page size")
            return response.code, {k.lower(): v for k, v in response.headers.items()}, body

    def get(self, url: str) -> bytes:
        for attempt in range(self.retries + 1):
            wait = self._next_request - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            started = time.monotonic()
            event = {"url": url, "requested_at": utc_now(), "attempt": attempt + 1,
                     "transport": self.transport}
            status, headers, body = None, {}, b""
            try:
                status, headers, body = self._request(url)
                event.update(status=status, bytes=len(body))
            except (FetchError, urllib.error.URLError, OSError, subprocess.TimeoutExpired) as error:
                event["error"] = str(error)
            finally:
                self._next_request = time.monotonic() + self.delay
                event["elapsed_seconds"] = round(time.monotonic() - started, 3)
                self.events.append(event)
            if status == 200:
                return body
            # Authentication / forbidden responses stop immediately, without evasion.
            if status is not None and status not in {429, 500, 502, 503, 504}:
                raise FetchError(f"HTTP {status}: {url}")
            reason = event.get("error", f"HTTP {status}")
            if attempt == self.retries:
                raise FetchError(f"Retry budget exhausted: {reason}")
            wait = max(self.delay, retry_after_seconds(headers.get("retry-after")) or 5 * 2 ** attempt)
            if wait > 60:
                raise FetchError(f"Source requests a {wait:.0f}s cooldown; stop and retry later: {url}")
            print(f"{reason}; waiting {wait:.1f}s before retry", file=sys.stderr, flush=True)
            time.sleep(wait)
        raise AssertionError("unreachable")


def parse_atom(body: bytes) -> tuple[list[dict], int, int, int]:
    """Return valid papers, raw entry count, result count and invalid count."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError as error:
        raise FetchError("Expected Atom XML; response may be a gateway or challenge page") from error
    if root.tag != f"{{{NS['a']}}}feed":
        raise FetchError("Response is not an Atom feed")
    entries = root.findall("a:entry", NS)
    papers, invalid = [], 0
    for entry in entries:
        raw_id = entry.findtext("a:id", "", NS)
        title = " ".join(entry.findtext("a:title", "", NS).split())
        abstract = " ".join(entry.findtext("a:summary", "", NS).split())
        if "/api/errors" in raw_id or title.lower() == "error":
            raise FetchError(f"arXiv API error: {abstract}")
        match = re.fullmatch(r"https?://arxiv\.org/abs/((?:\d{4}\.\d{4,5}|[A-Za-z.-]+/\d{7}))(?:v(\d+))?", raw_id)
        if not match or not title or not abstract:
            invalid += 1
            continue
        paper_id, version = match.group(1), int(match.group(2) or 1)
        categories = [c.get("term") for c in entry.findall("a:category", NS) if c.get("term")]
        primary = entry.find("arxiv:primary_category", NS)
        papers.append({
            "paper_id": paper_id, "version": version, "title": title,
            "abstract": abstract, "text": title + "\n\n" + abstract,
            "authors": [" ".join(a.findtext("a:name", "", NS).split()) for a in entry.findall("a:author", NS)],
            "categories": categories,
            "primary_category": primary.get("term") if primary is not None else None,
            "published": entry.findtext("a:published", "", NS),
            "updated": entry.findtext("a:updated", "", NS),
            "url": "https://arxiv.org/abs/" + paper_id,
            "pdf_url": f"https://arxiv.org/pdf/{paper_id}v{version}",
            "doi": entry.findtext("arxiv:doi", None, NS), "source": "arxiv_api",
        })
    try:
        total = int(root.findtext("op:totalResults", str(len(entries)), NS))
    except ValueError as error:
        raise FetchError("Invalid totalResults in Atom response") from error
    return papers, len(entries), total, invalid


def harvest(query: str, limit: int, page_size: int, output: Path, cache: Path,
            client: HttpClient) -> dict:
    if not query.strip() or not 1 <= limit <= 1000 or not 1 <= page_size <= 200:
        raise ValueError("query is required; limit must be 1..1000 and page_size 1..200")
    cache.mkdir(parents=True, exist_ok=True)
    papers: dict[str, dict] = {}
    pages, offset, duplicates, invalid = [], 0, 0, 0
    started = time.monotonic()
    total = 0
    # Bound work even if the remote service returns invalid or repeated records.
    for _ in range(math.ceil(limit / page_size) + 5):
        url = API + "?" + urllib.parse.urlencode({
            "search_query": query, "start": offset, "max_results": min(page_size, limit - len(papers)),
            "sortBy": "submittedDate", "sortOrder": "descending"})
        key = hashlib.sha256(url.encode()).hexdigest()
        raw_path, meta_path = cache / (key + ".xml"), cache / (key + ".json")
        cached = raw_path.exists() and meta_path.exists()
        body = raw_path.read_bytes() if cached else client.get(url)
        records, count, total, dropped = parse_atom(body)
        if cached:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata["sha256"] != hashlib.sha256(body).hexdigest():
                raise FetchError(f"Cache checksum mismatch: {raw_path}")
        else:
            metadata = {"url": url, "retrieved_at": utc_now(), "sha256": hashlib.sha256(body).hexdigest()}
            raw_path.write_bytes(body)
            write_json(meta_path, metadata)
        try:
            raw_reference = raw_path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            raw_reference = raw_path.resolve().as_posix()
        pages.append({**metadata, "cache_hit": cached, "raw_file": raw_reference,
                      "entries": count, "invalid_records": dropped})
        invalid += dropped
        for paper in records:
            paper["fetched_at"] = metadata["retrieved_at"]
            previous = papers.get(paper["paper_id"])
            if previous:
                duplicates += 1
            if previous is None or (paper["version"], paper["updated"]) > (previous["version"], previous["updated"]):
                papers[paper["paper_id"]] = paper
        print(f"{'cache' if cached else 'fetched'} start={offset}: {count} entries, {len(papers)} unique papers", flush=True)
        offset += count
        if len(papers) >= limit or not count or offset >= total:
            break
    rows = list(papers.values())[:limit]
    if not rows:
        raise FetchError("No usable papers; existing output was left unchanged")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(output)
    lengths = [len(row["abstract"].split()) for row in rows]
    report = {
        "completed_at": utc_now(), "query": query, "requested": limit, "saved": len(rows),
        "target_reached": len(rows) == limit, "total_matching_at_last_page": total,
        "duplicates_removed": duplicates, "invalid_records_dropped": invalid,
        "missing_title": sum(not row["title"] for row in rows),
        "missing_abstract": sum(not row["abstract"] for row in rows),
        "primary_categories": dict(Counter(row["primary_category"] for row in rows)),
        "all_categories": dict(Counter(c for row in rows for c in row["categories"])),
        "published_range": [min(row["published"] for row in rows), max(row["published"] for row in rows)],
        "abstract_words": {"min": min(lengths), "mean": round(statistics.mean(lengths), 1), "max": max(lengths)},
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "transport": client.transport, "delay_seconds": client.delay,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "relative_paths_base": "project_root", "pages": pages, "requests": client.events,
    }
    write_json(output.with_suffix(".manifest.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="cat:cs.CL")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--page-size", type=int, default=25)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/crawling/samples/arxiv_cs_cl_100.jsonl",
                        help="Output JSONL (default: project data/crawling/samples)")
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "data/crawling/raw/arxiv",
                        help="Raw response cache (default: project data/crawling/raw/arxiv)")
    parser.add_argument("--delay", type=float, default=3.2)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--transport", choices=["auto", "curl", "urllib"], default="auto")
    args = parser.parse_args()
    try:
        client = HttpClient(args.delay, args.timeout, args.retries, args.transport)
    except ValueError as error:
        parser.error(str(error))
    try:
        report = harvest(args.query, args.limit, args.page_size, args.output, args.cache, client)
    except (FetchError, ValueError, OSError) as error:
        write_json(args.output.with_suffix(".failure.json"), {"failed_at": utc_now(),
                   "query": args.query, "error": str(error), "requests": client.events})
        parser.exit(1, f"Stopped: {error}\nSuccessful pages remain cached; rerun the same command to resume.\n")
    print(f"Saved {report['saved']} papers to {args.output}")
    if not report["target_reached"]:
        parser.exit(2, "Saved a partial sample; see the manifest before using it.\n")


if __name__ == "__main__":
    main()
