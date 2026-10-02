"""Offline checks for failures that could corrupt a corpus or violate pacing."""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from arxiv_paper_recommend.corpus.crawling.arxiv import FetchError, HttpClient, harvest, parse_atom, retry_after_seconds


def feed(ids):
    entries = "".join(f"""<entry><id>http://arxiv.org/abs/{paper_id}</id>
        <title> A  paper\n title </title><summary> A real\n abstract. </summary>
        <published>2025-01-01T00:00:00Z</published><updated>2025-01-02T00:00:00Z</updated>
        <author><name>A. Author</name></author><category term="cs.CL"/>
        </entry>""" for paper_id in ids)
    return (f'<feed xmlns="http://www.w3.org/2005/Atom" '
            f'xmlns:op="http://a9.com/-/spec/opensearch/1.1/">'
            f'<op:totalResults>100</op:totalResults>{entries}</feed>').encode()


class ArxivTests(unittest.TestCase):
    def test_normalizes_versioned_and_legacy_ids(self):
        papers, count, _, invalid = parse_atom(feed(["2501.12345v2", "cs/9901001v1"]))
        self.assertEqual((count, invalid), (2, 0))
        self.assertEqual(papers[0]["paper_id"], "2501.12345")
        self.assertEqual(papers[0]["version"], 2)
        self.assertEqual(papers[0]["text"], "A paper title\n\nA real abstract.")
        self.assertEqual(papers[1]["paper_id"], "cs/9901001")

    def test_html_challenge_is_not_paper_data(self):
        with self.assertRaises(FetchError):
            parse_atom(b"<html><title>Verify you are human</title></html>")

    def test_api_error_feed_is_not_paper_data(self):
        body = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors</id><title>Error</title><summary>bad query</summary></entry></feed>'
        with self.assertRaisesRegex(FetchError, "bad query"):
            parse_atom(body)

    def test_missing_abstract_is_counted_and_excluded(self):
        body = feed(["2501.12345v1"]).replace(b" A real\n abstract. ", b"")
        papers, count, _, invalid = parse_atom(body)
        self.assertEqual((papers, count, invalid), ([], 1, 1))

    @patch("arxiv_paper_recommend.corpus.crawling.arxiv.time.sleep")
    def test_rate_limit_respects_retry_after(self, sleep):
        client = HttpClient(retries=1)
        with patch.object(client, "_request", side_effect=[(429, {"retry-after": "12"}, b""), (200, {}, b"ok")]):
            self.assertEqual(client.get("https://example.org"), b"ok")
        sleep.assert_any_call(12.0)
        self.assertEqual([event["status"] for event in client.events], [429, 200])

    def test_forbidden_stops_without_retry(self):
        client = HttpClient()
        with patch.object(client, "_request", return_value=(403, {}, b"Forbidden")) as request:
            with self.assertRaisesRegex(FetchError, "403"):
                client.get("https://example.org")
        self.assertEqual(request.call_count, 1)

    @patch("arxiv_paper_recommend.corpus.crawling.arxiv.time.sleep")
    def test_long_cooldown_stops_instead_of_retrying_early(self, sleep):
        client = HttpClient()
        with patch.object(client, "_request", return_value=(503, {"retry-after": "120"}, b"")) as request:
            with self.assertRaisesRegex(FetchError, "120s cooldown"):
                client.get("https://example.org")
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

    @patch("arxiv_paper_recommend.corpus.crawling.arxiv.time.sleep")
    def test_successive_requests_are_paced(self, sleep):
        client = HttpClient(delay=3.2)
        with patch.object(client, "_request", return_value=(200, {}, b"ok")):
            client.get("https://example.org/1")
            client.get("https://example.org/2")
        self.assertGreater(sleep.call_args.args[0], 3.0)

    def test_http_date_retry_after(self):
        value = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30))
        self.assertTrue(28 <= retry_after_seconds(value) <= 31)
        self.assertIsNone(retry_after_seconds("not a date"))

    def test_interrupted_run_reuses_completed_pages(self):
        with tempfile.TemporaryDirectory() as temp:
            cache, output = Path(temp) / "cache", Path(temp) / "papers.jsonl"
            first = HttpClient()
            with patch.object(first, "get", side_effect=[feed(["2501.10001v1"]), FetchError("offline")]):
                with self.assertRaises(FetchError):
                    harvest("cat:cs.CL", 2, 1, output, cache, first)
            second = HttpClient()
            with patch.object(second, "get", return_value=feed(["2501.10002v1"])) as request:
                result = harvest("cat:cs.CL", 2, 1, output, cache, second)
            self.assertEqual(request.call_count, 1)
            self.assertTrue(result["pages"][0]["cache_hit"])
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)

    def test_deduplicates_versions_and_fetches_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            output, client = Path(temp) / "papers.jsonl", HttpClient()
            with patch.object(client, "get", side_effect=[feed(["2501.10001v1", "2501.10001v2"]), feed(["2501.10002v1"])]):
                report = harvest("cat:cs.CL", 2, 2, output, Path(temp) / "cache", client)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(report["duplicates_removed"], 1)
            self.assertEqual(rows[0]["version"], 2)
            self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
