#!/usr/bin/env python3
"""
test_gpqa_answer_extraction.py — unit tests for answer extraction, shuffling and scoring stats.

  python3 evals/gpqa/test_gpqa_answer_extraction.py

Status: works.
Issues: None
Todo: None
"""
import http.server
import json
import threading
import unittest

from gpqa_answer_extraction import extract_answer_letter
from gpqa_dataset import build_shuffled_item, choice_permutation
from gpqa_openai_stream_client import ServeRequestError, stream_chat_completion
from gpqa_score import mean_with_clustered_ci, proportion_with_ci


class AnswerExtractionTests(unittest.TestCase):
    def assert_letter(self, content, letter, method=None, finish_reason="stop"):
        got, how = extract_answer_letter(content, finish_reason)
        self.assertEqual(got, letter, f"{content!r} -> {how}")
        if method:
            self.assertEqual(how, method)

    def test_simple_evals_answer_line(self):
        self.assert_letter("Reasoning...\nAnswer: C", "C", "answer_line")
        self.assert_letter("Answer:B", "B", "answer_line")
        self.assert_letter("Answer: $D$", "D", "answer_line")

    def test_markdown_and_brackets(self):
        self.assert_letter("**Answer:** (B)", "B", "answer_line")
        self.assert_letter("**Final Answer: A**", "A", "answer_line")
        self.assert_letter("Answer: **[C]**", "C", "answer_line")

    def test_last_answer_line_wins(self):
        self.assert_letter("Answer: A\nWait, recheck.\nAnswer: D", "D", "answer_line")

    def test_lowercase_word_is_not_a_letter(self):
        self.assert_letter("Answer: a mixture of both.\n\nAnswer: B", "B")
        self.assert_letter("Answer: a mixture of both.", None)

    def test_letter_followed_by_word_is_not_a_letter(self):
        self.assert_letter("Answer: Based on the above, B", None)

    def test_boxed(self):
        self.assert_letter("so \\boxed{C}", "C", "boxed")
        self.assert_letter("\\boxed{\\text{A}}", "A", "boxed")

    def test_answer_is(self):
        self.assert_letter("Therefore the correct option is (D).", "D", "answer_is")

    def test_lone_letter_last_line(self):
        self.assert_letter("Long text\n**B**", "B", "lone_letter")

    def test_leaked_think_is_stripped(self):
        self.assert_letter("<think>Answer: A maybe</think>\nAnswer: C", "C")
        self.assert_letter("Answer: A maybe\n</think>\nAnswer: B", "B")

    def test_failures_are_named(self):
        self.assertEqual(extract_answer_letter("", "length"), (None, "truncated_no_answer"))
        self.assertEqual(extract_answer_letter(None, "stop"), (None, "empty_content"))
        self.assertEqual(extract_answer_letter("I am not sure.", "stop"), (None, "no_match"))
        self.assertEqual(extract_answer_letter("I am not sure.", "length"), (None, "truncated_no_match"))


class ShuffleTests(unittest.TestCase):
    ROW = {"record_id": "recX", "question": "q", "correct": "right", "incorrect": ["w1", "w2", "w3"]}

    def test_deterministic_per_seed(self):
        self.assertEqual(choice_permutation(42, "recX"), choice_permutation(42, "recX"))

    def test_seeds_differ_somewhere(self):
        ids = [f"rec{i}" for i in range(20)]
        self.assertNotEqual([choice_permutation(42, i) for i in ids], [choice_permutation(43, i) for i in ids])

    def test_correct_letter_points_at_correct_text(self):
        for seed in range(30):
            item = build_shuffled_item(self.ROW, seed)
            line = [l for l in item["prompt"].splitlines() if l.startswith(item["correct_letter"] + ") ")][0]
            self.assertEqual(line, f"{item['correct_letter']}) right")

    def test_correct_position_is_roughly_uniform(self):
        counts = {}
        for i in range(400):
            letter = build_shuffled_item(dict(self.ROW, record_id=f"r{i}"), 42)["correct_letter"]
            counts[letter] = counts.get(letter, 0) + 1
        self.assertEqual(sorted(counts), list("ABCD"))
        self.assertTrue(all(60 < c < 140 for c in counts.values()), counts)


class StatsTests(unittest.TestCase):
    def test_proportion_ci(self):
        p, half = proportion_with_ci(150, 198)
        self.assertAlmostEqual(p, 150 / 198)
        self.assertAlmostEqual(half, 1.96 * (p * (1 - p) / 198) ** 0.5, places=6)

    def test_clustered_ci_identical_seeds_matches_proportion(self):
        scores = {f"r{i}": [1.0 if i < 150 else 0.0] * 2 for i in range(198)}
        mean, half = mean_with_clustered_ci(scores)
        p, phalf = proportion_with_ci(150, 198)
        self.assertAlmostEqual(mean, p)
        self.assertAlmostEqual(half, phalf, places=2)


class FakeStreamingServeHandler(http.server.BaseHTTPRequestHandler):
    """Streams 2 reasoning chunks, 2 content chunks, a finish chunk and a usage chunk."""
    status = 200

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["stream"] and body["stream_options"]["include_usage"]
        self.send_response(self.status)
        self.end_headers()
        if self.status != 200:
            return
        deltas = [{"reasoning_content": "think "}, {"reasoning_content": "more"},
                  {"content": "So.\n"}, {"content": "Answer: B"}]
        for delta in deltas:
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode())
        self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n')
        self.wfile.write(b'data: {"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":40}}\n\n')
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


class StreamClientTests(unittest.TestCase):
    def serve(self, status):
        handler = type("Handler", (FakeStreamingServeHandler,), {"status": status})
        server = http.server.HTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def test_stream_is_assembled_with_usage_and_timing(self):
        reply = stream_chat_completion(self.serve(200), {"model": "m", "messages": []})
        self.assertEqual(reply["content"], "So.\nAnswer: B")
        self.assertEqual(reply["reasoning"], "think more")
        self.assertEqual(reply["finish_reason"], "stop")
        self.assertEqual(reply["completion_tokens"], 40)
        self.assertTrue(reply["completion_tokens_from_usage"])
        self.assertEqual(reply["stream_token_chunks"], 4)
        self.assertIsNotNone(reply["ttft_s"])

    def test_http_error_raises(self):
        with self.assertRaises(ServeRequestError):
            stream_chat_completion(self.serve(500), {"model": "m", "messages": []})


if __name__ == "__main__":
    unittest.main()
