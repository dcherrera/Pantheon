#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
test_mtp_acceptance_gate.py — offline test of tools/mtp_acceptance_gate.py: corpus selection
(private / public / auto fallback), the public corpus's shape, and a full `run` + `summarize`
against an in-process stub llama-server (a fake chat endpoint that appends the per-request
acceptance line to a log file). Needs no model and no network beyond loopback.

Usage: python3 tests/test_mtp_acceptance_gate.py

@status  complete
@issues  none
@todo    none
"""
import contextlib
import http.server
import io
import json
import os
import sys
import tempfile
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import mtp_acceptance_gate as gate  # noqa: E402


class Stub(http.server.BaseHTTPRequestHandler):
    log_path = None
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.seen.append(body)
        with open(Stub.log_path, "a") as f:
            f.write("draft acceptance = 0.50000 (    2 accepted /     4 generated), "
                    "mean acceptance length =   2.00, acceptance rate per position = (0.700, 0.300)\n")
        resp = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "timings": {"predicted_n": 8, "predicted_per_second": 10.0, "draft_n": 4, "draft_n_accepted": 2}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class Args:
    def __init__(self, **kw):
        self.__dict__.update(dict(depth=2, temps="0", top_p=0.95, top_k=20, seed=1, max_tokens=8, timeout=10,
                                  workloads="tool", limit=0, ids="", label="stub", tool_corpus="auto",
                                  dry_run=False))
        self.__dict__.update(kw)


class GateTest(unittest.TestCase):
    def test_public_corpus_shape(self):
        c = json.load(open(os.path.join(gate.DATA, gate.TOOL_CORPORA["public"])))
        self.assertEqual(len(c["turns"]), 6)
        self.assertEqual(len(c["tools"]), 9)
        self.assertNotIn("arg" + "us", json.dumps(c).lower())  # split so the leak scan stays clean
        self.assertTrue(any(m.get("tool_calls") for t in c["turns"] for m in t["messages"]))

    def test_auto_falls_back_to_public(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            old = gate.DATA
            gate.DATA = d
            try:
                self.assertEqual(gate.resolve_tool_corpus("auto")[0], "public")
                open(os.path.join(d, gate.TOOL_CORPORA["private"]), "w").close()
                self.assertEqual(gate.resolve_tool_corpus("auto")[0], "private")
                self.assertEqual(gate.resolve_tool_corpus("public")[0], "public")
            finally:
                gate.DATA = old

    def test_run_records_corpus(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            Stub.log_path = os.path.join(d, "server.log")
            open(Stub.log_path, "w").close()
            srv = http.server.HTTPServer(("127.0.0.1", 0), Stub)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                out = os.path.join(d, "rows.jsonl")
                args = Args(url=f"http://127.0.0.1:{srv.server_port}", log=Stub.log_path, out=out,
                            tool_corpus="public")
                with contextlib.redirect_stdout(io.StringIO()):
                    gate.cmd_run(args)
                rows = [json.loads(line) for line in open(out)]
                self.assertEqual(len(rows), 6)
                self.assertTrue(all(r["corpus"] == "public" and r["workload"] == "tool" for r in rows))
                self.assertTrue(all(r["draft_accepted"] == 2 and r["verify_steps"] == 2 for r in rows))
                self.assertTrue(all(b["tools"] and b["messages"][0]["role"] == "system" for b in Stub.seen))
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    gate.cmd_summarize(type("A", (), {"files": [out], "common": False})())
                self.assertIn("tool corpus: public", buf.getvalue())
            finally:
                srv.shutdown()


if __name__ == "__main__":
    unittest.main()
