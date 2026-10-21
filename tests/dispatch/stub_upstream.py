#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
stub_upstream.py — a tiny OpenAI-compatible upstream for the dispatch-layer
end-to-end test. Speaks just enough of POST /v1/chat/completions to prove the
router proxies + streams correctly:

  - Non-streaming: returns a single chat.completion JSON whose content encodes
    which upstream answered (so the test can assert *which* instance was picked).
  - Streaming (stream:true): emits SSE `data:` chunks then `data: [DONE]`, with a
    short inter-chunk sleep so a BUFFERING proxy would be caught (the test reads
    chunks with timestamps).

Usage:
  stub_upstream.py --port N --tag NAME [--mode dead]

--mode dead: accept the TCP connection but immediately close without replying,
to exercise the router's failover path (upstream error before any body).
"""
import argparse
import json
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TAG = "stub"
MODE = "ok"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def do_POST(self):
        body = self._read_body()
        try:
            req = json.loads(body or b"{}")
        except Exception:
            req = {}
        streaming = bool(req.get("stream"))

        if streaming:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            # Emit a few chunks with a gap so buffering would be detectable.
            for i in range(3):
                chunk = {
                    "id": "chatcmpl-stub",
                    "object": "chat.completion.chunk",
                    "model": req.get("model", "?"),
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"{TAG}-{i} "},
                        "finish_reason": None,
                    }],
                }
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
                time.sleep(0.15)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            # Echo the forwarded X-Session-Id so the test can assert the router
            # threads it upstream (where the serve keys its prefix cache).
            seen_sid = self.headers.get("X-Session-Id", "")
            resp = {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "model": req.get("model", "?"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant",
                                "content": f"answered-by:{TAG} session:{seen_sid}"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            payload = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


def run_dead(port):
    """Accept connections and immediately close — no HTTP reply (failover test)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port))
    s.listen(16)
    while True:
        conn, _ = s.accept()
        try:
            conn.recv(4096)
        except Exception:
            pass
        conn.close()


def main():
    global TAG, MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tag", default="stub")
    ap.add_argument("--mode", default="ok", choices=["ok", "dead"])
    args = ap.parse_args()
    TAG = args.tag
    MODE = args.mode

    if MODE == "dead":
        run_dead(args.port)
        return

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.serve_forever()


if __name__ == "__main__":
    main()
