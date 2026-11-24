#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
@file serve_long_prompt_smoke.py
@brief Long-prompt gate: a serve must answer or cleanly refuse every prompt length
       and stay up.

@status Gate for the 1,891-token TT crash (tt-long-prompt-crash).
@issues Prompt lengths are approximate (filler words, ~1 token each); the serve's
        own [admit] n= log line is the exact count.
@todo none

Usage: serve_long_prompt_smoke.py <base_url> <model> <approx_tokens> [...]
  e.g. serve_long_prompt_smoke.py http://127.0.0.1:8411 q3 19 1900 4000

Each request carries a system message and a fresh "session_id", as Argus CHAT
does: that is the prefix cache's system-prompt path (a new per-session state),
the path the 1,891-token TT crash was on.
For each length: POST /v1/chat/completions (max_tokens 16). PASS when the reply is
a 200 with text, or a 4xx/5xx JSON error (a clean refusal). FAIL when the
connection drops (the serve died) or /v1/models stops answering afterwards.
A 401/403 is a FAIL, not a refusal: it never reached the model. For a key-gated
serve (Pantheon Local) set PANTHEON_API_KEY; it is sent as a Bearer token.
Exit 0 only when every length passed.
"""
import json
import os
import sys
import urllib.error
import urllib.request

FILLER = ("the quick brown fox jumps over the lazy dog while a small cat "
          "watches from the warm window ledge and counts every falling leaf ").split()


def build_prompt(approx_tokens):
    words = [FILLER[i % len(FILLER)] for i in range(max(1, approx_tokens - 20))]
    return " ".join(words) + "\n\nIn one short sentence, what animal jumps?"


def auth_headers():
    key = os.environ.get("PANTHEON_API_KEY")
    return {"Authorization": f"Bearer {key}"} if key else {}


def post_chat(base_url, model, prompt):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": "You are a concise assistant."},
                     {"role": "user", "content": prompt}],
        "max_tokens": 16,
        "temperature": 0,
        "session_id": f"long-prompt-smoke-{os.getpid()}-{len(prompt)}",
    }).encode()
    req = urllib.request.Request(base_url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          **auth_headers()})
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            return resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode(errors="replace")


def serve_alive(base_url):
    try:
        req = urllib.request.Request(base_url + "/v1/models", headers=auth_headers())
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ConnectionError, OSError):
        return False


def check_length(base_url, model, approx_tokens):
    try:
        status, text = post_chat(base_url, model, build_prompt(approx_tokens))
    except (urllib.error.URLError, ConnectionError, OSError) as err:
        print(f"FAIL n~{approx_tokens}: connection lost ({err})")
        return False
    alive = serve_alive(base_url)
    if status == 200:
        message = json.loads(text)["choices"][0]["message"]
        reply = (message.get("content") or "") + (message.get("reasoning_content") or "")
        verdict = "answered" if reply.strip() else "EMPTY"
    else:
        verdict = f"refused {status}"
    ok = alive and verdict != "EMPTY" and status not in (401, 403)
    print(f"{'PASS' if ok else 'FAIL'} n~{approx_tokens}: {verdict} "
          f"serve_alive={alive} body={text[:160]!r}")
    return ok


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    base_url, model = sys.argv[1].rstrip("/"), sys.argv[2]
    results = [check_length(base_url, model, int(n)) for n in sys.argv[3:]]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
