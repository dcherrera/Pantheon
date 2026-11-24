#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
mtp_serve_client.py — a real OpenAI-compatible client for MTP serve
measurement (tests/local/speed_bench_k80.sh): streams chat completions from a running
`pantheon serve`, saves every reply's text for an exact comparison, and records
TTFT and decode tok/s per request.

Per (depth, temperature) it sends ONE request: the user message is a fixed
filler cut to about `depth` tokens (calibrated once against the serve's own
prompt_tokens) followed by a fixed prompt; temperature 0 is greedy, > 0 samples
with top_p 0.95, top_k 20 and a fixed `seed`, so the reply is reproducible.

--tools offers a read_file tool (tool_choice "auto"); streamed tool-call deltas are
saved with the text, so a call is compared like any other output.
--sessions a,b sends each (depth, temperature) once per session id, in turn
(prefix-cache sessions; back-to-back session switches); none = no session.

Outputs in --out: reply_<depth>_<temp>[_<session>].txt (reasoning + content, exactly as
streamed) and results.jsonl (one line per request: prompt/completion tokens,
ttft_ms, decode_tok_s, sha1 of the reply). Exit 0 when every request completed.

  mtp_serve_client.py --port 8207 --model qwen36-mtp --filler F --prompt P \
      --depths 1024,8192 --temps 0,0.7 --out DIR [--sessions a,b] [--max-tokens 256]

@status complete  @issues none  @todo none
"""
import argparse
import hashlib
import http.client
import json
import os
import sys
import time


def stream_chat(host, port, body, timeout):
    """POST a streaming chat completion; return (text, usage, t_first, t_last, t_sent)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    payload = json.dumps(body).encode()
    t_sent = time.monotonic()
    conn.request("POST", "/v1/chat/completions", payload,
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError("HTTP %d: %s" % (resp.status, resp.read()[:300]))
    parts, usage, t_first, t_last = [], None, None, None
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            for call in delta.get("tool_calls") or []:
                fn = call.get("function") or {}
                piece += "<call %s>%s" % (fn.get("name") or "", fn.get("arguments") or "")
            if piece:
                now = time.monotonic()
                t_first = t_first or now
                t_last = now
                parts.append(piece)
    conn.close()
    return "".join(parts), usage, t_first, t_last, t_sent


def build_user(filler, prompt, n_chars):
    if n_chars <= 0:
        return prompt
    return "Reference material:\n\n" + filler[:n_chars] + "\n\n---\n\n" + prompt


TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file from the workspace and return its text.",
    "parameters": {"type": "object", "required": ["path"],
                   "properties": {"path": {"type": "string", "description": "file path"}}}}}]


def request_body(args, user, temp, max_tokens, session=""):
    body = {"model": args.model, "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": temp}
    if args.tools:
        body.update({"tools": TOOLS, "tool_choice": args.tool_choice})
    if args.no_think:
        body["enable_thinking"] = False
    if temp > 0:
        body.update({"top_p": 0.95, "top_k": 20, "seed": args.seed})
    if session:
        body["session_id"] = session
    return body


def calibrate(args, filler, prompt):
    """Chars per prompt token for this filler, from one short request."""
    probe_chars = 4000
    body = request_body(args, build_user(filler, prompt, probe_chars), 0.0, 1)
    _, usage, _, _, _ = stream_chat(args.host, args.port, body, args.timeout)
    base = request_body(args, build_user(filler, prompt, 0), 0.0, 1)
    _, usage0, _, _, _ = stream_chat(args.host, args.port, base, args.timeout)
    extra = usage["prompt_tokens"] - usage0["prompt_tokens"]
    return probe_chars / max(extra, 1), usage0["prompt_tokens"]


def run_request(args, user, depth, temp, session, res):
    """One streamed request: save its text, log its numbers. Returns success."""
    body = request_body(args, user, temp, args.max_tokens, session)
    try:
        text, usage, t_first, t_last, t_sent = stream_chat(args.host, args.port, body,
                                                           args.timeout)
    except Exception as exc:  # report and keep going: the gate counts it
        print("REQUEST depth=%d temp=%.1f session=%s FAILED: %s" % (depth, temp, session, exc),
              flush=True)
        return False
    name = "reply_%d_%s%s.txt" % (depth, ("%.1f" % temp).replace(".", "p"),
                                  "_" + session if session else "")
    with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
        f.write(text)
    n_out = usage["completion_tokens"] if usage else 0
    decode_s = (t_last - t_first) if t_first and t_last else 0.0
    row = {"depth": depth, "temp": temp, "session": session,
           "prompt_tokens": usage["prompt_tokens"] if usage else None,
           "completion_tokens": n_out,
           "ttft_ms": round((t_first - t_sent) * 1e3, 1) if t_first else None,
           "decode_tok_s": round((n_out - 1) / decode_s, 2) if decode_s > 0 and n_out > 1
           else None,
           "sha1": hashlib.sha1(text.encode()).hexdigest(), "file": name}
    res.write(json.dumps(row) + "\n")
    res.flush()
    print("REQUEST %s" % json.dumps(row), flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8207)
    ap.add_argument("--model", required=True)
    ap.add_argument("--filler", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--depths", required=True)
    ap.add_argument("--temps", default="0,0.7")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--sessions", default="")
    ap.add_argument("--no-think", action="store_true", help="enable_thinking false")
    ap.add_argument("--tool-choice", default="auto", help="with --tools: auto | required")
    ap.add_argument("--tools", action="store_true",
                    help="offer a read_file tool (tool_choice auto) on every request")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    filler = open(args.filler, encoding="utf-8", errors="replace").read()
    prompt = open(args.prompt, encoding="utf-8").read().strip()
    cpt, base_tokens = calibrate(args, filler, prompt)
    print("calibrated: %.3f chars/token, bare prompt %d tokens" % (cpt, base_tokens), flush=True)
    sessions = [x for x in args.sessions.split(",") if x] or [""]
    ok = True
    with open(os.path.join(args.out, "results.jsonl"), "w") as res:
        for depth in [int(d) for d in args.depths.split(",")]:
            user = build_user(filler, prompt, int(max(depth - base_tokens - 16, 0) * cpt))
            for temp in [float(t) for t in args.temps.split(",")]:
                for session in sessions:
                    ok &= run_request(args, user, depth, temp, session, res)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
