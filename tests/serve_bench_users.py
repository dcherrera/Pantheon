#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""serve_bench_users.py — end-to-end multi-user bench against `pantheon serve`
(HTTP + SSE), stdlib only.  Drives K concurrent streaming chat completions and
reports per-user TTFT, per-user tok/s, windowed aggregate tok/s, and max
inter-token gaps — the client-side truth for the multi-tenant-speed-opt plan's
M5 (join stall) / M6 (slow-client isolation) / end-to-end serve re-bench.

Modes:
  steady  K clients, identical short prompt, max_tokens N.  Reports TTFT +
          per-user tok/s + windowed aggregate (window = all K streams live).
  join    K-1 clients decode long completions; after --join-delay s the K-th
          joins with a ~--join-words words prompt.  Reports the decoders' max
          inter-token gap inside the join window (prompt send → joiner TTFT)
          vs their baseline gap, + joiner TTFT.
  slow    K clients; client 0 sets a tiny SO_RCVBUF and stops reading for
          --stall s mid-stream (a stalled SSE consumer).  Reports the other
          users' max gap during the stall (isolation check).

Examples (CUDA host):
  python3 tests/serve_bench_users.py --port 8105 steady -K 8 -n 200
  python3 tests/serve_bench_users.py --port 8105 join -K 8 --join-words 1400
  python3 tests/serve_bench_users.py --port 8105 slow -K 4 --stall 20
"""
import argparse
import json
import socket
import sys
import threading
import time

HOST = "127.0.0.1"


def sse_request(port, prompt, max_tokens, events, tag, rcvbuf=0,
                stall_at=None, stall_s=0.0, model="default"):
    """POST a streaming chat completion; append (t, tag) to events per content
    delta.  Returns dict with ttft, times list, err."""
    out = {"tag": tag, "t_send": None, "ttft": None, "times": [], "err": None,
           "n": 0, "usage": None}
    body = json.dumps({
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # Phase 6: the usage chunk carries prompt_tokens (the context depth
        # this run actually started at) and kv_cap — the per-session context
        # cap the model was LOADED with (pn_session_ctx_cap, i.e. the
        # configured window pre-ladder).  It confirms per request that a sweep
        # point's ctx_cap really landed.  It is NOT the post-ladder effective
        # window: that is only visible in the serve log's [window] line.
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = (b"POST /v1/chat/completions HTTP/1.1\r\n"
           b"Host: bench\r\nContent-Type: application/json\r\n"
           b"Content-Length: " + str(len(body)).encode() +
           b"\r\nConnection: close\r\n\r\n" + body)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(900.0)              # hang guard (per recv)
    if rcvbuf:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    try:
        s.connect((HOST, port))
        out["t_send"] = time.monotonic()
        s.sendall(req)
        buf = b""
        stalled = False
        while True:
            if (stall_at is not None and not stalled and
                    out["n"] >= stall_at):
                stalled = True
                time.sleep(stall_s)          # stop reading: kernel window fills
            chunk = s.recv(65536)
            if not chunk:
                break
            t = time.monotonic()
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    return out
                try:
                    j = json.loads(payload)
                except ValueError:
                    continue
                if j.get("usage"):
                    out["usage"] = j["usage"]
                if not j.get("choices"):
                    continue
                delta = j.get("choices", [{}])[0].get("delta", {})
                if "content" in delta or "reasoning_content" in delta:
                    if out["ttft"] is None:
                        out["ttft"] = t - out["t_send"]
                    out["times"].append(t)
                    out["n"] += 1
                    if events is not None:
                        events.append((t, tag))
    except OSError as e:
        out["err"] = str(e)
    finally:
        s.close()
    return out


def gaps(times):
    return [b - a for a, b in zip(times, times[1:])]


def report_user(u):
    ts = u["times"]
    if len(ts) < 2:
        print(f"  {u['tag']}: err={u['err']} n={u['n']}")
        return
    dur = ts[-1] - ts[0]
    g = gaps(ts)
    # STEADY-STATE decode rate: first token -> last token, so session setup and
    # prefill are excluded.  This is the number phases are compared against.
    # END-TO-END wall clock: request sent -> last token, as a client sees it
    # (session create + prefill + first-token latency + decode).  The two are
    # NOT interchangeable — see the plan's measurement protocol.
    e2e = ts[-1] - u["t_send"]
    us = u.get("usage") or {}
    # ctx depth + the effective window read back off the stream (kv_cap).
    extra = ""
    if us:
        extra = (f" | prompt_tok={us.get('prompt_tokens')} "
                 f"kv_used={us.get('kv_used')} kv_cap={us.get('kv_cap')}")
    print(f"  {u['tag']}: n={u['n']} ttft={u['ttft']:.2f}s "
          f"tok/s={(len(ts) - 1) / dur:.2f} max_gap={max(g) * 1e3:.0f}ms "
          f"| e2e={e2e:.2f}s e2e_tok/s={u['n'] / e2e:.2f}{extra}")


def run_threads(specs):
    res = [None] * len(specs)

    def go(i, kw):
        res[i] = sse_request(**kw)

    th = [threading.Thread(target=go, args=(i, kw))
          for i, kw in enumerate(specs)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return res


def pad_prompt(base, words):
    """Prefix `base` with `words` filler words so the request starts at a
    chosen CONTEXT DEPTH.  Phase 6 needs per-user tok/s as a function of
    context length, and decode depth is the one axis the campaign never swept:
    attention is the per-user term that grows with context, so the batching
    advantage is expected to shrink with depth.  The filler is a fixed
    lorem-style cycle (deterministic across runs and across K, so every point
    is comparable) and is framed as a document the model is told to ignore, so
    the generation stays a normal answer rather than a summarisation."""
    if words <= 0:
        return base
    cycle = ("lighthouse keeper logbook entry brass lamp storm wave gull "
             "horizon rope timber tide lantern fog rock beam salt cliff "
             "harbour rain wind night watch signal").split()
    filler = " ".join(cycle[i % len(cycle)] for i in range(words))
    return ("Here is a reference document; do not summarise it.\n"
            + filler + "\n\nNow, ignoring the document above: " + base)


def mode_steady(a):
    prompt = pad_prompt(
        ("You are a helpful assistant. Please explain, step by step, "
         "why the sky appears blue during the day and red at sunset, "
         "mentioning Rayleigh scattering."), a.prompt_words)
    specs = [dict(port=a.port, model=a.model, prompt=prompt,
                  max_tokens=a.n, events=None,
                  tag=f"u{i}") for i in range(a.K)]
    res = run_threads(specs)
    print(f"steady K={a.K} max_tokens={a.n} prompt_words={a.prompt_words} "
          f"prompt_chars={len(prompt)}")
    for u in res:
        report_user(u)
    live = [u for u in res if len(u["times"]) >= 2]
    if len(live) == len(res) and live:
        w0 = max(u["times"][0] for u in live)
        w1 = min(u["times"][-1] for u in live)
        if w1 > w0:
            n = sum(sum(1 for t in u["times"] if w0 < t <= w1) for u in live)
            print(f"  AGGREGATE (windowed, all-{a.K}-live "
                  f"{w1 - w0:.1f}s): {n / (w1 - w0):.2f} tok/s")


def mode_join(a):
    K = a.K
    prompt = "Please write a long, detailed story about a lighthouse keeper."
    filler = ("lighthouse keeper story detail wave storm lamp brass stairs "
              "logbook gull horizon ").split()
    join_prompt = " ".join(filler[i % len(filler)]
                           for i in range(a.join_words)) + \
        " Summarize the above in one sentence."
    events = []
    res = [None] * K
    t_join = [None]

    def decoder(i):
        res[i] = sse_request(a.port, prompt, a.n, events, f"u{i}",
                             model=a.model)

    def joiner():
        time.sleep(a.join_delay)
        t_join[0] = time.monotonic()
        res[K - 1] = sse_request(a.port, join_prompt, 32, events,
                                 "joiner")

    th = [threading.Thread(target=decoder, args=(i,)) for i in range(K - 1)]
    th.append(threading.Thread(target=joiner))
    for t in th:
        t.start()
    for t in th:
        t.join()
    print(f"join K={K} join_words={a.join_words} join_delay={a.join_delay}s")
    for u in res:
        report_user(u)
    jo = res[K - 1]
    if t_join[0] and jo["ttft"] is not None:
        w0, w1 = t_join[0], t_join[0] + jo["ttft"]
        print(f"  joiner prompt_ttft={jo['ttft']:.2f}s")
        worst = base = 0.0
        for u in res[:K - 1]:
            ts = u["times"]
            g = gaps(ts)
            # any gap OVERLAPPING the join window counts (a freeze gap ends
            # only after the joiner's prefill completes)
            worst = max([worst] + [b - x for x, b in zip(ts, ts[1:])
                                   if x < w1 and b > w0])
            pre = [d for x, d in zip(ts[1:], g) if x < w0]
            if pre:
                base = max(base, sorted(pre)[len(pre) // 2])
        print(f"  M5: decoders' max token-gap DURING JOIN = "
              f"{worst * 1e3:.0f} ms (baseline median gap {base * 1e3:.0f} ms)")


def mode_slow(a):
    prompt = "Count slowly from one to fifty, with a short comment each."
    specs = [dict(port=a.port, model=a.model, prompt=prompt,
                  max_tokens=a.n, events=None,
                  tag=f"u{i}") for i in range(a.K)]
    specs[0].update(tag="SLOW", rcvbuf=1024, stall_at=5, stall_s=a.stall)
    res = run_threads(specs)
    print(f"slow K={a.K} stall={a.stall}s (client 0 stops reading, "
          f"rcvbuf=1KB)")
    for u in res:
        report_user(u)
    worst = 0.0
    for u in res[1:]:
        g = gaps(u["times"])
        if g:
            worst = max(worst, max(g))
    print(f"  M6: fast users' max token-gap while client 0 stalled = "
          f"{worst * 1e3:.0f} ms (stall was {a.stall:.0f}s — isolation "
          f"{'OK' if worst < a.stall / 2 else 'FAILED'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8105)
    ap.add_argument("--model", default="default",
                    help="model name for the request body (fleet models are "
                         "named, so \"default\" is ambiguous once >1 is loaded)")
    ap.add_argument("mode", choices=["steady", "join", "slow"])
    ap.add_argument("-K", type=int, default=8)
    ap.add_argument("-n", type=int, default=200, help="max_tokens")
    ap.add_argument("--join-delay", type=float, default=20.0)
    ap.add_argument("--join-words", type=int, default=1400)
    ap.add_argument("--stall", type=float, default=20.0)
    ap.add_argument("--prompt-words", type=int, default=0,
                    help="steady mode: pad the prompt with N filler words so "
                         "decode starts at a chosen context depth (0 = the "
                         "short ~40-token prompt every earlier phase used)")
    a = ap.parse_args()
    {"steady": mode_steady, "join": mode_join, "slow": mode_slow}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
