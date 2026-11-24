#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""serve_prefill_fairness_bench.py — neighbours' decode speed while another
session on the same model prefills a long prompt (the batched scheduler's
prefill/decode interleave, QWEN35_BS_PREFILL_DECODE_SHARE). Stdlib only.

N-1 "neighbour" sessions with --ctx-words-word prompts start decoding long
greedy completions; once all are streaming (+ --settle s) one more session
sends a --prefill-words prompt (the re-prefill a compacted agent causes).
Reports, per neighbour, tok/s before the prefill and inside the prefill window
(prompt sent -> prefiller's first token), the prefiller's TTFT, and writes every
session's full streamed text to --out (JSON) so two runs can be compared for
bit-exactness (`--compare a.json b.json`).

No system message and fresh session ids per run, so no prefix cache applies and
every run does the same full prefills.

  python3 tests/serve_prefill_fairness_bench.py --port 8611 --tag s0 --out s0.json
  python3 tests/serve_prefill_fairness_bench.py --compare s0.json s50.json
"""
import argparse
import http.client
import json
import random
import sys
import threading
import time


def build_words(n_words, seed):
    """Deterministic filler text: numbered lines of pseudo-random words."""
    rng = random.Random(seed)
    vocab = ("alpha beta gamma delta river stone engine copper lantern orbit "
             "harbor signal meadow vector cipher garden falcon quartz ember "
             "tunnel summit canyon ledger beacon").split()
    lines, line = [], []
    for i in range(n_words):
        line.append(rng.choice(vocab))
        if len(line) == 12:
            lines.append(f"{len(lines) + 1}. " + " ".join(line) + ".")
            line = []
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def stream_chat(port, model, sid, prompt, max_tokens, rec):
    """Stream one chat completion; rec gets t_send, per-delta times and text.
    `prompt` is the user text, or a full messages list."""
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    body = json.dumps({
        "model": model, "stream": True, "max_tokens": max_tokens,
        "temperature": 0.0, "session_id": sid,
        "stream_options": {"include_usage": True},
        "messages": messages,
    })
    rec.update(t_send=time.time(), times=[], text=[], usage=None, err=None,
               stop=False)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
    try:
        conn.request("POST", "/v1/chat/completions", body,
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            rec["err"] = f"HTTP {resp.status}: {resp.read()[:300]!r}"
            return
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            if rec.get("stop"):
                break
            ev = json.loads(data)
            if ev.get("usage"):
                rec["usage"] = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if piece:
                    rec["times"].append(time.time())
                    rec["text"].append(piece)
    except Exception as e:  # report, never hide
        rec["err"] = repr(e)
    finally:
        conn.close()   # a hang-up: the serve stops this session's decode


def release_session(port, model, sid):
    """POST /session/release so the next run finds every seat free."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        body = {"session_id": sid}
        if model != "default":
            body["name"] = model
        conn.request("POST", "/session/release", json.dumps(body),
                     {"Content-Type": "application/json"})
        conn.getresponse().read()
    except Exception as e:  # report, never hide
        print(f"release {sid} failed: {e!r}")


def release_after_hangup(args, sids):
    """Release sessions whose streams were just hung up. The serve notices a
    hang-up within 8 tokens, and a release only frees a seat whose turn has
    ended, so release after a pause and once more after another."""
    for _ in range(2):
        time.sleep(3)
        for sid in sids:
            release_session(args.port, args.model, sid)


def rate(times, t0, t1):
    n = sum(1 for t in times if t0 <= t < t1)
    alive = [t for t in times if t0 <= t < t1]
    end = min(t1, alive[-1]) if alive else t1
    return n / (end - t0) if end > t0 else 0.0, n


def wait_seats(args):
    """A previous run's sessions may still be generating after their clients hung
    up: until the model takes --sessions new sessions at once (each probe holds its
    seat until released), release them and retry."""
    deadline = time.time() + args.setup_timeout
    sids = [f"fair-{args.tag}-probe{i}" for i in range(args.sessions)]
    while time.time() < deadline:
        ok = True
        for sid in sids:
            rec = {}
            stream_chat(args.port, args.model, sid, "ok", 1, rec)
            if (rec.get("err") or "").startswith("HTTP 429"):
                ok = False
                break
        for sid in sids:
            release_session(args.port, args.model, sid)
        if ok:
            return True
        time.sleep(5)
    return False


def run(args):
    if not wait_seats(args):
        print("seats never freed")
        return 2
    neighbours = args.sessions - 1
    recs = [dict() for _ in range(args.sessions)]
    threads = []
    ctx_prompt = ("Write a very long, detailed story about the numbered list "
                  "below, one paragraph per line, and do not stop early.\n\n")
    for i in range(neighbours):
        p = ctx_prompt + build_words(args.ctx_words, seed=100 + i)
        th = threading.Thread(target=stream_chat, args=(
            args.port, args.model, f"fair-{args.tag}-n{i}", p, args.max_tokens, recs[i]))
        th.start()
        threads.append(th)
    deadline = time.time() + args.setup_timeout
    while time.time() < deadline:
        if all(r.get("times") for r in recs[:neighbours]) or any(r.get("err") for r in recs):
            break
        time.sleep(0.5)
    if not all(r.get("times") for r in recs[:neighbours]):
        print("neighbours never all streamed:", [r.get("err") for r in recs])
        for r in recs:
            r["stop"] = True
        for t in threads:
            t.join()
        release_after_hangup(args, [f"fair-{args.tag}-n{i}" for i in range(neighbours)])
        return 2
    t_all = max(r["times"][0] for r in recs[:neighbours])
    time.sleep(args.settle)
    pf = recs[-1]
    pf_prompt = "Summarise the list below in one sentence.\n\n" + build_words(args.prefill_words, seed=7)
    th = threading.Thread(target=stream_chat, args=(
        args.port, args.model, f"fair-{args.tag}-pf", pf_prompt, args.pf_max_tokens, pf))
    th.start()
    th.join()
    time.sleep(args.tail)          # neighbours keep decoding, then hang up
    for r in recs[:neighbours]:
        r["stop"] = True
    for t in threads:
        t.join()
    release_after_hangup(args, [f"fair-{args.tag}-n{i}" for i in range(neighbours)] +
                         [f"fair-{args.tag}-pf"])
    if pf.get("err") or not pf.get("times"):
        print("prefiller failed:", pf.get("err"))
        return 3
    t_send, t_first = pf["t_send"], pf["times"][0]
    print(f"prefiller: prompt_tokens={pf['usage'] and pf['usage'].get('prompt_tokens')} "
          f"TTFT={t_first - t_send:.1f}s")
    report = {"tag": args.tag, "ttft_s": t_first - t_send, "neighbours": [],
              "prompt_tokens": pf["usage"] and pf["usage"].get("prompt_tokens")}
    for i in range(neighbours):
        r = recs[i]
        base, nb = rate(r["times"], t_all, t_send)
        during, nd = rate(r["times"], t_send, t_first)
        gaps = [b - a for a, b in zip(r["times"], r["times"][1:]) if t_send <= a < t_first]
        mg = max(gaps) if gaps else 0.0
        print(f"neighbour {i}: before {base:5.2f} tok/s ({nb})  during prefill "
              f"{during:5.2f} tok/s ({nd})  max gap {mg:.2f}s  err={r.get('err')}")
        report["neighbours"].append({"before_tok_s": base, "during_tok_s": during,
                                     "max_gap_s": mg, "deltas": len(r["times"])})
    report["texts"] = ["".join(r["text"]) for r in recs]
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
    return 0


def div_gate(args):
    """Rung 1d exactness: A, then B rewriting history after a kept head (saves
    the divergence checkpoint), then C rewriting at the same point (resumes from
    it). C's greedy output must equal a cold C on a fresh session. Check the
    serve log for `divergence-checkpoint saved` / `... resume`."""
    head = [{"role": "user", "content": "Remember this list.\n\n" + build_words(args.ctx_words, 11)},
            {"role": "assistant", "content": "I have read the list."}]
    def turn(text):
        return head + [{"role": "user", "content": text + "\n\n" + build_words(1500, len(text))}]
    a = turn("Alpha request: count the words that start with a.")
    b = turn("Beta request: which word appears most often?")
    c = turn("Gamma request: name three words from the list.")
    sid = f"divgate-{args.tag}"
    out = {}
    for name, msgs, s_id in (("A", a, sid), ("B", b, sid), ("C", c, sid),
                             ("C_cold", c, sid + "-cold")):
        rec = {}
        stream_chat(args.port, args.model, s_id, msgs, args.pf_max_tokens, rec)
        if rec.get("err"):
            print(name, "failed:", rec["err"])
            return 3
        out[name] = "".join(rec["text"])
        print(f"{name}: prompt_tokens={rec['usage'] and rec['usage'].get('prompt_tokens')} "
              f"TTFT={rec['times'][0] - rec['t_send']:.1f}s")
    release_session(args.port, args.model, sid)
    release_session(args.port, args.model, sid + "-cold")
    same = out["C"] == out["C_cold"]
    print("C vs cold C:", "IDENTICAL" if same else "DIFFERS")
    print(repr(out["C"][:160]))
    return 0 if same else 1


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    ok = True
    for i, (ta, tb) in enumerate(zip(a["texts"], b["texts"])):
        n = min(len(ta), len(tb))
        diff = next((k for k in range(n) if ta[k] != tb[k]), n)
        same = diff == n
        ok &= same
        print(f"session {i}: {'IDENTICAL' if same else 'DIFFERS at char ' + str(diff)} "
              f"over the common {n} chars (lengths {len(ta)} / {len(tb)})")
    print("BIT-EXACT" if ok else "MISMATCH")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8611)
    ap.add_argument("--model", default="default")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--ctx-words", type=int, default=4000)
    ap.add_argument("--prefill-words", type=int, default=15000)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--pf-max-tokens", type=int, default=32)
    ap.add_argument("--settle", type=float, default=10.0)
    ap.add_argument("--tail", type=float, default=10.0)
    ap.add_argument("--setup-timeout", type=float, default=1800.0)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--div-gate", action="store_true")
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if args.div_gate:
        return div_gate(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
