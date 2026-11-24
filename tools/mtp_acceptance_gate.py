#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
mtp_acceptance_gate.py — K80 MTP Gate 1: MTP draft acceptance
on Qwen3.6-35B-A3B, measured through llama.cpp's MTP speculative decode (the oracle engine;
Pantheon's CPU qwen35moe path has no MTP forward).

Drives a running llama-server (started with --spec-type draft-mtp
and a fixed --spec-draft-n-max) over three workloads, one request at a time:
  code  tests/data/code_edit_prompts.txt        (agent read_file -> re-emit, 8 prompts)
  chat  tests/data/ngram_chat_prompts.txt       (plain technical chat, 6 prompts)
  tool  tool-calling turns (generic agent system prompt + file/shell toolset, 6 turns); see --tool-corpus
Per request it records the server's draft counters (timings.draft_n / draft_n_accepted) and the
per-request log line "draft acceptance = ... mean acceptance length = ..., acceptance rate per
position = (...)", which carries the verify-step count the JSON does not.

  run:        mtp_acceptance_gate.py run --url http://127.0.0.1:8291 --log server.log \
                  --depth 2 --temps 0,0.7 --out rows.jsonl [--max-tokens 512] [--workloads code,chat,tool]
              Each prompt runs at every temperature back to back, so the slot reuses its prefill.
  --tool-corpus auto|private|public|<path>   which tool-calling corpus the `tool` workload uses.
    private  tests/data/mtp_gate1_toolcall_turns.json  (the maintainer's corpus; not shipped publicly)
    public   tests/data/mtp_toolcall_turns_public.json (neutral corpus, same shape and turn count)
    auto     (default) private when the file is present, else public.
  Every output row records the corpus used ("corpus"), and the run prints it. Numbers measured on the
  public corpus are NOT directly comparable with the private corpus's historical numbers (different
  system prompt, tool descriptions and tasks); compare like corpus with like corpus.
  --dry-run   list the prompts that would run (workload, id, corpus) and exit; needs no server.
  summarize:  mtp_acceptance_gate.py summarize [--common] rows.jsonl [...]   (markdown on stdout;
              --common keeps only prompts measured at every depth)
  project:    mtp_acceptance_gate.py project --step-ms 41.5 --verify-fixed-ms 41.5 --verify-row-ms 22.2 \
                  --draft-ms 3.5 [--temp 0.7] [--common] rows.jsonl
              projected K=1 speedup per depth D: tau * step / (fixed + row * D + draft * D), where
              a D-draft round verifies D+1 rows (fixed covers the first) and tau is measured here.

Acceptance = accepted drafts / drafted tokens. tau = tokens per verify forward = 1 + accepted /
verify steps. A sampled run (temp > 0) accepts a draft iff the target's own draw equals it.

@status  run + summarize complete
@issues  none
@todo    none
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "tests", "data")
ACCEPT_LINE = re.compile(r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), "
                         r"mean acceptance length =\s*([0-9.]+), acceptance rate per position = \(([^)]*)\)")


def load_marker_corpus(path):
    """A ===SYSTEM=== / ===PROMPT=== file as a list of message lists."""
    convs, system, kind = [], None, "===PROMPT==="  # a file may open without a marker
    for block in re.split(r"^(===SYSTEM===|===PROMPT===)\n", open(path).read(), flags=re.M):
        if block in ("===SYSTEM===", "===PROMPT==="):
            kind = block
            continue
        if not block.strip():
            continue
        if kind == "===SYSTEM===":
            system = block.strip()
        else:
            msgs = [{"role": "system", "content": system}] if system else []
            convs.append(msgs + [{"role": "user", "content": block.strip()}])
    return convs


TOOL_CORPORA = {"private": "mtp_gate1_toolcall_turns.json", "public": "mtp_toolcall_turns_public.json"}


def resolve_tool_corpus(choice):
    """(name, path) of the tool-calling corpus: private when present for auto, else public."""
    if choice == "auto":
        choice = "private" if os.path.exists(os.path.join(DATA, TOOL_CORPORA["private"])) else "public"
    if choice in TOOL_CORPORA:
        return choice, os.path.join(DATA, TOOL_CORPORA[choice])
    return os.path.basename(choice), choice  # an explicit path


def load_workloads(names, tool_corpus="auto"):
    """[(workload, id, messages, tools)] for the requested workload names."""
    out = []
    if "code" in names:
        for i, m in enumerate(load_marker_corpus(os.path.join(DATA, "code_edit_prompts.txt"))):
            out.append(("code", f"code{i}", m, None))
    if "chat" in names:
        for i, m in enumerate(load_marker_corpus(os.path.join(DATA, "ngram_chat_prompts.txt"))):
            out.append(("chat", f"chat{i}", m, None))
    if "tool" in names:
        corpus = json.load(open(resolve_tool_corpus(tool_corpus)[1]))
        for t in corpus["turns"]:
            msgs = [{"role": "system", "content": corpus["system"]}] + t["messages"]
            out.append(("tool", t["id"], msgs, corpus["tools"]))
    return out


def post_chat(url, body, timeout):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def read_new_accept_line(log_path, offset):
    """The first per-request acceptance line written to the server log after `offset`."""
    for _ in range(50):  # the slot prints it right after the response is sent
        with open(log_path, errors="replace") as f:
            f.seek(offset)
            m = ACCEPT_LINE.search(f.read())
        if m:
            return m
        time.sleep(0.1)
    return None


def run_one(args, temp, workload, rid, msgs, tools, corpus=None):
    body = {"messages": msgs, "max_tokens": args.max_tokens, "temperature": temp,
            "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed, "stream": False}
    if temp == 0:
        body.update({"top_k": 1, "top_p": 1.0})
    if tools:
        body["tools"] = tools
    offset = os.path.getsize(args.log)
    resp = post_chat(args.url, body, args.timeout)
    tm = resp.get("timings", {})
    m = read_new_accept_line(args.log, offset)
    accepted = int(m.group(2)) if m else tm.get("draft_n_accepted", 0)
    mean_len = float(m.group(4)) if m else None
    # verify steps: exact from the mean acceptance length; with nothing accepted that is 0/0, and
    # every round drafts exactly `depth` tokens (p_min 0), so drafted / depth.
    steps = round(accepted / (mean_len - 1.0)) if mean_len and mean_len > 1.0 \
        else round(tm.get("draft_n", 0) / args.depth)
    msg = resp["choices"][0]["message"]
    return {"workload": workload, "id": rid, "corpus": corpus if workload == "tool" else None, "depth": args.depth, "temp": temp, "label": args.label,
            "predicted_n": tm.get("predicted_n"), "predicted_per_second": tm.get("predicted_per_second"),
            "draft_n": tm.get("draft_n", 0), "draft_accepted": accepted, "verify_steps": steps,
            "per_pos": [float(x) for x in m.group(5).split(",")] if m and m.group(5).strip() else [],
            "finish": resp["choices"][0].get("finish_reason"),
            "has_tool_call": bool(msg.get("tool_calls")),
            "reasoning_chars": len(msg.get("reasoning_content") or ""), "content_chars": len(msg.get("content") or "")}


def cmd_run(args):
    rows = load_workloads(args.workloads.split(","), args.tool_corpus)
    if args.ids:
        rows = [r for r in rows if r[1] in args.ids.split(",")]
    if args.limit:
        rows = [r for w in dict.fromkeys(r[0] for r in rows) for r in [x for x in rows if x[0] == w][:args.limit]]
    corpus = resolve_tool_corpus(args.tool_corpus)[0] if any(r[0] == "tool" for r in rows) else None
    print(f"tool corpus: {corpus or 'n/a (no tool workload)'}", flush=True)
    if args.dry_run:
        for workload, rid, msgs, tools in rows:
            print(f"{workload:5s} {rid:24s} messages={len(msgs)} tools={len(tools or [])}")
        return
    with open(args.out, "a") as out:
        for workload, rid, msgs, tools in rows:
            for temp in (float(t) for t in args.temps.split(",")):
                row = run_one(args, temp, workload, rid, msgs, tools, corpus)
                out.write(json.dumps(row) + "\n")
                out.flush()
                acc = row["draft_accepted"] / row["draft_n"] if row["draft_n"] else float("nan")
                print(f"{args.label} D={args.depth} T={temp} {rid:24s} n={row['predicted_n']:4d} "
                      f"acc={acc:.3f} steps={row['verify_steps']} {row['predicted_per_second']:.1f} tok/s", flush=True)


def aggregate(rows):
    n = sum(r["predicted_n"] for r in rows)
    drafted = sum(r["draft_n"] for r in rows)
    acc = sum(r["draft_accepted"] for r in rows)
    steps = sum(r["verify_steps"] or 0 for r in rows)
    return n, drafted, acc, steps


def load_rows(files, common):
    """All JSONL rows; with `common`, only prompts measured at EVERY depth (like for like)."""
    rows = [json.loads(line) for p in files for line in open(p) if line.strip()]
    if common:
        depths = {r["depth"] for r in rows}
        have = {}
        for r in rows:
            have.setdefault((r["label"], r["temp"], r["id"]), set()).add(r["depth"])
        rows = [r for r in rows if have[(r["label"], r["temp"], r["id"])] == depths]
    return rows


def cmd_summarize(args):
    rows = load_rows(args.files, args.common)
    keys = sorted({(r["label"], r["temp"]) for r in rows})
    for label, temp in keys:
        corpora = sorted({r["corpus"] for r in rows if r["label"] == label and r["temp"] == temp and r.get("corpus")})
        print(f"\n### {label}, temp {temp}" + (f" (tool corpus: {', '.join(corpora)})" if corpora else "") + "\n")
        print("| D | workload | tokens | drafted | accepted | acceptance | tau (tokens/verify) | per-position |")
        print("|---|---|---|---|---|---|---|---|")
        for depth in sorted({r["depth"] for r in rows if r["label"] == label and r["temp"] == temp}):
            sel = [r for r in rows if r["label"] == label and r["temp"] == temp and r["depth"] == depth]
            for workload in ("code", "chat", "tool", "ALL"):
                ws = sel if workload == "ALL" else [r for r in sel if r["workload"] == workload]
                if not ws:
                    continue
                n, drafted, acc, steps = aggregate(ws)
                tau = 1.0 + acc / steps if steps else float("nan")
                pos = [sum(r["per_pos"][i] * (r["verify_steps"] or 0) for r in ws if len(r["per_pos"]) > i) / steps
                       for i in range(depth)] if steps else []
                name = f"**{workload}**" if workload == "ALL" else workload
                print(f"| {depth} | {name} | {n} | {drafted} | {acc} | {acc / drafted:.3f} | {tau:.2f} | "
                      f"{' / '.join(f'{p:.2f}' for p in pos)} |")


def cmd_project(args):
    rows = [r for r in load_rows(args.files, args.common) if r["temp"] == args.temp]
    print(f"| D | tau (temp {args.temp}) | round ms | break-even tau | projected speedup |")
    print("|---|---|---|---|---|")
    for depth in sorted({r["depth"] for r in rows}):
        _, _, acc, steps = aggregate([r for r in rows if r["depth"] == depth])
        tau = 1.0 + acc / steps
        round_ms = args.verify_fixed_ms + args.verify_row_ms * depth + args.draft_ms * depth
        print(f"| {depth} | {tau:.2f} | {round_ms:.1f} | {round_ms / args.step_ms:.2f} | "
              f"**{tau * args.step_ms / round_ms:.2f}x** |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--log", required=True, help="the llama-server log file (for the per-request line)")
    r.add_argument("--depth", type=int, required=True, help="the server's --spec-draft-n-max (recorded only)")
    r.add_argument("--temps", default="0,0.7", help="comma list; 0 = greedy")
    r.add_argument("--top-p", type=float, default=0.95)
    r.add_argument("--top-k", type=int, default=20)
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--timeout", type=float, default=1800)
    r.add_argument("--workloads", default="code,chat,tool")
    r.add_argument("--tool-corpus", default="auto", help="auto|private|public|<path> (see module doc)")
    r.add_argument("--dry-run", action="store_true", help="list the prompts and the corpus, then exit")
    r.add_argument("--limit", type=int, default=0, help="first N prompts of each workload (0 = all)")
    r.add_argument("--ids", default="", help="comma list of prompt ids to run (default all)")
    r.add_argument("--label", default="q8_0")
    r.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--common", action="store_true", help="only prompts present at every depth")
    s.add_argument("files", nargs="+")
    pr = sub.add_parser("project")
    pr.add_argument("--step-ms", type=float, required=True, help="plain K=1 decode step")
    pr.add_argument("--verify-fixed-ms", type=float, required=True, help="verify cost of the first row")
    pr.add_argument("--verify-row-ms", type=float, required=True, help="each further verify row")
    pr.add_argument("--draft-ms", type=float, required=True, help="one MTP draft token")
    pr.add_argument("--temp", type=float, default=0.7)
    pr.add_argument("--common", action="store_true", help="only prompts present at every depth")
    pr.add_argument("files", nargs="+")
    args = ap.parse_args()
    return {"run": cmd_run, "summarize": cmd_summarize, "project": cmd_project}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
