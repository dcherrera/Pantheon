#!/usr/bin/env python3
"""
gpqa_run.py — run GPQA Diamond against an OpenAI-compatible serve; resumable, one JSONL per seed.

  python3 evals/gpqa/gpqa_run.py --base-url http://HOST:8319 --model ornith \\
      --run-name ornith_tt_full [--seeds 42 43] [--limit N]

Defaults reproduce the tt-hous / Ornith AI recipe: temperature 1.0, top_p 0.95, top_k 20,
presence_penalty 1.5, max_tokens 32768, thinking on (template default), seeds 42 and 43.
Results go to evals/gpqa/results/<run-name>/ (gitignored): seed<N>.jsonl (one finished question
per line, appended + fsynced), request_errors.jsonl (transport/HTTP failures; those questions
stay pending and are retried on the next invocation), run_config.json (the protocol; a resume
with a different protocol is refused). Seeds run one after the other so a full per-seed number
lands first. The console never shows question text — record id, letters, tokens, speed only.

Status: works.
Issues: None
Todo: None
"""
import argparse
import datetime
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from gpqa_answer_extraction import extract_answer_letter
from gpqa_dataset import DIAMOND_CSV, EVAL_DIR, build_shuffled_item, load_diamond_questions, request_sampling_seed
from gpqa_openai_stream_client import ServeRequestError, stream_chat_completion
from gpqa_score import render_markdown, score_run

PROTOCOL_KEYS = ("model", "temperature", "top_p", "top_k", "presence_penalty", "max_tokens", "dataset_csv_name")


def parse_args():
    parser = argparse.ArgumentParser(description="GPQA Diamond against an OpenAI-compatible serve")
    parser.add_argument("--base-url", required=True, help="serve root, e.g. http://<host>:8319")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--run-name", required=True, help="results/<run-name>/")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--limit", type=int, default=0, help="first N questions only (0 = all)")
    parser.add_argument("--dataset-csv", default=DIAMOND_CSV)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--concurrency", type=int, default=1, help="in-flight requests (TT serve: 1)")
    parser.add_argument("--idle-timeout-s", type=float, default=900, help="max silence on the stream")
    parser.add_argument("--api-key-env", default="", help="NAME of an env var holding the API key")
    return parser.parse_args()


def protocol_of(args):
    return {
        "model": args.model, "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "presence_penalty": args.presence_penalty, "max_tokens": args.max_tokens,
        "dataset_csv_name": os.path.basename(args.dataset_csv),
        "prompt": "simple-evals QUERY_TEMPLATE_MULTICHOICE, single user turn, no system prompt",
        "shuffle": "random.Random(sha256('gpqa-diamond-shuffle:{seed}:{record_id}')).sample(range(4), 4)",
    }


def check_or_write_run_config(run_dir, args):
    """Write run_config.json, or refuse to resume a run made with a different protocol."""
    path = os.path.join(run_dir, "run_config.json")
    protocol = protocol_of(args)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            previous = json.load(f)["protocol"]
        changed = [k for k in PROTOCOL_KEYS if previous.get(k) != protocol[k]]
        if changed:
            sys.exit(f"gpqa_run: {run_dir} was run with a different {changed}; use a new --run-name")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"protocol": protocol, "base_url": args.base_url,
                   "started": datetime.datetime.now().isoformat(timespec="seconds")}, f, indent=2)


def finished_record_ids(results_path):
    if not os.path.exists(results_path):
        return set()
    with open(results_path, encoding="utf-8") as f:
        return {json.loads(line)["record_id"] for line in f if line.strip()}


def append_jsonl(path, record, lock):
    with lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def build_request_body(args, item, seed):
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "presence_penalty": args.presence_penalty, "max_tokens": args.max_tokens,
        "seed": request_sampling_seed(seed, item["record_id"]),
    }


def run_one_question(args, item, seed, paths, lock, api_key):
    """Ask one question; append its result (or its transport error) to the run's JSONL files."""
    body = build_request_body(args, item, seed)
    try:
        reply = stream_chat_completion(args.base_url, body, api_key, args.idle_timeout_s)
    except ServeRequestError as e:
        append_jsonl(paths["errors"], {"record_id": item["record_id"], "seed": seed, "error": str(e),
                                        "at": datetime.datetime.now().isoformat(timespec="seconds")}, lock)
        print(f"seed {seed} {item['record_id']}: REQUEST ERROR {e}", flush=True)
        return
    letter, method = extract_answer_letter(reply["content"], reply["finish_reason"])
    record = {
        "record_id": item["record_id"], "seed": seed, "request_seed": body["seed"],
        "permutation": item["permutation"], "correct_letter": item["correct_letter"],
        "extracted_letter": letter, "extraction_method": method,
        "is_correct": letter == item["correct_letter"],
        **{k: v for k, v in reply.items() if k not in ("content", "reasoning")},
        "content": reply["content"], "reasoning": reply["reasoning"],
    }
    append_jsonl(paths["results"], record, lock)
    rate = f"{reply['decode_tok_s']:.1f}" if reply["decode_tok_s"] else "-"
    print(f"seed {seed} {item['record_id']}: got {letter or '-'} ({method}) want {item['correct_letter']} "
          f"{'OK ' if record['is_correct'] else 'BAD'} tokens {reply['completion_tokens']} "
          f"finish {reply['finish_reason']} ttft {reply['ttft_s'] or 0:.2f}s decode {rate} tok/s "
          f"wall {reply['total_s']:.0f}s", flush=True)


def main():
    args = parse_args()
    questions = load_diamond_questions(args.dataset_csv)
    if args.limit:
        questions = questions[:args.limit]
    run_dir = os.path.join(EVAL_DIR, "results", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    check_or_write_run_config(run_dir, args)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    lock = threading.Lock()
    errors_path = os.path.join(run_dir, "request_errors.jsonl")
    for seed in args.seeds:
        paths = {"results": os.path.join(run_dir, f"seed{seed}.jsonl"), "errors": errors_path}
        done = finished_record_ids(paths["results"])
        pending = [build_shuffled_item(q, seed) for q in questions if q["record_id"] not in done]
        print(f"seed {seed}: {len(done)} done, {len(pending)} pending", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = [pool.submit(run_one_question, args, item, seed, paths, lock, api_key) for item in pending]
            for future in futures:
                future.result()  # re-raise anything unexpected instead of dropping it
    print(render_markdown(score_run(run_dir)))


if __name__ == "__main__":
    main()
