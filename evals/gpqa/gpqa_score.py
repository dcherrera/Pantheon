#!/usr/bin/env python3
"""
gpqa_score.py — score a GPQA Diamond run directory: per-seed accuracy, mean, 95% CI, extraction
failures, truncation, and the per-request speed figures (TTFT, decode tok/s, output length).

  python3 evals/gpqa/gpqa_score.py results/<run_name> [--markdown]

Scoring: an extraction failure counts as WRONG (as in simple-evals) and is reported separately.
Per-seed CI: normal approximation sqrt(p(1-p)/n). Mean-over-seeds CI: clustered on the
question (each question's mean score over seeds is one sample; SE = sd/sqrt(n)) — the questions
are the unit of sampling, so seeds of the same question are not counted as independent.

Status: works.
Issues: None
Todo: None
"""
import argparse
import glob
import json
import os
import statistics

Z95 = 1.96


def proportion_with_ci(correct, total):
    """(p, 95% half-width) for a binomial proportion."""
    p = correct / total
    return p, Z95 * (p * (1 - p) / total) ** 0.5


def mean_with_clustered_ci(scores_by_question):
    """(mean, 95% half-width) over questions of each question's mean score across seeds."""
    per_question = [sum(s) / len(s) for s in scores_by_question.values()]
    n = len(per_question)
    mean = sum(per_question) / n
    sd = statistics.pstdev(per_question) if n > 1 else 0.0
    return mean, Z95 * sd / n ** 0.5


def load_seed_results(run_dir):
    """{seed: [record, ...]} from seed<N>.jsonl files (last record per record_id wins)."""
    results = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "seed*.jsonl"))):
        by_id = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    by_id[record["record_id"]] = record
        if by_id:
            results[int(os.path.basename(path)[4:-6])] = list(by_id.values())
    return results


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize_speed(records):
    tokens = [r["completion_tokens"] for r in records]
    ttft = [r["ttft_s"] for r in records if r["ttft_s"] is not None]
    rates = [r["decode_tok_s"] for r in records if r["decode_tok_s"]]
    total_s = sum(r["total_s"] for r in records)
    return {
        "tokens_mean": statistics.mean(tokens), "tokens_median": statistics.median(tokens),
        "tokens_p90": percentile(tokens, 0.9), "tokens_max": max(tokens),
        "ttft_median_s": statistics.median(ttft) if ttft else None,
        "decode_tok_s_median": statistics.median(rates) if rates else None,
        "decode_tok_s_aggregate": sum(tokens) / total_s if total_s else None,
        "wall_s_total": total_s, "wall_s_mean": total_s / len(records),
    }


def score_run(run_dir):
    seeds = load_seed_results(run_dir)
    per_seed, scores_by_question = {}, {}
    for seed, records in sorted(seeds.items()):
        correct = sum(r["is_correct"] for r in records)
        p, half = proportion_with_ci(correct, len(records))
        failures = {}
        for r in records:
            if r["extracted_letter"] is None:
                failures[r["extraction_method"]] = failures.get(r["extraction_method"], 0) + 1
            scores_by_question.setdefault(r["record_id"], []).append(1.0 if r["is_correct"] else 0.0)
        per_seed[seed] = {
            "n": len(records), "correct": correct, "accuracy": p, "ci95": half,
            "extraction_failures": failures,
            "truncated": sum(r["finish_reason"] == "length" for r in records),
            "speed": summarize_speed(records),
        }
    complete = {q: s for q, s in scores_by_question.items() if len(s) == len(seeds)}
    mean, half = mean_with_clustered_ci(complete) if complete else (None, None)
    return {"per_seed": per_seed, "mean": mean, "mean_ci95": half, "n_questions_all_seeds": len(complete)}


def format_optional(value, spec):
    return "-" if value is None else format(value, spec)


def render_markdown(summary):
    lines = ["| seed | n | correct | accuracy | 95% CI | extraction failures | truncated | "
             "tokens mean / median / p90 / max | TTFT median | decode tok/s median | wall |",
             "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |"]
    for seed, s in summary["per_seed"].items():
        sp = s["speed"]
        fails = ", ".join(f"{k} {v}" for k, v in s["extraction_failures"].items()) or "0"
        lines.append(
            f"| {seed} | {s['n']} | {s['correct']} | {100 * s['accuracy']:.1f} | ±{100 * s['ci95']:.1f} | "
            f"{fails} | {s['truncated']} | {sp['tokens_mean']:.0f} / {sp['tokens_median']:.0f} / "
            f"{sp['tokens_p90']} / {sp['tokens_max']} | {format_optional(sp['ttft_median_s'], '.2f')} s | "
            f"{format_optional(sp['decode_tok_s_median'], '.1f')} | {sp['wall_s_total'] / 3600:.2f} h |")
    if summary["mean"] is not None:
        lines.append(f"\nMean over seeds: **{100 * summary['mean']:.1f} ± {100 * summary['mean_ci95']:.1f}** "
                     f"(95% CI, clustered on {summary['n_questions_all_seeds']} questions).")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="score a GPQA Diamond run directory")
    parser.add_argument("run_dir")
    parser.add_argument("--markdown", action="store_true", help="print a markdown table")
    args = parser.parse_args()
    summary = score_run(args.run_dir)
    print(render_markdown(summary) if args.markdown else json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
