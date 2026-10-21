#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# eval_quality.sh — A/B quality gate for two engine builds whose logits are not
# bit-identical (relaxed float-reduction order, Q8_0 hot tensors, ...).
#
# Runs the fixed corpus perplexity and the greedy prompt suite through build A
# (reference) and build B (candidate) via tests/eval_ppl, then prints metrics
# and a single PASS/FAIL verdict.
#
# Usage:
#   evals/eval_quality.sh <buildA_bin> <buildB_bin> [model.gguf]
#
# Env knobs:
#   EVAL_MODEL       model path (overrides arg 3)
#   EVAL_MODEL_A/B   per-side model path (overrides EVAL_MODEL) — for a
#                    model-file A/B: same binary twice, two GGUFs
#   EVAL_PROMPTS     prompt suite      (default: evals/prompts/quality_suite.txt)
#   EVAL_CORPUS      ppl corpus        (default: evals/corpus/held_out.txt)
#   EVAL_MAX_NEW     greedy tokens per prompt (default: 48)
#   EVAL_PPL_TOKENS  cap on corpus tokens, 0 = all (default: 0)
#   EVAL_ENV_A/B     extra env for each build, e.g. EVAL_ENV_A="QWEN35_GRAPH=0"
#                    (lets one binary A/B its own runtime variants)
#   EVAL_DIE_ARG     optional leading barswap die arg (legacy rigs)
#   EVAL_OUT         artifact dir (default: evals/out/<timestamp>)
#   EVAL_SKIP_PPL=1 / EVAL_SKIP_GEN=1  skip a stage (for quick iteration)
#
# GPU runs: export the usual QWEN35_GPU_SPLIT_LAYER / QWEN35_GPU_DIE_A /
# QWEN35_GPU_DIE_B before invoking (shared by both builds), and use
# EVAL_ENV_A/EVAL_ENV_B for anything that must differ between them.
#
# ---------------------------------------------------------------------------
# Verdict thresholds (rationale)
#
#   PPL_DELTA_MAX_PCT = 0.5   Perplexity is the core quality number. Reduction
#                             reordering / KV rounding moves ppl by ~0.01-0.1%;
#                             a real bug (wrong tensor, broken expert, bad scale)
#                             moves it percents to orders of magnitude. 0.5% is
#                             far above harness noise (0 for a fixed corpus —
#                             runs are deterministic) yet far below any real
#                             regression.
#
#   EARLY_DIV_POS = 8         A benign near-tie flip almost never happens in the
#   EARLY_DIV_MAX_FRAC = 0.10 first few forced tokens of a templated answer;
#                             systematic early divergence (>=10% of prompts
#                             diverging before token 8) means the candidate's
#                             logits are shifted, not just reordered.
#
#   DIV_MAX_FRAC = 0.60       Once one near-tie token flips, all later tokens
#   (warn at 0.30)            differ textually, so "any divergence" over-counts;
#                             still, if >60% of a 60-prompt greedy suite diverges
#                             within 48 tokens the change is not a rounding-order
#                             effect. 30-60% prints a WARN: eyeball the TXT lines
#                             in the artifact dir for semantic equivalence.
#
#   PASS = ppl delta < 0.5%  AND  early-divergence fraction < 10%
#          AND  diverged fraction <= 60%.
#   A bit-exact pair (golden vs golden, GRAPH=0 vs GRAPH=1) must report
#   0 divergence and 0.000% ppl delta — that is the harness self-check.
# ---------------------------------------------------------------------------
set -u

PPL_DELTA_MAX_PCT=0.5
EARLY_DIV_POS=8
EARLY_DIV_MAX_FRAC=0.10
DIV_MAX_FRAC=0.60
DIV_WARN_FRAC=0.30

usage() { echo "usage: $0 <buildA_bin> <buildB_bin> [model.gguf]" >&2; exit 2; }
[ $# -ge 2 ] || usage

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
bin_a="$1"; bin_b="$2"
model="${EVAL_MODEL:-${3:-}}"
# A model-file A/B (one binary, two GGUFs — e.g. a retyped tensor) is the mirror
# image of the build A/B this script was written for: EVAL_MODEL_A/EVAL_MODEL_B
# override the shared model per side, and you pass the same binary twice.
model_a="${EVAL_MODEL_A:-$model}"
model_b="${EVAL_MODEL_B:-$model}"
prompts="${EVAL_PROMPTS:-$repo_root/evals/prompts/quality_suite.txt}"
corpus="${EVAL_CORPUS:-$repo_root/evals/corpus/held_out.txt}"
max_new="${EVAL_MAX_NEW:-48}"
ppl_tokens="${EVAL_PPL_TOKENS:-0}"
out_dir="${EVAL_OUT:-$repo_root/evals/out/$(date +%Y%m%d_%H%M%S)}"

[ -x "$bin_a" ] || { echo "FAIL: build A not executable: $bin_a" >&2; exit 2; }
[ -x "$bin_b" ] || { echo "FAIL: build B not executable: $bin_b" >&2; exit 2; }
[ -n "$model_a" ] && [ -f "$model_a" ] || { echo "FAIL: model A not found: '$model_a'" >&2; usage; }
[ -n "$model_b" ] && [ -f "$model_b" ] || { echo "FAIL: model B not found: '$model_b'" >&2; usage; }
[ -f "$prompts" ] || { echo "FAIL: prompts not found: $prompts" >&2; exit 2; }
[ -f "$corpus" ]  || { echo "FAIL: corpus not found: $corpus" >&2; exit 2; }
mkdir -p "$out_dir"

echo "== eval_quality =="
echo "build A : $bin_a ${EVAL_ENV_A:+(env: $EVAL_ENV_A)}"
echo "build B : $bin_b ${EVAL_ENV_B:+(env: $EVAL_ENV_B)}"
echo "model A : $model_a"
echo "model B : $model_b"
echo "out     : $out_dir"

# run_build <A|B> <mode> <input> <limit> <outfile>
run_build() {
    side="$1"; mode="$2"; input="$3"; limit="$4"; outfile="$5"
    bin="$bin_a"; extra="${EVAL_ENV_A:-}"; mdl="$model_a"
    [ "$side" = B ] && { bin="$bin_b"; extra="${EVAL_ENV_B:-}"; mdl="$model_b"; }
    # Perplexity needs the FULL logit vector: the GPU LM head defaults to a
    # sparse top-k copy (-inf elsewhere), under which the true next token can be
    # unrepresented -> -inf log-prob. Force full logits for ppl (the var is
    # ignored on the CPU build). Greedy gen is unaffected — argmax survives top-k.
    stage_env="$extra"
    [ "$mode" = ppl ] && stage_env="$stage_env QWEN35_GPU_FULL_LOGITS=1"
    echo "-- [$side] $mode ($(basename "$input"))"
    # shellcheck disable=SC2086
    if ! env $stage_env "$bin" ${EVAL_DIE_ARG:-} "$mdl" "$mode" "$input" "$limit" \
            >"$outfile" 2>"$outfile.log"; then
        echo "FAIL: build $side $mode run failed — see $outfile.log" >&2
        tail -5 "$outfile.log" >&2
        exit 1
    fi
}

fail=0

# ---- perplexity ------------------------------------------------------------
ppl_delta_pct="skipped"
if [ "${EVAL_SKIP_PPL:-0}" != 1 ]; then
    run_build A ppl "$corpus" "$ppl_tokens" "$out_dir/ppl_a.txt"
    run_build B ppl "$corpus" "$ppl_tokens" "$out_dir/ppl_b.txt"
    ppl_a=$(sed -n 's/.*ppl=\([0-9.eE+-]*\).*/\1/p' "$out_dir/ppl_a.txt")
    ppl_b=$(sed -n 's/.*ppl=\([0-9.eE+-]*\).*/\1/p' "$out_dir/ppl_b.txt")
    [ -n "$ppl_a" ] && [ -n "$ppl_b" ] || { echo "FAIL: missing PPL output" >&2; exit 1; }
    ppl_delta_pct=$(awk -v a="$ppl_a" -v b="$ppl_b" \
        'BEGIN { d = (b - a) / a * 100; printf "%.4f", (d < 0 ? -d : d) }')
    echo "ppl A=$ppl_a  B=$ppl_b  |delta|=$ppl_delta_pct%"
    awk -v d="$ppl_delta_pct" -v m="$PPL_DELTA_MAX_PCT" 'BEGIN { exit !(d < m) }' \
        || { echo "GATE: ppl delta $ppl_delta_pct% >= $PPL_DELTA_MAX_PCT%"; fail=1; }

    # BATCHED perplexity — the path the serve actually runs.
    #
    # `ppl` above feeds one token at a time. Serve prefills a whole prompt in
    # one batched call, and that is a DIFFERENT kernel path. Scoring only the
    # sequential path is how the q4 window-ring hazard survived every gate:
    # it cost +7.53% ppl batched (15x this threshold) while sequential ppl was
    # bit-identical, so the gate that exists to catch exactly this saw nothing.
    # A quality gate that measures a path the product does not use is not a
    # quality gate.
    run_build A pplb "$corpus" "$ppl_tokens" "$out_dir/pplb_a.txt"
    run_build B pplb "$corpus" "$ppl_tokens" "$out_dir/pplb_b.txt"
    pplb_a=$(sed -n 's/.*ppl=\([0-9.eE+-]*\).*/\1/p' "$out_dir/pplb_a.txt")
    pplb_b=$(sed -n 's/.*ppl=\([0-9.eE+-]*\).*/\1/p' "$out_dir/pplb_b.txt")
    [ -n "$pplb_a" ] && [ -n "$pplb_b" ] || { echo "FAIL: missing batched PPL output" >&2; exit 1; }
    pplb_delta_pct=$(awk -v a="$pplb_a" -v b="$pplb_b" \
        'BEGIN { d = (b - a) / a * 100; printf "%.4f", (d < 0 ? -d : d) }')
    echo "ppl(batched) A=$pplb_a  B=$pplb_b  |delta|=$pplb_delta_pct%"
    awk -v d="$pplb_delta_pct" -v m="$PPL_DELTA_MAX_PCT" 'BEGIN { exit !(d < m) }' \
        || { echo "GATE: batched ppl delta $pplb_delta_pct% >= $PPL_DELTA_MAX_PCT%"; fail=1; }

    # Sequential-vs-batched spread WITHIN each build. A/B deltas cancel a defect
    # present on both sides; this catches one that is present on both.
    for side in A B; do
        seq_v=$([ "$side" = A ] && echo "$ppl_a"  || echo "$ppl_b")
        bat_v=$([ "$side" = A ] && echo "$pplb_a" || echo "$pplb_b")
        spread=$(awk -v a="$seq_v" -v b="$bat_v" \
            'BEGIN { d = (b - a) / a * 100; printf "%.4f", (d < 0 ? -d : d) }')
        echo "build $side: sequential=$seq_v batched=$bat_v spread=$spread%"
        awk -v d="$spread" -v m="${PPL_BATCH_SPREAD_MAX_PCT:-1.0}" 'BEGIN { exit !(d < m) }' \
            || { echo "GATE: build $side batched-vs-sequential spread $spread% >= ${PPL_BATCH_SPREAD_MAX_PCT:-1.0}% — the batched path disagrees with itself"; fail=1; }
    done
fi

# ---- A/B greedy divergence -------------------------------------------------
if [ "${EVAL_SKIP_GEN:-0}" != 1 ]; then
    run_build A gen "$prompts" "$max_new" "$out_dir/gen_a.txt"
    run_build B gen "$prompts" "$max_new" "$out_dir/gen_b.txt"
    awk -v early_pos="$EARLY_DIV_POS" -v early_max="$EARLY_DIV_MAX_FRAC" \
        -v div_max="$DIV_MAX_FRAC" -v div_warn="$DIV_WARN_FRAC" '
        # Parse "GEN <i> prompt_tokens=<n>: id id ..." from both files, then
        # report per-prompt first-diff position + flip counts and gate.
        /^GEN / {
            idx = $2
            n = 0
            for (f = 4; f <= NF; f++) tok[n++] = $f
            if (FILENAME == ARGV[1]) {
                a_len[idx] = n
                for (i = 0; i < n; i++) a_tok[idx, i] = tok[i]
                if (idx + 1 > n_prompts) n_prompts = idx + 1
            } else {
                b_len[idx] = n
                for (i = 0; i < n; i++) b_tok[idx, i] = tok[i]
            }
        }
        END {
            if (n_prompts == 0) { print "GATE: no GEN output parsed"; exit 1 }
            diverged = 0; early = 0; flips = 0; compared = 0
            printf "%-8s %6s %6s %10s %6s\n", "prompt", "lenA", "lenB", "first_diff", "flips"
            for (p = 0; p < n_prompts; p++) {
                la = a_len[p]; lb = b_len[p]
                ml = la < lb ? la : lb
                fd = -1; fl = 0
                for (i = 0; i < ml; i++)
                    if (a_tok[p, i] != b_tok[p, i]) { fl++; if (fd < 0) fd = i }
                if (fd < 0 && la != lb) fd = ml   # one build stopped early
                flips += fl; compared += ml
                if (fd >= 0) {
                    diverged++
                    if (fd < early_pos) early++
                    printf "%-8d %6d %6d %10d %6d\n", p, la, lb, fd, fl
                }
            }
            div_frac = diverged / n_prompts
            early_frac = early / n_prompts
            flip_pct = compared ? flips * 100.0 / compared : 0
            printf "prompts=%d diverged=%d (%.1f%%) early(<%d)=%d (%.1f%%) token_flips=%.2f%%\n",
                   n_prompts, diverged, div_frac * 100, early_pos, early,
                   early_frac * 100, flip_pct
            rc = 0
            if (early_frac >= early_max) {
                printf "GATE: early divergence %.1f%% >= %.1f%%\n",
                       early_frac * 100, early_max * 100; rc = 1
            }
            if (div_frac > div_max) {
                printf "GATE: diverged fraction %.1f%% > %.1f%%\n",
                       div_frac * 100, div_max * 100; rc = 1
            } else if (div_frac >= div_warn)
                printf "WARN: diverged fraction %.1f%% — review TXT lines for semantic equivalence\n",
                       div_frac * 100
            exit rc
        }' "$out_dir/gen_a.txt" "$out_dir/gen_b.txt" | tee "$out_dir/divergence.txt"
    [ "${PIPESTATUS[0]}" -eq 0 ] || fail=1
    # Side-by-side text for human review of diverged prompts.
    grep '^TXT' "$out_dir/gen_a.txt" >"$out_dir/txt_a.txt" || true
    grep '^TXT' "$out_dir/gen_b.txt" >"$out_dir/txt_b.txt" || true
fi

echo "=================================================="
if [ "$fail" -eq 0 ]; then
    echo "VERDICT: PASS (ppl |delta|=$ppl_delta_pct% < $PPL_DELTA_MAX_PCT%, no systematic divergence)"
else
    echo "VERDICT: FAIL — see gates above and artifacts in $out_dir"
fi
exit "$fail"
