# GPQA Diamond eval

Runs GPQA Diamond (198 questions) against any OpenAI-compatible serve, to compare our
Ornith-1.5-9B TT serve with tt-hous's published number. Doubles as a real-workload speed
measurement (per-request TTFT, decode tok/s, output length).

## Comparison target

- **tt-hous/ornith-1.5-9b** (tt-metal + vLLM v0.26.0, `p150` profile): **78.3**, mean of 2 runs
  (seeds 42 / 43: 75.3 / 81.3).
- **86.4 reference**: the Ornith AI model card (`ornith-ai/Ornith-1.5-9B`), "averaged over five
  independent runs". No GPQA-specific settings are published there.

## Protocol

| | used here | source |
| --- | --- | --- |
| sampling | temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5 | tt-hous card ("the model card's recipe"); Ornith card's general-task settings |
| output budget | `max_tokens` 32768 (reasoning + answer) | tt-hous card |
| reasoning | on (template default); thinking returned in `reasoning_content` | tt-hous serves with `--reasoning_parser qwen3` |
| runs | seeds 42, 43 | tt-hous card |
| prompt | simple-evals `QUERY_TEMPLATE_MULTICHOICE`, one user turn, no system prompt | **not published**; simple-evals is the de-facto zero-shot CoT GPQA prompt |
| shuffle | per (seed, Record ID): `random.Random(sha256(...)).sample(range(4), 4)` | **not published**; ours is deterministic and order-independent |
| extraction | `content` only; simple-evals `Answer: X` regex (widened for markdown), then `\boxed{}`, "answer is X", lone final letter; last match wins | **not published** |
| failures | unextractable = wrong, reported separately by reason (`truncated_no_answer`, `no_match`, ...) | simple-evals scores them wrong |
| request seed | `sha256("gpqa-diamond-sample:{seed}:{record_id}") mod 2^31` | ours; makes a question reproducible on the serve |
| CI | per seed: normal approx; mean: clustered on question | Miller 2024, "Adding error bars to evals" |

Because tt-hous's prompt, shuffle and extraction are unpublished, their seed 42 is not our seed
42; a match is expected only within run-to-run noise (their own two seeds differ by 6 points).

### Known deviations of the Pantheon serve (12b314c) from vLLM

1. **presence_penalty window.** Pantheon applies it over the last `repeat_last_n` = 64 tokens
   (`include/pantheon.h`); vLLM applies it over every generated token. Not settable per request.
2. **Server repeat_penalty 1.1** over the same 64-token window, always on
   (`pn_sampling_defaults`). vLLM's `repetition_penalty` defaults to 1.0. Not settable per
   request.
3. **presence_penalty does not force plain decode** at 12b314c. Sampled speculation
   (`sg_spec_init`) needs temperature > 0, no grammar, and sparse top-64 rows; a positive
   penalty keeps the rows sparse (`pn_qwen35moe_tt_select_rows`; only a negative one forces a
   dense refill). Measured: 28.0 tok/s with 1.5 vs 32.4 without (n=1). Speculative sampling is
   by construction the same draw as plain sparse sampling: speed, not accuracy. The MTP head is
   whatever `<artifact>.mtp.fttw` points at when the serve starts.

(1) and (2) can move accuracy; they are inherent to the serve and are not corrected by the harness.

## Data

`Idavidrein/gpqa` is gated on Hugging Face (CC BY 4.0, plus a request that examples not be
revealed online). `gpqa_dataset_fetch.py` uses the Mac's HF login and writes `data/`
(gitignored). Nothing prints, logs or commits question text; `results/` (gitignored) holds model
outputs. Needs the HF account to have accepted the dataset terms and a token with
**read access to public gated repos**.

## Usage

```sh
python3 evals/gpqa/gpqa_dataset_fetch.py
python3 evals/gpqa/gpqa_run.py --base-url http://<host>:8319 --model ornith --run-name ornith_tt_full
python3 evals/gpqa/gpqa_score.py evals/gpqa/results/ornith_tt_full --markdown
python3 evals/gpqa/test_gpqa_answer_extraction.py
```

Interrupt any time; rerun the same command to resume (finished questions are skipped; request
errors are retried). A changed protocol under the same `--run-name` is refused.

`fixtures/smoke_questions_not_gpqa.csv` holds five of our own GPQA-style questions (not GPQA
data), for exercising the pipeline without the gated dataset.
