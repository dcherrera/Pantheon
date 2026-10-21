# Pantheon — HTTP API Reference

The server speaks HTTP/1.1 over TCP. Every response carries
`Access-Control-Allow-Origin: *` and `Connection: close`. No authentication.
All bodies are JSON unless noted (SSE for streaming).

---

## Routes

| Method | Path | Handler | Audience |
|--------|------|---------|----------|
| `GET` | `/` | health | client / operator |
| `OPTIONS` | `*` | CORS preflight | client |
| `GET` | `/v1/models` | list models | client |
| `POST` | `/v1/chat/completions` | chat completion | client |
| `GET` | `/admin/models` | list live model instances | operator |
| `POST` | `/admin/models/load` | load a model | operator |
| `POST` | `/admin/models/unload` | unload a model | operator |
| `GET` | `/admin/dies` | die status | operator |
| `POST` | `/admin/session/export` | export a session's state as a blob | operator |
| `POST` | `/session/restore` | restore a session from a blob | client |
| `POST` | `/session/release` | free a session's slot | client |
| `GET` | `/session/estimate` | warm-restore vs re-prefill time estimate | client |
| `GET` | `/session/watch` | hold a session alive while connected | client |

Any other path → `404 {"error":{"message":"unknown route","type":"not_found","code":null}}`.

---

## GET /

Health check. Always 200.

**Response:**
```json
{"service":"pantheon","status":"ok","backend":"cuda",
 "build":"5d818bfd1c9a","accelerator":true,"dies_visible":8}
```

| Field | Type | Meaning |
|-------|------|---------|
| `backend` | string | The accelerator backend this binary was COMPILED with: `cuda`, `tt`, `metal`, `cpu`. Compile-time constant, not a probe. |
| `build` | string | Commit this binary was built from (`-dirty` when the tree was not clean), or `unknown`. |
| `accelerator` | bool | False means every forward pass in this process runs on the host CPU, whatever the flags say. |
| `dies_visible` | int | Dies the die manager holds (`pn_dm_count()`). `0` on a CUDA build means no driver / no card / empty `CUDA_VISIBLE_DEVICES`. |

The cheapest route on the server is also the one that answers "what am I
running?" — see [`GET /v1/models`](#backend--build--what-this-binary-actually-is)
for why that question needed an endpoint at all.

---

## OPTIONS \*

CORS preflight. Always 200 with `{}` body.

---

## GET /v1/models

**Serve's own endpoint** (`openai_chat.c:oai_handle_models`, `serve_command.c:dispatch`):
same-name replica slots (multi-die LB copies) are deduplicated — clients see one
logical id per served name. The response does **not** include `live` or `free_slots`.

**Response — 200:**
```json
{
  "object": "list",
  "backend": "cuda",
  "build": "5d818bfd1c9a",
  "data": [
    {
      "id": "my-model",
      "object": "model",
      "created": 0,
      "owned_by": "pantheon",
      "max_sessions": 4,
      "max_concurrent": 4,
      "context_window": 32000,
      "tool_choice_required_with_thinking": true,
      "graphs": {"live": 30, "total": 40, "failed": 0,
                 "ineligible": 10, "pending": 0},
      "attention": {"ondie": 10, "host": 0, "total": 10}
    }
  ]
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `backend` | string | **Top-level.** The accelerator backend this binary was COMPILED with: `cuda`, `tt`, `metal` or `cpu`. Compile-time constant, not a probe. |
| `build` | string | **Top-level.** The commit this binary was built from (`-dirty` suffixed when the tree was not clean), or `unknown`. |
| `id` | string | The routing key for `model` on a completion |
| `max_sessions` | int | RESIDENCY: how many conversations may hold KV at once |
| `max_concurrent` | int | CONCURRENCY: how many requests this model ACTUALLY decodes at once — the **served** value (`bs_get_load` capacity), not the configured cap. See below. |
| `context_window` | int | **Per-session context window in tokens** — see below. Omitted when unknown. |
| `tool_choice_required_with_thinking` | bool | `true` when `tool_choice: "required"` may be sent with thinking on: the think block is unconstrained and the turn ends in a parsed tool call (the think-gated grammar models ORNITH_JSON and QWEN_XML; `false` for MiniCPM's attribute dialect, which no grammar models, and for every model when `PANTHEON_TOOL_GRAMMAR=0`). |
| `graphs` | object | **CUDA-graph residency** — see below. Omitted when the model has no graph slots. |
| `attention` | object | **Attention-path residency** — see below. Omitted until the forward has reached an attention layer. |

### `backend` / `build` — what this binary actually is

Top-level on the response, not per-model, because it is a property of the
process: the accelerator backend it was **compiled** with (`cuda`, `tt`, `metal`,
`cpu`) and the commit it came from.

Compile-time constant, deliberately — not a runtime probe. A serve whose binary
has no CUDA compiled in cannot use a GPU whatever the driver reports, and the
completion text is identical either way, so nothing downstream can detect it
from the output.

The same values are on `GET /` (plus `dies_visible`), and out of band from
`pantheon --backend`, which prints one lowercase token for a deploy script to
assert against.

### `max_concurrent` — the SERVED number, not the configured cap

`max_concurrent` on `/v1/models` is what the model actually decodes at once
(`bs_get_load`'s capacity), which is **not** always what `--max-concurrent` asked
for. It reads `1` whenever there is no batch scheduler — no CUDA compiled in,
`QWEN35_BATCH_SLOTS` off, `bs_create` failed — and it drops below the cap when
the memory-pressure ladder clamps the slot count.

Same argument as `context_window` below: a client sizes its in-flight requests
from this number, so advertising a cap the serve cannot honour (e.g. `2` with no
scheduler, which turns every second concurrent chat into a `429`) is worse than
advertising nothing.

Operators still get the configured value: `/admin/models` carries **both**, as
`max_concurrent` (served) and `max_concurrent_configured` (asked for). Seeing
them differ is the signal.

### `context_window` — the advertised window

`context_window` is the window a client may actually plan a prompt against, in
tokens. It is the **effective** window (`pn_effective_ctx_window`): the operator's
`--ctx-cap` **after** the reactive memory ladder's runtime clamp — *not* the
configured cap, and not the model's theoretical maximum.

That distinction is the whole point of the field. Publishing the configured cap
while admission enforces a smaller one hands the client a prefill the serve then
rejects — on a K80 that rejection lands after minutes of cold-prefill TTFT, and it
lands mid-task.

Two properties clients depend on:

- **It is the same number a completion reports as `usage.kv_cap`.** The model list
  is the client's INITIAL plan; `kv_cap` is the LIVE truth on every turn. A client
  that reads both needs no polling and no extra endpoint to notice a mid-session
  clamp.
- **It is additive and omitted when unknown.** A plain OpenAI-compatible client
  ignores the extra key; a client that reads it never sees a `0` it has to
  special-case. An absent field means *undetected*; a client should fall back to
  a conservative window rather than guess.

Field name: Pantheon emits `context_window`. Clients that detect the window from
other servers' spellings (`context_length`, `max_model_len`, `n_ctx`, ...) need
to accept this one.

### `graphs` — CUDA-graph residency

Each layer of the hybrid stack can hold a captured CUDA graph that collapses its
~13-30 kernel launches into one replay. Across 40 layers those graphs are worth
about 1.8 ms/step. A capture that **fails** falls back to the direct kernel body
and produces **correct output** — so losing them costs only speed, silently.
This object is the only surface that reports it.

| Key | Meaning |
|-----|---------|
| `live` | Slots holding a captured graph right now |
| `total` | Slots the forward has actually reached (see below) |
| `failed` | Attempted and did not stick — **the only value that is a fault** |
| `ineligible` | Gated out; this model never captures. **Not a fault.** |
| `pending` | Eligible, warming up, no attempt yet |
| `gave_up` | Subset of `failed` whose bounded retry budget is spent (omitted when 0) |

**`ineligible` is normal.** The captured path requires the grouped MoE kernel,
which requires Q3_K expert tensors. A Q5_K_M model reports `live 0 / total 40 /
ineligible 40 / failed 0` — correct behaviour, zero exposure, and also zero of
the +4%. Read `failed`, not `live/total`, to decide whether something is wrong.

**`total` counts only exercised slots.** A serve decoding batched never touches
its single-token graph slots, so those are not reported as missing graphs.

A failed capture is retried on a bounded backoff and only stops after the budget
is spent, so a transient `failed` on one read that is `live` on the next is the
system self-healing, not a flapping metric.

### `attention` — on-die vs host-softmax residency

The same argument as `graphs`, only much sharper. Each full-attention layer
normally runs the on-die flash kernels. When it cannot it falls back to host-side
causal GQA softmax on the serve host's CPU — the exact wall this engine was built
to escape (it was 87.5% of an 8k prefill; moving attention on-die took that
prefill from 26 minutes to 3.82). The fallback returns the **same tokens**, logs
nothing per step, and on the depth bench costs **13.02 -> 1.22 tok/s** (K=2,
depth 4096) — and it worsens with context, because host softmax is O(context)
per layer per token. Without
this object a user simply waits, and nothing anywhere says why.

| Key | Meaning |
|-----|---------|
| `ondie` | Attention layers running the on-die flash kernels right now. |
| `host` | Layers on the host-softmax fallback — **any non-zero value is a fault** |
| `total` | Layers the forward has actually reached (same rule as `graphs.total`) |
| `gave_up` | Layers whose KV-ring allocation spent its retry budget (omitted when 0) |
| `impossible` | Subset of `gave_up`: the rings cannot fit this die at all (omitted when 0) |

`ondie + host == total`. Unlike `graphs`, there is no benign state here: `host >
0` always means this model is serving degraded, and the `reasons` histogram on
`/admin/models` says why. The two common causes are a per-slot KV-ring allocation
that did not fit (lower `--ctx-cap` or `--max-sessions`) and a session that has
run past the KV window, which wraps the ring.

A ring allocation that fails is retried on a bounded backoff, so a `host` reading
that clears on the next poll is the system self-healing. `gave_up` is the number
that should page someone.

No error cases (returns an empty `data` array when nothing is loaded).

**Router's `/v1/models`** (`registry.c:reg_models_json`) is a different shape: it
aggregates across live serve instances and adds `live` (count of live instances serving
that id) and `free_slots` (summed across live replicas). Do not conflate the two.

The router also emits `context_window`, aggregated the only honest way: the
**minimum** across that model_id's live replicas (a dispatch may land on any of
them), and **omitted entirely** if any live replica has not reported one. Serve
instances report their window on both `/registry/register` and every
`/registry/heartbeat`, because the ladder can move it at runtime.

---

## POST /v1/chat/completions

OpenAI-compatible chat completion. Supports streaming (SSE) and non-streaming.
Includes reasoning (`<think>…</think>` split) and tool-call support.

### Request headers

| Header | Required | Effect |
|--------|----------|--------|
| `Content-Type: application/json` | no | Not checked; the body is parsed as JSON regardless. |
| `X-Session-Id: <sid>` | no | Prefix-cache session key. A body key (`session_id`, `prompt_cache_key`, `promptCacheKey`) takes precedence. opencode sends this header on every request. |

### Request body

```json
{
  "model": "my-model",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "hello"}
  ],
  "temperature":  0.7,
  "top_p":        0.95,
  "top_k":        40,
  "max_tokens":   1024,
  "stop":         ["<|im_end|>"],
  "stream":       false,
  "tools":        [...],
  "tool_choice":  "auto",
  "session_id":   "abc123"
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `model` | string | — | Routing key. With one loaded model, any value (or absent) routes to it. With multiple models, must match a served name exactly; unknown → 404. |
| `messages` | array | *required* | Non-empty array of `{role, content}`. Roles: `system`, `user`, `assistant`, `tool`; a missing role is treated as `user`. `content` must be a string — any other type (e.g. an array of content parts) is read as empty. |
| `messages[].reasoning_content` | string | unset | Assistant turns only. The reasoning this serve returned for that turn; replayed into the prompt so a session can resume from its prefix cache. |
| `messages[].tool_calls` | array | unset | Assistant turns only. Rendered back into the prompt in the model's tool dialect. |
| `temperature` | float | `0.7` | `<= 0` is greedy argmax. |
| `top_p` | float | `0.95` | Nucleus sampling. |
| `top_k` | int | `40` | Top-K sampling. |
| `max_tokens` | int | `1024` | Max completion tokens. `0` or negative → `512`. Clamped to 1048576. |
| `max_completion_tokens` | int | `1024` | Alias; `max_tokens` wins if both are present. |
| `stop` | string or array | unset | Stop strings. |
| `stream` | bool | `false` | `true` → SSE (`text/event-stream`). |
| `stream_options.include_usage` | bool | `false` | Streaming only: send a terminal `usage` chunk before `[DONE]`. |
| `seed` | number | unset | Makes sampling reproducible. With `n` > 1, choice *i* uses `seed + i`. |
| `presence_penalty` | float | `0` | Clamped to [-2, 2]. |
| `frequency_penalty` | float | `0` | Clamped to [-2, 2]. |
| `logit_bias` | object | unset | `{"<token id>": bias}`; bias clamped to [-100, 100]. Non-numeric keys or values are skipped. |
| `logprobs` | bool | `false` | Return per-token log-probabilities (`choices[].logprobs.content[]`, with `token`, `logprob`, `bytes`). |
| `top_logprobs` | int | `0` | Alternatives per token; clamped to [0, 20]. |
| `n` | int | `1` | Number of completions. Each is a full separate decode (only the first uses the session's prefix cache). Capped at 8; a larger `n` returns 8 and sets `n_status: "capped_to_8"`. |
| `response_format` | object | `{"type":"text"}` | `json_object` constrains the answer to valid JSON. `json_schema` gets the same JSON constraint but the schema's shape is **not** enforced. See `response_format_status` below. |
| `enable_thinking` | bool | unset | Think-style templates: `true` opens a think block, `false` prefills a closed one. Absent → the `PANTHEON_NO_THINK` env decides. |
| `reasoning_budget` | int | per-model | Force `</think>` after N reasoning tokens. Overrides the model's `--reasoning-budget`; `0` or negative disables it. `max_reasoning_tokens` is an alias (`reasoning_budget` wins). |
| `tools` | array | unset | OpenAI-format tool definitions. Rendered into a system turn in the model's tool dialect (below). Ignored when `tool_choice` is `"none"`. |
| `tool_choice` | string or object | `"auto"` | `"none"` disables tools. `"required"` forces a call: the grammar is armed by default (`PANTHEON_TOOL_GRAMMAR=0` opts out). With thinking on the think block is unconstrained and the call is compelled from `</think>` (models advertising `tool_choice_required_with_thinking`). `{"type":"function","function":{"name":"X"}}` is enforced only when `PANTHEON_TOOL_GRAMMAR=1`; otherwise it behaves like `"auto"`. |
| `session_id` | string | unset | Prefix-cache session key (overrides `X-Session-Id`). Enables delta prefill on later turns of the same session. |
| `prompt_cache_key` / `promptCacheKey` | string | unset | Aliases for `session_id` (OpenAI's name; opencode's camelCase body field). Order: `session_id`, `prompt_cache_key`, `promptCacheKey`, then the `X-Session-Id` header. |

### Tool calling

Tools are rendered into a system turn (merged with a leading `system` message)
in the dialect the routed model emits. The dialect is chosen from the model name
and architecture (`tool_calling.c:tc_dialect_for_model`), not from the GGUF chat
template:

| Dialect | Used for | Call syntax |
|---------|----------|-------------|
| Qwen3-Coder XML | default | `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>` |
| MiniCPM XML | Llama-arch models whose template adds BOS (MiniCPM5) | `<function name="NAME"><param name="KEY">VALUE</param>…` |
| JSON | served names starting `ornith` | fenced or bare `{"function"\|"name": NAME, "arguments": {…}}` |

The model's output is parsed into OpenAI `tool_calls`, with arguments typed from
each tool's JSON schema. `PANTHEON_TOOL_GRAMMAR` decides which requests are
grammar-constrained:

| Value | Constrained requests |
|-------|----------------------|
| unset (default) | `tool_choice: "required"` only — after reasoning, the output is one well-formed call to an offered tool, arguments typed and `required` keys present. Everything else decodes **unconstrained**. |
| `1` | every request with tools: `"auto"` masks once a call opener appears; `"required"` and a named function compel a call. |
| `0` | none — `"required"` is unconstrained too. |

On the Tenstorrent serve a `"required"` turn keeps speculative decoding: drafts
are checked against the grammar and the output equals non-speculative
constrained decoding. On a `"required"` turn (every dialect, single-user and
batched) an end token is refused before the call opens and, inside the call,
allowed only where the call can still be closed; the serve then appends the
closing text. Any other call the model started but did not finish is left
out of `tool_calls` and reported in `tool_call_error` (`code`, optional
`function`, `parameter`, `message`) on the choice.

`tools` and `response_format` cannot both constrain one decode: when the request
carries tools, tools win and `response_format_status` is `"ignored_tools_present"`.

### Non-streaming response — 200

```json
{
  "id": "chatcmpl-a1b2c3d4",
  "object": "chat.completion",
  "created": 1720000000,
  "model": "my-model",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello!",
      "reasoning_content": "...",
      "tool_calls": [...]
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 7,
    "total_tokens": 19,
    "prompt_tokens_details": {"cached_tokens": 0},
    "kv_used": 12,
    "kv_cap": 32000
  }
}
```

- `reasoning_content` — the text inside a `<think>…</think>` block; present only
  when non-empty. Only think-style chat templates with thinking on produce it.
  Non-think templates (plain ChatML such as SmolLM2, Mistral, Phi-3) return the
  whole answer as `content`.
- `tool_calls` — present when a call was parsed; `finish_reason` is then
  `"tool_calls"`. `content` is `null` for a pure tool call.
- `finish_reason` — `stop`, `length`, `tool_calls`, or `repetition_loop` (the
  decode was stopped because the model was looping; outranks `tool_calls`).
- `logprobs` — on the choice, when requested.
- `prompt_tokens_details.cached_tokens` — prompt tokens reused from the prefix
  cache rather than computed. Always present (`0` on a cold turn).
- `kv_used` — this session's KV occupancy in tokens after this turn.
- `kv_cap` — the **effective** per-session context window, the same number
  `GET /v1/models` advertises as `context_window`, re-read every turn. If the
  memory ladder moves the window mid-session, the next completion carries the
  new value. Treat the model list as the initial plan and this as authoritative.
- `session_conflict: true` (inside `usage`) — two conversations are sharing one
  session id, so neither can resume from cache.
- `response_format_status` (top-level, when the request sent `response_format`)
  — `applied`, `schema_downgraded`, `ignored_tools_present`,
  `ignored_unknown_type`, or `not_enforced` (a constraint was armed but the
  output did not close a complete JSON object).
- `n_status` (top-level) — why fewer choices came back than `n` asked for:
  `capped_to_8`, `partial_slot_unavailable`, `partial_generation_failed`.
- `usage` covers the whole request: the prompt is counted once, completion
  tokens are summed across choices.

A prompt longer than the effective window is **refused** with
`context_over_budget` (below); nothing is silently dropped. The
`context_capped` field the writer supports is never set by the current
generation path.

### Streaming response — 200 (SSE)

`Content-Type: text/event-stream`. Each chunk is:

```
data: {"id":"chatcmpl-...", "object":"chat.completion.chunk", "created":...,
       "model":"...", "choices":[{"index":0,"delta":{...},"finish_reason":null}]}

```

Delta fields emitted in order:

1. `{"role":"assistant"}` — first chunk of each choice.
2. `{"reasoning_content":"<fragment>"}` — inside a think block (think-style templates only).
3. `{"content":"<fragment>"}` — answer text. `logprobs` rides the chunk carrying those tokens.
4. Tool-call chunks (when a call was parsed):
   - Opening: `{"tool_calls":[{"index":0,"id":"...","type":"function","function":{"name":"...","arguments":""}}]}`
   - Arguments: `{"tool_calls":[{"index":0,"function":{"arguments":"..."}}]}`
5. Closing chunk: `{"delta":{},"finish_reason":"..."}`. Carries `tool_call_error`
   on the choice and `response_format_status` top-level when applicable.
6. With `stream_options.include_usage`: a chunk with `"choices":[]` and `usage`.
7. Terminal sentinel: `data: [DONE]`

With `n` > 1 the choices stream one after another, told apart by `choices[].index`.
A client that disconnects aborts the turn and frees its slot.

Errors detected after the stream opened cannot change the status code:
`context_over_budget` is sent as a terminal event
`data: {"error":{"message":"...","type":"context_over_budget","code":null,"kv_used":N,"kv_cap":M}}`
and the stream ends; other generation failures close the choice with
`finish_reason: "stop"`.

### Error responses

| Status | `type` | Condition |
|--------|--------|-----------|
| 400 | `invalid_request_error` | Empty body, malformed JSON, body not an object, `messages` missing or empty |
| 404 | `model_not_found` | Several models loaded and `model` matches none |
| 422 | `context_over_budget` | Prompt exceeds the effective per-session window. Not transient: shorten the conversation and retry. Non-streaming: `kv_used`/`kv_cap` are in the message text; streaming: they are fields of the error object. |
| 429 | `server_busy` | Single-user path: a generation is already running on this model. Batched path: all slots busy and the queue full, or 30 s queue wait timed out. `Retry-After` when the scheduler supplies one. |
| 429 | `model_at_session_capacity` | A new `session_id` while every session slot (`max_sessions`) holds a live session. `Retry-After: 30`; the message estimates the re-prefill time. |
| 503 | `no_models_loaded` | No model loaded |
| 503 | `server_memory_constrained` | Host memory pressure: admission refused the prompt, or the queue is full under critical pressure |
| 503 | `server_shutting_down` | Server is draining for shutdown. `Retry-After: 1` when refused at the route. |
| 500 | `server_error` | Generation failed / out of memory |

Source: `openai_chat.c:oai_handle_chat`, `serve_generation.c:sg_run_completion`,
`serve_command.c:dispatch`.

---

## POST /admin/models/load

Operator-only. Load a model at runtime (Phase 7). Serialized against other
uploads via `reg->upload_lock` (one GPU upload in flight at a time).

### Request body

```json
{
  "path":             "/models/my-model.gguf",
  "name":             "my-model",
  "dies":             [0, 1],
  "quantize":         false,
  "ctx_cap":          32768,
  "max_sessions":     4,
  "max_concurrent":   2,
  "reasoning_budget": 1024
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `path` | string | yes | GGUF file path |
| `name` | string | no | Served id (defaults to file basename) |
| `dies` | int array | no | Device ordinals (same `CUDA_VISIBLE_DEVICES`-relative semantics as `--dies`). Max `PN_MODEL_MAX_DIES=8`. |
| `quantize` | bool | no | INT8-quantize at load time |
| `ctx_cap` | int | no | Per-session context cap in tokens; a prompt past it is refused with `context_over_budget`. `0`/absent = engine default. Mirrors `--ctx-cap`. |
| `max_sessions` | int | no | **Residency** — how many named sessions may hold KV at once. Over the cap a new session id gets `429 model_at_session_capacity`. `0`/absent = engine default. Mirrors `--max-sessions`. |
| `max_concurrent` | int | no | **Concurrency** — how many requests may decode simultaneously (batch-scheduler slots). Surplus queues, then `429`/`503`. Distinct from `max_sessions`. `0`/absent = the resolved `max_sessions`. Mirrors `--max-concurrent`. |
| `reasoning_budget` | int | no | Force `</think>` after N tokens of unclosed reasoning. `0`/absent = unbounded. Mirrors `--reasoning-budget`. |

Every per-model option available on the `serve` command line is settable here, so a model
added at runtime ends up configured identically to one launched with the process.

### Response — 200

```json
{
  "id": "my-model",
  "object": "model",
  "path": "/models/my-model.gguf",
  "dies": [0, 1],
  "vram_bytes": 12345678,
  "max_sessions": 4,
  "max_concurrent": 4,
  "context_window": 32000,
  "context_window_configured": 32000,
  "graphs": {
    "live": 30, "total": 40, "failed": 0, "gave_up": 0,
    "ineligible": 10, "pending": 0,
    "captures": 30, "capture_failures": 0,
    "batched_attn": {"live": 10, "total": 10, "failed": 0, "ineligible": 0, "pending": 0},
    "batched_ssm":  {"live": 20, "total": 30, "failed": 0, "ineligible": 10, "pending": 0}
  },
  "attention": {
    "ondie": 10, "host": 0, "total": 10,
    "gave_up": 0, "impossible": 0,
    "host_engagements": 0, "ring_alloc_failures": 0
  }
}
```

The operator surface carries **both** context numbers, because an operator's
question is not the client's: `context_window` is what sessions actually get right
now, `context_window_configured` is what `--ctx-cap` asked for. They are equal
unless the memory ladder is clamping this instant — and seeing them differ is
exactly the signal that memory pressure is costing context. (`/v1/models`
advertises only the effective one; a client has no use for a window it cannot
have.) Both are omitted when unknown.

Falls back to `{"status":"loaded"}` if the row cannot be found in the snapshot (race).

### Error responses

| Status | `type` | Condition |
|--------|--------|-----------|
| 400 | `invalid_request_error` | Missing `path`, malformed JSON, too many dies |
| 400 | `load_failed` | `dies` named on a serve with no accelerator backend compiled in, or with no dies visible. Not retryable — the fix is a different binary (`pantheon --backend`) or an omitted `dies`. |
| 409 | `load_failed` | Model already loaded / already unloading |
| 507 | `load_failed` | Admission refused (VRAM or host), die binding conflict, registry full |
| 404 | `load_failed` | File not found |
| 500 | `load_failed` | Other load failure |

Source: `model_admin.c:ma_handle_load`.

---

## POST /admin/models/unload

Operator-only. Unload a model by served name. Drains any in-flight generation
before freeing.

### Request body

```json
{"name": "my-model"}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `name` | string | yes | Served model id to unload |

### Response — 200

```json
{"unloaded": "my-model"}
```

### Error responses

| Status | `type` | Condition |
|--------|--------|-----------|
| 400 | `invalid_request_error` | Missing `name`, malformed JSON |
| 404 | `unload_failed` | Name not loaded |
| 409 | `unload_failed` | Already unloading |
| 500 | `unload_failed` | Other failure |

Source: `model_admin.c:ma_handle_unload`.

---

## GET /admin/models

Operator-only. Every **live instance**, one row each. Unlike `/v1/models` these
are **not** deduplicated by name: two instances of one model on different dies
are two rows, because to an operator they are two different things.

The row is the same one `POST /admin/models/load` returns — this route exists so
its operator-only fields can be read at any time rather than only as a side
effect of loading a model.

### Response — 200

```json
{
  "object": "list",
  "data": [
    {
      "id": "my-model",
      "object": "model",
      "path": "/models/my-model.gguf",
      "dies": [2, 3],
      "vram_bytes": 12345678,
      "max_sessions": 4,
      "max_concurrent": 4,
      "max_concurrent_configured": 4,
      "context_window": 32000,
      "context_window_configured": 32000,
      "graphs": { "live": 30, "total": 40, "failed": 0, "gave_up": 0,
                  "ineligible": 10, "pending": 0,
                  "captures": 30, "capture_failures": 0,
                  "batched_attn": {"live": 10, "total": 10, "failed": 0,
                                   "ineligible": 0, "pending": 0},
                  "batched_ssm": {"live": 20, "total": 30, "failed": 0,
                                  "ineligible": 10, "pending": 0} },
      "attention": { "ondie": 8, "host": 2, "total": 10,
                     "gave_up": 2, "impossible": 0,
                     "host_engagements": 2, "ring_alloc_failures": 10,
                     "reasons": {"ring_alloc_failed": 2} }
    }
  ]
}
```

The operator row carries **both** concurrency numbers: `max_concurrent` is what
the model serves right now (`bs_get_load` capacity, the same value `/v1/models`
advertises) and `max_concurrent_configured` is what `--max-concurrent` asked for.
They differ when there is no batch scheduler at all (no CUDA compiled in,
`QWEN35_BATCH_SLOTS` off, `bs_create` failed) or when the memory-pressure ladder
clamped the slot count — the same split, for the same reason, as
`context_window` vs `context_window_configured`.

The operator `graphs` object extends the `/v1/models` one with `gave_up`, the
lifetime `captures` / `capture_failures` counters, and a per-class split:
`batched_attn` and `batched_ssm` (the multi-user decode path), `single_ssm` (the
one-user per-layer path) and `segment` (the per-die full-step graph). A class
the model has never exercised is omitted rather than reported as an empty
deficit. Field meanings: see `graphs` under `GET /v1/models`.

The operator `attention` object adds `gave_up` / `impossible`, the lifetime
`host_engagements` / `ring_alloc_failures` counters, and `reasons` — a
histogram, present only when `host > 0`, naming why each layer is off-die
(`ring_alloc_failed`, `dec_scratch_alloc_failed`, `kv_geometry_or_ring_wrap`,
`no_slot_decode_kernel`, `no_flash_kernels`, `no_batch_scratch`,
`flash_slots_off`). Field meanings: see `attention` under `GET /v1/models`.

No error cases (returns an empty `data` array when nothing is loaded).

Source: `model_admin.c:ma_handle_models`.

---

## GET /admin/dies

Operator-only. Snapshot of all probed accelerator dies, their VRAM budget/used,
and which model occupies them.

### Response — 200

```json
{
  "object": "list",
  "n_dies": 8,
  "data": [
    {
      "die": 0,
      "card": 0,
      "model": "my-model",
      "n_models": 1,
      "vram_budget": 12884901888,
      "vram_used": 11674869760
    }
  ]
}
```

| Field | Type | Notes |
|-------|------|-------|
| `die` | int | Die manager ordinal (0-indexed across all probed dies) |
| `card` | int | Physical card index (`pn_dm_card_of(die)`) |
| `model` | string | Name of the model occupying this die, or empty string |
| `n_models` | int | Number of models on this die (typically 0 or 1) |
| `vram_budget` | int (bytes) | Die's total VRAM minus `PN_DIE_RESERVE_BYTES` |
| `vram_used` | int (bytes) | Bytes allocated by the model manager |

Source: `model_admin.c:ma_handle_dies`.

---

## Session routes

A session is a live prefix-cache / decode slot keyed by `session_id`. In each
route below the session id comes from the body's `session_id`, else the
`X-Session-Id` header (export reads the body only). `name` selects the model;
absent → the default model. Unknown model → `404 model_not_found`.

One session id may carry several conversations (opencode sends its title
request and its real turns under one id). Each keeps its own cache: a turn
resumes from the conversation its prompt continues. `max_sessions` counts ids,
not conversations. On the single-user path (no batch scheduler) export takes the
id's most recently used conversation, and restore replaces the conversation the
restored `tokens` continue (else adds one); release frees all of them.

### POST /admin/session/export

Operator. Serialize a live session's state so it can be restored later.

Request: `{"name": "my-model", "session_id": "abc123"}` (`session_id` required).

Response — 200:

```json
{"session_id":"abc123","weights_version":123,"n_tok":4096,
 "blob_bytes":31457280,"blob_b64":"<base64>","tokens":[...]}
```

`tokens` is the committed history the blob covers; it must be sent back on
restore.

| Status | `type` | Condition |
|--------|--------|-----------|
| 400 | `invalid_request_error` | Empty/malformed body, missing `session_id` |
| 400 | `session_export_unsupported` | The model's architecture has no session snapshot format |
| 400 | `session_caching_disabled` | The model has no session cache |
| 404 | `session_not_found` | No live session with that id |
| 429 | `session_busy` | A turn is in flight on that session; retry after it completes |
| 500 | `server_error` | Serialize / out-of-memory failure |

Source: `model_admin.c:ma_handle_session_export`.

### POST /session/restore

Install an exported session into a live slot.

Request:

```json
{"name":"my-model","session_id":"abc123","blob":"<blob_b64 from export>","tokens":[...]}
```

`session_id`, `blob` (base64) and `tokens` are required. The blob must come from
the same model architecture and the same weights.

Response — 200: `{"restored":true,"session_id":"abc123","n_tok":4096}`

| Status | `type` | Condition |
|--------|--------|-----------|
| 400 | `invalid_request_error` | Empty/malformed body, missing `session_id`/`blob`/`tokens` |
| 400 | `session_caching_disabled` | The model has no session cache |
| 400 | `bad_session_blob` | Not base64, corrupt, wrong architecture, or does not fit this model |
| 409 | `weights_version_mismatch` | Blob captured under different weights — replay the conversation instead |
| 429 | `model_at_session_capacity` | No free session slot |
| 500 | `server_error` | Restore failed |

Source: `openai_chat.c:oai_handle_session_restore`.

### POST /session/release

Free a session's slot now instead of waiting for the idle timeout (12 h default,
`PN_SESSION_IDLE_TTL_MS`).

Request: `{"name": "my-model", "session_id": "abc123"}`

Response — 200: `{"released":true,"session_id":"abc123"}`. Idempotent: an unknown
or already-freed session also returns 200.

Errors: `400 invalid_request_error` (empty/malformed body, no session id),
`404 model_not_found`.

Source: `openai_chat.c:oai_handle_session_release`.

### GET /session/estimate

`GET /session/estimate?tokens=N[&name=M][&blob_bytes=B][&upload_bps=R]`

Arithmetic estimate of restoring a session from a blob vs re-prefilling it.
`tokens` (> 0) is required. Without `blob_bytes`, the blob is estimated at
16 KiB per token; without `upload_bps` (bytes/s), 12.5 MB/s is assumed.

Response — 200:

```json
{"model":"my-model","tokens":4096,"prefill_tok_per_s":17.00,
 "cold_reprefill_s":240.94,"warm_restore_s":5.391,"blob_bytes_est":67108864}
```

`prefill_tok_per_s` is the serve's measured prefill rate (a fixed fallback
before one is measured). An unknown `name` is not an error; `model` is then `""`.

Errors: `400 invalid_request_error` when `tokens` is missing or not positive.

Source: `openai_chat.c:oai_handle_session_estimate`.

### GET /session/watch

`GET /session/watch?session_id=S[&name=M]` (or `X-Session-Id`).

Liveness hold. On success the server answers `200` with
`Content-Type: text/event-stream`, a `: watch established` comment, and keeps the
connection open. While it is held the session is never reaped for idleness;
when it closes, the session is released after a grace period (45 s default,
`PN_SESSION_WATCH_GRACE_MS`) unless a new watch for it arrives first.

| Status | `type` | Condition |
|--------|--------|-----------|
| 400 | `invalid_request_error` | No session id |
| 400 | `session_caching_disabled` | The model has no session cache |
| 404 | `model_not_found` | Unknown model |
| 503 | `watch_unavailable` | Watch manager not running, or its table is full |

Source: `serve_command.c:handle_session_watch`, `session_watch.c`.

---

## Error body format

All error responses use:

```json
{"error":{"message":"<text>","type":"<code>","code":null}}
```

The `code` field is always `null`.

---

## Concurrency and connection model

- One detached pthread per accepted TCP connection. `accept()` never blocks
  behind a generation.
- On `pthread_create` failure the connection is handled inline (no request
  is dropped).
- SIGPIPE is ignored so a client disconnect mid-stream cannot kill the server.
- Generation serialization:
  - **Legacy path** (`sched == NULL`): `gen_lock` trylock per model. Second
    concurrent request to the same model → immediate 429.
  - **Batched path** (`QWEN35_BATCH_SLOTS=1`): the C3 scheduler owns
    concurrency. Up to 8 concurrent users; overflow waits in a per-model FIFO
    queue (default depth 32, `BS_QUEUE_DEPTH`). 30-second queue wait timeout.
