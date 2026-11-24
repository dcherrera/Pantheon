# Serve Layer Architecture

Pantheon's serve layer implements an OpenAI-compatible HTTP inference server.
Entry point: `src/serve/serve_command.c`. The serve layer sits above the engine
(`src/core/engine.c`) and is reached via `pantheon serve`.

---

## Overview

At startup `pn_serve_main` (`serve_command.c:206`) parses one or more `--model`
flags, loads each model into an `mr_registry`, binds a TCP socket, and enters an
infinite `accept()` loop. Every accepted connection is handed to a detached
`pthread` (`spawn_connection`, `serve_command.c:185`); if `pthread_create` fails
it falls back to handling the request inline so no connection is ever silently
dropped.

The connection thread calls `handle_connection` → `dispatch`. Dispatch is a
straight O(route-count) `strcmp` chain (`serve_command.c:124`):

| Method | Path | Handler |
|--------|------|---------|
| `OPTIONS` | any | CORS preflight: `hs_send_json 200 {}` |
| `GET` | `/v1/models` | `oai_handle_models` |
| `POST` | `/v1/chat/completions` | `oai_handle_chat` |
| `POST` | `/admin/models/load` | `ma_handle_load` |
| `POST` | `/admin/models/unload` | `ma_handle_unload` |
| `GET` | `/admin/dies` | `ma_handle_dies` |
| `GET` | `/` | health: `{"service":"pantheon","status":"ok"}` |

`GET /v1/models` and admin routes never touch any generation lock; they can
respond concurrently with an in-flight generation on any model.

### Request lifecycle (socket → SSE)

```
accept()
  └─ spawn detached pthread
       └─ hs_read_request()   (http_server.c — reads headers + body)
            └─ dispatch()
                 └─ oai_handle_chat()
                      ├─ jr_parse(body)
                      ├─ parse_chat_request()   (extracts model, params, session_id)
                      ├─ route_model()           (mr_acquire → entry + refcount++)
                      │
                      ├─ [batched path: QWEN35_BATCH_SLOTS=1]
                      │    ├─ bs_slot_acquire_or_wait()   (blocks up to 30 s)
                      │    └─ sg_run_completion_batched() → SSE
                      │
                      └─ [single-user path: sched == NULL]
                           ├─ pthread_mutex_trylock(&entry->gen_lock) → 429 if busy
                           └─ sg_run_completion() → SSE
                                └─ mr_release(reg, entry)  (refcount--)
```

---

## Model Registry and Routing

**Files:** `src/serve/model_registry.h`, `src/serve/model_registry.c`

### Data structures

```c
/* model_registry.h:54 */
typedef struct {
    char             name[MR_MAX_NAME];   /* served model id */
    pn_model        *model;
    pn_state        *state;               /* single-user state (sched==NULL) */
    pthread_mutex_t  gen_lock;            /* single-user generation serializer */
    pn_batch_sched  *sched;               /* C3: non-NULL when batching is on */
    pn_prefix_cache *prefix_cache;        /* PC1: session prefix cache */
    pthread_mutex_t  pc_lock;             /* serializes prefix-cache ops */
    int              dies[PN_MODEL_MAX_DIES];
    int              n_dies;              /* 0 = CPU-resident */
    uint64_t         vram_bytes;
    int              refs;                /* in-flight acquires */
    bool             unloading;
    bool             used;
} mr_entry;

/* model_registry.h:86 */
typedef struct {
    mr_entry        entries[MR_MAX_MODELS];  /* MR_MAX_MODELS = 16 */
    int             n;
    pthread_mutex_t lock;
    pthread_cond_t  drained;     /* signaled on every mr_release */
    pthread_mutex_t upload_lock; /* A3: one GPU weight upload at a time */
} mr_registry;
```

### Loading (`mr_load`, `model_registry.c:90`)

Load is a four-phase operation designed to keep the registry serviceable
throughout:

1. **Slot reservation** (under `reg->lock`): a slot is marked `used=true` with
   no model pointer yet — the name is taken but the entry is not yet acquirable
   (`require_ready` skips it in `find_locked`).
2. **Placement + VRAM admission** (lock-free): auto-placement via the die manager
   (`pn_dm_place`), or explicit dies from `--dies`. `pn_admit_model_load` gates
   against the per-die budget using a conservative 8 k context guess
   (`MR_ADMIT_CTX_GUESS = 8192`, `model_registry.c:36`). `pn_dm_bind` commits the
   reservation.
3. **Weight upload** (under `reg->upload_lock`): `pn_load_on` runs while the
   registry stays fully serviceable. `upload_lock` ensures at most one GPU upload
   is in flight at a time, preventing two concurrent 2.4 GB transients stacking
   on an 8 GB host (`model_registry.c:213`).
4. **Publish** (under `reg->lock`): model, state, locks, and die info are written
   atomically and `reg->n` is incremented. After publish, the batch scheduler
   (`bs_create`) and prefix cache (`pn_prefix_cache_new`) are created and
   published under the lock. Both are fail-soft: failure leaves the model in
   single-user mode or full-prefill mode respectively.

### Acquire / release refcounting

```c
mr_entry *mr_acquire(mr_registry *reg, const char *name);  /* model_registry.c:322 */
void      mr_release(mr_registry *reg, mr_entry *entry);   /* model_registry.c:358 */
```

`mr_acquire` locks `reg->lock`, scans entries, increments `refs`, and returns
the pointer. Every user of an entry MUST pair `mr_acquire` with `mr_release`.
`mr_release` decrements `refs` and broadcasts `reg->drained`; `mr_unload` waits
on that condvar until `refs == 0` before freeing.

### Multi-card load-balancing (`model_registry.c:329`)

An operator may place one model id on several cards (`--dies 4,5` and `--dies
6,7` both under `--name my-model`). `mr_acquire` finds all ready entries that
share the requested name and picks the one with the fewest in-flight `refs` —
round-robin load balance at the reference granularity:

```c
for (int i = 0; i < MR_MAX_MODELS; i++) {
    mr_entry *c = &reg->entries[i];
    ...
    if (!e || c->refs < e->refs)
        e = c;
}
```

Empty/null name resolves to the first (lowest slot index) resident entry —
the back-compat single-model shortcut (`model_registry.c:344`).

### Route model (`openai_chat.c:499`)

```c
static mr_entry *route_model(int client_fd, mr_registry *reg, const char *name)
{
    mr_entry *entry = (mr_count(reg) <= 1) ? mr_acquire(reg, NULL)
                                           : mr_acquire(reg, name);
    ...
}
```

With exactly one resident model, every request routes to it regardless of the
`model` field — preserving single-model server behavior. With more than one,
an exact name match is required; an unknown name returns 404, no loaded model
returns 503.

### `/v1/models` dedupe (`openai_chat.c:266`)

`oai_handle_models` calls `mr_snapshot` to get all resident entries, then
dedupes by name before serializing JSON — a model placed on two cards appears
as one logical id:

```c
bool seen = false;
for (int j = 0; j < i; j++)
    if (strcmp(info[j].name, info[i].name) == 0) { seen = true; break; }
if (seen) continue;
```

### Unload (`mr_unload`, `model_registry.c:369`)

Marks `unloading=true` (blocking new acquires), decrements `reg->n`, then
blocks on `reg->drained` until `refs == 0`. Detaches all pointers under the
lock and frees them outside: prefix cache → batch scheduler → die binding →
state → model.

---

## Batched Decode (`batch_sched.c`)

**Feature gate:** `QWEN35_BATCH_SLOTS=1` (env var, CUDA build only).
`bs_enabled()` (`batch_sched.c:44`) caches the result.

When enabled, `mr_load` calls `bs_create(model, BS_MAX_USERS)` where
`BS_MAX_USERS = 8` (`batch_sched.h:173`). The scheduler owns 8 `pn_state*`
instances (one per slot) and a background driver thread.

### Slot lifecycle

```
IDLE -(bs_slot_acquire/acquire_or_wait)-> PREFILL
     -(bs_slot_bind / bs_slot_bind_ex)-> BOUND
     -(driver first step)--------------> DECODE
     -(bs_slot_release)----------------> IDLE
```

Only the owner thread moves a slot back to IDLE. The driver marks `finished=true`
on EOS/budget but never calls `bs_slot_release` — it always signals the owner
to drain and release.

### GPU-ownership turnstile (`batch_sched.c:72`)

The model's single-user forward machinery (C2 async phase machine, shared live
SSM buffers, shared GPU KV ring) supports exactly one in-flight GPU forward at
a time. Pre-C3 the per-model `gen_lock` enforced this. The batch scheduler uses
a FIFO ticket turnstile instead — `gpu_ticket_next` / `gpu_ticket_serving` /
`cv_gpu` — which prevents the free-running L3 driver from barging ahead of a
waiting prefiller:

```c
static void bs_gpu_acquire_locked(pn_batch_sched *sched) {  /* batch_sched.c:72 */
    uint64_t my = sched->gpu_ticket_next++;
    while (sched->gpu_ticket_serving != my)
        pthread_cond_wait(&sched->cv_gpu, &sched->mu);
}
```

A turn owner never `cond_wait` (chunks and steps are bounded GPU work), so
the turnstile is deadlock-free by construction.

### D1 overflow queue + D3 admission backpressure

`bs_slot_acquire_or_wait` (`batch_sched.c:1475`, called from `openai_chat.c:577`
with a 30 s timeout) replaces the immediate 429:

- **Fast path:** slot free AND `pressure_level < BS_PRESSURE_HOLD_LEVEL` (3 =
  CRITICAL) → acquire immediately.
- **Slow path:** enqueue (`n_queued++`) and block on `cv_idle`.
  - Queue full + pressure < CRITICAL → 429 + `Retry-After: 5`.
  - Queue full + CRITICAL → 503.
  - Timeout after `wait_ms` → 429 + `Retry-After: 2`.
  - Shutdown → 503.

`BS_QUEUE_DEPTH_DEFAULT = 32` (`batch_sched.h:274`). Override with `BS_QUEUE_DEPTH`.

`bs_notify_pressure` (`batch_sched.c:1589`) is called by the serve path before
every `bs_slot_acquire_or_wait` (`openai_chat.c:569`) to push the live
`pn_mem_state` into the scheduler, so the gate sees the current host state at
decision time.

### D2 — continuous batching / dynamic-width step

The driver compacts the active (BOUND/DECODE) slots into GPU indices
`[0, n_active)` each step. Idle slots pay zero GPU compute. When `active_set_changed`
is set (join, leave, or finish), `rebuild_gpu_map` (`batch_sched.c:237`) runs
a greedy permutation algorithm that handles arbitrary reorderings — including
cycles that require a scratch slot (`cap` index, `batch_sched.c:279`). This
fixed the K≥5-concurrent-joiners bug where the old downward-only move refusal
caused all active users to emit empty content.

### L3 — in-driver sampling (default on)

`bs_slot_bind_ex` (`batch_sched.c:1294`) passes the request's sampling params,
repeat-penalty window, and token budget to the driver. After each `decode_slots`
step the driver samples every slot itself — sequentially, deterministically —
and pushes the result to a per-slot output ring sized to the full budget:

```c
uint32_t next = pn_sample_next_token(lo[g], sched->n_vocab,
                                     &sl->sp, sl->recent, sl->n_recent);
sl->ring[sl->ring_w++] = next;   /* batch_sched.c:545 */
```

Owners drain the ring with `bs_slot_pop_token` and do detokenize/emitter/SSE
entirely off the step's critical path. A slow or stalled client cannot stall
the rest of the batch.

The ring is never resized after bind (`ring_cap == budget`), so it can never
overflow and the driver never blocks on a slow consumer.

Stop-string detection stays owner-side (text level); a slot may over-generate
past a stop string until its owner drains and releases — surplus ring tokens
are discarded.

Disable with `QWEN35_BS_DRIVER_SAMPLE=0` to restore legacy lockstep
logits/token exchange.

### L2 — chunked prefill with yield windows

Prefill runs in chunks of `QWEN35_BS_PREFILL_CHUNK` tokens (default 128).
After each chunk (except the last) `bs_prefill_yield` (`batch_sched.c:1040`)
makes the slot's host mirror authoritative via `gpu_state_to_host`, releases
the GPU turn, and re-acquires it FIFO — giving the driver one decode step
between chunks. This bounds a joiner's stall on concurrent decoders to one
chunk instead of one full prompt. It also yields when a finished slot is queued
in `bs_slot_finalize` (`n_finalize_waiting`): that owner sends its reply only
after finalize, so it waits one chunk, not the rest of the other prompt. A
queued prefill alone does not trigger a yield.

Every batched request ends with one `[req]` line in the serve log
(`request_timing_line.h`): wall time, prompt / prefilled tokens, TTFT from
request arrival, decode tok/s over the request's own decode time, the
active-slot range it decoded in, whether MTP ran, and the finalize wait.

### Single-user fallback

When `QWEN35_BATCH_SLOTS` is not set (or the build has no CUDA), `entry->sched`
is `NULL`. The serve path (`openai_chat.c:658`) falls back to:

```c
if (pthread_mutex_trylock(&entry->gen_lock) != 0) {
    hs_send_error(client_fd, 429, "server_busy",
                  "a generation is already in progress on this model; "
                  "retry shortly");
    ...
}
```

A second concurrent request to the same model gets an immediate 429. Requests
to different models run concurrently (each holds only its own entry's lock).

---

## Generation and Streaming (`serve_generation.c`)

### Prompt building

`build_prompt` (`serve_generation.c:120`) tokenizes the full `messages[]` array
into ChatML format using the model's tokenizer:
`<|im_start|>role\ncontent<|im_end|>\n` for each turn, with `tool` messages
wrapped in `<tool_response>`. Opens the assistant turn and prefills `<think>`
for reasoning models (`PN_CHAT_CHATML_THINK`).

### Admission

Before any prefill, `pn_admit` checks live host + die memory and either passes,
caps the context (sliding-window oldest-token drop), or denies. A denied
admission returns `SG_ADMIT_DENIED`, which the serve path converts to a 503
(`openai_chat.c:683`). A capped admission sets `result->context_capped = 1` and
disables prefix caching for that turn; the client sees `"context_capped": true`
on the response.

### Single-user completion (`sg_run_completion`, `serve_generation.c:525`)

```
build_prompt → pn_admit → [prefix cache acquire] → prefill chunks
  → sample first token → decode loop (sg_forward_step per token)
  → [prefix cache commit] → return sg_result
```

`sg_forward_step` (`serve_generation.c:411`) uses an async non-blocking launch +
event-poll when the model supports it (`pn_forward_step_begin/poll`), sleeping
50 µs → 500 µs between polls. This allows up to 8 per-card connection threads
to run their cards simultaneously from a 2-core host without blocking each other.

### Batched completion (`sg_run_completion_batched`, `serve_generation.c:744`)

Same flow as single-user but:
- Prefill takes the GPU turn via `bs_prefill_begin` or `bs_prefill_resume`.
- Chunked prefill with `bs_prefill_yield` windows (L2).
- After the last chunk, `bs_slot_bind_ex` (L3) or `bs_slot_bind` (legacy) hands
  the slot to the driver.
- L3 path: drain the ring with `bs_slot_pop_token` → detokenize → emitter → SSE,
  all off the critical path.
- `bs_slot_finalize` (if caching) then `bs_slot_release`.

### Streaming (SSE)

`run_streaming` / `run_streaming_batched` (`openai_chat.c:434,464`) call
`hs_begin_sse` and register `stream_delta_cb` as the `delta_fn`. The callback is
invoked by the emitter for each text fragment:

```c
static void stream_delta_cb(const char *text, int is_reasoning, void *ud) {
    /* openai_chat.c:397 */
    jw_raw(&jw, is_reasoning ? "\"reasoning_content\":" : "\"content\":");
    jw_string(&jw, text);
    send_chunk(sc, jw.data, NULL);  /* hs_send_sse */
}
```

A final chunk carries `finish_reason`; then `hs_end_sse` writes `data: [DONE]`.

### Reasoning / stop-sequence emitter

The `emitter` struct (`serve_generation.c:179`) handles three channels:
- **in_think**: text before `</think>` → `reasoning_content`; switches to content
  channel on `</think>`.
- **content**: after `</think>`, before a stop string or `<tool_call>`.
- **in_tool**: raw text after `<tool_call>` → captured in `tool` channel for
  structured parsing by `tool_calling.c`.

A `carry` buffer holds the longest suffix that could start any marker or stop
string across a token boundary.

### Prefix cache (`src/core/prefix_cache.c`, used via `serve_generation.c:574`)

Keyed by `session_id` (from body field `session_id` or header `X-Session-Id`).
Single-user path: `pn_prefix_cache_acquire` returns the session's `pn_state` at
the delta boundary; on commit, `pn_prefix_cache_commit` records the full token
history. The `pc_lock` is not needed on the single-user path (`gen_lock` already
serializes), so `pcx.pc_lock = NULL` (`openai_chat.c:674`).

Batched path (PC-batched): session affinity is slot-resident. `bs_slot_acquire_biased`
prefers the slot that already holds this session's warm state. `bs_prefill_resume`
checks if the slot's committed history (`hist`) is an exact prefix of the new
prompt and skips the already-committed tokens. `bs_slot_finalize` snapshots the
GPU state to host and publishes the driver-maintained `ct` (committed-token log)
as `hist` for the next turn.

---

## Concurrency Model

| Lock | Scope | What it protects |
|------|-------|-----------------|
| `reg->lock` | `mr_registry` | slot table (used/refs/unloading/n); brief critical sections only |
| `reg->drained` | condvar on `reg->lock` | unload drain wait |
| `reg->upload_lock` | `mr_registry` | one GPU weight upload in flight at a time |
| `entry->gen_lock` | `mr_entry` | single-user generation serializer (trylock → 429) |
| `entry->pc_lock` | `mr_entry` | prefix cache ops on the single-user path (unused — gen_lock covers it; kept for the batched concurrent-slot case) |
| `sched->mu` | `pn_batch_sched` | ALL slot state: status words, token/logits flags, rings, turnstile counters, D1 queue depth, D3 pressure level |
| `sched->cv_gpu` (turnstile) | under `sched->mu` | one GPU forward in flight at a time (prefill chunk, bind, or decode step) |
| `sched->cv_work` | under `sched->mu` | driver wake (gate may have opened) |
| `sched->cv_idle` | under `sched->mu` | D1 waiter wake (slot freed or pressure dropped) |
| `sl->cv` | per-slot, under `sched->mu` | driver → owner wakeup (logits/ring/finished) |

Connection threads are otherwise independent: each owns its own
`hs_request`, `chat_request`, `sg_result`, and `emitter`. The sampler scratch
(`pn_sample_next_token`) is `_Thread_local`.

The `accept()` loop is single-threaded but non-blocking per iteration. Heavy
work (parsing, prefill, decode, SSE write) runs entirely in per-connection
threads, so cheap routes (`/v1/models`, `/`) are never queued behind an
in-flight generation.

---

## Admin API (`model_admin.c`)

| Route | Body | Action |
|-------|------|--------|
| `POST /admin/models/load` | `{"path":"..","name":"..","dies":[..],"quantize":false}` | `mr_load` — may block ~55 s for a GPU upload |
| `POST /admin/models/unload` | `{"name":".."}` | `mr_unload` — drains in-flight refs then frees |
| `GET /admin/dies` | — | `pn_dm_get` snapshot of all dies (budget/used/model) |

Error codes from `status_for_reason` (`model_admin.c:28`): admission/table-full
→ 507, duplicate/unloading → 409, not-found → 404, other → 500.

A runtime `/admin/models/load` arriving while a startup upload is in flight
blocks on `reg->upload_lock` and is refused promptly by the slot scan once the
prior load finishes — never stacks two 2.4 GB transients.

---

## Router Registration (`serve_registry.c`)

**Gating:** fully off unless `PANTHEON_REGISTRY_URL` is set. `rc_enabled()` is checked at
`serve_registry_start` entry; with it unset the function returns immediately and the
serve process is byte-for-byte unchanged (`serve_command.c:350`, `serve_registry.c:78`).

When enabled, at startup `serve_registry_start` (`serve_registry.c:76`) iterates every
loaded model (via `mr_snapshot`) and calls `rc_register` for each with:

```
model_id    = info[i].name          /* the shared served identity */
instance_id = host:port:model       /* serve_registry.c:98 — unique per process */
endpoint    = http://host:port
capacity    = batch-slot count (1 if single-user)
```

A background heartbeat thread then re-acquires each entry every
`RW_HEARTBEAT_SECONDS` to read live load (`bs_get_load`) and POST it to the router.

### What this means for in-process replicas

Two replicas of `ornith` on dies 2 and 3 in **the same process** share the same
`host:port`, so they produce the same `instance_id = host:port:ornith` and **collapse
to one instance** at the router. The router sees one `ornith` endpoint, not two.

This is correct and intentional: **in-instance replica load-balancing is handled
entirely by `mr_acquire`** (least-busy across same-name entries — see above). The router
is the **multi-instance dispatch layer** for tying separate `pantheon serve` processes
together. A single self-routing instance does not need it.

> Cross-link: docs/README.md "Multi-Model Smart Routing" — "one instance is fully
> self-routing and needs nothing in front of it."

---

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `QWEN35_BATCH_SLOTS` | off | `1` enables multi-user batched decode (C3) |
| `QWEN35_BS_DRIVER_SAMPLE` | `1` | `0` restores legacy owner-side lockstep sampling |
| `QWEN35_BS_PREFILL_BATCH` | `1` | `0` disables batched chunk prefill (token-by-token) |
| `QWEN35_BS_PREFILL_CHUNK` | `128` | chunk size in tokens for L2 yield windows |
| `BS_QUEUE_DEPTH` | `32` | max D1 queue depth before 429 |
| `QWEN35_BS_STATS` | off | `1` logs per-step timing to stderr |
| `QWEN35_KV_Q4` | `1` | `0` disables Q4 KV quantization in admission math |
| `QWEN35_GPU_FULL_LOGITS` | off | Refused at serve startup — forces 248k-vocab D2H copy |
| `PN_ASYNC_POLL_US` | `500` | Max sleep between async forward-step event polls (µs) |
| `PANTHEON_REGISTRY_URL` | unset | Set to enable router registration (`serve_registry.c:78`); unset = no-op |
| `PANTHEON_NODE_ID` | `node0` | Node identifier sent in router registration (`serve_registry.c:83`) |
| `PANTHEON_REGISTRY_TRANSPORT` | `local` | Transport hint sent to router on registration (`serve_registry.c:84`) |

---

## Known Sharp Edges

- **`mr_unload` blocks its caller** until all in-flight `refs` drain
  (`model_registry.c:389`). A long generation on the model being unloaded will
  hold the admin response for its full duration. There is no timeout.

- **Duplicate names are explicitly allowed** (`model_registry.c:119`). An
  operator placing the same model id on cards 4 and 5, then unloading by name,
  will unload whichever slot `find_locked` finds first — the other remains. There
  is no unload-by-slot-index API; the operator must call unload twice.

- **`make_completion_id` uses `rand()`** without seeding (`openai_chat.c:87`).
  `rand()` is not seeded with `srand()`, so id generation is not
  cryptographically random and starts from the same sequence on every process
  restart. Collision risk is low (hex8 = 4B permutations) but completions ids
  are not guaranteed unique across restarts.

- **`pc_lock` is allocated but never taken on the single-user path**
  (`openai_chat.c:674`, `pcx.pc_lock = NULL`). On the batched path each slot
  owns its own `hist`/`ct` structures guarded by `sched->mu`, so `pc_lock` is
  also unused there. The lock exists as a slot for a future concurrent-slot
  single-cache design.

- **Stop-string detection fires owner-side** after the driver has already
  stepped tokens into the ring (L3 path). The driver steps until EOS/budget;
  the owner drains until a stop string is detected in the decoded text.
  Over-generation is bounded by `max_tokens` but the surplus ring tokens are
  silently discarded (`serve_generation.c:930-942`). This means
  `completion_tokens` in the usage field may be slightly lower than the number
  of steps the driver ran.

- **GPU lazy init hangs if called too early** (`batch_sched.c:714`). GPU lane
  registration (`bs_lazy_init_gpu`) is deferred to the first `bs_slot_bind` to
  avoid a CUDA context timing issue right after a long weight upload. If the
  context is still unsettled when the first request arrives, the bind will hang
  or fail. The scheduler is fail-soft here (bind failure → -1 → request gets a
  500), but a burst of concurrent first requests could all fail before the
  context settles.

- **The `reg->lock` is brief but coarse** during `mr_load`'s slot reservation
  phase (`model_registry.c:123`). A concurrent load + multiple request threads
  all briefly contend on the same mutex. In practice this is negligible (the
  critical sections are O(MR_MAX_MODELS) array scans at most), but worth noting
  for any future lock-free refactor.

- **Cross-replica session affinity is not implemented.** `mr_acquire` selects the
  least-busy replica by `refs` with no stickiness (`model_registry.c:329-341`). A
  multi-turn session with `session_id` can hop replicas between turns, missing its
  prefix cache on the new replica and paying a full reprefill. The single-die batched
  path has slot-resident affinity (`bs_slot_acquire_biased`), but that only applies
  within one replica. Cross-replica affinity is designed but not yet shipped.

- **`QWEN35_GPU_FULL_LOGITS` is a hard startup guard** (`serve_command.c:279`).
  Setting it causes the server to refuse to start with an explicit error. It is
  a debug knob that forces the full 248k-vocab logit tensor D2H per step and
  caps throughput on the K80 host link (~0.4 GB/s). The guard is intentional and
  correct; do not remove it.
