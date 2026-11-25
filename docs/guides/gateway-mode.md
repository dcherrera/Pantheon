# Gateway mode

`pantheon serve --gateway` runs the serve behind a trusted gateway that terminates customer TLS and
auth and forwards OpenAI requests on one shared key. Without `--gateway` nothing below applies and the
serve behaves as before.

```sh
pantheon serve --model model.gguf --name my-model --host 0.0.0.0 --port 8080 \
    --gateway --api-key-file /etc/pantheon/gateway.key \
    --admin-port 8081            # optional; admin routes on 127.0.0.1:8081
```

## Flags

| Flag | Meaning |
|---|---|
| `--gateway` | Turn gateway mode on. Cannot be combined with `--local`. |
| `--api-key-file <path>` | Required. One key, surrounding whitespace trimmed, at most 256 bytes. Read once at start, never logged. Keep it `chmod 600`. |
| `--admin-host <addr>` | Address of the admin listener (default `127.0.0.1`). |
| `--admin-port <n>` | Port of the admin listener. Without it the admin and session routes are off. |

## Routes on the API port

Only these are routed; every other path is `404 unknown_route` (checked before the key, so admin paths
look absent):

| Route | Notes |
|---|---|
| `POST /v1/chat/completions` | |
| `GET /v1/models` | |
| `GET /health/live` | 200 while the process runs. |
| `GET /health/ready` | 200 when a model is loaded and the serve is not draining; 503 otherwise, including during a SIGTERM drain. |

Every request needs `Authorization: Bearer <key>`; a missing or wrong key is `401 invalid_api_key`.
The admin listener (`/admin/*`, `/session/*`, `/v1/session/release`) takes no key: keep it on loopback.

## What the gateway sends

| Header | Meaning |
|---|---|
| `Authorization: Bearer <key>` | The key in `--api-key-file`. |
| `X-Pantheon-Tenant: <id>` | The customer. 1-48 characters of `A-Z a-z 0-9 . _ -`; anything else is `400 invalid_tenant`. Optional. |

The client's `X-Session-Id` and `X-Pantheon-Replica` headers and the body `replica` field are ignored.

## Sessions

A body session key (`session_id`, `prompt_cache_key`, `promptCacheKey`) is scoped to the tenant: the
serve uses `<tenant>:<key>`, so two customers can never share, evict or probe each other's session. Without
`X-Pantheon-Tenant` the key is ignored and the turn runs anonymously (full prefill).

A session seat is a cache, not a reservation. When a request needs a slot and every seat is held, the least
recently used idle seat of another session is taken (that session re-prefills on its next turn). A seat
mid-turn is never taken; the request queues for it (30 s, then `429 server_busy`). Idle seats also expire
after 120 s (`PN_SESSION_IDLE_TTL_MS` overrides). One customer's key can never lock out other customers or
anonymous turns, whatever `max_sessions` is.

## Refused instead of degraded

| Request | Answer |
|---|---|
| `response_format: {"type": "json_schema"}` | `400`, `code: unsupported_parameter`, `param: response_format`. `json_object` works. |
| Image, audio, video or file content parts | `400`, `code: unsupported_content`, `param: messages`. |
| A `model` that is not loaded | `404`, `code: model_not_found`, `param: model`. |

## Billing

- Completion ids are `chatcmpl-` plus 24 random base62 characters; tool-call ids are `call_` plus 24.
  They do not repeat across restarts or replicas.
- Non-streamed `usage` is exact. With `stream_options.include_usage` a stream carries exactly one usage
  chunk (`choices: []`) after the last completion, aggregated over `n` (prompt counted once, completions
  summed).
- A stream that ends abnormally (`model_unloaded`, `generation_stalled`) gets its usage chunk before the
  error event. A refused request (`context_over_budget`, `server_memory_constrained`, any 4xx/429) computed
  nothing and reports no usage. A non-streamed request that fails returns an error and no usage.
- Every request writes one line to stderr, whatever happened:

  ```
  [usage] id=chatcmpl-… tenant=acme model=my-model prompt=812 cached=640 completion=96 choices=1 finish=stop
  ```

  `finish` is the wire finish reason (`stop`, `length`, `tool_calls`, …) or what ended the request:
  `client_gone`, `model_unloaded`, `stalled`, `shutdown`, `context_over_budget`, `memory_constrained`,
  `session_capacity`, `slot_unavailable`, `error`. It holds no prompt or output text. When a stream dies
  (the client or the gateway hung up), this line is the record to reconcile against.

## Disconnects

A client that disconnects stops its turn and frees the slot within 8 tokens or one prefill chunk, streaming
or not. A client that disconnects while queued for a slot leaves the queue and never takes one.
