#!/usr/bin/env python3
"""Mutation reds for datacenter gateway serving: each mutation puts back one
pre-fix behaviour (or breaks one guard) and must turn its host test red.
Every file is restored from the copy read before mutating.
Run from the repo root: python3 tests/mutations/dc_gateway_mutations.py
(The non-streaming disconnect test loads a model; set PANTHEON_TEST_MODEL and
run this through foundry's metal/run_guarded.sh on macOS to include it.)"""
import os
import subprocess
import sys

OC = 'src/serve/openai_chat.c'
TC = 'src/serve/tool_calling.c'
BS = 'src/serve/batch_sched.c'
GM = 'src/serve/gateway_mode.c'
GA = 'src/serve/gateway_auth.c'
T_CHAT = 'tests/test_chat_request'
T_GW = 'tests/test_gateway_mode'
T_BS = 'tests/test_prefix_cache_sys'
T_DISC = 'tests/test_serve_nonstream_disconnect'

MUTS = [
    ("completion id from rand()", OC, 'pn_request_id_new("chatcmpl-", buf, len);',
     'snprintf(buf, len, "chatcmpl-%08x", (unsigned)rand());', T_CHAT),
    ("tool-call id from rand()", TC, 'pn_request_id_new("call_", buf, len);',
     'snprintf(buf, len, "call_%08x", (unsigned)rand());', T_GW),
    ("usage chunk per choice", OC, "    stream_finish_choice(sc, req, res, rc);\n    if (rc != 0)",
     "    stream_finish_choice(sc, req, res, rc);\n    if (req->include_usage)\n"
     "        send_usage_chunk(sc, res);\n    if (rc != 0)", T_CHAT),
    ("no usage before unload error", OC,
     "        sc->end_reason = \"model_unloaded\";\n        if (req->include_usage)\n"
     "            stream_send_usage_once(sc);\n",
     "        sc->end_reason = \"model_unloaded\";\n", T_CHAT),
    ("second [DONE] on over-budget", OC, "             kv_used, kv_cap);\n    hs_send_sse(client_fd, json);\n}",
     "             kv_used, kv_cap);\n    hs_send_sse(client_fd, json);\n    hs_end_sse(client_fd);\n}", T_CHAT),
    ("json_schema not refused", OC, "    if (req->wants_json_schema) {", "    if (0) {", T_CHAT),
    ("media parts not flagged", OC, "                req->has_media_parts = true;\n", "", T_CHAT),
    ("seats not evictable", BS, "    if (!sched->sessions_evictable)\n        return false;\n    pthread_mutex_unlock",
     "    if (1)\n        return false;\n    pthread_mutex_unlock", T_BS),
    ("queued waiter ignores its client", BS, "        if (alive && !alive(alive_ud)) {",
     "        if (0) {", T_BS),
    ("gateway routes everything", GM, '        return path_is(path, "/v1/chat/completions");',
     "        return true;", T_GW),
    ("key matches a prefix", GA, "    return diff == 0 && plen == key->len;", "    return diff == 0;", T_GW),
    ("tenant ':' allowed", GM, "c == '.' || c == '_' || c == '-';", "c == '.' || c == '_' || c == '-' || c == ':';", T_GW),
    ("non-stream turn ignores its client", OC,
     "oai_turn_abort_ctx turn_abort = { .client_fd = &client_fd, .entry = entry };",
     "oai_turn_abort_ctx turn_abort = { .client_fd = NULL, .entry = entry };", T_DISC),
]


def run(test):
    """Build from scratch (make 3.81 skips same-second edits) and run; True = green."""
    subprocess.run(['rm', '-f', test])
    r = subprocess.run(['make', '-s', 'USE_BLAS=1', 'USE_THREADS=1', test],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None, 'BUILD FAIL ' + r.stderr[-300:]
    p = subprocess.run(['./' + test], capture_output=True, text=True)
    tail = (p.stdout.strip().splitlines() or [''])[-1]
    if 'SKIP' in p.stdout:
        return None, 'SKIPPED (no model)'
    return p.returncode == 0, tail[-120:]


survivors = 0
for name, f, a, b, t in MUTS:
    if t == T_DISC and not os.environ.get('PANTHEON_TEST_MODEL'):
        print(f"{name:36s} SKIPPED (set PANTHEON_TEST_MODEL)")
        continue
    original = open(f).read()
    if original.count(a) != 1:
        print(f"{name:36s} PATTERN NOT FOUND")
        survivors += 1
        continue
    open(f, 'w').write(original.replace(a, b))
    try:
        green, line = run(t)
    finally:
        open(f, 'w').write(original)
    red = green is False
    survivors += 0 if red else 1
    print(f"{name:36s} {'RED' if red else 'NOT RED'}  {line}")
for t in sorted({m[4] for m in MUTS}):
    if t == T_DISC and not os.environ.get('PANTHEON_TEST_MODEL'):
        continue
    green, line = run(t)
    print(f"restored {t}: {'GREEN' if green else 'NOT GREEN'}  {line}")
    survivors += 0 if green else 1
sys.exit(1 if survivors else 0)
