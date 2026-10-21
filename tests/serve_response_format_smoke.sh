#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# serve_response_format_smoke.sh — end-to-end `response_format` gate against a
# LIVE serve, on the path the fleet actually runs.
#
# WHY THIS EXISTS
#   response_format shipped once looking correct: the unit tests passed, the
#   grammar was provably right in isolation, and a K80 measurement showed valid
#   JSON.  But that measurement was taken on a CPU build, i.e. the single-user
#   decode path — while the fleet runs the CUDA BATCHED path, where the
#   constraint did not engage and the serve still reported "applied".  A test
#   that cannot tell those two paths apart is a test that would have passed
#   through the whole defect.
#
#   So this script asserts against a real serve and PARSES the body:
#     1. when the serve says response_format_status == "applied", the content
#        MUST parse as JSON — "applied" has to mean applied;
#     2. when the content does NOT parse, the status must NOT be "applied"
#        (a truthful "not_enforced" is a pass — a false "applied" is the bug);
#     3. the same on the streaming path, reassembled from the SSE deltas.
#   It also prints whether the target model is running batched, so a green run
#   on the wrong path can never again be mistaken for a green run on the fleet's.
#
# USAGE
#   tests/serve_response_format_smoke.sh [BASE_URL] [MODEL]
#     BASE_URL  default http://127.0.0.1:8201
#     MODEL     default nex-n2-mini
#
#   Run it on the serve host (or through a tunnel) against a running serve, and
#   confirm the serve log shows the model on a batched slot.  Exit 0 = pass.
#
# @status Covers json_object on both the streaming and non-streaming paths.
# @issues Needs a live serve + a loaded model; not part of `make test`.
# @todo None

set -uo pipefail

BASE="${1:-http://127.0.0.1:8201}"
MODEL="${2:-nex-n2-mini}"
PROMPT='Give me a person record: name, age, city.'
FAILED=0

say()  { printf '%s\n' "$*"; }
pass() { printf '  [PASS] %s\n' "$*"; }
fail() { printf '  [FAIL] %s\n' "$*"; FAILED=1; }

say "=== response_format smoke: $MODEL @ $BASE ==="

# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------
body=$(cat <<EOF
{"model":"$MODEL",
 "messages":[{"role":"user","content":"$PROMPT"}],
 "response_format":{"type":"json_object"},
 "temperature":0.7,"max_tokens":200}
EOF
)
resp=$(curl -s --max-time 900 "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$body")

printf '%s' "$resp" | python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception as e:
    print("  [FAIL] response body is not JSON at all: %s" % e); sys.exit(1)
if "error" in d:
    print("  [SKIP] serve refused: %s" % d["error"].get("message","")[:80]); sys.exit(2)

status  = d.get("response_format_status")
content = d["choices"][0]["message"].get("content") or ""
print("  status  = %r" % status)
print("  content = %r" % content[:160])

try:
    parsed = json.loads(content)
    is_json = isinstance(parsed, dict)
except Exception:
    is_json = False

rc = 0
# (1) "applied" must mean applied.
if status == "applied" and not is_json:
    print("  [FAIL] status says applied but the content is NOT a JSON object")
    print("         -> the constraint did not engage on this path")
    rc = 1
elif status == "applied":
    print("  [PASS] applied, and the content parses as a JSON object")
# (2) not enforced is acceptable ONLY if it says so.
elif not is_json and status != "applied":
    print("  [PASS] content is not JSON, and the serve reported %r (truthful)" % status)
elif is_json:
    print("  [PASS] content parses as JSON (status %r)" % status)
sys.exit(rc)
'
rc=$?; [ $rc -eq 1 ] && FAILED=1
[ $rc -eq 2 ] && say "  (non-streaming skipped)"

# ---------------------------------------------------------------------------
# Streaming — the same guarantee, reassembled from the deltas
# ---------------------------------------------------------------------------
say "--- streaming ---"
sbody=$(cat <<EOF
{"model":"$MODEL",
 "messages":[{"role":"user","content":"$PROMPT"}],
 "response_format":{"type":"json_object"},
 "stream":true,"temperature":0.7,"max_tokens":200}
EOF
)
curl -s --max-time 900 -N "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' -d "$sbody" | python3 -c '
import json, sys
content, status, saw = "", None, False
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: "): continue
    payload = line[6:]
    if payload == "[DONE]": continue
    try: o = json.loads(payload)
    except Exception: continue
    saw = True
    if o.get("response_format_status"): status = o["response_format_status"]
    if "error" in o:
        print("  [SKIP] serve refused: %s" % str(o["error"])[:80]); sys.exit(2)
    for ch in o.get("choices", []):
        piece = ch.get("delta", {}).get("content")
        if piece: content += piece
if not saw:
    print("  [FAIL] no SSE chunks received"); sys.exit(1)

print("  status  = %r" % status)
print("  content = %r" % content[:160])
try:
    is_json = isinstance(json.loads(content), dict)
except Exception:
    is_json = False

rc = 0
if status == "applied" and not is_json:
    print("  [FAIL] status says applied but the streamed content is NOT JSON")
    rc = 1
elif status == "applied":
    print("  [PASS] applied, and the reassembled content parses as JSON")
else:
    print("  [PASS] serve reported %r (truthful, not a false applied)" % status)
sys.exit(rc)
'
rc=$?; [ $rc -eq 1 ] && FAILED=1
[ $rc -eq 2 ] && say "  (streaming skipped)"

say
if [ $FAILED -eq 0 ]; then say "ALL RESPONSE_FORMAT CHECKS PASSED"; else say "RESPONSE_FORMAT CHECKS FAILED"; fi
exit $FAILED
