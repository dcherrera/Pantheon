#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# serve_session_export_smoke.sh — regression test for "session export/restore
# only ever worked on the single-user path".
#
# The defect: /admin/session/export looked the session up in the model's
# pn_prefix_cache.  A BATCHED model (QWEN35_BATCH_SLOTS=1 — every model a K80
# deployment runs) keeps its session state in the batch scheduler's warm decode slot,
# and nothing ever puts an entry in that model's prefix cache, so the lookup
# returned NULL and EVERY export against a batched model answered 404
# session_not_found.  /session/restore had the mirror-image defect: it imported
# into the prefix cache, so it reported 200 restored:true while writing to a
# store no batched generation ever reads.  /session/release cleared only the
# prefix cache too, leaving a departed session's context bound to its slot.
#
# Since the host KV mirror was removed (Phase 10b / INV3), export is the ONLY
# way a session's state can leave the GPU: a serve restart loses all warm
# context and the client never receives the blob the design says it owns.
#
# Asserts, against a LIVE session on the model under test:
#   1. export returns HTTP 200 with a non-empty blob, n_tok > 0, a
#      weights_version, and the committed token ids.  (This alone is the
#      assertion that would have caught the defect.)
#   2. weights_version guard: the same blob with a corrupted weights_version
#      is refused 409 weights_version_mismatch, and nothing is restored.
#   3. release actually drops the session: a second export answers 404.
#   4. restore of the exported blob answers 200 restored:true.
#   5. ROUND TRIP IS LOSSLESS: re-exporting after the restore yields a blob
#      byte-identical to the original, with the same n_tok and token ids.
#      This is the discriminating check — it proves the state left the GPU,
#      crossed the wire format, and came back into a live slot intact.
#   6. The session still generates coherently after the restore (recall of a
#      planted token that was in the context before the export).
#
# Usage (server must already be running):
#   BASE_URL=http://127.0.0.1:8201 MODEL=saber-orch \
#     tests/serve_session_export_smoke.sh
#
# Env:
#   BASE_URL     default http://127.0.0.1:8201
#   MODEL        default saber-orch
#   SECRET       planted recall token (default ZEPHYR-7-QUARTZ)
#   FILLER       words of real context in front of the planted token (default 200)
#   MAX_TOKENS   default 300
#   REQ_TIMEOUT  per-request curl timeout, seconds (default 600)

set -u

BASE_URL="${BASE_URL:-http://127.0.0.1:8201}"
MODEL="${MODEL:-saber-orch}"
SECRET="${SECRET:-ZEPHYR-7-QUARTZ}"
FILLER="${FILLER:-200}"
MAX_TOKENS="${MAX_TOKENS:-300}"
REQ_TIMEOUT="${REQ_TIMEOUT:-600}"

WORK="$(mktemp -d)"
SID="expsmoke-$$-$(date +%s)"
PASS=0
FAIL=0

cleanup() {
    curl -s -m 30 -o /dev/null "$BASE_URL/session/release" \
        -H 'Content-Type: application/json' \
        -d "{\"name\":\"$MODEL\",\"session_id\":\"$SID\"}" || true
    rm -rf "$WORK"
}
trap cleanup EXIT

ok()   { PASS=$((PASS+1)); echo "  PASS  $*"; }
bad()  { FAIL=$((FAIL+1)); echo "  FAIL  $*"; }

# jq is not assumed — pull scalars with python3, which the K80 test machine has.
jget() {  # jget <file> <expr>   e.g. jget f 'd["n_tok"]'
    python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
v=eval(sys.argv[2])
print("" if v is None else v)
' "$1" "$2" 2>/dev/null
}

post() {  # post <path> <json-file-in> <json-file-out> -> echoes http code
    curl -s -m "$REQ_TIMEOUT" -o "$3" -w '%{http_code}' \
         "$BASE_URL$1" -H 'Content-Type: application/json' --data-binary "@$2"
}

echo "== session export/restore round trip: model=$MODEL sid=$SID =="

# ── Turn 1: plant a distinctive token behind real context ──────────────────
python3 - "$WORK/turn1.json" "$MODEL" "$SID" "$SECRET" "$FILLER" "$MAX_TOKENS" <<'PY'
import json, sys
out, model, sid, secret, filler, maxtok = sys.argv[1:7]
words = ("The maintenance log records routine checks of the cooling loop, the "
         "power rails and the interconnect fabric. ").split()
body = " ".join(words[i % len(words)] for i in range(int(filler)))
msg = (f"{body}\n\nRemember this access code exactly: {secret}. "
       "Reply with only the word: stored")
json.dump({"model": model, "session_id": sid, "max_tokens": int(maxtok),
           "messages": [{"role": "user", "content": msg}]}, open(out, "w"))
PY

code=$(post /v1/chat/completions "$WORK/turn1.json" "$WORK/turn1.out")
if [ "$code" = "200" ] && [ "$(jget "$WORK/turn1.out" 'd["usage"]["completion_tokens"]')" -gt 0 ] 2>/dev/null; then
    ok "turn 1 generated (${code}, $(jget "$WORK/turn1.out" 'd["usage"]["prompt_tokens"]') prompt tokens)"
else
    bad "turn 1 did not generate (HTTP $code)"; cat "$WORK/turn1.out"; echo
    echo "== $PASS passed, $FAIL failed =="; exit 1
fi

# ── 1. Export ──────────────────────────────────────────────────────────────
cat > "$WORK/exp.json" <<EOF
{"name":"$MODEL","session_id":"$SID"}
EOF
code=$(post /admin/session/export "$WORK/exp.json" "$WORK/exp1.out")
N_TOK=$(jget "$WORK/exp1.out" 'd.get("n_tok")')
WV=$(jget "$WORK/exp1.out" 'd.get("weights_version")')
BLOB_BYTES=$(jget "$WORK/exp1.out" 'd.get("blob_bytes")')
if [ "$code" = "200" ] && [ -n "$N_TOK" ] && [ "$N_TOK" -gt 0 ] 2>/dev/null \
   && [ -n "$WV" ] && [ "${BLOB_BYTES:-0}" -gt 0 ] 2>/dev/null; then
    ok "export 200: n_tok=$N_TOK blob_bytes=$BLOB_BYTES weights_version=$WV"
else
    bad "export failed (HTTP $code) — THE REGRESSION"; head -c 300 "$WORK/exp1.out"; echo
    echo "== $PASS passed, $FAIL failed =="; exit 1
fi
if [ "$(jget "$WORK/exp1.out" 'len(d.get("tokens") or [])')" = "$N_TOK" ]; then
    ok "export returned the $N_TOK committed token ids"
else
    bad "export tokens[] length != n_tok"
fi

# Build the restore bodies (good + weights_version-corrupted).
python3 - "$WORK/exp1.out" "$MODEL" "$SID" "$WORK/res_good.json" "$WORK/res_badwv.json" <<'PY'
import base64, json, sys, struct
src, model, sid, good, badwv = sys.argv[1:6]
d = json.load(open(src))
raw = base64.b64decode(d["blob_b64"])
json.dump({"name": model, "session_id": sid, "blob": d["blob_b64"],
           "tokens": d["tokens"]}, open(good, "w"))
# The blob header is {magic, format_version, weights_version, ...}; flip a bit
# in the weights_version field so only the identity tag differs.
b = bytearray(raw)
wv_off = None
tag = struct.pack("<Q", int(d["weights_version"]))
for off in range(0, min(len(b), 64) - 8):
    if bytes(b[off:off + 8]) == tag:
        wv_off = off
        break
assert wv_off is not None, "weights_version tag not found in the blob header"
b[wv_off] ^= 0xFF
json.dump({"name": model, "session_id": sid,
           "blob": base64.b64encode(bytes(b)).decode(),
           "tokens": d["tokens"]}, open(badwv, "w"))
PY

# ── 2. weights_version guard ───────────────────────────────────────────────
code=$(post /session/restore "$WORK/res_badwv.json" "$WORK/badwv.out")
if [ "$code" = "409" ] && grep -q weights_version_mismatch "$WORK/badwv.out"; then
    ok "weights_version guard: corrupted tag refused 409 weights_version_mismatch"
else
    bad "weights_version guard did not fire (HTTP $code): $(head -c 200 "$WORK/badwv.out")"
fi

# ── 3. Release really drops the session ────────────────────────────────────
post /session/release "$WORK/exp.json" "$WORK/rel.out" >/dev/null
code=$(post /admin/session/export "$WORK/exp.json" "$WORK/exp_gone.out")
if [ "$code" = "404" ]; then
    ok "release dropped the session (export now 404)"
else
    bad "release did not drop the session (export still HTTP $code)"
fi

# ── 4. Restore ─────────────────────────────────────────────────────────────
code=$(post /session/restore "$WORK/res_good.json" "$WORK/res.out")
if [ "$code" = "200" ] && [ "$(jget "$WORK/res.out" 'd.get("n_tok")')" = "$N_TOK" ]; then
    ok "restore 200: n_tok=$N_TOK"
else
    bad "restore failed (HTTP $code): $(head -c 300 "$WORK/res.out")"
    echo "== $PASS passed, $FAIL failed =="; exit 1
fi

# ── 5. Round trip is lossless ──────────────────────────────────────────────
code=$(post /admin/session/export "$WORK/exp.json" "$WORK/exp2.out")
if [ "$code" != "200" ]; then
    bad "re-export after restore failed (HTTP $code)"
else
    if python3 - "$WORK/exp1.out" "$WORK/exp2.out" <<'PY'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
same = (a["n_tok"] == b["n_tok"] and a["tokens"] == b["tokens"]
        and a["blob_b64"] == b["blob_b64"])
sys.exit(0 if same else 1)
PY
    then
        ok "round trip lossless: re-exported blob byte-identical to the original"
    else
        bad "round trip LOSSY: re-exported blob differs from the original"
        python3 - "$WORK/exp1.out" "$WORK/exp2.out" <<'PY'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
print(f"    n_tok {a['n_tok']} -> {b['n_tok']}; "
      f"blob_bytes {a['blob_bytes']} -> {b['blob_bytes']}; "
      f"tokens equal: {a['tokens'] == b['tokens']}")
PY
    fi
fi

# ── 6. The restored session still answers ──────────────────────────────────
python3 - "$WORK/turn2.json" "$WORK/turn1.json" "$WORK/turn1.out" "$MAX_TOKENS" <<'PY'
import json, sys
out, t1, t1o, maxtok = sys.argv[1:5]
req = json.load(open(t1)); res = json.load(open(t1o))
msgs = list(req["messages"])
msgs.append({"role": "assistant",
             "content": res["choices"][0]["message"].get("content") or "stored"})
msgs.append({"role": "user",
             "content": "What was the access code? Reply with only the code."})
req["messages"] = msgs
req["max_tokens"] = int(maxtok)
json.dump(req, open(out, "w"))
PY
code=$(post /v1/chat/completions "$WORK/turn2.json" "$WORK/turn2.out")
ANS=$(jget "$WORK/turn2.out" '(d["choices"][0]["message"].get("content") or "") + " " + (d["choices"][0]["message"].get("reasoning_content") or "")')
if [ "$code" = "200" ] && printf '%s' "$ANS" | grep -q "$SECRET"; then
    ok "post-restore turn recalls the planted token ($SECRET)"
else
    bad "post-restore turn did not recall the planted token (HTTP $code): $(printf '%s' "$ANS" | head -c 200)"
fi

echo "== $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
