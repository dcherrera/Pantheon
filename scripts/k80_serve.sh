# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# Shared body of the Tesla K80 run scripts (k80_qwen36.sh, k80_nexn2.sh). Sourced, not run.
# The caller sets NAME, GGUF, GGUF_SHA, GGUF_GB, MIRROR, UPSTREAM, DIR, CTX_CAP, MAX_SESSIONS,
# REASONING_BUDGET and PORT, then calls `k80_main "$@"`.
#
# One model takes both dies of one K80 (DIES, default 0,1). Prefix caching is on: the serve
# keeps MAX_SESSIONS session slots, and a client that sends `session_id` (or X-Session-Id) and
# replays each assistant turn's `reasoning_content` only prefills the new tokens on later turns.

die() { echo "$NAME: $*" >&2; exit 1; }

# The flags the K80 fleet runs with (measured settings, not defaults).
k80_env() {
    K80_ENV=(PN_THREADS=2 PN_DIE_RESERVE_BYTES=512M PANTHEON_TOOL_GRAMMAR=1
             QWEN35_KV_Q4=1 QWEN35_BATCH_SLOTS=1 QWEN35_VOCAB_SPLIT=1
             QWEN35_VOCAB_SPLIT_FULL_LOGITS=1 QWEN35_PREFILL_PIPELINE=1 QWEN35_MOE_DEDUP=1
             QWEN35_SSM_WARP=1 PN_SPARSE_SAMPLER=1 PN_GRAMMAR_ARM_WINDOW=1
             FC_MOE_DEDUP_SHX=1 FC_MOE_DEDUP_COMPACT=1 FC_ATTN_KTILE_REG=1 FC_ATTN_VSTAGE=1)
}

k80_main() {
    local download_only=0
    [ "${1:-}" = "--download-only" ] && { download_only=1; shift; }
    if [ "$download_only" -eq 0 ]; then
        [ -x ./pantheon-cuda ] || die "no ./pantheon-cuda; run ./install.sh --k80 first"
        command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1 ||
            die "no NVIDIA driver/GPU visible (nvidia-smi -L); a K80 needs the 470 driver"
    fi
    . scripts/model_fetch.sh
    mkdir -p "$DIR"
    [ -f "$DIR/$GGUF" ] || mf_need_disk "$((GGUF_GB + 1))"
    mf_fetch "$GGUF" "$GGUF_SHA" "$MIRROR" "$UPSTREAM" || die "could not download $GGUF"
    [ "$download_only" -eq 1 ] && { echo "$NAME: files ready in $DIR"; exit 0; }

    k80_env
    echo "$NAME: starting on ${HOST:-127.0.0.1}:$PORT, dies ${DIES:-0,1} (loading takes a few minutes)"
    exec env "${K80_ENV[@]}" ./pantheon-cuda serve \
        --model "$DIR/$GGUF" --name "$NAME" --dies "${DIES:-0,1}" \
        --ctx-cap "$CTX_CAP" --max-sessions "$MAX_SESSIONS" --reasoning-budget "$REASONING_BUDGET" \
        --host "${HOST:-127.0.0.1}" --port "$PORT" "$@"
}
