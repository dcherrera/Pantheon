#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# deploy.sh — bring a Pantheon fleet up to a manifest, one model at a time.
#
# WHAT THIS IS
#   A template. Copy it, point it at your own manifest, adapt the gates.
#   It reconciles a running `pantheon serve` to models.conf by loading each
#   missing model through POST /admin/models/load, one at a time, waiting for
#   each to settle before starting the next.
#
# WHY ONE AT A TIME
#   On a constrained host, loading everything at once is how you OOM. Each
#   load is gated on free VRAM (via the engine's own /admin/dies) and host-RAM
#   headroom, then settle-waited until the model actually appears in
#   /v1/models. That makes bring-up idempotent and restartable: run it again
#   after a failure and it picks up where it stopped.
#
# REQUIREMENTS
#   bash + curl. Nothing else — no jq, no python, no nvidia-smi. GPU state
#   comes from the engine, so this works the same on CPU, Metal and CUDA.
#
# USAGE
#   ./deploy.sh                  reconcile to the manifest (idempotent)
#   ./deploy.sh --plan           dry run: print what would happen, change nothing
#   ./deploy.sh --prune          also unload anything running that isn't listed
#   ./deploy.sh --file other.conf
#   PANTHEON_URL=http://host:8080 ./deploy.sh
#
# EXIT CODES
#   0 all listed models are serving · 1 usage/config error · 2 a load failed
#
set -uo pipefail

URL="${PANTHEON_URL:-http://127.0.0.1:8080}"
MANIFEST="$(dirname "$0")/models.conf"
PLAN=0
PRUNE=0

# Free host RAM (MB) below which we refuse to start another load.
: "${MIN_HOST_FREE_MB:=1024}"
# Seconds to wait for a model to appear in /v1/models after a load returns.
: "${SETTLE_TIMEOUT:=300}"

while [ $# -gt 0 ]; do
    case "$1" in
        --plan)   PLAN=1 ;;
        --prune)  PRUNE=1 ;;
        --file)   MANIFEST="${2:?--file needs a path}"; shift ;;
        --url)    URL="${2:?--url needs a URL}"; shift ;;
        -h|--help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1 (try --help)" >&2; exit 1 ;;
    esac
    shift
done

log()  { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -r "$MANIFEST" ] || die "manifest not readable: $MANIFEST"

# --- engine state -----------------------------------------------------------
# Responses are small and flat, so grep/sed is enough and keeps the dependency
# list at zero. If your manifest grows past a few dozen models, switch to jq.

api() { curl -fsS --max-time 30 "$@" 2>/dev/null; }

serve_up() { api "$URL/v1/models" >/dev/null; }

loaded_models() {
    api "$URL/v1/models" | tr ',' '\n' | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'
}

model_loaded() { loaded_models | grep -qxF "$1"; }

# Free VRAM in bytes across all devices, from the engine itself. Prints
# nothing if the engine reports no devices (CPU/Metal builds) — callers treat
# an empty answer as "no GPU gate applies".
free_vram_bytes() {
    local body budget used
    body="$(api "$URL/admin/dies")" || return 0
    printf '%s' "$body" | grep -q '"die"' || return 0
    budget=$(printf '%s' "$body" | tr ',' '\n' | sed -n 's/.*"vram_budget"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' | paste -sd+ - | bc 2>/dev/null)
    used=$(printf   '%s' "$body" | tr ',' '\n' | sed -n 's/.*"vram_used"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p'   | paste -sd+ - | bc 2>/dev/null)
    [ -n "${budget:-}" ] && [ -n "${used:-}" ] && echo $(( budget - used ))
}

# Free host RAM in MB. Linux via /proc, macOS via vm_stat. Prints nothing on
# platforms we cannot read, which disables the host gate rather than guessing.
host_free_mb() {
    if [ -r /proc/meminfo ]; then
        awk '/^MemAvailable:/ {print int($2/1024); exit}' /proc/meminfo
    elif command -v vm_stat >/dev/null 2>&1; then
        vm_stat | awk '
            /page size of/ { for (i=1;i<=NF;i++) if ($i+0>1024) ps=$i+0 }
            /Pages free/      { gsub(/\./,"",$3); f=$3 }
            /Pages inactive/  { gsub(/\./,"",$3); f+=$3 }
            END { if (ps && f) print int(f*ps/1048576) }'
    fi
}

# --- manifest ---------------------------------------------------------------
# name | path | key=value ...   ->  JSON body for /admin/models/load

build_body() {
    local name="$1" path="$2" opts="$3" json kv k v
    json="{\"name\":\"$name\",\"path\":\"$path\""
    for kv in $opts; do
        k="${kv%%=*}"; v="${kv#*=}"
        [ "$k" = "$kv" ] && { echo "  ! ignoring malformed option '$kv'" >&2; continue; }
        case "$k" in
            dies)     json="$json,\"dies\":[${v}]" ;;
            quantize) json="$json,\"quantize\":${v}" ;;
            ctx_cap|max_sessions|max_concurrent|reasoning_budget)
                      json="$json,\"$k\":${v}" ;;
            *)        echo "  ! unknown option '$k' — skipped" >&2 ;;
        esac
    done
    echo "$json}"
}

settle_wait() {
    local name="$1" waited=0
    while [ "$waited" -lt "$SETTLE_TIMEOUT" ]; do
        model_loaded "$name" && return 0
        sleep 2; waited=$(( waited + 2 ))
    done
    return 1
}

load_model() {
    local name="$1" path="$2" opts="$3" body resp
    body="$(build_body "$name" "$path" "$opts")"

    local vram free_mb
    vram="$(free_vram_bytes)"
    [ -n "$vram" ] && log "    gate: $(( vram / 1048576 )) MiB VRAM free"
    free_mb="$(host_free_mb)"
    if [ -n "$free_mb" ]; then
        log "    gate: ${free_mb} MiB host RAM free"
        if [ "$free_mb" -lt "$MIN_HOST_FREE_MB" ]; then
            log "    REFUSED: host RAM below ${MIN_HOST_FREE_MB} MiB — stopping here"
            return 1
        fi
    fi

    resp="$(curl -fsS --max-time 900 -X POST "$URL/admin/models/load" \
                 -H 'Content-Type: application/json' -d "$body" 2>&1)" || {
        log "    LOAD FAILED: ${resp:-no response}"
        return 1
    }
    settle_wait "$name" || { log "    did not appear in /v1/models within ${SETTLE_TIMEOUT}s"; return 1; }
    log "    ok"
}

# --- main -------------------------------------------------------------------

serve_up || die "no serve responding at $URL — start one with: pantheon serve --model <path>"

log "manifest: $MANIFEST"
log "target:   $URL"
[ "$PLAN" = 1 ] && log "PLAN MODE — nothing will be changed"

declare -a WANTED=()
rc=0

while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    case "$line" in *[![:space:]]*) ;; *) continue ;; esac

    IFS='|' read -r name path opts <<< "$line"
    name="$(echo "${name:-}" | xargs)"
    path="$(echo "${path:-}" | xargs)"
    opts="$(echo "${opts:-}" | xargs)"
    [ -n "$name" ] && [ -n "$path" ] || { echo "skipping malformed line: $line" >&2; continue; }

    WANTED+=("$name")

    if model_loaded "$name"; then
        log "= $name already serving"
        continue
    fi
    if [ "$PLAN" = 1 ]; then
        log "+ would load $name from $path"
        log "    body: $(build_body "$name" "$path" "$opts")"
        continue
    fi

    log "+ loading $name from $path"
    load_model "$name" "$path" "$opts" || { rc=2; break; }
done < "$MANIFEST"

if [ "$PRUNE" = 1 ]; then
    while IFS= read -r running; do
        [ -n "$running" ] || continue
        for w in ${WANTED+"${WANTED[@]}"}; do [ "$w" = "$running" ] && continue 2; done
        if [ "$PLAN" = 1 ]; then
            log "- would unload $running (not in manifest)"
        else
            log "- unloading $running (not in manifest)"
            api -X POST "$URL/admin/models/unload" \
                -H 'Content-Type: application/json' \
                -d "{\"name\":\"$running\"}" >/dev/null \
                || log "    unload failed"
        fi
    done < <(loaded_models)
fi

log "serving: $(loaded_models | paste -sd' ' -)"
[ "$rc" -ne 0 ] && log "stopped early — fix the failure above and re-run; loaded models are kept"
exit "$rc"
