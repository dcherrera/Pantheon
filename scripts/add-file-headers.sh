#!/usr/bin/env bash
# Insert the standard file-header block into sources that lack one.
#
#   ./scripts/add-file-headers.sh            # dry run — report only
#   ./scripts/add-file-headers.sh --apply    # write the blocks
#   ./scripts/add-file-headers.sh --apply src/core   # limit to a subtree
#
# Adds the SPDX pair when missing, then:
#
#   /**
#    * @file   name.c
#    * @brief  TODO
#    *
#    * @status None
#    * @issues None
#    * @todo   None
#    */
#
# @brief is seeded from the paired header's @brief when there is one, otherwise
# left as TODO — a one-line summary is not mechanically derivable, and a wrong
# one is worse than an absent one. scripts/check-headers.sh rejects TODO briefs
# on files a PR adds.
#
# Exempt: tests, generated sources, third-party-derived files (those keep their
# upstream attribution).
set -euo pipefail

APPLY=0
[ "${1-}" = "--apply" ] && { APPLY=1; shift; }
SCOPE="${1-.}"

SPDX='/* SPDX-License-Identifier: AGPL-3.0-or-later */'
COPYRIGHT='/* Copyright (C) 2026 David (Corey) Herrera */'

is_exempt() {
    case "$1" in
        tests/*|*/tests/*)          return 0 ;;  # test files
        *_ptx.h|*ptx_modules*)      return 0 ;;  # generated
        *) ;;
    esac
    # third-party attribution present
    head -5 "$1" | grep -qE 'SPDX-FileCopyrightText|Apache-2\.0' && return 0
    return 1
}

added=0 todo=0 seeded=0 skipped=0

while IFS= read -r f; do
    head -12 "$f" | grep -q '@file' && continue
    if is_exempt "$f"; then skipped=$((skipped+1)); continue; fi

    base=$(basename "$f")
    brief="TODO"
    pair="${f%.*}.h"
    if [ "$pair" != "$f" ] && [ -f "$pair" ]; then
        p=$(grep -m1 '@brief' "$pair" 2>/dev/null | sed 's/.*@brief[[:space:]]*//' || true)
        [ -n "$p" ] && brief="Implementation of ${p%.}."
    fi
    [ "$brief" = "TODO" ] && todo=$((todo+1)) || seeded=$((seeded+1))
    added=$((added+1))

    [ "$APPLY" = "1" ] || continue

    tmp=$(mktemp)
    if head -1 "$f" | grep -q 'SPDX-License-Identifier'; then
        # SPDX already present — insert the block after the copyright line
        head -2 "$f" > "$tmp"
        rest=3
    else
        printf '%s\n%s\n' "$SPDX" "$COPYRIGHT" > "$tmp"
        rest=1
    fi
    {
        printf '/**\n'
        printf ' * @file   %s\n' "$base"
        printf ' * @brief  %s\n' "$brief"
        printf ' *\n'
        printf ' * @status None\n'
        printf ' * @issues None\n'
        printf ' * @todo   None\n'
        printf ' */\n'
    } >> "$tmp"
    tail -n "+$rest" "$f" >> "$tmp"
    mv "$tmp" "$f"
done < <(git ls-files "$SCOPE" | grep -E '\.(c|h|cu|cuh|m)$')

printf 'headers %s: %d file(s) — %d brief seeded from paired header, %d left TODO; %d exempt\n' \
    "$([ "$APPLY" = 1 ] && echo written || echo needed)" "$added" "$seeded" "$todo" "$skipped"

# --- SPDX pass -------------------------------------------------------------
# Independent of the header block: a file can carry @file and still have no SPDX
# line. Only files that are unambiguously ours get stamped. Anything whose header
# cites upstream sources is listed for manual triage instead — a port of an
# Apache-2.0 source needs Apache-2.0 plus the upstream copyright, and that is a
# per-file judgement, not a sed.
spdx_added=0 triage=0
while IFS= read -r f; do
    head -3 "$f" | grep -q 'SPDX-License-Identifier' && continue
    case "$f" in tests/*|*/tests/*|*_ptx.h) continue ;; esac
    if head -20 "$f" | grep -qiE 'tt-metal|tt-kmd|Tenstorrent|transliteration|rendering of|upstream'; then
        triage=$((triage+1))
        [ "$APPLY" = 1 ] || printf '  triage: %s\n' "$f"
        continue
    fi
    spdx_added=$((spdx_added+1))
    [ "$APPLY" = 1 ] || continue
    tmp=$(mktemp)
    printf '%s\n%s\n' "$SPDX" "$COPYRIGHT" > "$tmp"
    cat "$f" >> "$tmp"
    mv "$tmp" "$f"
done < <(git ls-files "$SCOPE" | grep -E '\.(c|h|cu|cuh|m)$')

printf 'SPDX %s: %d file(s); %d need manual triage (upstream-derived)\n' \
    "$([ "$APPLY" = 1 ] && echo written || echo needed)" "$spdx_added" "$triage"
[ "$APPLY" = 1 ] || printf 'dry run — pass --apply to write\n'
