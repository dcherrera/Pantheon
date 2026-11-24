#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# tools/local/sign_catalog.sh — sign a Pantheon Local catalog manifest (spec §11.2).
# Run on David's Mac. Everything secret stays inside minisign: this script never
# reads, copies or prints the secret key file; it only passes its path to
# `minisign -S` (the keys have no password, the same as TeamIDE's ~/.tauri updater key).
#
#   tools/local/sign_catalog.sh catalog.json                # current key
#   tools/local/sign_catalog.sh catalog.json --backup       # backup key
#   PANTHEON_CATALOG_KEY=/path/to/key tools/local/sign_catalog.sh catalog.json
#
# Default key paths: ~/.minisign/pantheon-catalog.key (current) and
# ~/.minisign/pantheon-catalog-backup.key (backup; keep it on separate media).
#
# Keys (done 2026-10-27, no password, same pattern as TeamIDE's ~/.tauri key):
#   minisign -G -W -p ~/.minisign/pantheon-catalog.pub        -s ~/.minisign/pantheon-catalog.key
#   minisign -G -W -p ~/.minisign/pantheon-catalog-backup.pub -s ~/.minisign/pantheon-catalog-backup.key
# Then paste the second line of each .pub ("RW...") into src/local/catalog_keys.h
# (PN_CATALOG_KEY_CURRENT / PN_CATALOG_KEY_BACKUP). The .pub files are public;
# the .key files never go on the VPS, in CI or in any repo.
#
# The manifest is checked first with the same parser the box uses
# (`pantheon local check-catalog`), and the signature is verified after signing.
# Publish catalog.json and catalog.json.minisig side by side.
set -euo pipefail

manifest=${1:?usage: sign_catalog.sh <catalog.json> [--backup]}
which_key=${2:-}
here=$(cd "$(dirname "$0")/../.." && pwd)
if [ "$which_key" = "--backup" ]; then
    key=${PANTHEON_CATALOG_KEY:-$HOME/.minisign/pantheon-catalog-backup.key}
else
    key=${PANTHEON_CATALOG_KEY:-$HOME/.minisign/pantheon-catalog.key}
fi

command -v minisign > /dev/null || { echo "minisign is not installed (brew install minisign)"; exit 1; }
[ -f "$key" ] || { echo "no secret key at $key (see the setup notes at the top of this script)"; exit 1; }
[ -x "$here/pantheon" ] || { echo "build pantheon first (make pantheon) to validate the manifest"; exit 1; }

"$here/pantheon" local check-catalog "$manifest"
if grep -Eq '"revision":[[:space:]]*"0{40}"' "$manifest"; then
    echo "placeholder revision (40 zeros) in $manifest: put each file's HF upload commit in first"
    exit 1
fi
version=$(sed -n 's/^[[:space:]]*"version":[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$manifest" | head -1)
minisign -S -s "$key" -m "$manifest" -x "$manifest.minisig" -t "pantheon catalog version $version"
pub="${key%.key}.pub"
if [ -f "$pub" ]; then
    minisign -V -p "$pub" -m "$manifest" -x "$manifest.minisig" > /dev/null
    echo "signed and verified: $manifest.minisig (version $version)"
else
    echo "signed: $manifest.minisig (version $version); no $pub next to the key to verify with"
fi
