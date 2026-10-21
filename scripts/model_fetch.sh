# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# Download-and-check helpers for the model run scripts (run_ornith.sh, k80_*.sh). Sourced,
# not run. The caller sets DIR (where model files live) and defines die().
#
# A file is good once its sha256 matched (or it was built locally); a `<file>.ok` marker
# records that so later runs don't re-hash many GB. Downloads resume into `<file>.part` and
# move into place only after the hash matches. HF_TOKEN, if set, is sent to Hugging Face.

# mf_verified <file>: already checked.
mf_verified() {
    [ -f "$DIR/$1" ] && [ -f "$DIR/$1.ok" ]
}

# mf_verify <file> <sha256>: hash it; keep and mark it only if it matches.
mf_verify() {
    echo "fetch: checking $1"
    [ "$(sha256sum "$DIR/$1" | cut -d' ' -f1)" = "$2" ] || { rm -f "$DIR/$1"; return 1; }
    touch "$DIR/$1.ok"
}

# mf_download <base-url> <file> <sha256>
mf_download() {
    local auth=()
    echo "fetch: downloading $2"
    [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
    curl -fL --retry 5 -C - ${auth[@]+"${auth[@]}"} -o "$DIR/$2.part" "$1/$2" || return 1
    mv "$DIR/$2.part" "$DIR/$2"
    mf_verify "$2" "$3"
}

# mf_fetch <file> <sha256> <mirror-url> [upstream-url]: make <file> present and checked.
# An upstream URL is tried when the mirror fails. Returns 1 if every source failed.
mf_fetch() {
    mf_verified "$1" && return 0
    [ -f "$DIR/$1" ] && mf_verify "$1" "$2" && return 0
    mf_download "$3" "$1" "$2" && return 0
    rm -f "$DIR/$1.part"
    [ -n "${4:-}" ] || return 1
    echo "fetch: mirror failed, trying $4"
    mf_download "$4" "$1" "$2" && return 0
    rm -f "$DIR/$1.part"
    return 1
}

# mf_need_disk <gb>: die unless DIR has <gb> free.
mf_need_disk() {
    local free_gb
    free_gb=$(df -Pk "$DIR" | awk 'NR == 2 { print int($4 / 1048576) }')
    [ "$free_gb" -ge "$1" ] || die "$DIR has ${free_gb} GB free; this needs about $1 GB (set the model dir)"
}
