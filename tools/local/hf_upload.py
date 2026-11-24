#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# @file hf_upload.py
# @brief Upload catalog files to a Hugging Face repo, then check each one's size and sha256
#        there and print the commit to pin as the catalog `revision`.
#
# @status Used for the Qwen3.6 Q8_0 K80 file (2026-10-28) and the Qwen3.6 p150 files
#         (2026-11-03, --add).
# @issues none
# @todo none
#
# Usage (needs huggingface_hub[hf_xet]; the token comes ONLY from the HF_TOKEN env var,
# never argv or a file, so it never shows in `ps` or a log):
#   hf_upload.py <repo_id> <local file> <expected sha256> [--readme <README.md>] [--message <msg>]
#   hf_upload.py <repo_id> --readme <README.md> [--message <msg>]       (README only)
#   hf_upload.py <repo_id> --create --add <local> <name in repo> <sha256> [--add ...]
#                [--readme <README.md>] [--license <LICENSE>] [--message <msg>]
# --add puts every file (plus README/LICENSE) in ONE commit, so one revision pins them all;
# --create makes the (public) repo first if it does not exist.
# On the home uplink set HF_XET_FIXED_UPLOAD_CONCURRENCY=6: xet's adaptive concurrency sat at one
# stream (~0.8 MB/s from a LAN host); six gave ~3.2 MB/s.
# Prints `revision=<40 hex>` (the commit holding the files) and `sha256_ok=1` when every
# file's LFS sha256 (and, for --add, size) on the repo match; exits 1 otherwise.
import argparse
import os
import sys

from huggingface_hub import CommitOperationAdd, HfApi


def check(info, name, sha256, size):
    f = next((s for s in info.siblings if s.rfilename == name), None)
    got = f.lfs.sha256 if f is not None and f.lfs is not None else None
    got_size = f.size if f is not None else None
    print(f"{name}: remote_sha256={got} size={got_size}", flush=True)
    return got == sha256 and (size is None or got_size == size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("path", nargs="?")
    ap.add_argument("sha256", nargs="?")
    ap.add_argument("--add", nargs=3, action="append", default=[],
                    metavar=("LOCAL", "NAME", "SHA256"))
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--readme")
    ap.add_argument("--license")
    ap.add_argument("--message", default=None)
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("hf_upload: HF_TOKEN is not set")
    api = HfApi(token=token)
    if a.create:
        api.create_repo(a.repo_id, repo_type="model", private=False, exist_ok=True)
    want = []
    if a.add:
        ops = []
        for local, name, sha in a.add:
            size = os.path.getsize(local)
            print(f"adding {name} ({size} B) from {local}", flush=True)
            ops.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=local))
            want.append((name, sha, size))
        if a.readme:
            ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=a.readme))
        if a.license:
            ops.append(CommitOperationAdd(path_in_repo="LICENSE", path_or_fileobj=a.license))
        api.create_commit(repo_id=a.repo_id, operations=ops,
                          commit_message=a.message or "Add the model files")
    elif a.path:
        name = os.path.basename(a.path)
        print(f"uploading {name} ({os.path.getsize(a.path)} B) to {a.repo_id}", flush=True)
        api.upload_file(path_or_fileobj=a.path, path_in_repo=name, repo_id=a.repo_id,
                        commit_message=a.message or f"Add {name}")
        want.append((name, a.sha256, None))
    if a.readme and not a.add:
        api.upload_file(path_or_fileobj=a.readme, path_in_repo="README.md", repo_id=a.repo_id,
                        commit_message=a.message or "README: describe the files")
    info = api.model_info(a.repo_id, files_metadata=True)
    print(f"revision={info.sha}", flush=True)
    if want:
        ok = all([check(info, name, sha, size) for name, sha, size in want])
        print(f"sha256_ok={1 if ok else 0}", flush=True)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
