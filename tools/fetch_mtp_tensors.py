#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
fetch_mtp_tensors.py — fetch ONLY the `mtp.*` tensors of a Qwen3.5/3.6 HF repo by HTTP range
reads of the safetensors shards, instead of downloading every shard for a fraction of a GB.
Used for Qwen/Qwen3.5-9B (STEP 4 S2: the grafted MTP draft head for Ornith) and
Qwen/Qwen3.6-35B-A3B (K80 MTP Gate 1, ~1.7 GB of the ~70 GB checkpoint).

Uses the logged-in huggingface_hub token. Writes one safetensors file (all mtp tensors,
original dtypes) into the given directory (gitignored: models/); config.json,
model.safetensors.index.json and the tokenizer files are fetched next to it when absent
(llama.cpp's convert_hf_to_gguf.py --mtp reads them).

Usage: fetch_mtp_tensors.py <out_dir> [repo]   (default repo Qwen/Qwen3.5-9B;
       run with the python that has huggingface_hub)
"""
import json
import os
import struct
import sys
import urllib.request

from huggingface_hub import get_token, hf_hub_download, hf_hub_url

DEFAULT_REPO = "Qwen/Qwen3.5-9B"
SIDE_FILES = ("config.json", "model.safetensors.index.json", "tokenizer.json",
              "tokenizer_config.json", "generation_config.json", "chat_template.jinja")


def ranged(url, token, start, end):
    """Bytes [start, end] of `url` (inclusive), following redirects."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}",
                                               "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def fetch_side_files(repo, out):
    """The small non-weight files, skipping ones already present or absent from the repo."""
    for name in SIDE_FILES:
        if os.path.exists(os.path.join(out, name)):
            continue
        try:
            hf_hub_download(repo, name, local_dir=out)
        except Exception as err:  # optional files (chat_template.jinja) may not exist
            print(f"note: {name} not fetched from {repo}: {type(err).__name__}")


def main():
    out = sys.argv[1]
    repo = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_REPO
    os.makedirs(out, exist_ok=True)
    fetch_side_files(repo, out)
    token = get_token()
    index = json.load(open(os.path.join(out, "model.safetensors.index.json")))
    wanted = {k: v for k, v in index["weight_map"].items() if k.startswith("mtp.")}
    by_shard = {}
    for name, shard in wanted.items():
        by_shard.setdefault(shard, []).append(name)

    tensors = {}
    for shard, names in sorted(by_shard.items()):
        url = hf_hub_url(repo, shard)
        head_len = struct.unpack("<Q", ranged(url, token, 0, 7))[0]
        header = json.loads(ranged(url, token, 8, 8 + head_len - 1))
        base = 8 + head_len
        for name in sorted(names):
            meta = header[name]
            begin, end = meta["data_offsets"]
            data = ranged(url, token, base + begin, base + end - 1)
            assert len(data) == end - begin, (name, len(data), end - begin)
            tensors[name] = (meta["dtype"], meta["shape"], data)
            print(f"{name:55s} {meta['dtype']:5s} {meta['shape']} {len(data)} B", flush=True)

    # One safetensors file, tensors in name order.
    header = {}
    offset = 0
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    path = os.path.join(out, "mtp.safetensors")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name in sorted(tensors):
            f.write(tensors[name][2])
    print(f"wrote {path}: {len(tensors)} tensors, {offset} B of data")


if __name__ == "__main__":
    main()
