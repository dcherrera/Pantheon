#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""gguf_extract_chat_template.py: read a GGUF's KV header (never the tensor
data) and write `tokenizer.chat_template` plus the added-token list (token_type
CONTROL=3 or USER_DEFINED=4) the tokenizer splits out before BPE.  Given an HF
model directory instead, copy its chat_template.jinja and list the
tokenizer.json `added_tokens`.

Stdlib only.  The KV block is walked sequentially with seeks; the tokens and
token_type arrays are the only large values and are read in place.

Usage:
  python3 tests/host/gguf_extract_chat_template.py MODEL.gguf|HF_DIR \
      --template OUT.jinja --added-tokens OUT.txt
"""

import argparse
import json
import os
import struct
import sys

GGUF_TYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
SCALAR_FORMATS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
                  7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_exact(handle, count):
    data = handle.read(count)
    if len(data) != count:
        raise EOFError("GGUF header truncated")
    return data


def read_string(handle):
    (length,) = struct.unpack("<Q", read_exact(handle, 8))
    return read_exact(handle, length).decode("utf-8")


def read_value(handle, value_type, keep):
    """Read one value; return it when `keep`, else skip it and return None."""
    if value_type == GGUF_TYPE_STRING:
        if keep:
            return read_string(handle)
        (length,) = struct.unpack("<Q", read_exact(handle, 8))
        handle.seek(length, 1)
        return None
    if value_type == GGUF_TYPE_ARRAY:
        (elem_type,) = struct.unpack("<I", read_exact(handle, 4))
        (count,) = struct.unpack("<Q", read_exact(handle, 8))
        if elem_type != GGUF_TYPE_STRING and not keep:
            handle.seek(GGUF_TYPE_SIZES[elem_type] * count, 1)
            return None
        return [read_value(handle, elem_type, keep) for _ in range(count)]
    raw = read_exact(handle, GGUF_TYPE_SIZES[value_type])
    return struct.unpack(SCALAR_FORMATS[value_type], raw)[0] if keep else None


def read_gguf_kv(path, wanted_keys):
    """Return {key: value} for the wanted keys found in the GGUF KV header."""
    found = {}
    with open(path, "rb") as handle:
        if read_exact(handle, 4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version, _tensor_count, kv_count = struct.unpack("<IQQ", read_exact(handle, 20))
        if version < 2:
            raise ValueError(f"{path}: GGUF v{version} unsupported")
        for _ in range(kv_count):
            key = read_string(handle)
            (value_type,) = struct.unpack("<I", read_exact(handle, 4))
            value = read_value(handle, value_type, key in wanted_keys)
            if key in wanted_keys:
                found[key] = value
    return found


def read_hf_directory(path):
    """Return (chat template, added-token strings) from an HF model directory."""
    with open(os.path.join(path, "chat_template.jinja"), encoding="utf-8") as handle:
        template = handle.read()
    with open(os.path.join(path, "tokenizer.json"), encoding="utf-8") as handle:
        added = [entry["content"] for entry in json.load(handle)["added_tokens"]]
    return template, added


def read_gguf_template(path):
    """Return (chat template, added-token strings) from a GGUF header."""
    keys = {"tokenizer.chat_template", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type"}
    kv = read_gguf_kv(path, keys)
    if "tokenizer.chat_template" not in kv:
        sys.exit(f"{path}: no tokenizer.chat_template")
    tokens = kv.get("tokenizer.ggml.tokens") or []
    types = kv.get("tokenizer.ggml.token_type") or []
    added = [tok for tok, ttype in zip(tokens, types) if ttype in (3, 4)]
    return kv["tokenizer.chat_template"], added


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("gguf", help="GGUF file or HF model directory")
    parser.add_argument("--template", required=True)
    parser.add_argument("--added-tokens", required=True)
    args = parser.parse_args()
    if os.path.isdir(args.gguf):
        template, added = read_hf_directory(args.gguf)
    else:
        template, added = read_gguf_template(args.gguf)
    with open(args.template, "w", encoding="utf-8", newline="") as out:
        out.write(template)
    with open(args.added_tokens, "w", encoding="utf-8", newline="") as out:
        for tok in added:
            out.write(tok + "\n")
    print(f"template {len(template)} B, {len(added)} added tokens")


if __name__ == "__main__":
    main()
