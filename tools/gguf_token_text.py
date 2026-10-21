#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
gguf_token_text.py — print the vocab text of token ids from a GGUF's tokenizer.ggml.tokens
(no dependencies), to read what a token-id diff in a serve log actually says.

Usage: gguf_token_text.py <file.gguf> <id>...
"""
import struct
import sys

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def rd_str(f):
    (n,) = rd(f, "Q")
    return f.read(n).decode("utf-8", "replace")


def skip_value(f, t):
    if t in SIZES:
        f.read(SIZES[t])
    elif t == 8:
        rd_str(f)
    elif t == 9:
        (et,) = rd(f, "I")
        (n,) = rd(f, "Q")
        if et in SIZES:
            f.read(SIZES[et] * n)
        else:
            for _ in range(n):
                skip_value(f, et)
    else:
        raise ValueError(f"gguf value type {t}")


def main():
    path, ids = sys.argv[1], [int(x) for x in sys.argv[2:]]
    with open(path, "rb") as f:
        magic, _version = rd(f, "4sI")
        assert magic == b"GGUF", magic
        _n_tensors, n_kv = rd(f, "QQ")
        for _ in range(n_kv):
            key = rd_str(f)
            (t,) = rd(f, "I")
            if key != "tokenizer.ggml.tokens":
                skip_value(f, t)
                continue
            (_et,) = rd(f, "I")
            (n,) = rd(f, "Q")
            vocab = [rd_str(f) for _ in range(n)]
            for i in ids:
                print(i, repr(vocab[i]) if 0 <= i < n else "(out of range)")
            return
    sys.exit("no tokenizer.ggml.tokens")


if __name__ == "__main__":
    main()
