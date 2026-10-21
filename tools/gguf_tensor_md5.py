#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
gguf_tensor_md5.py — md5 of named tensors' raw bytes in a GGUF (no dependencies), to check that
two files carry byte-identical tensors (e.g. the Ornith MTP head's output.weight vs the trunk's).

Usage: gguf_tensor_md5.py <file.gguf> <tensor name>...
"""
import hashlib
import struct
import sys

# ggml type -> (block elements, block bytes)
TYPES = {0: (1, 4), 1: (1, 2), 8: (32, 34), 30: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22),
         7: (32, 24), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
         14: (256, 210), 15: (256, 292)}


def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def rd_str(f):
    (n,) = rd(f, "Q")
    return f.read(n).decode("utf-8", "replace")


def skip_value(f, t):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in sizes:
        f.read(sizes[t])
    elif t == 8:
        rd_str(f)
    elif t == 9:
        (et,) = rd(f, "I")
        (n,) = rd(f, "Q")
        if et in sizes:
            f.read(sizes[et] * n)
        else:
            for _ in range(n):
                skip_value(f, et)
    else:
        raise ValueError(f"gguf value type {t}")


def main():
    path, names = sys.argv[1], set(sys.argv[2:])
    with open(path, "rb") as f:
        magic, version = rd(f, "4sI")
        assert magic == b"GGUF", magic
        n_tensors, n_kv = rd(f, "QQ")
        alignment = 32
        for _ in range(n_kv):
            key = rd_str(f)
            (t,) = rd(f, "I")
            if key == "general.alignment":
                (alignment,) = rd(f, "I")
            else:
                skip_value(f, t)
        infos = []
        for _ in range(n_tensors):
            name = rd_str(f)
            (nd,) = rd(f, "I")
            dims = rd(f, "Q" * nd)
            (typ,) = rd(f, "I")
            (off,) = rd(f, "Q")
            infos.append((name, dims, typ, off))
        base = (f.tell() + alignment - 1) // alignment * alignment
        for name, dims, typ, off in infos:
            if name not in names:
                continue
            n = 1
            for d in dims:
                n *= d
            be, bb = TYPES[typ]
            size = n // be * bb
            f.seek(base + off)
            h = hashlib.md5()
            left = size
            while left:
                chunk = f.read(min(left, 1 << 24))
                h.update(chunk)
                left -= len(chunk)
            print(f"{h.hexdigest()}  {name} type {typ} dims {list(dims)} {size} B")


if __name__ == "__main__":
    main()
