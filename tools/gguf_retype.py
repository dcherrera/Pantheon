#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""gguf_retype.py — retype selected tensors of a GGUF in place-of-copy, leaving
every other tensor byte-identical.

Why this exists: `llama-quantize --allow-requantize` rebuilds *every* tensor
(dequantise -> requantise), so a one-tensor change cannot be isolated from the
rounding it does to the other 700. Phase 2 of the batched-decode campaign needs
exactly two tensors moved Q5_K -> Q4_K (`blk.0/blk.1.ffn_down_exps.weight`, the
only two layers whose `ffn_down_exps` is not a type `gpu_moe_setup` can group)
with everything else preserved bit-for-bit.

How: the GGUF header + KV block + tensor-info block are copied verbatim; only
the `type` (u32) and `offset` (u64) fields of the info records are patched, both
fixed-width, so the header length is unchanged. Tensor data is streamed through
in info order, copied byte-for-byte except for the retyped tensors, which are
dequantised to f32 and requantised with ggml's own `ggml_quantize_chunk` (the
same code path `llama-quantize` uses, no imatrix).

Requires: libggml-base.so from a llama.cpp build (exports `ggml_quantize_chunk`
and `dequantize_row_*`). numpy is NOT required.

Usage:
  # list tensors (name, type, dims, offset, bytes)
  python3 tools/gguf_retype.py --in model.gguf --dump [--filter ffn_down_exps]

  # retype
  python3 tools/gguf_retype.py --in model.gguf --out model.retyped.gguf \
      --ggml-lib /path/to/libggml-base.so \
      --retype blk.0.ffn_down_exps.weight=Q4_K \
      --retype blk.1.ffn_down_exps.weight=Q4_K

  # re-parse the output and diff the tensor table against the input
  python3 tools/gguf_retype.py --in model.gguf --verify model.retyped.gguf

@status Working; used for Phase 2 (saber blk.0/blk.1 ffn_down_exps Q5_K -> Q4_K)
@issues Only the quant types listed in GGML_TYPES below are known to this tool.
@todo None
"""
import argparse
import ctypes
import os
import struct
import sys

# --- ggml type table: name -> (enum value, block size in elems, bytes/block) --
GGML_TYPES = {
    "F32":   (0,   1,   4),
    "F16":   (1,   1,   2),
    "Q4_0":  (2,  32,  18),
    "Q4_1":  (3,  32,  20),
    "Q5_0":  (6,  32,  22),
    "Q5_1":  (7,  32,  24),
    "Q8_0":  (8,  32,  34),
    "Q8_1":  (9,  32,  36),
    "Q2_K":  (10, 256, 84),
    "Q3_K":  (11, 256, 110),
    "Q4_K":  (12, 256, 144),
    "Q5_K":  (13, 256, 176),
    "Q6_K":  (14, 256, 210),
    "Q8_K":  (15, 256, 292),
    "BF16":  (30,  1,   2),
}
TYPE_BY_ID = {v[0]: k for k, v in GGML_TYPES.items()}

# GGUF metadata value types
(U8, I8, U16, I16, U32, I32, F32V, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)
_FIXED = {U8: 1, I8: 1, U16: 2, I16: 2, U32: 4, I32: 4, F32V: 4, BOOL: 1,
          U64: 8, I64: 8, F64: 8}


class Reader:
    def __init__(self, buf):
        self.b = buf
        self.p = 0

    def read_u32(self):
        v = struct.unpack_from("<I", self.b, self.p)[0]; self.p += 4; return v

    def read_u64(self):
        v = struct.unpack_from("<Q", self.b, self.p)[0]; self.p += 8; return v

    def read_string(self):
        n = self.read_u64()
        v = self.b[self.p:self.p + n]; self.p += n
        return v.decode("utf-8", "replace")

    def skip_value(self, t):
        if t in _FIXED:
            self.p += _FIXED[t]
        elif t == STRING:
            # NB: `self.p += self.read_u64()` would drop the length prefix —
            # augmented assignment reads self.p before the RHS advances it.
            n = self.read_u64()
            self.p += n
        elif t == ARRAY:
            et = self.read_u32()
            n = self.read_u64()
            if et in _FIXED:
                self.p += _FIXED[et] * n
            elif et == STRING:
                for _ in range(n):
                    ln = self.read_u64()
                    self.p += ln
            else:
                raise ValueError(f"nested array type {et}")
        else:
            raise ValueError(f"unknown value type {t}")


class Tensor:
    __slots__ = ("name", "dims", "type", "offset", "nbytes",
                 "type_pos", "off_pos", "index")

    @property
    def nelem(self):
        n = 1
        for d in self.dims:
            n *= d
        return n


def quant_row_bytes(type_name, ne0):
    _, blck, tsz = GGML_TYPES[type_name]
    if ne0 % blck:
        raise ValueError(f"row length {ne0} not a multiple of {type_name} "
                         f"block {blck}")
    return ne0 // blck * tsz


def tensor_total_bytes(type_name, dims):
    nrows = 1
    for d in dims[1:]:
        nrows *= d
    return quant_row_bytes(type_name, dims[0]) * nrows


def pad_to_alignment(offset, align):
    return (offset + align - 1) // align * align


def parse_gguf_metadata(path):
    """Parse header + KV + tensor infos. Returns (header_bytes, tensors,
    data_start, alignment, kv)."""
    with open(path, "rb") as f:
        head = f.read(64 * 1024 * 1024)      # metadata is far under this
    if head[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    r = Reader(head)
    r.p = 4
    version = r.read_u32()
    n_tensors = r.read_u64()
    n_kv = r.read_u64()
    kv = {}
    for _ in range(n_kv):
        key = r.read_string()
        t = r.read_u32()
        vpos = r.p
        if t == U32:
            kv[key] = struct.unpack_from("<I", head, vpos)[0]
        elif t == STRING:
            save = r.p
            kv[key] = r.read_string()
            r.p = save
        r.skip_value(t)
    tensors = []
    for i in range(n_tensors):
        t = Tensor()
        t.index = i
        t.name = r.read_string()
        nd = r.read_u32()
        t.dims = [r.read_u64() for _ in range(nd)]
        t.type_pos = r.p
        tid = r.read_u32()
        t.type = TYPE_BY_ID.get(tid, f"UNK{tid}")
        t.off_pos = r.p
        t.offset = r.read_u64()
        t.nbytes = tensor_total_bytes(t.type, t.dims)
        tensors.append(t)
    align = int(kv.get("general.alignment", 32))
    data_start = pad_to_alignment(r.p, align)
    return head[:data_start], tensors, data_start, align, version, kv


def cmd_dump(args):
    header, tensors, data_start, align, version, kv = parse_gguf_metadata(args.inp)
    print(f"gguf v{version} tensors={len(tensors)} align={align} "
          f"data_start={data_start} file={os.path.getsize(args.inp)}")
    counts = {}
    for t in tensors:
        counts[t.type] = counts.get(t.type, 0) + 1
        if args.filter and args.filter not in t.name:
            continue
        print(f"  {t.name:44s} {t.type:5s} dims={t.dims} "
              f"off={t.offset} bytes={t.nbytes}")
    print("type histogram: " + ", ".join(f"{k}={v}" for k, v in
                                         sorted(counts.items())))
    # self-check: recompute offsets with the standard packing rule
    off = 0
    bad = 0
    for t in tensors:
        if t.offset != off:
            bad += 1
            if bad < 5:
                print(f"  OFFSET MISMATCH {t.name}: file={t.offset} calc={off}")
        off = pad_to_alignment(off + t.nbytes, align)
    print(f"offset-model check: {bad} mismatches; "
          f"calc end={data_start + off} actual={os.path.getsize(args.inp)}")


def load_ggml_library(libpath):
    lib = ctypes.CDLL(libpath)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float)]
    return lib


def dequant_row_fn(lib, type_name):
    sym = f"dequantize_row_{type_name.lower()}"
    # ggml keeps the K-quant suffix capitalised: dequantize_row_q5_K
    if type_name.endswith("_K"):
        sym = f"dequantize_row_{type_name[:-2].lower()}_K"
    fn = getattr(lib, sym)
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                   ctypes.c_int64]
    return fn


def cmd_retype(args):
    header, tensors, data_start, align, version, kv = parse_gguf_metadata(args.inp)
    want = {}
    for spec in args.retype:
        name, _, ty = spec.partition("=")
        if ty not in GGML_TYPES:
            sys.exit(f"unknown target type {ty}")
        want[name] = ty
    by_name = {t.name: t for t in tensors}
    for name in want:
        if name not in by_name:
            sys.exit(f"tensor not in model: {name}")
    lib = load_ggml_library(args.ggml_lib)

    # 1. rebuild the info block: new types + repacked offsets
    hdr = bytearray(header)
    off = 0
    plan = []
    for t in tensors:
        new_type = want.get(t.name, t.type)
        nbytes = tensor_total_bytes(new_type, t.dims)
        plan.append((t, new_type, off, nbytes))
        struct.pack_into("<I", hdr, t.type_pos, GGML_TYPES[new_type][0])
        struct.pack_into("<Q", hdr, t.off_pos, off)
        off = pad_to_alignment(off + nbytes, align)
    total = data_start + off
    print(f"[retype] {len(want)} tensor(s); out size {total} "
          f"({total / 2**30:.3f} GiB) vs in {os.path.getsize(args.inp)}")

    # 2. stream the data section
    CHUNK = 8 << 20
    with open(args.inp, "rb") as fi, open(args.out, "wb") as fo:
        fo.write(hdr)
        written = 0
        for t, new_type, new_off, nbytes in plan:
            if written < new_off:                      # inter-tensor padding
                fo.write(b"\0" * (new_off - written))
                written = new_off
            fi.seek(data_start + t.offset)
            if new_type == t.type:
                left = t.nbytes
                while left:
                    b = fi.read(min(CHUNK, left))
                    if len(b) != min(CHUNK, left):
                        sys.exit(f"short read on {t.name}")
                    fo.write(b)
                    left -= len(b)
                written += t.nbytes
            else:
                n_per_row = t.dims[0]
                nrows = t.nelem // n_per_row
                src_rs = quant_row_bytes(t.type, n_per_row)
                dst_rs = quant_row_bytes(new_type, n_per_row)
                deq = dequant_row_fn(lib, t.type)
                rows_per_batch = max(1, (4 << 20) // (n_per_row * 4))
                fbuf = (ctypes.c_float * (rows_per_batch * n_per_row))()
                dbuf = ctypes.create_string_buffer(rows_per_batch * dst_rs)
                print(f"[retype] {t.name}: {t.type} -> {new_type}, "
                      f"{nrows} rows x {n_per_row} "
                      f"({t.nbytes} -> {nbytes} bytes)")
                done = 0
                while done < nrows:
                    nb = min(rows_per_batch, nrows - done)
                    raw = fi.read(nb * src_rs)
                    if len(raw) != nb * src_rs:
                        sys.exit(f"short read on {t.name}")
                    deq(ctypes.cast(ctypes.c_char_p(raw), ctypes.c_void_p),
                        fbuf, nb * n_per_row)
                    got = lib.ggml_quantize_chunk(
                        GGML_TYPES[new_type][0], fbuf,
                        ctypes.cast(dbuf, ctypes.c_void_p),
                        0, nb, n_per_row, None)
                    if got != nb * dst_rs:
                        sys.exit(f"quantize_chunk returned {got}, "
                                 f"expected {nb * dst_rs}")
                    fo.write(dbuf.raw[:nb * dst_rs])
                    done += nb
                written += nbytes
        if written < off:
            fo.write(b"\0" * (off - written))
    print(f"[retype] wrote {args.out} ({os.path.getsize(args.out)} bytes)")


def cmd_verify(args):
    ha, ta, dsa, aa, va, _ = parse_gguf_metadata(args.inp)
    hb, tb, dsb, ab, vb, _ = parse_gguf_metadata(args.verify)
    ok = True
    if len(ta) != len(tb) or dsa != dsb or aa != ab or va != vb:
        print(f"HEADER DIFF: n={len(ta)}/{len(tb)} data_start={dsa}/{dsb} "
              f"align={aa}/{ab} ver={va}/{vb}")
        ok = False
    fa = open(args.inp, "rb")
    fb = open(args.verify, "rb")
    changed = 0
    for x, y in zip(ta, tb):
        if x.name != y.name or x.dims != y.dims:
            print(f"TENSOR DIFF: {x.name} vs {y.name}")
            ok = False
            continue
        if x.type != y.type:
            changed += 1
            print(f"  retyped {x.name}: {x.type} -> {y.type} "
                  f"({x.nbytes} -> {y.nbytes} bytes)")
            continue
        if x.nbytes != y.nbytes:
            print(f"SIZE DIFF: {x.name}")
            ok = False
            continue
        # byte-compare the payload
        fa.seek(dsa + x.offset)
        fb.seek(dsb + y.offset)
        left = x.nbytes
        while left:
            n = min(8 << 20, left)
            if fa.read(n) != fb.read(n):
                print(f"BYTE DIFF in unchanged tensor {x.name}")
                ok = False
                break
            left -= n
    fa.close()
    fb.close()
    # metadata block: identical except the patched type/offset fields
    diffs = sum(1 for i in range(min(len(ha), len(hb))) if ha[i] != hb[i])
    print(f"metadata bytes differing: {diffs} "
          f"(expected: type+offset fields only)")
    print(f"retyped tensors: {changed}")
    print("VERIFY: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--verify", default=None, help="output file to check")
    ap.add_argument("--retype", action="append", default=[],
                    help="name=TYPE, repeatable")
    ap.add_argument("--ggml-lib", default=os.environ.get("GGML_LIB"),
                    help="path to libggml-base.so (default: $GGML_LIB)")
    a = ap.parse_args()
    if a.dump:
        return cmd_dump(a)
    if a.verify:
        return cmd_verify(a)
    if not a.out or not a.retype:
        ap.error("need --out and at least one --retype (or --dump/--verify)")
    if not a.ggml_lib:
        ap.error("retype needs --ggml-lib or GGML_LIB (libggml-base.so)")
    return cmd_retype(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
