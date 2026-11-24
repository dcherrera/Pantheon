#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
graft_mtp_block.py — give a trunk GGUF that has no MTP head the head of another GGUF of the
same model: copy the donor's trailing `blk.<N>.*` block (N = the trunk's block_count) byte for
byte, set `<arch>.block_count` to the donor's and copy `<arch>.nextn_predict_layers`.

Why: llama-quantize from the BF16 GGUF writes no head (the BF16 GGUF was converted without
--mtp), and the bank no longer has the head's BF16 tensors. bartowski's Qwen3.6-35B-A3B files
carry the head as blk.40.* with every matrix already Q8_0 (byte-identical to
`convert_hf_to_gguf.py --mtp --outtype q8_0`, results/mtp_gate1_acceptance.md §1), so a Q8_0
trunk + that block is the same file a Q8_0 conversion with --mtp would write.

Streams: no tensor is loaded whole (64 MiB copy chunks); peak RSS is the headers.
After writing, every tensor's bytes are re-read and md5-compared against its source
(GRAFT_RED=1 flips one byte of the first grafted tensor while copying: the check must FAIL).

Usage: graft_mtp_block.py <trunk.gguf> <donor.gguf> <out.gguf>
"""
import hashlib
import os
import struct
import sys

CHUNK = 64 << 20
# ggml type -> (block elements, block bytes)
TYPES = {0: (1, 4), 1: (1, 2), 8: (32, 34), 30: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22),
         7: (32, 24), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
         14: (256, 210), 15: (256, 292)}
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def rd_str(f):
    (n,) = rd(f, "Q")
    return f.read(n).decode("utf-8")


def skip_value(f, t):
    if t in SCALAR:
        f.read(SCALAR[t])
    elif t == 8:
        rd_str(f)
    elif t == 9:
        (et,) = rd(f, "I")
        (n,) = rd(f, "Q")
        if et in SCALAR:
            f.read(SCALAR[et] * n)
        else:
            for _ in range(n):
                skip_value(f, et)
    else:
        raise ValueError(f"unknown kv type {t}")


def read_value(f, t):
    pos = f.tell()
    if t in (4, 5, 6, 10, 11, 12, 0, 1, 2, 3, 7):
        fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q",
               11: "q", 12: "d"}[t]
        return rd(f, fmt)[0]
    if t == 8:
        return rd_str(f)
    f.seek(pos)
    skip_value(f, t)
    return None


def parse(path):
    """Header of a GGUF: kv list [(key, type, raw_bytes, value)], tensors, data offset."""
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", path
    (ver,) = rd(f, "I")
    n_t, n_kv = rd(f, "QQ")
    kvs = []
    for _ in range(n_kv):
        start = f.tell()
        key = rd_str(f)
        (t,) = rd(f, "I")
        val = read_value(f, t)
        end = f.tell()
        f.seek(start)
        raw = f.read(end - start)
        kvs.append((key, t, raw, val))
    tensors = []
    for _ in range(n_t):
        name = rd_str(f)
        (nd,) = rd(f, "I")
        dims = list(rd(f, "Q" * nd))
        (tt,) = rd(f, "I")
        (off,) = rd(f, "Q")
        tensors.append({"name": name, "dims": dims, "type": tt, "off": off})
    align = next((v for k, _, _, v in kvs if k == "general.alignment"), 32)
    data0 = (f.tell() + align - 1) // align * align
    f.close()
    for t in tensors:
        be, bb = TYPES[t["type"]]
        n = 1
        for d in t["dims"]:
            n *= d
        assert n % be == 0, t["name"]
        t["bytes"] = n // be * bb
    return ver, kvs, tensors, data0, align


def kv_u32(key, val):
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<II", 4, val)


def copy_range(src, dst, n, md5, red=False):
    first = True
    while n > 0:
        b = src.read(min(CHUNK, n))
        if red and first and b:
            b = bytes([b[0] ^ 0xFF]) + b[1:]
        first = False
        md5.update(b)
        dst.write(b)
        n -= len(b)


def tensor_md5(path, data0, t):
    h = hashlib.md5()
    with open(path, "rb") as f:
        f.seek(data0 + t["off"])
        n = t["bytes"]
        while n > 0:
            b = f.read(min(CHUNK, n))
            h.update(b)
            n -= len(b)
    return h.hexdigest()


def main():
    trunk, donor, out = sys.argv[1:4]
    ver, kvs, tts, tdata0, align = parse(trunk)
    _, dkvs, dts, ddata0, _ = parse(donor)
    arch = next(v for k, _, _, v in kvs if k == "general.architecture")
    bc_key, nx_key = f"{arch}.block_count", f"{arch}.nextn_predict_layers"
    tbc = next(v for k, _, _, v in kvs if k == bc_key)
    dbc = next(v for k, _, _, v in dkvs if k == bc_key)
    nx = next((raw for k, _, raw, _ in dkvs if k == nx_key), None)
    if any(t["name"].startswith(f"blk.{tbc}.") for t in tts):
        sys.exit(f"FAIL: {trunk} already has blk.{tbc}.*")
    if dbc != tbc + 1 or nx is None:
        sys.exit(f"FAIL: donor block_count {dbc} (want {tbc + 1}) / {nx_key} present={nx is not None}")
    head = [t for t in dts if t["name"].startswith(f"blk.{tbc}.")]
    if not head:
        sys.exit(f"FAIL: donor has no blk.{tbc}.* tensors")

    out_kvs = []
    for k, t, raw, v in kvs:
        if k == nx_key:
            continue
        out_kvs.append(kv_u32(bc_key, dbc) if k == bc_key else raw)
        if k == bc_key:
            out_kvs.append(nx)
    plan = [(trunk, tdata0, t) for t in tts] + [(donor, ddata0, t) for t in head]
    off = 0
    infos = []
    for _, _, t in plan:
        off = (off + align - 1) // align * align
        nm = t["name"].encode()
        infos.append(struct.pack("<Q", len(nm)) + nm + struct.pack("<I", len(t["dims"])) +
                     struct.pack("<" + "Q" * len(t["dims"]), *t["dims"]) +
                     struct.pack("<IQ", t["type"], off))
        t["new_off"] = off
        off += t["bytes"]

    red = os.environ.get("GRAFT_RED") == "1"
    want = {}
    with open(out, "wb") as o:
        o.write(b"GGUF" + struct.pack("<IQQ", ver, len(plan), len(out_kvs)))
        for raw in out_kvs:
            o.write(raw)
        for i in infos:
            o.write(i)
        hdr = o.tell()
        o.write(b"\0" * ((hdr + align - 1) // align * align - hdr))
        data0 = o.tell()
        srcs = {trunk: open(trunk, "rb"), donor: open(donor, "rb")}
        for idx, (path, sdata0, t) in enumerate(plan):
            pad = data0 + t["new_off"] - o.tell()
            o.write(b"\0" * pad)
            s = srcs[path]
            s.seek(sdata0 + t["off"])
            h = hashlib.md5()
            copy_range(s, o, t["bytes"], h, red=red and idx == len(tts))
            want[t["name"]] = (path, sdata0, t)
            if idx % 100 == 0:
                print(f"  {idx}/{len(plan)} {t['name']}", flush=True)
        for s in srcs.values():
            s.close()
    print(f"wrote {out}: {len(tts)} trunk + {len(head)} head tensors, {bc_key}={dbc}, "
          f"{os.path.getsize(out)} B", flush=True)

    _, okvs, ots, odata0, _ = parse(out)
    bad = 0
    for t in ots:
        path, sdata0, st = want[t["name"]]
        if t["type"] != st["type"] or t["dims"] != st["dims"] or \
                tensor_md5(out, odata0, t) != tensor_md5(path, sdata0, st):
            print(f"  MISMATCH {t['name']}")
            bad += 1
    if bad:
        sys.exit(f"FAIL: {bad} tensors differ from their source")
    print(f"verified: all {len(ots)} tensors byte-identical to their source")


if __name__ == "__main__":
    main()
