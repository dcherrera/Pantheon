# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""Readers shared by the offline draft-acceptance tools (mtp_accept.py, mtp_finetune.py).

  * Gguf: mmap a GGUF file and read F32 / Q8_0 tensors as float32 rows.
  * load_dump: read a card hidden-state dump (PN_TT_DUMP_HIDDEN): per position int32 pos,
    int32 token, `planes` x `hid` float32.
"""
import mmap
import struct

import numpy as np



# ----------------------------------------------------------------------------- gguf
class Gguf:
    TYPE_SIZES = {0: (1, 4), 1: (1, 2), 8: (32, 34), 30: (1, 2)}

    def __init__(self, path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.o = 0
        assert self._take(4) == b"GGUF"
        self._u32()
        n_tensors, n_kv = self._u64(), self._u64()
        align = 32
        for _ in range(n_kv):
            key = self._str()
            t = self._u32()
            val = self._value(t)
            if key == "general.alignment":
                align = val
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._str()
            nd = self._u32()
            dims = [self._u64() for _ in range(nd)]
            t = self._u32()
            off = self._u64()
            self.tensors[name] = (dims, t, off)
        self.data = (self.o + align - 1) // align * align

    def _take(self, n):
        v = self.mm[self.o:self.o + n]
        self.o += n
        return v

    def _u32(self):
        return struct.unpack("<I", self._take(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self._take(8))[0]

    def _str(self):
        return self._take(self._u64()).decode("utf-8", "replace")

    def _value(self, t):
        fixed = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                 10: "<Q", 11: "<q", 12: "<d"}
        if t in fixed:
            fmt = fixed[t]
            return struct.unpack(fmt, self._take(struct.calcsize(fmt)))[0]
        if t == 8:
            return self._str()
        if t == 9:
            et = self._u32()
            n = self._u64()
            return [self._value(et) for _ in range(n)]
        raise ValueError(f"gguf value type {t}")

    def rows(self, name, first, count):
        """Rows [first, first+count) of a 2-D tensor (ne0 = row length) as float32."""
        dims, t, off = self.tensors[name]
        ne0 = dims[0]
        per, size = self.TYPE_SIZES[t]
        row_bytes = ne0 // per * size
        start = self.data + off + first * row_bytes
        buf = np.frombuffer(self.mm, dtype=np.uint8, count=count * row_bytes, offset=start)
        if t == 0:
            return buf.view(np.float32).reshape(count, ne0).copy()
        if t == 8:
            blocks = buf.reshape(count, ne0 // 32, 34)
            d = blocks[:, :, :2].copy().view(np.float16).astype(np.float32)
            q = blocks[:, :, 2:].copy().view(np.int8).astype(np.float32)
            return (q * d).reshape(count, ne0)
        raise ValueError(f"gguf type {t} for {name}")

    def vector(self, name):
        dims, t, off = self.tensors[name]
        assert t == 0
        return np.frombuffer(self.mm, dtype=np.float32, count=dims[0],
                             offset=self.data + off).copy()


def load_dump(path, planes=9, hid=4096):
    """(pos[n], token[n], hidden[n, planes, hid]) from a PN_TT_DUMP_HIDDEN file."""
    rec = 8 + planes * hid * 4
    raw = np.fromfile(path, dtype=np.uint8)
    n = raw.size // rec
    raw = raw[: n * rec].reshape(n, rec)
    head = raw[:, :8].copy().view(np.int32)
    hidden = raw[:, 8:].copy().view(np.float32).reshape(n, planes, hid)
    return head[:, 0], head[:, 1], hidden
