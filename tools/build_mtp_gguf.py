#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
build_mtp_gguf.py — the Ornith MTP draft head (protoLabsAI/Ornith-1.0-9B-MTP-GGUF,
mtp-head/mtp-Ornith-1.0-9B-head-Q8_0.gguf) as a small GGUF the TT converter turns into a
second artifact: only the nextn block (blk.32.*), renamed `mtp.*`, with eh_proj split along its
input into the embedding half and the hidden half (fc(cat(enorm(emb), hnorm(h))) =
eh_proj_e @ enorm(emb) + eh_proj_h @ hnorm(h)), so the card needs no concat. Byte slices of Q8_0
or BF16 rows (both split cleanly at input 4096), no requantization. output.weight / token_embd
are dropped: the trunk's are used (tools/gguf_tensor_md5.py checks they are byte-identical).
After writing, the output is read back and eh_proj_e || eh_proj_h must equal the source rows
byte for byte (MTP_RED_SWAP_HALVES=1 writes the halves swapped: that check must then fail).

Usage: build_mtp_gguf.py <head.gguf> <out.gguf>   (python with the gguf package)
"""
import os
import sys

import numpy as np
from gguf import GGML_QUANT_SIZES, GGUFReader, GGUFWriter, GGMLQuantizationType

EH_PROJ = "blk.32.nextn.eh_proj.weight"
EH_PROJ_SPLIT_TYPES = (GGMLQuantizationType.Q8_0, GGMLQuantizationType.BF16)


def split_eh_proj_rows(t):
    """eh_proj's rows as bytes, cut at input column n_in / 2: (embedding half, hidden half)."""
    assert t.tensor_type in EH_PROJ_SPLIT_TYPES, t.tensor_type.name
    n_in, n_out = int(t.shape[0]), int(t.shape[1])  # ne0 = 8192 in, ne1 = 4096 out
    block_size, type_size = GGML_QUANT_SIZES[t.tensor_type]
    raw = np.asarray(t.data).reshape(n_out, -1)      # one row of bytes per output
    half = raw.shape[1] // 2
    assert n_in == 8192 and half == (n_in // 2) // block_size * type_size, (n_in, raw.shape)
    return raw[:, :half], raw[:, half:]


def verify_eh_proj_split(src_tensor, out_path):
    """Read the written GGUF back: eh_proj_e || eh_proj_h must be the source rows, byte for byte."""
    written = {t.name: np.asarray(t.data) for t in GGUFReader(out_path).tensors}
    n_out = int(src_tensor.shape[1])
    joined = np.concatenate([written["mtp.eh_proj_e.weight"].reshape(n_out, -1),
                             written["mtp.eh_proj_h.weight"].reshape(n_out, -1)], axis=1)
    if not np.array_equal(joined, np.asarray(src_tensor.data).reshape(n_out, -1)):
        sys.exit("FAIL: eh_proj_e || eh_proj_h is not the source eh_proj")
    print("verified: eh_proj_e || eh_proj_h == source eh_proj, byte for byte")

RENAME = {
    "blk.32.attn_norm.weight": "mtp.attn_norm.weight",
    "blk.32.post_attention_norm.weight": "mtp.post_attention_norm.weight",
    "blk.32.attn_q.weight": "mtp.attn_q.weight",
    "blk.32.attn_k.weight": "mtp.attn_k.weight",
    "blk.32.attn_v.weight": "mtp.attn_v.weight",
    "blk.32.attn_output.weight": "mtp.attn_output.weight",
    "blk.32.attn_q_norm.weight": "mtp.attn_q_norm.weight",
    "blk.32.attn_k_norm.weight": "mtp.attn_k_norm.weight",
    "blk.32.ffn_gate.weight": "mtp.ffn_gate.weight",
    "blk.32.ffn_up.weight": "mtp.ffn_up.weight",
    "blk.32.ffn_down.weight": "mtp.ffn_down.weight",
    "blk.32.nextn.enorm.weight": "mtp.enorm.weight",
    "blk.32.nextn.hnorm.weight": "mtp.hnorm.weight",
    "blk.32.nextn.shared_head_norm.weight": "mtp.shared_head_norm.weight",
}


def main():
    src, out = sys.argv[1], sys.argv[2]
    r = GGUFReader(src)
    w = GGUFWriter(out, "qwen35")
    for key, field in r.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture" or key.startswith("tokenizer."):
            continue
        val = field.contents()
        vtype = field.types[0]
        if vtype.name == "ARRAY":
            w.add_key_value(key, val, vtype, sub_type=field.types[-1])
        else:
            w.add_key_value(key, val, vtype)
    eh_proj = None
    for t in r.tensors:
        data = np.asarray(t.data)
        if t.name == EH_PROJ:
            eh_proj = t
            emb_half, hid_half = split_eh_proj_rows(t)
            if os.environ.get("MTP_RED_SWAP_HALVES") == "1":
                emb_half, hid_half = hid_half, emb_half
            for name, part in (("mtp.eh_proj_e.weight", emb_half), ("mtp.eh_proj_h.weight", hid_half)):
                w.add_tensor(name, np.ascontiguousarray(part), raw_shape=part.shape,
                             raw_dtype=t.tensor_type)
                print(f"{name:34s} {t.tensor_type.name} [{int(t.shape[0]) // 2}, {int(t.shape[1])}]")
            continue
        if t.name not in RENAME:
            continue
        w.add_tensor(RENAME[t.name], np.ascontiguousarray(data), raw_shape=data.shape,
                     raw_dtype=t.tensor_type)
        print(f"{RENAME[t.name]:34s} {t.tensor_type.name} {list(t.shape)}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", out)
    if eh_proj is None:
        sys.exit(f"FAIL: {src} has no {EH_PROJ}")
    verify_eh_proj_split(eh_proj, out)


if __name__ == "__main__":
    main()
