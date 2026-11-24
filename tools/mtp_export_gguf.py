#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""mtp_export_gguf.py — a fine-tuned MTP block (tools/mtp_finetune.py .npz) as the MTP GGUF the
TT converter takes (`ft_tt_convert_gguf <out.gguf> <out>.mtp.fttw --ledger bfp8`): the names and
split of tools/build_mtp_gguf.py (mtp.*, eh_proj as eh_proj_e | eh_proj_h), matrices BF16 (round
to nearest even from the float32 merge), norms F32, the source GGUF's metadata (no tokenizer).

Usage: mtp_export_gguf.py <trunk.gguf (metadata)> <ft.npz> <out.gguf>
"""
import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

NAMES = {
    "eh_e": "mtp.eh_proj_e.weight", "eh_h": "mtp.eh_proj_h.weight",
    "attn_norm": "mtp.attn_norm.weight", "post_norm": "mtp.post_attention_norm.weight",
    "wq": "mtp.attn_q.weight", "wk": "mtp.attn_k.weight", "wv": "mtp.attn_v.weight",
    "wo": "mtp.attn_output.weight", "q_norm": "mtp.attn_q_norm.weight",
    "k_norm": "mtp.attn_k_norm.weight", "wg": "mtp.ffn_gate.weight", "wu": "mtp.ffn_up.weight",
    "wd": "mtp.ffn_down.weight", "enorm": "mtp.enorm.weight", "hnorm": "mtp.hnorm.weight",
    "head_norm": "mtp.shared_head_norm.weight",
}


def to_bf16(x):
    """float32 -> bf16 bits, round to nearest even (finite inputs)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def main():
    src, npz, out = sys.argv[1:4]
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
    z = np.load(npz)
    for k, name in NAMES.items():
        a = z[k].astype(np.float32)
        if a.ndim == 1:
            w.add_tensor(name, a)
        else:
            b = to_bf16(a)
            w.add_tensor(name, b, raw_shape=b.shape, raw_dtype=GGMLQuantizationType.BF16)
        print(f"{name:34s} {'F32' if a.ndim == 1 else 'BF16'} {list(a.shape)}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
