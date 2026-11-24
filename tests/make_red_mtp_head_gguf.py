#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
make_red_mtp_head_gguf.py — the RED arm of K80 MTP Gate 1: a standalone MTP-head GGUF cut from
the Qwen3.6-35B-A3B trunk (its blk.40.* + token_embd / output_norm / output, every KV) with
nextn.eh_proj's two input halves SWAPPED, so the head reads the embedding as the hidden state
and vice versa. Served with `-md`, tools/mtp_acceptance_gate.py must then report acceptance
collapsing; if it does not, the harness is not measuring the head.

MTP_RED_SWAP_HALVES=0 writes the halves in order (the GREEN control: same acceptance as the
trunk's bundled head, which also proves the -md path loads the cut head correctly).

Usage: make_red_mtp_head_gguf.py <trunk.gguf> <out.gguf>   (llama.cpp/gguf-py on PYTHONPATH)

@status  complete
@issues  none
@todo    none
"""
import os
import sys

import numpy as np
from gguf import GGUFReader, GGUFWriter

KEEP_GLOBAL = ("token_embd.weight", "output_norm.weight", "output.weight")
EH_PROJ = "blk.40.nextn.eh_proj.weight"


def swapped_halves(t):
    """eh_proj rows with the embedding half and the hidden half exchanged (Q8_0 rows split at 2048)."""
    raw = np.asarray(t.data).reshape(int(t.shape[1]), -1)
    half = raw.shape[1] // 2
    return np.ascontiguousarray(np.concatenate([raw[:, half:], raw[:, :half]], axis=1))


def main():
    trunk_path, out_path = sys.argv[1], sys.argv[2]
    swap = os.environ.get("MTP_RED_SWAP_HALVES", "1") == "1"
    r = GGUFReader(trunk_path)
    w = GGUFWriter(out_path, r.fields["general.architecture"].contents())
    for key, field in r.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        vtype = field.types[0]
        if vtype.name == "ARRAY":
            w.add_key_value(key, field.contents(), vtype, sub_type=field.types[-1])
        else:
            w.add_key_value(key, field.contents(), vtype)
    for t in r.tensors:
        if not (t.name.startswith("blk.40.") or t.name in KEEP_GLOBAL):
            continue
        data = swapped_halves(t) if (t.name == EH_PROJ and swap) else np.asarray(t.data)
        w.add_tensor(t.name, np.ascontiguousarray(data), raw_shape=data.shape, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path} (eh_proj halves {'SWAPPED' if swap else 'in order'})")


if __name__ == "__main__":
    main()
