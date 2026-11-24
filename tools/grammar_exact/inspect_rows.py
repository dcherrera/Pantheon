#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""inspect_rows.py — print today's-undecided masked rows from a PN_GRAMMAR_ROW_DUMP file:
the allowed tokens inside the top-64 (rank, tempered prob among the allowed), the allowed
count outside, and the outside mass at the 64th logit relative to the inside mass.

  inspect_rows.py <dump.grd> [--limit N]
"""
import struct
import sys

import numpy as np

HDR = struct.Struct("<Iiffifffiiii")


def records(path):
    with open(path, "rb") as f:
        while True:
            raw = f.read(HDR.size)
            if len(raw) < HDR.size:
                return
            (magic, nv, temp, top_p, top_k, rp, pp, fp, bias, n_recent, n_allowed,
             has_row) = HDR.unpack(raw)
            recent = np.frombuffer(f.read(4 * n_recent), dtype="<u4")
            ids = np.frombuffer(f.read(256), dtype="<u4")
            vals = np.frombuffer(f.read(256), dtype="<f4")
            allowed = np.frombuffer(f.read(64), dtype="u1")
            row = verdict = None
            if has_row:
                row = np.frombuffer(f.read(4 * nv), dtype="<f4")
                verdict = np.frombuffer(f.read(nv), dtype="u1")
            yield dict(nv=nv, temp=temp, top_p=top_p, top_k=top_k, rp=rp, recent=recent,
                       ids=ids, vals=vals, allowed=allowed, n_allowed=n_allowed, row=row,
                       verdict=verdict)


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 1 << 30
    shown = 0
    for r in records(sys.argv[1]):
        if r["row"] is None or shown >= limit:
            continue
        if not r["allowed"].any():
            print(f"inside  0 out {r['n_allowed']:6d}  (no allowed token in the top-64)")
            shown += 1
            continue
        shown += 1
        t, bound = r["temp"], r["vals"][-1]
        inside = [(k, r["ids"][k], r["vals"][k]) for k in range(64) if r["allowed"][k]]
        out = r["n_allowed"] - len(inside)
        m = max(v for _, _, v in inside) if inside else bound
        w_in = sum(np.exp((v - m) / t) for _, _, v in inside)
        mask = (r["verdict"] == 1)
        mask[r["ids"]] = False
        true_out = float(np.exp((r["row"][mask].astype(np.float64) - m) / t).sum())
        print(f"inside {len(inside):2d} out {out:6d} m-bound {m - bound:6.2f}  "
              f"B_N/W {out * np.exp((bound - m) / t) / w_in:9.2e} true/W {true_out / w_in:9.2e}  "
              f"probs {[round(float(np.exp((v - m) / t) / w_in), 3) for _, _, v in inside][:6]}")


if __name__ == "__main__":
    main()
