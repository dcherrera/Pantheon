#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""MTP draft acceptance on Ornith, offline, greedy — the measuring stick for an MTP fine-tune.

Inputs:
  * target dumps from the card (PN_TT_DUMP_HIDDEN; foundry tt/tests/run_dflash_dump.sh): per
    position int32 pos, int32 token, 9 x 4096 float32; the LAST plane is the residual after
    layer 31 (the hidden the MTP head chains on). Greedy trajectories: token[p+1] is the trunk's
    argmax at p from the prompt's last position on.
  * the MTP block (`blk.32.*`) from Ornith's GGUF (Q8_0), or `--head` an .npz written by a
    fine-tune (same names, float32), and the trunk's token_embd / output.weight.

Mirrors the card (qwen35moe_resident_tt.c pn_qwen35_tt_mtp_step / qwen35_tt_mtp_head and
qwen35moe_forward_tt.c): MTP at position p = eh_proj_e @ enorm(emb(x_{p+1})) + eh_proj_h @
hnorm(h_p), a full-attention block (gated q, q/k head norms, partial rope 64 of 256 at theta
1e7, 16 q / 4 kv heads) whose KV cache holds every committed position (the catch-up on trunk
hiddens), a SwiGLU FFN, then shared_head_norm + the LM head over the DRAFT VOCABULARY (ids <
65,536 + the special-token tail, PN_TT_MTP_VOCAB_TILES' default). Draft i >= 1 chains on the
MTP's own output at position s - 1 + i. ADAPTIVE COUNT: drafting stops after a draft whose
softmax over the head's top-8 values is below tau (PN_TT_SPEC_ADAPT, default 0.6).

Reports, per dump and overall: the walked trajectory's tokens a cycle (cycles from the prompt
end, s -> s + accepted + 1, as the card), per-position reach / accept, and the rank of the
trunk's token among the head's top 8 where a draft was refused (the card's TREE PROBE).

  python3 tools/mtp_accept.py --gguf G --dump D.bin --prompt-len P [--dump ...] [--k 3]
Run Mac model loads under foundry/metal/run_guarded.sh (~3 GB).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offline_model_readers import Gguf, load_dump  # noqa: E402

HID = 4096
N_HEAD, N_KV, HEAD_DIM, ROPE_DIM = 16, 4, 256, 64
THETA = 1.0e7
EPS = 1e-6
FFN = 12288
SUB_IDS = 65536          # draft vocabulary: ids < 2048 tiles x 32 ...
TAIL_FIRST = 248044 // 32 * 32  # ... + the special-token tail (from its tile)
TOP = 8

MTP_NAMES = {
    "eh_e": None, "eh_h": None,  # split from blk.32.nextn.eh_proj below
    "enorm": "blk.32.nextn.enorm.weight", "hnorm": "blk.32.nextn.hnorm.weight",
    "attn_norm": "blk.32.attn_norm.weight", "post_norm": "blk.32.post_attention_norm.weight",
    "wq": "blk.32.attn_q.weight", "wk": "blk.32.attn_k.weight", "wv": "blk.32.attn_v.weight",
    "wo": "blk.32.attn_output.weight", "q_norm": "blk.32.attn_q_norm.weight",
    "k_norm": "blk.32.attn_k_norm.weight", "wg": "blk.32.ffn_gate.weight",
    "wu": "blk.32.ffn_up.weight", "wd": "blk.32.ffn_down.weight",
    "head_norm": "blk.32.nextn.shared_head_norm.weight",
}


def rms(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * w


def matrix(gg, name):
    dims = gg.tensors[name][0]
    return gg.rows(name, 0, dims[1])


def load_head(gg, npz=None):
    """The MTP block as float32 arrays (out x in)."""
    if npz:
        z = np.load(npz)
        return {k: z[k].astype(np.float32) for k in z.files}
    w = {}
    for k, name in MTP_NAMES.items():
        if name is None:
            continue
        w[k] = gg.vector(name) if len(gg.tensors[name][0]) == 1 else matrix(gg, name)
    eh = matrix(gg, "blk.32.nextn.eh_proj.weight")  # [4096 out, 8192 in] = [emb | hidden]
    w["eh_e"], w["eh_h"] = eh[:, :HID].copy(), eh[:, HID:].copy()
    return w


def draft_vocab(gg):
    vocab = gg.tensors["output.weight"][0][1]
    ids = np.concatenate([np.arange(SUB_IDS), np.arange(TAIL_FIRST, vocab)])
    head = np.concatenate([gg.rows("output.weight", 0, SUB_IDS),
                           gg.rows("output.weight", TAIL_FIRST, vocab - TAIL_FIRST)])
    return ids, head


def rope(x, pos):
    """x [heads, 256]: rotate the first 64 dims, pairing j with j + 32 (the card's NEOX order)."""
    half = ROPE_DIM // 2
    inv = THETA ** (-np.arange(0, ROPE_DIM, 2, dtype=np.float64) / ROPE_DIM)
    ang = pos * inv
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    y = x.copy()
    x0, x1 = x[:, :half], x[:, half:ROPE_DIM]
    y[:, :half] = x0 * c - x1 * s
    y[:, half:ROPE_DIM] = x0 * s + x1 * c
    return y


class Mtp:
    def __init__(self, w, emb_rows, head_ids, head_w, max_seq):
        self.w, self.emb, self.ids, self.head = w, emb_rows, head_ids, head_w
        self.k = np.zeros((max_seq, N_KV, HEAD_DIM), np.float32)
        self.v = np.zeros((max_seq, N_KV, HEAD_DIM), np.float32)

    def step(self, h, token, pos):
        """One MTP step at `pos`: writes KV at pos, returns the MTP hidden."""
        w = self.w
        x = w["eh_e"] @ rms(self.emb(token), w["enorm"]) + w["eh_h"] @ rms(h, w["hnorm"])
        a = rms(x, w["attn_norm"])
        qg = (w["wq"] @ a).reshape(N_HEAD, 2 * HEAD_DIM)
        q, z = qg[:, :HEAD_DIM], qg[:, HEAD_DIM:]
        k = (w["wk"] @ a).reshape(N_KV, HEAD_DIM)
        v = (w["wv"] @ a).reshape(N_KV, HEAD_DIM)
        q = rope(rms(q, w["q_norm"]), pos)
        k = rope(rms(k, w["k_norm"]), pos)
        self.k[pos], self.v[pos] = k, v
        K, V = self.k[:pos + 1], self.v[:pos + 1]           # [T, kv, d]
        qh = q.reshape(N_KV, N_HEAD // N_KV, HEAD_DIM)      # [kv, g, d]
        sc = np.einsum("kgd,tkd->kgt", qh, K) / np.sqrt(HEAD_DIM)
        sc = np.exp(sc - sc.max(axis=-1, keepdims=True))
        sc /= sc.sum(axis=-1, keepdims=True)
        o = np.einsum("kgt,tkd->kgd", sc, V).reshape(N_HEAD, HEAD_DIM)
        o = o * (1.0 / (1.0 + np.exp(-z)))
        x = x + w["wo"] @ o.reshape(-1)
        f = rms(x, w["post_norm"])
        g = w["wg"] @ f
        x = x + w["wd"] @ (g / (1.0 + np.exp(-g)) * (w["wu"] @ f))
        return x

    def top(self, m):
        """The head's top-8 (ids, values) by value, and the adaptive count's confidence."""
        logits = self.head @ rms(m, self.w["head_norm"])
        t = np.argpartition(-logits, TOP)[:TOP]
        t = t[np.argsort(-logits[t], kind="stable")]
        vals = logits[t]
        conf = 1.0 / np.exp(vals - vals[0]).sum()
        return self.ids[t], vals, conf


def walk(mtp, hid, tokens, prompt_len, k, tau, stats):
    """The card's cycle walk over one greedy trajectory."""
    n = len(tokens)
    hcache = {}
    # catch-up over the prompt (positions 0 .. P-2: token p+1 is known), then cycle from P-1
    for p in range(prompt_len - 1):
        mtp.step(hid[p], tokens[p + 1], p)
    s = prompt_len  # the seed position: token[s] is the trunk's argmax at s - 1
    while s + 1 < n:
        m = mtp.step(hid[s - 1], tokens[s], s - 1)   # catch-up row = draft 0's step
        drafts, tops, confs = [], [], []
        for i in range(k):
            ids, vals, conf = mtp.top(m)
            drafts.append(int(ids[0]))
            tops.append(ids)
            confs.append(float(conf))
            if i + 1 == k or (tau > 0 and conf < tau) or s + 1 + i >= n - 1:
                break
            m = mtp.step(m, drafts[-1], s + i)
        acc = 0
        while acc < len(drafts) and s + 1 + acc < n and drafts[acc] == tokens[s + 1 + acc]:
            acc += 1
        for j in range(min(acc + 1, len(drafts))):
            stats["reach"][j] += 1
        for j in range(acc):
            stats["acc"][j] += 1
        if acc < len(drafts) and s + 1 + acc < n:
            hit = np.nonzero(tops[acc] == tokens[s + 1 + acc])[0]
            stats["rank"][acc][hit[0] if hit.size else TOP] += 1
        if stats.get("log"):
            ranks = []
            for j in range(min(acc + 1, len(drafts))):
                if s + 1 + j >= n:
                    break
                hit = np.nonzero(tops[j] == tokens[s + 1 + j])[0]
                ranks.append(int(hit[0]) if hit.size else -1)
            stats["log"].write(json.dumps({"s": s, "acc": acc, "nd": len(drafts),
                                           "conf": [round(c, 4) for c in confs],
                                           "rank": ranks}) + "\n")
        stats["cycles"] += 1
        stats["rows"] += len(drafts) + 1
        stats["tokens"] += acc + 1
        # committed: the seed + accepted drafts; the catch-up rewrites KV s .. s+acc-1 next cycle
        for j in range(acc):
            mtp.step(hid[s + j], tokens[s + j + 1], s + j)
        s += acc + 1


def report(tag, st):
    c = max(st["cycles"], 1)
    out = [f"{tag}: cycles {st['cycles']} tokens a cycle {st['tokens'] / c:.3f} rows a cycle "
           f"{st['rows'] / c:.2f}"]
    for j in range(len(st["reach"])):
        if not st["reach"][j]:
            break
        r = st["rank"][j]
        out.append(f"  pos {j}: reach {st['reach'][j]} acc {st['acc'][j]} "
                   f"({st['acc'][j] / st['reach'][j]:.3f}) refused-rank 1..7,miss "
                   f"{','.join(str(x) for x in r[1:])}")
    return "\n".join(out)


def new_stats(k):
    return {"cycles": 0, "rows": 0, "tokens": 0, "reach": [0] * k, "acc": [0] * k,
            "rank": [[0] * (TOP + 1) for _ in range(k)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--head", help="fine-tuned MTP block (.npz); default the GGUF's blk.32")
    ap.add_argument("--dump", action="append", required=True)
    ap.add_argument("--prompt-len", type=int, action="append", required=True)
    ap.add_argument("--planes", type=int, default=9, help="9 = PN_TT_DUMP_HIDDEN, 1 = _LAST=1")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--log", help="one JSON line a cycle: drafts' confidences, the trunk "
                    "token's rank in each reached head's top 8 (-1 miss), accepted")
    a = ap.parse_args()
    gg = Gguf(a.gguf)
    w = load_head(gg, a.head)
    ids, head = draft_vocab(gg)
    emb = lambda t: gg.rows("token_embd.weight", int(t), 1)[0]  # noqa: E731
    total = new_stats(a.k)
    log = open(a.log, "w") if a.log else None
    for path, plen in zip(a.dump, a.prompt_len):
        pos, tokens, hid = load_dump(path, a.planes)
        assert (pos == np.arange(len(pos))).all(), "dump positions are not 0..n-1"
        st = new_stats(a.k)
        st["log"] = log
        walk(Mtp(w, emb, ids, head, len(tokens) + 16), hid[:, -1, :], tokens, plen, a.k, a.tau, st)
        print(report(os.path.basename(path), st), flush=True)
        for key in ("cycles", "rows", "tokens"):
            total[key] += st[key]
        for j in range(a.k):
            total["reach"][j] += st["reach"][j]
            total["acc"][j] += st["acc"][j]
            for r in range(TOP + 1):
                total["rank"][j][r] += st["rank"][j][r]
    print(report(f"ALL (k {a.k}, tau {a.tau})", total))


if __name__ == "__main__":
    main()
