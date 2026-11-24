#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""Fine-tune Ornith's MTP draft head on the trunk's own trajectories, in CHAINED mode.

Why: offline (tools/mtp_accept.py) the official head accepts 0.84 at draft position 0 but
0.69 / 0.65 / 0.55 at positions 1 / 2 / 3 — those drafts chain on the head's OWN output hidden,
which the head never saw in training (it was trained on trunk hiddens only). This trains the
head the way the card runs it (FastMTP-style recursive steps with shared weights):

  step 0 at p: m0 = MTP(h_p (trunk), x_{p+1}, pos p)      -> x_{p+2}; KV K0[p] (the catch-up)
  step j at p: mj = MTP(m_{j-1}, x_{p+1+j}, pos p+j)       -> x_{p+2+j}; attends K0[0..p] +
               the chain's own keys K1..Kj (what the card's MTP cache holds while drafting)

Targets are the trajectory's next tokens (greedy dumps: the trunk's argmax; sampled dumps: the
trunk's own draws, an unbiased estimate of its distribution); the loss counts only targets in
the generated part (after the prompt) and inside the DRAFT VOCABULARY (the card's head only
proposes ids < 65,536 + the special tail). LoRA on every matrix + full norms; the LM head is the
trunk's, frozen. Writes the merged block as an .npz (tools/mtp_accept.py --head).

Data: PN_TT_DUMP_HIDDEN + PN_TT_DUMP_HIDDEN_LAST=1 dumps (int32 pos, int32 token, 4096 float32
a record); sequences split where pos restarts at 0; `--meta` gives each dump's prompt lengths.

  python3 tools/mtp_finetune.py --gguf G --train a.bin [--train ...] --prompt-lens a.json
      --out ft.npz [--steps 3] [--epochs 2] [--rank 64] [--lr 2e-4] [--device mps]
Run under foundry/metal/run_guarded.sh.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from offline_model_readers import Gguf  # noqa: E402
from mtp_accept import (HID, N_HEAD, N_KV, HEAD_DIM, ROPE_DIM, THETA, EPS, SUB_IDS,  # noqa: E402
                        TAIL_FIRST, load_head, draft_vocab)

MATS = ("eh_e", "eh_h", "wq", "wk", "wv", "wo", "wg", "wu", "wd")
NORMS = ("enorm", "hnorm", "attn_norm", "post_norm", "q_norm", "k_norm", "head_norm")


def load_sequences(paths, prompt_lens, planes=1):
    """[(tokens int64 [n], hid float16 [n, 4096], prompt_len)] split at pos == 0; the last
    plane of each record is layer 31 (planes 9 = a PN_TT_DUMP_HIDDEN dump, 1 = _LAST=1)."""
    out = []
    rec = 8 + planes * HID * 4
    for path in paths:
        raw = np.fromfile(path, dtype=np.uint8)
        n = raw.size // rec
        raw = raw[:n * rec].reshape(n, rec)
        head = raw[:, :8].copy().view(np.int32)
        hid = raw[:, 8 + (planes - 1) * HID * 4:].copy().view(np.float32).reshape(n, HID)
        starts = list(np.nonzero(head[:, 0] == 0)[0]) + [n]
        plens = prompt_lens.get(os.path.basename(path), [])
        for i in range(len(starts) - 1):
            a, b = starts[i], starts[i + 1]
            if not (head[a:b, 0] == np.arange(b - a)).all() or i >= len(plens):
                print(f"skip {path} seq {i}: positions not contiguous / no prompt length")
                continue
            out.append((head[a:b, 1].astype(np.int64), hid[a:b].astype(np.float16), plens[i]))
    return out


class Lora(torch.nn.Module):
    def __init__(self, w, rank, alpha):
        super().__init__()
        self.register_buffer("w", torch.from_numpy(w).to(torch.bfloat16))
        out_f, in_f = w.shape
        self.a = torch.nn.Parameter(torch.randn(rank, in_f) / math.sqrt(in_f))
        self.b = torch.nn.Parameter(torch.zeros(out_f, rank))
        self.scale = alpha / rank

    def forward(self, x):
        return (x.to(torch.bfloat16) @ self.w.T).float() + (x @ self.a.T) @ self.b.T * self.scale

    def merged(self):
        return (self.w.float() + self.b @ self.a * self.scale).detach().cpu().numpy()


class MtpTrain(torch.nn.Module):
    def __init__(self, w, rank, alpha):
        super().__init__()
        self.mats = torch.nn.ModuleDict({k: Lora(w[k], rank, alpha) for k in MATS})
        self.norms = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(torch.from_numpy(w[k].copy())) for k in NORMS})
        half = ROPE_DIM // 2
        self.register_buffer("inv", torch.tensor(
            THETA ** (-np.arange(0, ROPE_DIM, 2, dtype=np.float64) / ROPE_DIM), dtype=torch.float32))
        self.half = half

    def rms(self, x, k):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * self.norms[k]

    def rope(self, x, pos):
        """x [T, heads, 256], pos [T]."""
        ang = pos.float()[:, None] * self.inv[None, :]
        c, s = torch.cos(ang)[:, None, :], torch.sin(ang)[:, None, :]
        x0, x1 = x[..., :self.half], x[..., self.half:ROPE_DIM]
        return torch.cat([x0 * c - x1 * s, x0 * s + x1 * c, x[..., ROPE_DIM:]], dim=-1)

    def qkv(self, x, pos):
        a = self.rms(x, "attn_norm")
        qg = self.mats["wq"](a).view(-1, N_HEAD, 2 * HEAD_DIM)
        q, z = qg[..., :HEAD_DIM], qg[..., HEAD_DIM:]
        k = self.mats["wk"](a).view(-1, N_KV, HEAD_DIM)
        v = self.mats["wv"](a).view(-1, N_KV, HEAD_DIM)
        return self.rope(self.rms(q, "q_norm"), pos), z, self.rope(self.rms(k, "k_norm"), pos), v

    def tail(self, x, o, z):
        o = o * torch.sigmoid(z)
        x = x + self.mats["wo"](o.reshape(o.shape[0], -1))
        f = self.rms(x, "post_norm")
        return x + self.mats["wd"](torch.nn.functional.silu(self.mats["wg"](f)) * self.mats["wu"](f))

    def embed_in(self, h, e):
        return self.mats["eh_e"](self.rms(e, "enorm")) + self.mats["eh_h"](self.rms(h, "hnorm"))

    def chain(self, h, emb, steps):
        """h [T, 4096] trunk hiddens, emb [T + steps, 4096] embeddings of x_1.. (emb[p] =
        emb(x_{p+1})). Returns the head inputs of every step: [steps, T, 4096]."""
        T = h.shape[0]
        pos = torch.arange(T, device=h.device)
        g = N_HEAD // N_KV
        scale = 1.0 / math.sqrt(HEAD_DIM)
        causal = torch.ones(T, T, device=h.device, dtype=torch.bool).tril()
        x = self.embed_in(h, emb[:T])
        q, z, K0, V0 = self.qkv(x, pos)
        own_k, own_v, outs, hid = [], [], [], x
        for j in range(steps):
            if j > 0:
                x = self.embed_in(hid, emb[j:T + j])
                q, z, kj, vj = self.qkv(x, pos + j)
                own_k.append(kj)
                own_v.append(vj)
            qh = q.view(T, N_KV, g, HEAD_DIM)
            s0 = torch.einsum("tkgd,ukd->kgtu", qh, K0) * scale
            s0 = s0.masked_fill(~causal, float("-inf"))
            extra = [torch.einsum("tkgd,tkd->kgt", qh, kk)[..., None] * scale for kk in own_k]
            sc = torch.cat([s0] + extra, dim=-1).softmax(-1)
            o = torch.einsum("kgtu,ukd->tkgd", sc[..., :T], V0)
            for i, vv in enumerate(own_v):
                o = o + sc[..., T + i].permute(2, 0, 1)[..., None] * vv[:, :, None, :]
            hid = self.tail(x, o.reshape(T, N_HEAD, HEAD_DIM), z)
            outs.append(self.rms(hid, "head_norm"))
        return outs

    def export(self):
        w = {k: m.merged() for k, m in self.mats.items()}
        w.update({k: p.detach().cpu().numpy() for k, p in self.norms.items()})
        return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--prompt-lens", required=True, help="json {dump basename: [prompt len a seq]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--chunk", type=int, default=2048, help="max positions a training window")
    ap.add_argument("--planes", type=int, default=1)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    gg = Gguf(a.gguf)
    w = load_head(gg)
    ids, head = draft_vocab(gg)
    vocab = gg.tensors["output.weight"][0][1]
    col = np.full(vocab, -1, dtype=np.int64)          # vocabulary id -> draft-vocabulary column
    col[ids] = np.arange(len(ids))
    head_t = torch.from_numpy(head).to(torch.bfloat16).to(dev)
    del head
    model = MtpTrain(w, a.rank, a.alpha).to(dev)
    seqs = load_sequences(a.train, json.load(open(a.prompt_lens)), a.planes)
    emb_cache = {}

    def emb(t):
        if t not in emb_cache:
            emb_cache[t] = gg.rows("token_embd.weight", int(t), 1)[0]
        return emb_cache[t]

    windows = []
    for tokens, hid, plen in seqs:
        n = len(tokens)
        for start in range(0, n - 2, a.chunk):
            windows.append((tokens, hid, plen, start, min(n, start + a.chunk + a.steps + 1)))
    print(f"{len(seqs)} sequences, {sum(len(s[0]) for s in seqs)} positions, {len(windows)} windows",
          flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                            weight_decay=0.0)
    total = a.epochs * len(windows)
    step = 0
    rng = np.random.default_rng(0)
    for ep in range(-1, a.epochs):
        stat = np.zeros((a.steps, 2))
        t0 = time.time()
        for wi in rng.permutation(len(windows)):
            tokens, hid, plen, s, e = windows[wi]
            # positions s..e-1 are context; queries p in [s, e - 2 - j) predict x_{p+2+j}
            T = e - s - 1 - a.steps if e - s - 1 - a.steps > 0 else 0
            if T < 1:
                continue
            # the window's own MTP KV starts at s: the card's cache holds 0..p; a window only
            # holds s..p (documented approximation for windows past the first)
            h = torch.from_numpy(hid[s:s + T].astype(np.float32)).to(dev)
            e_rows = torch.from_numpy(np.stack([emb(tokens[p + 1]) for p in range(s, s + T + a.steps)])).to(dev)
            with torch.set_grad_enabled(ep >= 0):
                outs = model.chain(h, e_rows, a.steps)
            loss = 0.0
            for j, o in enumerate(outs):
                tgt_pos = np.arange(s, s + T) + 2 + j
                ok = tgt_pos < len(tokens)
                tgt = np.full(T, -1, dtype=np.int64)
                tgt[ok] = col[tokens[tgt_pos[ok]]]
                tgt[tgt_pos < plen] = -1               # only the generated part
                m = torch.from_numpy(tgt >= 0).to(dev)
                if not m.any():
                    continue
                logits = (o[m].to(torch.bfloat16) @ head_t.T).float()
                tg = torch.from_numpy(tgt).to(dev)[m]
                l = torch.nn.functional.cross_entropy(logits, tg)
                loss = loss + l
                stat[j, 0] += (logits.argmax(-1) == tg).sum().item()
                stat[j, 1] += m.sum().item()
            if not torch.is_tensor(loss) or ep < 0:
                continue
            for g_ in opt.param_groups:
                g_["lr"] = a.lr * min(1.0, (step + 1) / 20) * 0.5 * (1 + math.cos(math.pi * step / total))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        acc = " ".join(f"step{j} {stat[j,0]/max(stat[j,1],1):.3f}" for j in range(a.steps))
        print(f"epoch {ep}{' (no update: the head as loaded)' if ep < 0 else ''}: train top-1 {acc} ({time.time() - t0:.0f} s)", flush=True)
    np.savez(a.out, **model.export())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
