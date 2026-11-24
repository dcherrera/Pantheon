/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   gguf_fttw_overlay.h
 * @brief  Attach a foundry FTTW artifact to an open GGUF so its tiled 2-D tensors read as
 *         GGUF tensors (PN_QUANT_TILED_*): the card-format CPU oracle's weight source.
 *
 * The card host keeps a METADATA GGUF (keys + 1-D tensors + conv taps, foundry
 * ft_tt_gguf_slim) beside the artifact; attaching the artifact supplies every matmul
 * weight, the embedding and the expert stacks from the bytes the card reads, so the model
 * loaders run unchanged. `PN_FTTW_ORACLE=<trunk.fttw>` makes pn_load attach it; without it,
 * pn_load attaches the one artifact found beside a metadata GGUF
 * (pn_gguf_attach_card_artifact).
 *
 * The GGUF and the artifact are mapped back to back in one reserved range (the GGUF is
 * re-mapped at its start; every pointer into its first mapping moves with it), and each
 * artifact tensor's `offset` is its distance from `file->data`, so `file->data + offset` —
 * what every loader computes — lands in the artifact. A tensor the GGUF already holds keeps
 * the GGUF's copy.
 *
 * @status None
 * @issues None
 * @todo   None
 */
#ifndef PN_GGUF_FTTW_OVERLAY_H
#define PN_GGUF_FTTW_OVERLAY_H

#include "gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Map `fttw_path` and append its tiled 2-D tensors to `file`'s tensor table.
 *
 * @return the number of tensors added (>= 0), or -1 (stderr says why: unreadable file,
 *         bad magic or version, a format/layout the CPU cannot decode, a tensor that is
 *         not whole tiles, or a mapping that landed below the GGUF)
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_gguf_attach_fttw(pn_gguf_file *file, const char *fttw_path);

/**
 * A card METADATA GGUF (no `token_embd.weight`: ft_tt_gguf_slim leaves the matmul weights and
 * the embedding out) loads with the prebuilt artifact beside it: exactly one
 * `<gguf_path minus .gguf>.<rung>.fttw` in the same directory (`<rung>` has no '.', so the
 * MTP head `<stem>.<rung>.mtp.fttw` is not it). That is the name Pantheon Local's companion
 * link gives a catalog's card artifact, and the name the TT artifact cache uses. Found, it
 * is attached with pn_gguf_attach_fttw and its path written to `out`.
 *
 * @return 1 attached (path in `out`); 0 not a metadata GGUF, or no artifact beside it (the
 *         file is untouched); -1 refused (two or more candidates, a path too long, or the
 *         attach failed; stderr says which)
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_gguf_attach_card_artifact(pn_gguf_file *file, const char *gguf_path, char *out,
                                 size_t out_bytes);

#ifdef __cplusplus
}
#endif

#endif /* PN_GGUF_FTTW_OVERLAY_H */
