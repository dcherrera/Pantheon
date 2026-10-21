/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file session_snapshot.c
 * @brief Family selection for session snapshot/restore — see session_snapshot.h
 *        for the contract and for why the two families are not interchangeable.
 *
 * Every function here is a two-way switch and nothing else. The rule the switch
 * enforces: an architecture is routed BY NAME, and an architecture nobody has
 * written a family for is a refusal. Falling through to "probably llama" would
 * hand an RWKVX recurrent state to a KV-cache reader.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include "session_snapshot.h"

#include "../models/llama.h"
#include "../models/qwen35moe.h"

/* Which snapshot implementation owns this architecture. */
typedef enum {
    SNAP_FAMILY_NONE = 0,
    SNAP_FAMILY_LLAMA_KV,      /* pure causal attention, fp32 KV only */
    SNAP_FAMILY_QWEN35_DUAL,   /* SSM recurrent state + attention KV */
} snap_family;

/* The llama-family list is deliberately ENUMERATED rather than expressed as
 * "everything except the three hybrids". pn_state_new() allocates a
 * pn_llama_kv_cache for every arch outside RWKVX/QWEN35MOE/BONSAI today, so a
 * default-allow would be correct today and wrong the first time a fourth state
 * shape lands — and it would be wrong by writing a KV reader over a state that
 * is not one. This is the same list lcp_rewind_supported() keeps in
 * prefix_cache.c, for the same reason: both statements are "this arch's whole
 * per-session state is a plain causal KV cache". */
static snap_family family_for(pn_arch arch)
{
    switch (arch) {
    case PN_ARCH_LLAMA:
    case PN_ARCH_QWEN2:
    case PN_ARCH_QWEN3:
    case PN_ARCH_PHI:
    case PN_ARCH_MISTRAL:
    case PN_ARCH_MIXTRAL:
        return SNAP_FAMILY_LLAMA_KV;
    case PN_ARCH_QWEN35MOE:
    case PN_ARCH_BONSAI:
        return SNAP_FAMILY_QWEN35_DUAL;
    default:
        return SNAP_FAMILY_NONE;   /* RWKVX and anything unknown */
    }
}

int pn_session_snapshot_supported(pn_arch arch)
{
    return family_for(arch) != SNAP_FAMILY_NONE;
}

void *pn_session_snapshot_capture(pn_arch arch, const void *kv_cache, int pos)
{
    switch (family_for(arch)) {
    case SNAP_FAMILY_LLAMA_KV:
        return pn_llama_kv_snapshot(kv_cache, pos);
    case SNAP_FAMILY_QWEN35_DUAL:
        return pn_qwen35moe_state_snapshot(kv_cache, pos);
    default:
        return NULL;
    }
}

int pn_session_snapshot_restore(pn_arch arch, void *kv_cache, const void *snap)
{
    switch (family_for(arch)) {
    case SNAP_FAMILY_LLAMA_KV:
        return pn_llama_kv_restore(kv_cache, snap);
    case SNAP_FAMILY_QWEN35_DUAL:
        return pn_qwen35moe_state_restore(kv_cache, snap);
    default:
        return -1;
    }
}

int pn_session_snapshot_pos(pn_arch arch, const void *snap)
{
    switch (family_for(arch)) {
    case SNAP_FAMILY_LLAMA_KV:
        return pn_llama_kv_snapshot_pos(snap);
    case SNAP_FAMILY_QWEN35_DUAL:
        return pn_qwen35moe_state_snapshot_pos(snap);
    default:
        return -1;
    }
}

void pn_session_snapshot_free(pn_arch arch, void *snap)
{
    switch (family_for(arch)) {
    case SNAP_FAMILY_LLAMA_KV:
        pn_llama_kv_snapshot_free(snap);
        return;
    case SNAP_FAMILY_QWEN35_DUAL:
        pn_qwen35moe_state_snapshot_free(snap);
        return;
    default:
        return;
    }
}

int pn_session_snapshot_serialize(pn_arch arch, const void *snap,
                                  uint64_t weights_version,
                                  unsigned char **bytes_out, size_t *len_out)
{
    switch (family_for(arch)) {
    case SNAP_FAMILY_LLAMA_KV:
        return pn_llama_kv_snap_serialize(snap, weights_version, bytes_out,
                                          len_out);
    case SNAP_FAMILY_QWEN35_DUAL:
        return pn_qwen35moe_snap_serialize(snap, weights_version, bytes_out,
                                           len_out);
    default:
        return -1;
    }
}

void *pn_session_snapshot_deserialize(pn_arch arch, const unsigned char *bytes,
                                      size_t len, uint64_t *weights_version_out)
{
    switch (family_for(arch)) {
    case SNAP_FAMILY_LLAMA_KV:
        return pn_llama_kv_snap_deserialize(bytes, len, weights_version_out);
    case SNAP_FAMILY_QWEN35_DUAL:
        return pn_qwen35moe_snap_deserialize(bytes, len, weights_version_out);
    default:
        return NULL;
    }
}
