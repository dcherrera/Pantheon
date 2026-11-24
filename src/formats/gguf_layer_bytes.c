/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gguf_layer_bytes.c
 * @brief What a range of transformer blocks actually weighs in a GGUF.
 *
 * Used by the multi-die admission gate, which places layers [lo,hi) on one die
 * and must project that die's resident weight bytes. See gguf_layer_bytes.h for
 * why a prorated share of mmap_size is the wrong number.
 */

#include "gguf_layer_bytes.h"
#include "gguf_dequant.h"

#include <stdio.h>
#include <string.h>

/* Element count of a tensor. 0 if the info is malformed (n_dims out of range,
 * or a zero dimension), which propagates to a 0 byte count. */
static uint64_t tensor_numel(const pn_gguf_tensor_info *t)
{
    if (!t || t->n_dims == 0 || t->n_dims > 4)
        return 0;
    uint64_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) {
        if (t->dims[i] == 0)
            return 0;
        n *= t->dims[i];
    }
    return n;
}

/* Layer index of a `blk.<i>.<something>` tensor, or -1 for anything else
 * (token_embd.weight, output.weight, output_norm.weight, …). The trailing dot
 * is required so a hypothetical `blk.12suffix` cannot pass as layer 12. */
static int blk_layer_index(const char *name)
{
    if (!name || strncmp(name, "blk.", 4) != 0)
        return -1;
    const char *p = name + 4;
    if (*p < '0' || *p > '9')
        return -1;
    long idx = 0;
    while (*p >= '0' && *p <= '9') {
        idx = idx * 10 + (*p - '0');
        if (idx > 1000000)   /* not a layer index; refuse rather than wrap */
            return -1;
        p++;
    }
    if (*p != '.')
        return -1;
    return (int)idx;
}

uint64_t pn_gguf_tensor_file_bytes(pn_gguf_file *gf, const char *name)
{
    if (!gf || !name)
        return 0;
    pn_gguf_tensor_info *t = pn_gguf_find_tensor(gf, name);
    if (!t)
        return 0;
    uint64_t numel = tensor_numel(t);
    if (numel == 0)
        return 0;
    return (uint64_t)pn_gguf_type_size(t->type, (size_t)numel);
}

uint64_t pn_gguf_tensor_numel(pn_gguf_file *gf, const char *name)
{
    if (!gf || !name)
        return 0;
    return tensor_numel(pn_gguf_find_tensor(gf, name));
}

int pn_gguf_layer_count(pn_gguf_file *gf)
{
    if (!gf || !gf->tensors)
        return 0;
    int max_idx = -1;
    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        if (!gf->tensors[i].name.data)
            continue;
        int layer = blk_layer_index(gf->tensors[i].name.data);
        if (layer > max_idx)
            max_idx = layer;
    }
    int n = max_idx + 1;

    /* Drop a trailing MTP / NextN head. Qwen3.6-35B-A3B ships block_count=41:
     * 40 transformer layers plus a speculative-draft block the autoregressive
     * forward never runs and never uploads. Same rule the loader applies
     * (qwen35moe.c), so a placement sized from this count splits the same
     * layers the forward will. Counting it put ~230 MiB of weight that is
     * never resident onto the last die's projection — enough, on its own, to
     * refuse a 2-die Q4_K_M load with 100 MiB to spare. */
    if (n >= 2) {
        char key[64];
        snprintf(key, sizeof(key), "blk.%d.nextn.eh_proj.weight", n - 1);
        if (pn_gguf_find_tensor(gf, key) != NULL)
            n -= 1;
    }
    return n;
}

uint64_t pn_gguf_layer_range_bytes(pn_gguf_file *gf, int lo, int hi)
{
    if (lo < 0)
        lo = 0;
    if (!gf || !gf->tensors || hi <= lo)
        return 0;
    uint64_t total = 0;
    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        const pn_gguf_tensor_info *t = &gf->tensors[i];
        if (!t->name.data)
            continue;
        int layer = blk_layer_index(t->name.data);
        if (layer < lo || layer >= hi)
            continue;
        uint64_t numel = tensor_numel(t);
        if (numel == 0)
            continue;
        uint64_t bytes = (uint64_t)pn_gguf_type_size(t->type, (size_t)numel);
        /* An UNRECOGNISED quant must not size as FREE. This sum is a safety
         * projection: a tensor we cannot measure contributes 0, which reads
         * downstream as "costs nothing" and admits a load that cannot fit.
         * Measured 2026-08-25: BF16 (type 30) was in the enum but not in
         * pn_gguf_block_size, so every expert tensor sized as zero and the gate
         * projected 10 MiB per die for a model that needs ~8,300. Returning 0
         * for the WHOLE range makes the caller fall back to its prorated
         * estimate instead, which is wrong but never optimistic. */
        if (bytes == 0) {
            char qn[32];
            fprintf(stderr, "[gguf] layer range: no size for quant %s (%d) on "
                    "'%s' — refusing to project this range\n",
                    pn_gguf_quant_name(t->type, qn, sizeof(qn)), (int)t->type,
                    t->name.data);
            return 0;
        }
        total += bytes;
    }
    return total;
}

uint64_t pn_gguf_layer_range_upload_bytes(pn_gguf_file *gf, int lo, int hi)
{
    if (lo < 0)
        lo = 0;
    if (!gf || !gf->tensors || hi <= lo)
        return 0;
    uint64_t total = 0;
    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        const pn_gguf_tensor_info *t = &gf->tensors[i];
        if (!t->name.data)
            continue;
        int layer = blk_layer_index(t->name.data);
        if (layer < lo || layer >= hi)
            continue;
        uint64_t numel = tensor_numel(t);
        if (numel == 0)
            continue;
        uint64_t bytes;
        /* Q8_0 on-die layout: (cols/32)*36 B per row, only when cols % 32 == 0.
         * All other types count at their file size. */
        if (t->type == GGUF_QUANT_Q8_0 && t->n_dims >= 2 && t->dims[0] % 32 == 0) {
            bytes = numel / 32 * 36;
        } else {
            bytes = (uint64_t)pn_gguf_type_size(t->type, (size_t)numel);
        }
        if (bytes == 0) {
            char qn[32];
            fprintf(stderr, "[gguf] layer range: no size for quant %s (%d) on "
                    "'%s' — refusing to project this range\n",
                    pn_gguf_quant_name(t->type, qn, sizeof(qn)), (int)t->type,
                    t->name.data);
            return 0;
        }
        total += bytes;
    }
    return total;
}

/* `<arch>.<suffix>` as a u32, 0 when absent. */
static uint32_t arch_u32(pn_gguf_file *gf, const char *suffix)
{
    const char *arch = pn_gguf_get_str(gf, "general.architecture");
    if (!arch)
        return 0;
    char key[96];
    snprintf(key, sizeof(key), "%s.%s", arch, suffix);
    return pn_gguf_get_u32(gf, key, 0);
}

uint64_t pn_gguf_cpu_forward_bytes(pn_gguf_file *gf)
{
    if (!gf || !gf->tensors)
        return 0;
    uint32_t n_expert = arch_u32(gf, "expert_count");
    uint32_t n_used = arch_u32(gf, "expert_used_count");
    bool routed = n_expert > 0 && n_used > 0 && n_used < n_expert;
    uint64_t total = 0;
    for (uint64_t i = 0; i < gf->n_tensors; i++) {
        const pn_gguf_tensor_info *t = &gf->tensors[i];
        uint64_t numel = tensor_numel(t);
        uint64_t bytes = numel ? (uint64_t)pn_gguf_type_size(t->type, (size_t)numel) : 0;
        if (bytes == 0)
            return 0;   /* a tensor we cannot size: "don't know", never "free" */
        if (routed && t->name.data && strstr(t->name.data, "_exps."))
            bytes = bytes / n_expert * n_used;
        total += bytes;
    }
    return total;
}
