/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdn_checkpoint.c
 * @brief The multi-turn checkpoint of the carried GDN state. See pn_tt_gdn_checkpoint.h.
 *
 * @status Per-layer capture / restore through the puller's COPY; sampled verify.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_gdn_checkpoint.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_dispatch.h"
#include "pn_tt_gdn.h"
#include "../../core/bench.h"

/** The bytes each end of a verified buffer is compared over. */
#define CKPT_VERIFY_BYTES (64u * 1024u)

/** RED `ckpt_copy_short`: the last 4 KiB of every buffer is left. */
#define CKPT_RED_SHORT_BYTES 4096u

static int ckpt_reserve(pn_tt_device *device)
{
    const uint32_t layers = device->gdn_shape.layer_count;
    pn_tt_gdn_region *set;
    uint32_t layer;

    if (device->gdn_checkpoint)
        return 0;
    set = calloc(layers, sizeof(*set));
    if (!set)
        return -1;
    for (layer = 0; layer < layers; layer++)
        if (pn_tt_gdn_alloc_twin(device, layer, &set[layer]) != 0) {
            /* The DRAM already taken stays with the allocator until close; only the host
             * array is dropped, and a later capture tries again. */
            free(set);
            return -1;
        }
    device->gdn_checkpoint = set;
    return 0;
}

/** Copy one buffer card side. Both ends are exactly `bytes` long and 64 B aligned. */
static int ckpt_copy(pn_tt_device *device, const pn_tt_gdn_buffer *from,
                     const pn_tt_gdn_buffer *to, size_t bytes)
{
    if (pn_tt_red("ckpt_copy_short") && bytes > CKPT_RED_SHORT_BYTES)
        bytes -= CKPT_RED_SHORT_BYTES;
    return pn_tt_copy_dram_oncard(device, from->bank, from->address, to->bank, to->address,
                                  bytes);
}

/** Copy one layer's window + state, `to_checkpoint` choosing the direction. */
static int ckpt_copy_layer(pn_tt_device *device, uint32_t layer, bool to_checkpoint)
{
    const pn_tt_gdn_region *live = &device->gdn_region[layer];
    const pn_tt_gdn_region *saved = &device->gdn_checkpoint[layer];
    const pn_tt_gdn_region *from = to_checkpoint ? live : saved;
    const pn_tt_gdn_region *to = to_checkpoint ? saved : live;
    size_t window, state;

    pn_tt_gdn_carried_bytes(device, &window, &state);
    if ((window & 63u) != 0u || (state & 63u) != 0u) {
        fprintf(stderr, "pn_tt: gdn checkpoint — the carried buffers (%zu / %zu B) are not "
                        "64 B multiples; the on-card copy cannot move them\n", window, state);
        return -1;
    }
    if (ckpt_copy(device, &from->window, &to->window, window) != 0 ||
        ckpt_copy(device, &from->state, &to->state, state) != 0)
        return -1;
    return 0;
}

int pn_tt_gdn_checkpoint_capture_layer(pn_tt_device *device, uint32_t layer)
{
    if (!pn_tt_gdn_is_armed(device) || layer >= device->gdn_shape.layer_count ||
        ckpt_reserve(device) != 0)
        return -1;
    if (ckpt_copy_layer(device, layer, true) != 0) {
        fprintf(stderr, "pn_tt: gdn checkpoint — the capture copy of layer %u failed\n", layer);
        return -1;
    }
    return 0;
}

int pn_tt_gdn_checkpoint_restore(pn_tt_device *device)
{
    const double started = pn_time_ms();
    uint32_t layer;

    if (!pn_tt_gdn_is_armed(device) || !device->gdn_checkpoint)
        return -1;
    /* Positions move backwards: the on-card dispatch model re-learns, as at a sequence start. */
    pn_tt_dispatch_sequence_reset();
    if (pn_tt_red("ckpt_restore_skip")) {
        fprintf(stderr, "pn_tt: gdn checkpoint — RED ckpt_restore_skip: nothing restored\n");
        return 0;
    }
    for (layer = 0; layer < device->gdn_shape.layer_count; layer++)
        if (ckpt_copy_layer(device, layer, false) != 0) {
            fprintf(stderr, "pn_tt: gdn checkpoint — the restore copy failed at layer %u; the "
                            "live state is undefined\n", layer);
            return -1;
        }
    fprintf(stderr, "pn_tt: gdn checkpoint — restored %u layers in %.1f ms\n",
            device->gdn_shape.layer_count, pn_time_ms() - started);
    return 0;
}

bool pn_tt_gdn_checkpoint_verify_wanted(void)
{
    return pn_tt_env_u32("PN_TT_CKPT_VERIFY", 0u) != 0u;
}

/** Count differing words between the first and last CKPT_VERIFY_BYTES of two buffers. */
static long long ckpt_verify_buffer(pn_tt_device *device, const pn_tt_gdn_buffer *a,
                                    const pn_tt_gdn_buffer *b, size_t bytes, uint32_t *ha,
                                    uint32_t *hb, unsigned long long *words)
{
    const size_t span = bytes < CKPT_VERIFY_BYTES ? bytes : CKPT_VERIFY_BYTES;
    const size_t offsets[2] = { 0u, bytes - span };
    long long differ = 0;
    size_t end, i;

    for (end = 0; end < 2u; end++) {
        const uint32_t at = (uint32_t)offsets[end];

        if (pn_tt_read_dram(device, a->bank, a->address + at, ha, span) != 0 ||
            pn_tt_read_dram(device, b->bank, b->address + at, hb, span) != 0)
            return -1;
        for (i = 0; i < span / 4u; i++)
            differ += ha[i] != hb[i];
        *words += span / 4u;
    }
    return differ;
}

long long pn_tt_gdn_checkpoint_verify_layer(pn_tt_device *device, uint32_t layer,
                                            unsigned long long *words)
{
    uint32_t *ha, *hb;
    long long window_differ, state_differ;
    size_t window, state;

    if (!pn_tt_gdn_checkpoint_verify_wanted() || !device || !device->gdn_checkpoint ||
        layer >= device->gdn_shape.layer_count)
        return 0;
    ha = malloc(CKPT_VERIFY_BYTES);
    hb = malloc(CKPT_VERIFY_BYTES);
    if (!ha || !hb) {
        free(ha);
        free(hb);
        return -1;
    }
    pn_tt_gdn_carried_bytes(device, &window, &state);
    window_differ = ckpt_verify_buffer(device, &device->gdn_region[layer].window,
                                       &device->gdn_checkpoint[layer].window, window, ha, hb,
                                       words);
    state_differ = ckpt_verify_buffer(device, &device->gdn_region[layer].state,
                                      &device->gdn_checkpoint[layer].state, state, ha, hb,
                                      words);
    free(ha);
    free(hb);
    return window_differ < 0 || state_differ < 0 ? -1 : window_differ + state_differ;
}

#endif /* USE_TT */
