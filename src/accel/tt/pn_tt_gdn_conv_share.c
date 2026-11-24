/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdn_conv_share.c
 * @brief Implementation of one conv weight for both GDN families (pn_tt_gdn_conv_share.h).
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_gdn_conv_share.h"

#include <stdio.h>
#include <stdlib.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_gdn_layout.h"
#include "pn_tt_gdnc.h"

/** Is `name` set to an explicit 0? Read straight from the environment (no cache), and any
 *  non-number counts as 0, so an unreadable knob always falls back to main's placement. */
static bool share_env_declines(const char *name)
{
    const char *value = getenv(name);

    return value && *value && atoi(value) == 0;
}

bool pn_tt_gdn_conv_share_wanted(bool fit_policy)
{
    /* The strip is what makes the decode CONV blind to rows 1..31; the chunkwise family is
     * what holds the constant. Without either, sharing would not be bit-exact or not exist. */
    return fit_policy && !share_env_declines(PN_TT_GDN_CONV_SHARE_ENV) &&
           !share_env_declines("PN_TT_GDN_CONV_STRIP") &&
           !share_env_declines("PN_TT_PREFILL_GDNC");
}

void pn_tt_gdn_conv_share_request(pn_tt_device *device, bool share)
{
    if (device)
        device->gdn_conv_share = share ? PN_TT_GDN_CONV_PENDING : PN_TT_GDN_CONV_OWN;
}

bool pn_tt_gdn_conv_weight_is_shared(const pn_tt_device *device)
{
    return device && device->gdn_conv_share == PN_TT_GDN_CONV_SHARED;
}

/** Bytes of one layer's decode weight: `taps` planes of `conv_dim/32` fp32 tiles. */
static size_t share_weight_bytes(const pn_tt_device *device)
{
    return (size_t)(device->gdn_shape.conv_dim / PN_TT_GDN_LAYOUT_TILE_DIM) *
           device->gdn_shape.conv_taps * PN_TT_GDN_LAYOUT_TILE_BYTES;
}

/** Point every layer's decode weight at its chunkwise copy 0. */
static void share_alias_layers(pn_tt_device *device)
{
    const uint32_t layers = device->gdn_shape.layer_count;
    /* RED `gdn_conv_share_shift`: one tap plane in, so decode reads the wrong taps. */
    const uint32_t shift = pn_tt_red("gdn_conv_share_shift")
                               ? (device->gdn_shape.conv_dim / PN_TT_GDN_LAYOUT_TILE_DIM) *
                                     PN_TT_GDN_LAYOUT_TILE_BYTES
                               : 0u;

    for (uint32_t layer = 0; layer < layers; layer++) {
        device->gdn_region[layer].weight = device->gdnc_constant[layer];
        device->gdn_region[layer].weight.address += shift;
    }
    device->gdn_conv_share = PN_TT_GDN_CONV_SHARED;
    fprintf(stderr, "pn_tt: gdn — CONV WEIGHT SHARED: the decode CONV reads row 0 of the "
                    "chunkwise constant copy 0; %u layers x %.2f MiB not allocated "
                    "(%.2f MiB)%s\n",
            layers, (double)share_weight_bytes(device) / (1024.0 * 1024.0),
            (double)share_weight_bytes(device) * layers / (1024.0 * 1024.0),
            shift ? " [RED gdn_conv_share_shift]" : "");
}

int pn_tt_gdn_conv_share_bind(pn_tt_device *device)
{
    if (!device || device->gdn_conv_share != PN_TT_GDN_CONV_PENDING)
        return 0;
    if (!device->gdn_region) {
        device->gdn_conv_share = PN_TT_GDN_CONV_OWN;
        return 0;
    }
    if (pn_tt_gdnc_is_armed(device)) {
        share_alias_layers(device);
        return 0;
    }
    /* No chunkwise constant to read: the decode family takes its own weights after all. Late,
     * so they land wherever the cursor is now — the placement is the fit policy's either way. */
    device->gdn_conv_share = PN_TT_GDN_CONV_OWN;
    for (uint32_t layer = 0; layer < device->gdn_shape.layer_count; layer++)
        if (pn_tt_gdn_alloc_conv_weight(device, &device->gdn_region[layer].weight) != 0)
            return -1;
    fprintf(stderr, "pn_tt: gdn — the chunkwise family did not arm, so the decode CONV keeps "
                    "its own conv weight (allocated now)\n");
    return 0;
}

#endif /* USE_TT */
