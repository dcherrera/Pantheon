/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_unmirrored_probe.c
 * @brief Timing-only spread of an unmirrored projection input. See pn_tt_unmirrored_probe.h.
 * @status Probe; off by default.
 * @issues None
 * @todo None
 */
#ifdef USE_TT

#include "pn_tt_unmirrored_probe.h"

#include <stdio.h>

#include "pn_tt_device_internal.h"

uint32_t pn_tt_unmirrored_probe_mode(void)
{
    static int cached = -1;

    if (cached < 0) {
        cached = (int)pn_tt_env_u32("PN_TT_UNMIRRORED_PROBE", 0u);
        if (cached)
            fprintf(stderr, "pn_tt: UNMIRRORED PROBE %d — unmirrored multi-row inputs read A "
                            "from 8 banks (%s); TIMING ONLY, the text DIFFERS\n", cached,
                    cached == 1 ? "on the grid" : "on the decode group");
    }
    return (uint32_t)cached;
}

bool pn_tt_unmirrored_probe_view(const pn_tt_activation *input, uint32_t tile_rows,
                                 uint32_t bank_count, pn_tt_activation *out)
{
    uint32_t replica;

    if (!pn_tt_unmirrored_probe_mode() || !input || input->replica_count > 1u ||
        tile_rows <= 1u || bank_count < 2u)
        return false;
    *out = *input;
    out->replica_count = bank_count < PN_TT_ACTIVATION_MAX_REPLICAS
                             ? bank_count : PN_TT_ACTIVATION_MAX_REPLICAS;
    for (replica = 0; replica < out->replica_count; replica++) {
        out->replica_bank[replica] = (input->bank + replica) % bank_count;
        out->replica_address[replica] = input->address;
    }
    return true;
}

#endif /* USE_TT */
