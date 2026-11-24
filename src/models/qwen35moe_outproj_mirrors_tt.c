/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_outproj_mirrors_tt.c
 * @brief L2: mirror the DECODE out-projection inputs, so the out-projection's 16 matmul
 *        members stop reading A from one DRAM bank.
 *
 * Measured on a p150 (Qwen3.8-27B, `PN_TT_MATMUL_CLASS_CLOCK=1`): every decode matmul class reads its
 * A strip from 8 mirrors and runs 400-466 GB/s, except the out-projection (ssm_out and
 * attn_output, 64 a token, 46 MB a flush) at 248 GB/s. Its inputs, `ssm_gated` and attention
 * `gated`, are the only decode A strips with one copy: 16 members x 2 column blocks x 192 A
 * tiles = 12.6 MB a flush out of ONE bank (~60-70 GB/s), the whole flush's floor.
 *
 * Both strips are per-layer scratch written by the elementwise residency (silu_mul and
 * sigmoid+mul), whose shared writer fans out to a range record (pn_tt_replica_map.h). They are
 * mirrored here, at the END of the arm, from leftover DRAM (pn_tt_late_mirrors.h): the served
 * MTP load leaves its fullest bank 3 MiB free, so an early mirror would move every later
 * allocation. The matmul reads identical bytes from any copy, so the output is unchanged by
 * construction; the card A/B checks logits bit-identical.
 *
 * `PN_TT_DECODE_OUTPROJ_MIRRORS` = copies wanted, primary included (default 8; 0 or 1 off).
 * `PN_TT_DECODE_OUTPROJ_RESERVE_MIB` = free MiB every used bank keeps (default 16: a prefill
 * allocates ~2 MiB after the arm). Declined with `PN_TT_GATE_NARROW` on: that residency arms
 * after the map is published and writes `ssm_gated` with an empty table. A refused allocation
 * leaves that strip single; only a failed map publish (mirrors that would go stale) fails the
 * arm.
 *
 * @status L2. Card A/B staged, not run.
 * @issues None
 * @todo None
 */
#ifdef USE_TT

#include "qwen35moe_tt_internal.h"

#include <stdio.h>
#include <stdlib.h>

#include "../accel/tt/pn_tt_gate.h"
#include "../accel/tt/pn_tt_late_mirrors.h"
#include "../accel/tt/pn_tt_replica_map.h"

#define QWEN35_TT_OUTPROJ_MIRRORS_DEFAULT 8u
#define QWEN35_TT_OUTPROJ_RESERVE_MIB_DEFAULT 16u

static uint32_t outproj_env_u32(const char *name, uint32_t fallback)
{
    const char *value = getenv(name);

    return value && *value ? (uint32_t)strtoul(value, NULL, 10) : fallback;
}

int pn_qwen35_tt_decode_outproj_mirrors(pn_qwen35_tt *g)
{
    const uint32_t wanted =
        outproj_env_u32("PN_TT_DECODE_OUTPROJ_MIRRORS", QWEN35_TT_OUTPROJ_MIRRORS_DEFAULT);
    const uint64_t reserve = (uint64_t)outproj_env_u32("PN_TT_DECODE_OUTPROJ_RESERVE_MIB",
                                                       QWEN35_TT_OUTPROJ_RESERVE_MIB_DEFAULT) *
                             1024ull * 1024ull;
    const char *gate = getenv(PN_TT_GATE_NARROW_ENV);
    pn_tt_activation *strips[2];
    const pn_tt_activation *slots[2];
    uint32_t count = 0u, index;
    int copies[2] = { 1, 1 };

    if (!g || !g->device)
        return -1;
    if (wanted < 2u)
        return 0;
    if (gate && *gate && atoi(gate) != 0) {
        fprintf(stderr, "qwen35_tt: decode out-projection mirrors OFF — %s=1 writes ssm_gated "
                        "from a residency that arms after the replica map\n",
                PN_TT_GATE_NARROW_ENV);
        return 0;
    }
    strips[count++] = &g->card.ssm_gated;
    if (pn_qwen35_tt_attn_is_on_card(g))
        strips[count++] = &g->attn_card.gated;
    for (index = 0u; index < count; index++) {
        /* A strip taller than 4 bands would get no decode range record
         * (pn_tt_replica_map_set_decode_ranged): its mirrors would go stale. */
        if (strips[index]->tile_rows <= 4u)
            copies[index] = pn_tt_activation_mirror_late(g->device, strips[index], wanted,
                                                         reserve);
        /* A refused allocation leaves the strip unmirrored, which is the shipped path. */
        if (copies[index] < 0) {
            fprintf(stderr, "qwen35_tt: decode out-projection mirror %u refused — that strip "
                            "stays single\n", index);
            copies[index] = 1;
        }
        slots[index] = strips[index];
    }
    pn_tt_replica_map_set_decode_ranged(slots, count);
    fprintf(stderr, "qwen35_tt: decode out-projection inputs mirrored late — ssm_gated x%d, "
                    "attention gated x%d (of %u wanted, %u MiB kept free a bank); "
                    "PN_TT_DECODE_OUTPROJ_MIRRORS=1 off\n",
            copies[0], count > 1u ? copies[1] : 0, wanted,
            (unsigned)(reserve / (1024ull * 1024ull)));
    {
        const pn_tt_activation *mirrored[2] = { &g->card.normed, &g->card.ffn_swiglu };

        return pn_tt_publish_replica_map(g->device, mirrored, 2u, &g->card.ffn_swiglu);
    }
}

#endif /* USE_TT */
