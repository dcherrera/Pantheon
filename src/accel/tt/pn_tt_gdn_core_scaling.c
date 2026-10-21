/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdn_core_scaling.c
 * @brief Implementation of the GDN multi-core probe. See pn_tt_gdn_core_scaling.h.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_gdn_core_scaling.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_gdn.h"

/** Extra cores this probe will ever hold. 90 is the GDN's own and 91 is the head norm's, so
 *  the extras start after them and eight of them clears every other residency's default. */
#define PN_TT_GDN_PROBE_MAX_EXTRA 7u

/** First Tensix the extras take. A default, not a constant — `PN_TT_GDN_PROBE_CORE_FIRST`
 *  moves it, and `pn_tt_prepare_family_on_core()` still refuses a matmul member by name. */
#define PN_TT_GDN_PROBE_CORE_FIRST_ENV     "PN_TT_GDN_PROBE_CORE_FIRST"
#define PN_TT_GDN_PROBE_CORE_FIRST_DEFAULT 92u

/** Descriptor sets timed at each width. Enough that the spread is visible and few enough
 *  that the probe is over in under a second. */
#define PN_TT_GDN_PROBE_ROUNDS 16u

static double probe_now(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint32_t probe_env_u32(const char *name, uint32_t fallback)
{
    const char *value = getenv(name);
    long parsed;

    if (!value || !*value)
        return fallback;
    parsed = strtol(value, NULL, 10);
    return parsed > 0 ? (uint32_t)parsed : fallback;
}

/** An fp32 operand descriptor for one of this family's DRAM buffers. */
static ft_tt_op_operand probe_operand(const pn_tt_device *device,
                                      const pn_tt_gdn_buffer *buffer)
{
    ft_tt_op_operand operand;

    operand.x = (uint16_t)device->dram[buffer->bank].x;
    operand.y = (uint16_t)device->dram[buffer->bank].y;
    operand.address = buffer->address;
    operand.tile_bytes = FT_TT_OP_GDN_FP32_TILE_BYTES;
    return operand;
}

/**
 * Arm one extra GDN residency on `core`, with the SAME circular-buffer placement the shipped
 * one has.
 *
 * The placement comes from a CONV-shaped request even though the probe submits DELTA
 * descriptors, because one placement serves both phases — that is a property of this family
 * and `pn_tt_gdn.c` arms the real residency exactly the same way. Arming it any other way
 * would be measuring a different residency from the one the model runs.
 */
static bool probe_arm_one(pn_tt_device *device, pn_tt_residency *residency, uint32_t core,
                          const ft_tt_op_gdn_request *shape)
{
    ft_tt_launch_cb_entry placed[FT_TT_OP_GDN_CB_COUNT];
    ft_tt_op_gdn_request request = *shape;

    request.phase = FT_TT_OP_GDN_PHASE_CONV;
    request.activation_in.tile_bytes = FT_TT_OP_GDN_FP32_TILE_BYTES;
    request.activation_out.tile_bytes = FT_TT_OP_GDN_FP32_TILE_BYTES;
    if (!pn_tt_prepare_family_on_core(device, residency, PN_TT_GDN_PROGRAM, "gdn-probe",
                                      PN_TT_GDN_PROBE_CORE_FIRST_ENV, core,
                                      PN_TT_ONCARD_TRIAD_MASK))
        return false;
    if (ft_tt_op_gdn_place_cbs(&request, placed, residency->cb_buffer,
                               sizeof(residency->cb_buffer), &residency->cb_mask) != FT_TT_OK) {
        pn_tt_residency_release(residency);
        return false;
    }
    if (!pn_tt_start_family(residency)) {
        pn_tt_residency_release(residency);
        return false;
    }
    return true;
}

/**
 * One timed set: `width` descriptors started before any is waited on, `rounds` times.
 *
 * THE ENQUEUES COME FIRST AND THE WAITS COME SECOND. Enqueue-wait per core would measure
 * `width` SERIAL descriptors and print a flat line at every width — the shape of "it does
 * not parallelise" produced by the harness rather than by the card, which is the failure this
 * whole project keeps finding. The two loops being separate is the measurement.
 */
static double probe_time_width(pn_tt_device *device, pn_tt_residency **residency,
                               uint32_t width, uint32_t rounds,
                               const ft_tt_op_gdn_request *template_request, bool *out_ok)
{
    uint32_t words[PN_TT_GDN_PROBE_MAX_EXTRA + 1u][FT_TT_OP_GDN_WORDS];
    double started;
    uint32_t round;
    uint32_t index;

    /* Each core gets its OWN layer's carried state, so no two descriptors of a set touch the
     * same DRAM and the timing is not a serialisation on one address. */
    for (index = 0; index < width; index++) {
        ft_tt_op_gdn_request request = *template_request;

        request.state_recurrent = probe_operand(device, &device->gdn_region[index].state);
        if (ft_tt_op_gdn_parameters(&request, words[index]) != FT_TT_OK) {
            *out_ok = false;
            return 0.0;
        }
    }

    started = probe_now();
    for (round = 0; round < rounds; round++) {
        for (index = 0; index < width; index++)
            if (pn_tt_residency_enqueue(residency[index], FT_TT_QUEUE_OP_GDN, words[index],
                                        1u) != 0) {
                *out_ok = false;
                return 0.0;
            }
        for (index = 0; index < width; index++)
            if (pn_tt_residency_await(residency[index], PN_TT_OP_TIMEOUT_MS) != 0) {
                *out_ok = false;
                return 0.0;
            }
    }
    *out_ok = true;
    return probe_now() - started;
}

void pn_tt_gdn_core_scaling_probe(pn_tt_device *device,
                                  const ft_tt_op_gdn_request *template_request)
{
    const uint32_t asked = probe_env_u32("PN_TT_GDN_CORES", 0u);
    const uint32_t first = probe_env_u32(PN_TT_GDN_PROBE_CORE_FIRST_ENV,
                                         PN_TT_GDN_PROBE_CORE_FIRST_DEFAULT);
    pn_tt_residency extra[PN_TT_GDN_PROBE_MAX_EXTRA];
    pn_tt_residency *set[PN_TT_GDN_PROBE_MAX_EXTRA + 1u];
    double one_core = 0.0;
    uint32_t armed = 0;
    uint32_t width;
    uint32_t index;

    if (!device || !template_request || asked < 2u || device->gdn_probe_done)
        return;
    device->gdn_probe_done = true;
    if (!pn_tt_gdn_is_armed(device))
        return;

    width = asked > PN_TT_GDN_PROBE_MAX_EXTRA + 1u ? PN_TT_GDN_PROBE_MAX_EXTRA + 1u : asked;
    /* A width past the layer count would give two cores the same state region, which is the
     * one thing this probe promises not to do. */
    if (width > device->gdn_shape.layer_count)
        width = device->gdn_shape.layer_count;

    memset(extra, 0, sizeof(extra));
    set[0] = &device->gdn;
    for (index = 0; index + 1u < width; index++) {
        if (!probe_arm_one(device, &extra[index], first + index, template_request))
            break;
        set[index + 1u] = &extra[index];
        armed++;
    }
    if (armed + 1u < width) {
        fprintf(stderr, "pn_tt: gdn CORE SCALING — only %u of the %u cores armed; the sweep "
                        "stops there\n", armed + 1u, width);
        width = armed + 1u;
    }
    fprintf(stderr, "pn_tt: gdn CORE SCALING — %u cores from %u, %u rounds each. THIS WRITES "
                    "THE CARRIED STATE OF %u LAYERS: the run's text is not evidence.\n",
            width, first, PN_TT_GDN_PROBE_ROUNDS, width);

    for (index = 1u; index <= width; index <<= 1) {
        bool ok = false;
        const double seconds =
            probe_time_width(device, set, index, PN_TT_GDN_PROBE_ROUNDS, template_request, &ok);
        const double per_set = seconds / (double)PN_TT_GDN_PROBE_ROUNDS;
        const double per_descriptor = per_set / (double)index;

        if (!ok) {
            fprintf(stderr, "pn_tt: gdn CORE SCALING — width %u FAILED; the residencies are "
                            "stranded and the sweep stops\n", index);
            break;
        }
        if (index == 1u)
            one_core = per_set;
        fprintf(stderr, "pn_tt: gdn CORE SCALING   width %2u — %.4f s a set, %.1f us a "
                        "descriptor, %.2fx the throughput of one core\n",
                index, per_set, 1e6 * per_descriptor,
                per_set > 0.0 ? one_core * (double)index / per_set : 0.0);
    }

    for (index = 0; index < armed; index++)
        pn_tt_residency_release(&extra[index]);
    fprintf(stderr, "pn_tt: gdn CORE SCALING — %u extra cores given back\n", armed);
}

#endif /* USE_TT */
