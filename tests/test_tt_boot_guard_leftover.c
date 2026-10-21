/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_boot_guard_leftover.c
 * @brief The boot guard against a REAL leftover — two processes, one card, NO MODEL.
 *
 * ---------------------------------------------------------------------------
 * WHY IT HAS TO BE TWO PROCESSES
 * ---------------------------------------------------------------------------
 * `pn_tt_residency_stop()`'s grace and the `retired` flag already stop the process that
 * CAUSED a spin from re-arming onto it. The residual they do not reach is that the spinning
 * compute thread OUTLIVES the process: the next `serve`, or a model reload, starts with an
 * empty residency table and boots firmware onto a core that is still running. That is a
 * cross-process failure and no single-process arm can produce it.
 *
 * So this is deliberately two invocations of one binary:
 *
 *   --leak CORE    open the card, arm the reduce residency on CORE, submit a descriptor the
 *                  reduce program does not implement — the reader refuses it, sets
 *                  kErrBadOpcode and leaves, and the compute thread is left spinning out its
 *                  bound in a CB wait that will never be satisfied — then `_exit()` WITHOUT
 *                  closing the device. That is exactly the state a crashed or killed serve
 *                  leaves behind, produced by the same mechanism rather than imitated.
 *   --arm CORE     open the card and arm the same residency on the same core, through the
 *                  ordinary `pn_tt_prepare_family_on_core()` path. THIS is the process that
 *                  used to wedge the card.
 *   --read CORE    neither: report what `pn_tt_core_boot_is_permitted()` would decide, then
 *                  return. Writes nothing to that core on any path, so it is the safe first
 *                  stage — run it between --leak and --arm and the whole question is
 *                  answered before anything is booted.
 *
 * ---------------------------------------------------------------------------
 * IT LOADS NO MODEL, AND THAT IS A HARD PROPERTY OF THIS FILE
 * ---------------------------------------------------------------------------
 * Two `pantheon` processes each mapping a 9.5 GB GGUF on a 7.1 GB host took the TT host off the
 * network and needed a physical power action. Nothing here opens a GGUF, allocates weights
 * or calls the engine: it is `pn_tt_device_open()`, one residency, one descriptor. Host RSS
 * is a few MB. It is still a CARD ARM and still belongs behind the flock runner, because the
 * card is a single resource — but it is not a model load and must never become one.
 *
 * ---------------------------------------------------------------------------
 * THE CORE TO USE
 * ---------------------------------------------------------------------------
 * Not a core any family defaults to. Matmul members run from 75 and the set MOVES with
 * `PN_TT_MATMUL_CORES`; GDN is 90 with its head-split members above it; the head-norm slots
 * are 91 and 103; attention is 82/83/84; reduce 72, eltwise 73, topk 74; the device's own
 * core is 71. `--leak` refuses a core below 110 unless `--i-mean-it` is given, so a
 * mistyped argument cannot strand a core the next real run needs.
 *
 * @status New 2026-08-24.
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef USE_TT

int main(void)
{
    fprintf(stderr, "test_tt_boot_guard_leftover: built without USE_TT — nothing to do\n");
    return 0;
}

#else

#include "../src/accel/tt/pn_tt_core_boot_guard.h"
#include "../src/accel/tt/pn_tt_device_internal.h"
#include "tt/ft_tt_core_readiness.h"
#include "tt/ft_tt_op_rms_norm.h"

/** Below this, every core belongs to a family that defaults to it. */
#define GUARD_TEST_LOWEST_SAFE_CORE 110u

/** An opcode no resident program implements, so the reader refuses the descriptor by the
 *  documented path (`kErrBadOpcode`) rather than by a shape nobody can reproduce. */
#define GUARD_TEST_UNIMPLEMENTED_OPCODE 0x7Eu

/** Long enough that a descriptor which was going to complete has. */
#define GUARD_TEST_TIMEOUT_MS 4000u

/**
 * Arm the reduce residency on `core`, the same way `pn_tt_arm_reduce()` does.
 *
 * The same family, the same program and the same circular-buffer placement, so what this
 * proves is about the arming path the model uses and not about a second one written here.
 */
static bool arm_reduce_on(pn_tt_device *device, uint32_t core)
{
    static const ft_tt_op_reduce_function members[] = { FT_TT_OP_REDUCE_FN_RMS_NORM };
    ft_tt_launch_cb_entry placed[FT_TT_OP_REDUCE_CB_COUNT];
    ft_tt_op_reduce_residency residency;

    if (ft_tt_op_reduce_residency_for(members, 1u, 4u, PN_TT_TILE_BF16, &residency) !=
        FT_TT_OK) {
        fprintf(stderr, "leftover: no reduce residency serves Wt=4\n");
        return false;
    }
    if (!pn_tt_prepare_family_on_core(device, &device->reduce, "reduce", "reduce",
                                      "PN_TT_CORE_REDUCE", core, PN_TT_ONCARD_TRIAD_MASK))
        return false;
    if (ft_tt_op_reduce_place_cbs(&residency, placed, device->reduce.cb_buffer,
                                  sizeof(device->reduce.cb_buffer),
                                  &device->reduce.cb_mask) != FT_TT_OK) {
        fprintf(stderr, "leftover: the reduce circular buffers do not fit\n");
        pn_tt_residency_release(&device->reduce);
        return false;
    }
    return pn_tt_start_family(&device->reduce);
}

/** Print what the card says about `core`, without writing to it. */
static void report_readiness(pn_tt_device *device, uint32_t core, const char *when)
{
    ft_tt_core_readiness readiness;
    unsigned processor;

    if (ft_tt_core_readiness_read(&device->runtime.noc, &device->runtime.grid, core,
                                  &readiness) != FT_TT_OK) {
        printf("[%s] core %u: the readiness read itself failed\n", when, core);
        return;
    }
    printf("[%s] core %u: %s — %s\n", when, core, readiness.ready ? "READY" : "NOT READY",
           ft_tt_core_readiness_string(readiness.state));
    printf("[%s]   verdict '%s'  instrn_buf 0x%08x  go 0x%02x  waiting %s help\n", when,
           ft_tt_core_verdict_string(readiness.status.verdict),
           readiness.status.state.instrn_buf_status,
           readiness.status.state.go_messages[0].signal,
           readiness.waiting_can_help ? "CAN" : "cannot");
    printf("[%s]   queue %s, CONTROL heartbeat %s\n", when,
           readiness.status.queue_present ? "present" : "absent",
           readiness.status.queue_heartbeat_ticking ? "TICKING" : "frozen");
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        if (!readiness.processor[processor].slot_read)
            continue;
        printf("[%s]   proc %u: state 0x%08x heartbeat %u %s\n", when, processor,
               readiness.processor[processor].state, readiness.processor[processor].heartbeat,
               readiness.processor[processor].ticking ? "TICKING" : "frozen");
    }
    fflush(stdout);
}

static int usage(void)
{
    fprintf(stderr,
            "usage: test_tt_boot_guard_leftover --leak CORE [--i-mean-it]\n"
            "       test_tt_boot_guard_leftover --read CORE\n"
            "       test_tt_boot_guard_leftover --arm  CORE\n"
            "\n"
            "  --leak  arm reduce on CORE, get a descriptor refused, exit WITHOUT closing\n"
            "          the device. Leaves a real spinning compute thread behind.\n"
            "  --read  report readiness for CORE. Writes nothing. Run this first.\n"
            "  --arm   arm reduce on CORE through the ordinary path — the process that\n"
            "          used to wedge the card. Exits 0 if it armed, 2 if the guard\n"
            "          refused, which is the PASS for the red arm.\n");
    return 1;
}

int main(int argc, char **argv)
{
    bool leak = false, read_only = false, arm = false, i_mean_it = false;
    pn_tt_device *device = NULL;
    uint32_t core = 0u;
    int argument;

    for (argument = 1; argument < argc; argument++) {
        if (strcmp(argv[argument], "--i-mean-it") == 0) {
            i_mean_it = true;
        } else if (argument + 1 < argc && strcmp(argv[argument], "--leak") == 0) {
            leak = true;
            core = (uint32_t)strtoul(argv[++argument], NULL, 10);
        } else if (argument + 1 < argc && strcmp(argv[argument], "--read") == 0) {
            read_only = true;
            core = (uint32_t)strtoul(argv[++argument], NULL, 10);
        } else if (argument + 1 < argc && strcmp(argv[argument], "--arm") == 0) {
            arm = true;
            core = (uint32_t)strtoul(argv[++argument], NULL, 10);
        } else {
            return usage();
        }
    }
    if ((leak ? 1 : 0) + (read_only ? 1 : 0) + (arm ? 1 : 0) != 1)
        return usage();
    if (leak && core < GUARD_TEST_LOWEST_SAFE_CORE && !i_mean_it) {
        fprintf(stderr, "refused: core %u is below %u, where every family's default lives. "
                        "Pass --i-mean-it if that really is the core you want stranded.\n",
                core, (unsigned)GUARD_TEST_LOWEST_SAFE_CORE);
        return 1;
    }

    if (pn_tt_device_open(0, &device) != 0 || !device) {
        fprintf(stderr, "leftover: the card did not open\n");
        return 1;
    }

    if (read_only) {
        report_readiness(device, core, "read");
        pn_tt_device_close(device);
        return 0;
    }

    if (arm) {
        bool armed;

        report_readiness(device, core, "before");
        armed = arm_reduce_on(device, core);
        printf("[arm] core %u: %s\n", core, armed ? "ARMED" : "REFUSED");
        report_readiness(device, core, "after");
        fflush(stdout);
        pn_tt_device_close(device);
        return armed ? 0 : 2;
    }

    /* --leak. Everything from here is deliberately one-way. */
    if (!arm_reduce_on(device, core)) {
        fprintf(stderr, "leftover: could not arm reduce on core %u — nothing was leaked\n",
                core);
        pn_tt_device_close(device);
        return 1;
    }
    report_readiness(device, core, "armed");
    {
        uint32_t parameters[FT_TT_QUEUE_PARAM_WORDS];
        int rc;

        memset(parameters, 0, sizeof(parameters));
        rc = pn_tt_residency_submit(&device->reduce, GUARD_TEST_UNIMPLEMENTED_OPCODE,
                                    parameters, GUARD_TEST_TIMEOUT_MS);
        printf("[leak] the refused descriptor returned %d (a -1 here is the point)\n", rc);
    }
    report_readiness(device, core, "refused");

    /* `_exit`, NOT `return`: an ordinary exit would run atexit handlers, and the whole
     * value of this arm is that NOTHING retires the residency. `pn_tt_device_close()` is
     * deliberately not called — a serve that segfaults does not call it either. */
    printf("[leak] leaving core %u with a live residency and exiting WITHOUT closing. "
           "The compute thread is spinning out its bound now.\n", core);
    fflush(stdout);
    _exit(0);
}

#endif /* USE_TT */
