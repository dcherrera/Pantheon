/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_core_boot_guard.c
 * @brief Implementation of the pre-boot readiness gate (see pn_tt_core_boot_guard.h).
 *
 * @status New 2026-08-24.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_core_boot_guard.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

#include "pn_tt_residency.h"
#include "tt/ft_tt_core_readiness.h"
#include "tt/ft_tt_launch.h"
#include "tt/ft_tt_memmap.h"

/**
 * Milliseconds this guard will wait for a busy core to come free.
 *
 * IT IS LONGER THAN THE STOP'S GRACE ON PURPOSE, AND THE DIFFERENCE IS MEASURED.
 * `pn_tt_residency_stop()` is issued straight after a refusal, so what it waits out is the
 * CB-wait bound — 100,000,000 spins — and 20 s covers it. This guard waits out whatever an
 * EARLIER PROCESS left, and a residency abandoned by a process that DIED sits in
 * `ftq_wait_for_work()` until the IDLE bound instead: 400,000,000 spins, recorded in
 * `pn_tt_device.c`'s own death-stop warning as **over 40 s**. A 20 s guard would refuse a
 * core that was 30 s from clearing, and that refusal costs the whole model the
 * ~26 s/position host forward — far more than the wait.
 *
 * The cost of the margin is zero on a healthy core: this wait does not run unless the card
 * has already said a processor is executing.
 */
#define PN_TT_BOOT_GUARD_WAIT_MS 90000u

/** Milliseconds between readiness re-reads while waiting out a spin.
 *
 *  The stop polls at one millisecond because it is waiting on a queue field it can read in
 *  3 us. A readiness read is a mailbox snapshot plus fourteen more reads, ~400 us, and the
 *  thing it waits for takes seconds — so polling it at 1 ms would spend a third of the wait
 *  inside the instrument. 50 ms costs at most 50 ms of extra wait on a spin that is orders
 *  of magnitude longer. */
#define PN_TT_BOOT_GUARD_POLL_MS 50u

/** Names in FT_TT_PROC_* order, for the ticking mask. */
static const char *const PN_TT_BOOT_GUARD_PROCESSOR_NAME[FT_TT_PROC_COUNT] = {
    "BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2"
};

/** "TRISC0 TRISC2" into `out`, or "none". Never NULL, never unterminated. */
static void describe_ticking(uint32_t mask, char *out, size_t out_bytes)
{
    unsigned processor;
    size_t used = 0;

    if (!out || out_bytes == 0u)
        return;
    out[0] = '\0';
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        int written;

        if ((mask & (1u << processor)) == 0u)
            continue;
        written = snprintf(out + used, out_bytes - used, "%s%s", used ? " " : "",
                           PN_TT_BOOT_GUARD_PROCESSOR_NAME[processor]);
        if (written < 0 || (size_t)written >= out_bytes - used)
            return;
        used += (size_t)written;
    }
    if (used == 0u)
        snprintf(out, out_bytes, "none");
}

/**
 * The whole refusal, in one block, on stderr.
 *
 * IT IS LONG ON PURPOSE. The failure mode this guard sits in front of is not "the arm did
 * not happen" — it is "the arm did not happen, the model quietly ran on the CPU at
 * ~26 s/position, and the numbers looked plausible for eleven minutes". A refusal that reads
 * like one more warning would reproduce exactly that. Every line here answers a question the
 * person reading it is about to ask: which core, what did the card say, can I wait, what do
 * I run, and what happens to the model now.
 */
static void print_refusal(const ft_tt_core_readiness *readiness, uint32_t core_index,
                          const char *who, const char *core_env, unsigned waited_ms)
{
    char ticking[64];

    describe_ticking(readiness->ticking_mask, ticking, sizeof(ticking));
    fprintf(stderr,
            "pn_tt: ==================================================================\n"
            "pn_tt: %s — Tensix %u is NOT READY, so the %s residency is not armed.\n"
            "pn_tt:   the card says: %s\n"
            "pn_tt:   verdict '%s' · reset 0x%08x · instrn_buf 0x%08x · go 0x%02x\n"
            "pn_tt:   resident queue %s · control heartbeat %s · executing: %s\n",
            PN_TT_BOOT_GUARD_REFUSAL_TOKEN, core_index, who,
            ft_tt_core_readiness_string(readiness->state),
            ft_tt_core_verdict_string(readiness->status.verdict),
            readiness->status.state.reset_register, readiness->status.state.instrn_buf_status,
            readiness->status.state.go_messages[0].signal,
            readiness->status.queue_present ? "present" : "absent",
            readiness->status.queue_heartbeat_ticking ? "ticking" : "frozen", ticking);
    if (waited_ms != 0u)
        fprintf(stderr, "pn_tt:   waited %u ms for it to clear and it did not.\n", waited_ms);

    if (readiness->state == FT_TT_CORE_NOT_READY_STILL_EXECUTING) {
        fprintf(stderr,
                "pn_tt:   THIS IS A LEFTOVER FROM AN EARLIER PROCESS. A refused descriptor\n"
                "pn_tt:   leaves a compute thread spinning out its 400,000,000-iteration\n"
                "pn_tt:   bound, and that thread outlives the process that started it. It\n"
                "pn_tt:   WILL finish — leave the core alone and try again.\n");
    }
    if (readiness->state == FT_TT_CORE_NOT_READY_INSTRUCTION_FIFO) {
        fprintf(stderr,
                "pn_tt:   THIS CORE IS ALREADY WEDGED. Rung 1 does NOT clear it. Only a\n"
                "pn_tt:   whole-card reset does, and that resets the WHOLE CARD:\n"
                "pn_tt:     sudo -n env FT_TT_DRIVER=udrv ./tt/tools/build/ft_tt_core \\\n"
                "pn_tt:       recover %u --rung 3 --write --then-probe --driver udrv\n",
                core_index);
    }

    /* NOTHING WAS WRITTEN, and saying so is load-bearing: the next question after a refusal
     * is always "is the card in a worse state now than before I ran this". */
    fprintf(stderr,
            "pn_tt:   NOTHING WAS WRITTEN TO THIS CORE. Booting firmware onto it is the\n"
            "pn_tt:   wedge (instrn_buf 0x66), which is what this refusal prevented.\n"
            "pn_tt:   CONSEQUENCE: with this family unarmed the model may fall back to the\n"
            "pn_tt:   HOST forward at ~26 s/position. Watch for 'using CPU forward' and\n"
            "pn_tt:   treat plausible-but-very-slow numbers after this line as INVALID.\n");
    if (core_env)
        fprintf(stderr, "pn_tt:   To arm somewhere else instead, set %s to a free core.\n",
                core_env);
    fprintf(stderr,
            "pn_tt:   `ft_tt_core status %u --detail` reads this core without touching it.\n"
            "pn_tt: ==================================================================\n",
            core_index);
}

/** One millisecond of sleep, `count` times. */
static void nap_ms(unsigned count)
{
    struct timespec nap = { 0, 1000000L };
    unsigned index;

    for (index = 0; index < count; index++)
        (void)nanosleep(&nap, NULL);
}

bool pn_tt_core_boot_is_permitted(ft_tt_runtime *runtime, uint32_t core_index, const char *who,
                                  const char *core_env)
{
    ft_tt_core_readiness readiness;
    unsigned waited_ms = 0u;

    if (!runtime || !who)
        return true;

    /* A READ THAT FAILS IS NOT A REFUSAL.
     *
     * This guard catches one positively-identified condition: a core the card says is busy.
     * Everything else must pass. If the coordinate does not resolve or the reading itself
     * errors, refusing would let a diagnostic path's own failure take the whole model to the
     * ~26 s/position host forward — the exact silent degradation this round is defending
     * against, arriving through the defence. The boot below has its own error path for a
     * core that genuinely cannot be reached. */
    if (ft_tt_core_readiness_read(&runtime->noc, &runtime->grid, core_index, &readiness) !=
        FT_TT_OK)
        return true;
    if (readiness.ready)
        return true;

    /* WAIT ONLY WHEN THE READING SAYS WAITING IS THE ANSWER. A spun-out bound finishes; a
     * 0x66 core, a stalled kernel and a stuck boot do not, and PN_TT_BOOT_GUARD_WAIT_MS
     * spent in front of an inevitable refusal is a minute and a half of a model load that
     * nobody gets back. */
    if (readiness.waiting_can_help) {
        fprintf(stderr, "pn_tt: Tensix %u is still executing — %s waits up to %u ms. This is a "
                        "leftover spinning compute thread, and it is bounded.\n",
                core_index, who, (unsigned)PN_TT_BOOT_GUARD_WAIT_MS);
        while (waited_ms < (unsigned)PN_TT_BOOT_GUARD_WAIT_MS) {
            nap_ms(PN_TT_BOOT_GUARD_POLL_MS);
            waited_ms += PN_TT_BOOT_GUARD_POLL_MS;
            if (ft_tt_core_readiness_read(&runtime->noc, &runtime->grid, core_index,
                                          &readiness) != FT_TT_OK)
                return true;
            if (readiness.ready) {
                fprintf(stderr, "pn_tt: Tensix %u came free after %u ms — %s arms normally\n",
                        core_index, waited_ms, who);
                return true;
            }
            /* The reading may CHANGE under us: a spin that ends inside a wedged core stops
             * ticking and starts reading 0x66. Re-asking every poll is what lets the refusal
             * name the state the core is actually in rather than the one it started in. */
            if (!readiness.waiting_can_help)
                break;
        }
    }

    print_refusal(&readiness, core_index, who, core_env, waited_ms);
    return false;
}

#endif /* USE_TT */
