/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_residency.c
 * @brief Implementation of one resident Tensix program on one core (see pn_tt_residency.h).
 *
 * @status Green: the reduce and elementwise residencies of the on-card forward run on it.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_residency.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pn_tt_name_list.h"

#include "tt/ft_tt_coord.h"
#include "tt/ft_tt_op_dispatch.h"
#include "tt/ft_tt_op_l1_layout.h"
#include "tt/ft_tt_program_resolve.h"

/** Consecutive empty polls before a resident kernel returns on its own, and spins inside
 *  ONE circular-buffer wait before it abandons that wait. Neither may be 0: an unbounded
 *  wait is the one thing a resident kernel may not be configured into. */
#define PN_TT_RESIDENCY_IDLE_SPINS    FT_TT_QUEUE_DEFAULT_IDLE_SPINS
#define PN_TT_RESIDENCY_CB_WAIT_SPINS 100000000u

/** Idle milliseconds after which an enqueue retires the residency first. See
 *  residency_idle_retire(). */
#define PN_TT_RESIDENCY_IDLE_RETIRE_ENV        "PN_TT_RESIDENCY_IDLE_RETIRE_MS"
/* 5000 ms: 1.6x under the fastest measured idle exit (~8 s), and past the 2.3-4.7 s load gap,
 * so only the families idle through a prefill retire — 237.2 vs 234.0 tok/s at 2000. */
#define PN_TT_RESIDENCY_IDLE_RETIRE_DEFAULT_MS 5000L
/** A keepalive NOP is the cheapest descriptor there is; a second is far past any healthy one. */
#define PN_TT_RESIDENCY_KEEPALIVE_TIMEOUT_MS 1000u

/** Polls of `pn_tt_residency_stop`'s exit wait. Each is a device read of ~3.5 µs, so this
 *  is comfortably longer than the longest descriptor and still bounded. */
#define PN_TT_RESIDENCY_STOP_POLLS 20000u

/** Every processor of a triad. */
#define PN_TT_RESIDENCY_TRIAD_MASK                                                        \
    ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1) | (1u << FT_TT_PROC_MATH0) |         \
     (1u << FT_TT_PROC_MATH1) | (1u << FT_TT_PROC_MATH2))

static double pn_tt_residency_now(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/**
 * THE AWAIT IS THE PER-DESCRIPTOR HOST COST OF EVERY FAMILY EXCEPT THE MATMUL, AND IT IS
 * SEVENTEEN NON-POSTED READS. `PN_TT_RESIDENCY_AWAIT=control|window|line`, default `window`.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE SEVENTEEN ARE, AND WHY THE MATMUL'S 6.9 us IS A DIFFERENT NUMBER
 * ---------------------------------------------------------------------------
 * `residency->runtime->noc` is the UC / WORD32 accessor: **one 32-bit PCIe access per word**
 * (ft_tt_runtime.h). Per descriptor per member this path issues
 *
 *   ISSUE   `write_op_parameters`   64 B -> 16 word writes, POSTED
 *           `write_ring_run`        32 B ->  8 word writes, POSTED
 *           `publish`                4 B ->  1 word write,  POSTED
 *   AWAIT   `read_field(COUNT)`      4 B ->  1 word read,   NON-POSTED
 *           `read_control`          64 B -> 16 word reads,  NON-POSTED
 *
 * The recorded matmul figures — a 64-byte parameter block at **2.08 us** and a 32-byte
 * descriptor plus a 4-byte head bump at **4.84 us**, ~6.9 us together — are the ISSUE side,
 * and issue is posted stores. **They were never a measurement of the await.** The matmul's
 * own await never came through here: `pn_tt_group_await()` reads
 * `read_completion_wide()` (two BLOCK16) and, with `PN_TT_FAST_COMPLETION` on, ONE
 * `read_completion_line()`. So the two figures do not disagree; they are different halves.
 *
 * At the recorded ~3.4 us a non-posted read, seventeen is **57.8 us** — which is the
 * ~58 us/descriptor the eltwise split's flat per-member await measured
 * (0.0079-0.0080 s/token over 136 descriptors = 58.1 us), identical to three digits at
 * N=2 and N=16 because it never touched the card's work in the first place.
 *
 * **SO THE PARAMETER CACHE IS 2.08 OF 58 us — 3.6%** — and the lever WAS the await. With the
 * await at ONE access, the 25 posted writes are what is left: `PN_TT_ISSUE=word|wide` takes
 * them to seven by moving the two 16-byte-aligned writes onto the BLOCK16 accessor. That knob,
 * and the WC/UC store-ordering seam it opens, are documented in `pn_tt_issue.h`.
 *
 * ---------------------------------------------------------------------------
 * THREE SHAPES, ONE KNOB, ONE BINARY — WHICH IS THE POINT OF THE KNOB
 * ---------------------------------------------------------------------------
 *   `control`  `read_field(COUNT)` + `read_control()`     1 + 16 = 17   what shipped
 *   `window`   `read_field(COUNT)` + `read_completion()`  1 +  5 =  6
 *   `line`     `read_completion_line()` on BLOCK16             1        DEFAULT
 *
 * `control` is kept ALIVE rather than deleted so the ladder's control can be re-taken on the
 * SAME BINARY in the same hour. A control measured on yesterday's build is how this program
 * produced a 19% prediction that measured 3.4%.
 *
 * `window` gives up NOTHING. The completion window is words 5..9 and all four of rule 1's
 * agreements are inside it — COUNT 5, TOKEN 6, STATE 7, ERROR 9 — on the same accessor, from
 * the same L1, checked by the same function. `tests/test_queue.c` reads both and asserts the
 * four words are equal, so this is a fact about the code and not a claim in a header. That is
 * why it is the default with no silicon arm behind it: there is no state in which it can
 * disagree with `control`, and one env var puts `control` back with no rebuild.
 *
 * ---------------------------------------------------------------------------
 * `line` IS NOW THE DEFAULT, AND WHAT IT GIVES UP IS ZERO ON *THIS* PATH — BY CONSTRUCTION
 * ---------------------------------------------------------------------------
 * `line` does not read `error` (word 9) on the success side. That was why it stayed opt-in.
 * The exposure is precisely: a descriptor whose COUNT, TOKEN and STATE all agree AND whose
 * `error` word is set. `resident_queue_protocol.h` rule 5 — "ABANDONED IS NOT COMPLETED: a
 * descriptor whose wait exited on stop or on its bound advances `tail` and does NOT advance
 * `completed_count`" — makes every kernel error path leave the count SHORT, where the poll
 * never succeeds and the 128th-poll error read names it. The protocol raises exactly ONE
 * fault after completion has been published: `ftq_group_post()`'s refused inline write, and
 * that function returns immediately when `ctx->group_address == 0`.
 *
 * **NO RESIDENCY ON THIS PATH ARMS THE GROUP POST.** `pn_tt_residency_start()` builds its
 * kernel binding from one `memset` and sets three fields, none of them `group_address`;
 * `pn_tt_matmul_group.c` is the only file in the tree that sets one, and the matmul's await
 * never comes through here — it uses `pn_tt_group_await()`. That is not left as a comment:
 * `residency->posts_group` is READ OFF the binding that was actually built, and a residency
 * that ever armed the post FALLS BACK to the window, loudly. `PN_TT_RED=residency_posts_group`
 * is the arm that proves the guard can fire, because nothing in the tree can make it.
 *
 * The remaining failure modes are all FAIL-SAFE rather than fail-silent. A stale BLOCK16 line
 * can only show an OLD count, which spins one more iteration; the count and the token come
 * out of the SAME 16 bytes of the SAME transaction, so there is no window between reading a
 * count and reading the token it entitled you to — strictly better ordered than
 * poll-then-check. And a refusal still NAMES `error`: the failure path re-reads the full
 * window before printing. The matmul group has run this same read as its default since
 * 2026-08-20 over 6.49 million poll reads in one run, byte-identical.
 *
 * `window` remains one env var away and is the shape to fall back to if a family ever does
 * arm the group post.
 */
#define PN_TT_RESIDENCY_AWAIT_ENV "PN_TT_RESIDENCY_AWAIT"

/** The knob, read ONCE. A `getenv` per descriptor is 136 a token on this family alone.
 *  An unrecognised value is REFUSED to the default and says so, rather than silently
 *  selecting one — a typo here would attribute a whole ladder to the wrong shape. */
static pn_tt_await_shape residency_await_shape(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv(PN_TT_RESIDENCY_AWAIT_ENV);

        cached = (int)PN_TT_AWAIT_LINE;
        if (value && *value) {
            if (strcmp(value, "control") == 0)
                cached = (int)PN_TT_AWAIT_CONTROL;
            else if (strcmp(value, "window") == 0)
                cached = (int)PN_TT_AWAIT_WINDOW;
            else if (strcmp(value, "line") == 0)
                cached = (int)PN_TT_AWAIT_LINE;
            else
                fprintf(stderr, "pn_tt: %s=%s is not control|window|line — using the default "
                                "(line)\n", PN_TT_RESIDENCY_AWAIT_ENV, value);
        }
    }
    return (pn_tt_await_shape)cached;
}

/**
 * RED PROOF ONLY. `PN_TT_RED=residency_await_early` asks the four-agreement CHECK about one
 * descriptor MORE than was issued, while leaving the POLL alone.
 *
 * It exists because a ladder over three completion shapes is only evidence if the check on
 * each of them is live. A shape that would be green with no check at all is measuring the
 * card's timing, not the protocol — and this backend has already caught twelve vacuous-gate
 * classes. The wait still ends when the card really is finished, so the arm is bounded and
 * the ring is untouched; only the verdict is under test. Every shape must REFUSE, on the
 * first descriptor, naming `completed_count`.
 */
static bool residency_red_await_early(void)
{
    static int cached = -1;

    if (cached < 0) {
        /* SELECTS, NOT CONTAINS. `pn_tt_name_list_contains()` answers TRUE for an UNSET list
         * — it is a filter, "unset means every family" — so arming a red arm with it armed
         * the arm on EVERY run and refused every descriptor of every family at token 0.
         * Measured on silicon 2026-08-25; pn_tt_name_list.h carries the account. */
        cached = pn_tt_name_list_selects(getenv("PN_TT_RED"), "residency_await_early") ? 1 : 0;
    }
    return cached != 0;
}

/**
 * RED PROOF ONLY. `PN_TT_RED=residency_posts_group` claims this residency arms the group
 * post, so the LINE shape's exclusivity check must fire and fall back to the window.
 *
 * The check it arms is the whole of the `line` argument turned into code, and a check that
 * cannot fire is the thirteenth vacuous-gate class this backend has now paid for twice.
 * `binding.group_address` is a literal zero out of ONE `memset` in ONE function, so nothing
 * in the tree can make the guard fire on its own — which is exactly why it needs an arm that
 * can. SELECTS, not CONTAINS: an unset list selects nothing.
 */
static bool residency_red_posts_group(void)
{
    static int cached = -1;

    if (cached < 0)
        cached = pn_tt_name_list_selects(getenv("PN_TT_RED"), "residency_posts_group") ? 1 : 0;
    return cached != 0;
}

int pn_tt_residency_load(pn_tt_residency *residency, ft_tt_runtime *runtime,
                         const char *program_name, const char *override_root, const char *name,
                         uint32_t mask)
{
    const char *where = override_root ? override_root : "this binary's embedded program blob";

    if (!residency || !runtime || !program_name || !name)
        return -1;
    memset(residency, 0, sizeof(*residency));
    /* A HANDLE THAT HAS NEVER RUN IS TRIVIALLY RETIRED. The memset above makes it false, and
     * a caller asking `pn_tt_residency_retired()` before anything started would then refuse to
     * arm anything at all. */
    residency->retired = true;
    residency->runtime = runtime;
    snprintf(residency->name, sizeof(residency->name), "%s", name);

    /* Firmware and kernels from the SAME program, always — and one call each, with the same
     * name and the same override, so the pairing is structural. A kernel's text is linked at
     * its own firmware's `__fw_export_text_end`, so a mixed pair is a wedged core and a
     * card-level reset, not a configuration. */
    if (ft_tt_program_open_firmware(program_name, override_root, &residency->firmware) !=
        FT_TT_OK) {
        fprintf(stderr, "pn_tt: no %s firmware in %s — %s stays on the host\n", program_name,
                where, name);
        return -1;
    }
    if (ft_tt_program_open_kernels(program_name, override_root, mask, &residency->program) !=
        FT_TT_OK) {
        fprintf(stderr, "pn_tt: no %s program in %s — that op stays on the host\n", name, where);
        ft_tt_runtime_images_release(&residency->firmware);
        return -1;
    }
    residency->images_loaded = true;
    residency->processor_mask = mask;
    return 0;
}

static int residency_bind_core(pn_tt_residency *residency, uint32_t core_index)
{
    ft_tt_noc_coord core;
    int rc;

    if (!residency || !residency->images_loaded)
        return -1;
    rc = ft_tt_coord_tensix_by_index(&residency->runtime->grid, core_index,
                                     FT_TT_COORD_TRANSLATED, &core);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: Tensix index %u is not on this part (%s)\n", core_index,
                residency->name);
        return -1;
    }
    residency->core_index = core_index;
    residency->core_x = (uint16_t)core.x;
    residency->core_y = (uint16_t)core.y;

    rc = ft_tt_runtime_boot_core(&residency->runtime->noc, residency->core_x, residency->core_y,
                                 &residency->firmware);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: %s core %u (%u,%u) firmware boot failed: %s\n", residency->name,
                core_index, core.x, core.y, ft_tt_error_string(rc));
        return -1;
    }
    residency->core_booted = true;
    return 0;
}

uint32_t pn_tt_residency_queue_capacity(void)
{
    static uint32_t cached;

    if (cached == 0u) {
        const char *value = getenv("PN_TT_QUEUE_CAPACITY");
        const long parsed = (value && *value) ? strtol(value, NULL, 10) : 0;

        cached = (parsed >= 2 && parsed <= (long)PN_TT_RESIDENCY_QUEUE_CAPACITY &&
                  (parsed & (parsed - 1)) == 0)
                     ? (uint32_t)parsed
                     : PN_TT_RESIDENCY_QUEUE_CAPACITY_DEFAULT;
    }
    return cached;
}

static int residency_start(pn_tt_residency *residency)
{
    static const unsigned processors[5] = { FT_TT_PROC_DM0, FT_TT_PROC_DM1, FT_TT_PROC_MATH0,
                                            FT_TT_PROC_MATH1, FT_TT_PROC_MATH2 };
    uint32_t args[5][FT_TT_QUEUE_KERNEL_ARG_COUNT];
    ft_tt_op_args blocks[5];
    unsigned index;
    int rc;

    if (!residency || !residency->core_booted || residency->running)
        return -1;

    rc = ft_tt_queue_init(&residency->runtime->noc, residency->core_x, residency->core_y,
                          FT_TT_OP_QUEUE_BASE, pn_tt_residency_queue_capacity(),
                          ft_tt_queue_make_nonce(), &residency->queue);
    /* ASKED, NOT ASSUMED. `FT_TT_OP_QUEUE_BASE` is 16-byte aligned today, so `line` is
     * reachable today — but a queue placed on a word boundary is legal to
     * `ft_tt_queue_init()`, works through the WORD32 accessor, and would make every BLOCK16
     * read fail. On this path that failure arrives as a completion that never completes: a
     * retire-and-re-arm storm, not a diagnosable refusal. So the shape FALLS BACK loudly
     * rather than arming something the accessor cannot reach. */
    if (rc == FT_TT_OK) {
        residency->await_shape = residency_await_shape();
        if (residency->await_shape == PN_TT_AWAIT_LINE &&
            !ft_tt_queue_completion_line_is_addressable(&residency->queue)) {
            fprintf(stderr, "pn_tt: %s control block at 0x%08x is not 16-byte aligned — the "
                            "completion LINE is unreachable; falling back to the window\n",
                    residency->name, residency->queue.control_address);
            residency->await_shape = PN_TT_AWAIT_WINDOW;
        }
        /* THE SAME QUESTION IN THE WRITE DIRECTION, asked once here for the same reason. A
         * demotion, not a refusal: `word` reaches any placement the queue is legal at. */
        residency->issue_shape = pn_tt_issue_shape_for(residency->runtime, &residency->queue);
        if (residency->issue_shape != pn_tt_issue_shape_selected())
            fprintf(stderr, "pn_tt: %s control block at 0x%08x cannot carry the WIDE issue "
                            "shape — falling back to word\n",
                    residency->name, residency->queue.control_address);
    }
    if (rc == FT_TT_OK)
        rc = ft_tt_op_program_load(&residency->runtime->noc, residency->core_x,
                                   residency->core_y, &residency->program.program, NULL,
                                   &residency->config);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: %s could not be staged: %s\n", residency->name,
                ft_tt_error_string(rc));
        return -1;
    }
    residency->config.local_cb_mask = residency->cb_mask;
    residency->config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    /* The reader is on BRISC and was built with NOC_INDEX=0; the firmware re-reads this
     * from the launch message at run time, so the two must agree. */
    residency->config.brisc_noc_id = 0u;

    for (index = 0; index < 5u; index++) {
        ft_tt_queue_kernel_binding binding;

        memset(&binding, 0, sizeof(binding));
        binding.idle_spin_limit = PN_TT_RESIDENCY_IDLE_SPINS;
        binding.cb_wait_spin_limit = PN_TT_RESIDENCY_CB_WAIT_SPINS;
        /* Per-processor status. "The triad is stuck" is not a diagnosis; "NCRISC is in
         * CB_WAIT_FRONT at descriptor 12" is, and only these slots can say it. */
        binding.status_address =
            FT_TT_QUEUE_STATUS_ADDRESS(residency->queue.control_address, processors[index]);
        /* THE GROUP BINDING, AND IT IS NO LONGER A LITERAL ZERO. Everything the `line`
         * shape's exclusivity argument says about this stays true — the reading below is
         * still taken off the binding that was actually built, and a residency that set one
         * still loses the line. What has changed is that a caller can now set it, which is
         * the whole of the split's completion lever. */
        binding.group_x = residency->group_x;
        binding.group_y = residency->group_y;
        binding.group_address = residency->group_address;
        binding.group_member = residency->group_member;
        rc = ft_tt_queue_kernel_args(&residency->queue, &binding, args[index]);
        if (rc != FT_TT_OK)
            return -1;
        /* READ OFF THE BINDING THAT WAS ACTUALLY BUILT, not off a comment. This is the ONE
         * fact the `line` shape's whole argument rests on — see the guard below. */
        if (binding.group_address != 0u)
            residency->posts_group = true;
        blocks[index].processor = processors[index];
        blocks[index].values = args[index];
        blocks[index].count = FT_TT_QUEUE_KERNEL_ARG_COUNT;
    }
    if (residency_red_posts_group()) {
        fprintf(stderr, "pn_tt: RED PROOF residency_posts_group — %s is claimed to arm the "
                        "group post, so the LINE completion shape must decline\n",
                residency->name);
        residency->posts_group = true;
    }
    /* THE ONE STATE THE LINE CANNOT SEE, MADE UNREACHABLE BY CONSTRUCTION RATHER THAN BY
     * DISCIPLINE — the same move `pn_tt_matmul_group.c` makes for `PN_TT_FAST_COMPLETION`.
     *
     * `line` reads words 4..7 and does not read `error` (word 9) on the success side. Rule 5
     * of `resident_queue_protocol.h` is that an ABANDONED descriptor advances `tail` and does
     * NOT advance `completed_count`, so every kernel error path leaves the count short, the
     * poll never succeeds, and the 128th-poll error read names it. The protocol has EXACTLY
     * ONE fault raised after completion is published: `ftq_group_post()`'s refused inline
     * write, which returns immediately when `ctx->group_address == 0`. So a residency that
     * never arms the group post has nothing to give up — and the loop above is where that
     * stops being an argument and becomes a reading. */
    if (residency->await_shape == PN_TT_AWAIT_LINE && residency->posts_group) {
        fprintf(stderr, "pn_tt: %s arms the group post, whose refusal is the one fault raised "
                        "AFTER completion is published — the completion LINE cannot see it; "
                        "falling back to the window\n", residency->name);
        residency->await_shape = PN_TT_AWAIT_WINDOW;
    }

    rc = ft_tt_op_stage(&residency->runtime->noc, residency->core_x, residency->core_y,
                        &residency->config, residency->cb_buffer, sizeof(residency->cb_buffer),
                        blocks, 5u);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_send_go(&residency->runtime->noc, residency->core_x,
                                  residency->core_y);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: %s did not kick: %s\n", residency->name,
                ft_tt_error_string(rc));
        return -1;
    }
    residency->running = true;
    residency->retired = false;
    residency->last_alive_seconds = pn_tt_residency_now();
    return 0;
}

static void residency_stop(pn_tt_residency *residency)
{
    /* THE MASK THE PROGRAM WAS LOADED WITH, not every processor of a triad.
     *
     * The memory family (EMBEDDING / KV_APPEND) has NO compute thread and enables DM0|DM1
     * only, so its three TRISCs never run this program and never write EXIT into their status
     * slots. Waiting on all five therefore spun the full 20,000 polls and then printed
     * `still running: 0x1c` — which is the exact shape of the wedge signature this project
     * has paid cores for, on a core that was perfectly healthy. MEASURED on 2026-08-15: the
     * KV-append residency reported it on every clean shutdown. A false alarm that looks like
     * the real alarm is worse than no alarm, because it teaches you to ignore the real one. */
    const uint32_t mask = residency ? residency->processor_mask : PN_TT_RESIDENCY_TRIAD_MASK;
    uint32_t pending = mask;
    unsigned attempt;

    /* THE EARLY RETURN MUST NOT TOUCH `retired`, AND THAT COST A SECOND WEDGED CORE.
     *
     * `pn_tt_residency_release()` calls this function AGAIN, and by then `running` is already
     * false — so an early return that set `retired = true` overwrote the answer the real stop
     * had just computed. MEASURED ON SILICON 2026-08-22: the stop printed `did not retire`,
     * the release cleared the flag, the guard read `true` and armed anyway, and core 90 wedged
     * a second time with the guard nominally in place. A fresh handle is `retired` from
     * `pn_tt_residency_load()`; a started one is not; only a confirmed exit sets it back. */
    if (!residency || !residency->running || mask == 0u)
        return;
    residency->running = false;
    residency->retired = false;
    if (ft_tt_queue_request_stop(&residency->runtime->noc, &residency->queue) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: could not ask %s to stop\n", residency->name);
        return;
    }
    for (attempt = 0; attempt < PN_TT_RESIDENCY_STOP_POLLS; attempt++) {
        if (ft_tt_queue_all_processors_exited(&residency->runtime->noc, &residency->queue, mask,
                                              &pending) == FT_TT_OK) {
            residency->retired = true;
            return;
        }
    }

    /* THE GRACE, AND IT IS THERE BECAUSE 20,000 POLLS IS 60 ms AND THE THING BEING WAITED FOR
     * TAKES SECONDS.
     *
     * A REFUSED descriptor makes the reader exit and leaves the compute thread to spin out its
     * whole 400,000,000-iteration bound — that is the family's design, not a fault — so a stop
     * issued straight after one finds a processor still running for as long as that bound
     * takes. The tight loop above cannot cover it: it is a NoC read a poll, ~3 us, so it gives
     * up after 60 ms. What used to happen next is that the caller booted firmware onto that
     * core, and MEASURED ON SILICON 2026-08-21 that is a WEDGE — core 90 came back
     * `instrn_buf_status = 0x66` and needed `ft_tt_core recover --rung 3` with sudo.
     *
     * A millisecond of sleep between polls, up to the grace, costs nothing on the path where
     * the core retires immediately (it never runs) and turns the other path from a wedge into
     * a wait. If the grace expires the residency is left NOT RETIRED and says so, and
     * `pn_tt_residency_retired()` is what a caller asks before it boots anything. */
    /* THE GRACE IS WALL TIME, not a count of naps: a signal (EINTR, e.g. the PN_PROFILE
     * sampler's SIGPROF) cuts a nap short, and a counted loop would give up early. */
    {
        struct timespec nap = { 0, 1000000L };
        const double grace_started = pn_tt_residency_now();
        unsigned millisecond;

        for (millisecond = 0;
             (pn_tt_residency_now() - grace_started) * 1000.0 < (double)PN_TT_RESIDENCY_STOP_GRACE_MS;
             millisecond++) {
            (void)nanosleep(&nap, NULL);
            if (ft_tt_queue_all_processors_exited(&residency->runtime->noc, &residency->queue,
                                                  mask, &pending) == FT_TT_OK) {
                residency->retired = true;
                if (millisecond > 100u)
                    fprintf(stderr, "pn_tt: %s retired after %u ms of grace — a refused "
                                    "descriptor leaves the compute thread spinning out its "
                                    "bound\n", residency->name, millisecond);
                return;
            }
        }
    }
    fprintf(stderr, "pn_tt: %s did not retire; processors still running: 0x%x "
                    "(the core is left alone, NOT parked and NOT reset). NOTHING MAY BOOT "
                    "FIRMWARE ON IT — that is the wedge.\n",
            residency->name, pending);
}

bool pn_tt_residency_retired(const pn_tt_residency *residency)
{
    return residency && residency->retired;
}

void pn_tt_residency_release(pn_tt_residency *residency)
{
    if (!residency)
        return;
    pn_tt_residency_stop(residency);
    if (residency->images_loaded) {
        ft_tt_runtime_images_release(&residency->program);
        ft_tt_runtime_images_release(&residency->firmware);
        residency->images_loaded = false;
    }
}

/**
 * Wait for `expected` descriptors to be reported EXECUTED — or for the kernel to say it
 * refused one, which is a different thing and must not be waited out.
 *
 * On the LINE shape the poll IS the check: `out_fields` comes back holding words 4..7 of the
 * iteration that saw the count, so the caller has the count and the token out of ONE
 * transaction with no window between them. On the slow path `out_fields` is untouched and
 * the caller reads the completion window itself.
 *
 * @param out_fields  filled on the LINE shape only; must not be NULL
 * @return FT_TT_OK; FT_TT_ERROR if the kernel reported an error; FT_TT_EIO on timeout
 */
static int pn_tt_residency_wait(pn_tt_residency *residency, uint32_t expected,
                                uint32_t timeout_ms,
                                uint32_t out_fields[FT_TT_QUEUE_FIELD_COUNT])
{
    const double deadline = pn_tt_residency_now() + (double)timeout_ms / 1000.0;
    unsigned poll = 0;

    if (residency->await_shape == PN_TT_AWAIT_LINE) {
        for (;;) {
            int rc = ft_tt_queue_read_completion_line(&residency->runtime->bulk_l1,
                                                      &residency->queue, out_fields);

            residency->await_poll_transactions += 1u;
            if (rc != FT_TT_OK)
                return rc;
            if (out_fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] >= expected)
                return FT_TT_OK;
            if ((++poll & 127u) == 0u) {
                uint32_t error = 0;

                /* `error` is word 9 and deliberately outside the line. This is where a
                 * REFUSAL is found — the count it is waiting for will never arrive. */
                rc = ft_tt_queue_read_field(&residency->runtime->noc, &residency->queue,
                                            FT_TT_QUEUE_FIELD_ERROR, &error);
                residency->await_poll_transactions += 1u;
                if (rc != FT_TT_OK)
                    return rc;
                if (error != FT_TT_QUEUE_KERR_NONE)
                    return FT_TT_ERROR;
                if (pn_tt_residency_now() > deadline)
                    return FT_TT_EIO;
            }
        }
    }

    for (;;) {
        uint32_t value = 0;
        int rc = ft_tt_queue_read_field(&residency->runtime->noc, &residency->queue,
                                        FT_TT_QUEUE_FIELD_COMPLETED_COUNT, &value);

        residency->await_poll_transactions += 1u;
        if (rc != FT_TT_OK)
            return rc;
        if (value >= expected)
            return FT_TT_OK;
        if ((++poll & 127u) == 0u) {
            uint32_t error = 0;

            rc = ft_tt_queue_read_field(&residency->runtime->noc, &residency->queue,
                                        FT_TT_QUEUE_FIELD_ERROR, &error);
            residency->await_poll_transactions += 1u;
            if (rc != FT_TT_OK)
                return rc;
            if (error != FT_TT_QUEUE_KERR_NONE)
                return FT_TT_ERROR;
            if (pn_tt_residency_now() > deadline)
                return FT_TT_EIO;
        }
    }
}

/**
 * `PN_TT_ONCARD_BATCH`, read once and remembered.
 *
 * THE SAME INSTRUMENT AS `PN_TT_GDN_BATCH`, aimed at every OTHER family — the reduce, the
 * SwiGLU, the residual add, the top-N, the attention, the per-head norms. The spec prices
 * those at "162 descriptors a token, 247 µs each, for ops whose Tensix work is microseconds"
 * and concludes the token is round-trip-shaped. That is an inference from a number, not a
 * measurement of one, and this is the measurement: if 247 µs is a PCIe round trip then K
 * copies of a descriptor cost what one costs, and if it is the card's work they cost K times.
 *
 * A stopwatch, not a fast path — an in-place residual add applied K times is not the residual
 * add the model wanted. The arm's timings are evidence; its text is not.
 */
static uint32_t oncard_batch(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv("PN_TT_ONCARD_BATCH");
        long parsed = (value && *value) ? strtol(value, NULL, 10) : 0;

        cached = (parsed > 0 && parsed <= (long)pn_tt_residency_queue_capacity()) ? (int)parsed : 1;
    }
    return (uint32_t)cached;
}

/**
 * Is `name` in `PN_TT_ONCARD_BATCH_FAMILY`, a comma-separated list?
 *
 * WHY THE FILTER EXISTS: batching is a stopwatch that repeats a descriptor, so it is only
 * ANSWER-PRESERVING for a family whose descriptor is idempotent. `add(a, b, out)` and
 * `silu_mul(a, b, out)` write a pure function of two operands they do not touch, so K copies
 * leave exactly what one leaves — the elementwise family and both renders can be batched with
 * the completion still comparable to the oracle. The GDN recurrence reads and writes its
 * carried state, and the per-head L2 norm runs IN PLACE; batching those computes a value the
 * model does not want, and a `cmp` taken over them says nothing.
 *
 * Unset means EVERY family, which is what `PN_TT_ONCARD_BATCH` did before this filter existed
 * and is the shape every recorded batch arm was taken in.
 */
static bool oncard_batch_wants(const char *name)
{
    return pn_tt_name_list_contains(getenv("PN_TT_ONCARD_BATCH_FAMILY"), name);
}

/**
 * Does `PN_TT_RED=issue_head_first` fire on THIS enqueue? ONCE per process, and only after the
 * residency's ring has wrapped exactly once.
 *
 * BOTH CONDITIONS ARE THE ARM, not caution about it. Firing on the FIRST descriptor would
 * publish a head over a ZEROED ring slot, which is opcode 0 — an unknown opcode the kernel
 * refuses by name, so the arm would prove that a *malformed* descriptor is caught and say
 * nothing about ordering. After one lap the slot holds a WELL-FORMED descriptor from the
 * previous lap: the count matches, the kernel reports no error, and the only disagreement left
 * is the per-index token. That is the failure a WC store overtaking a UC one would actually
 * produce, and it is the one the arm has to be able to see.
 *
 * ONCE, because a refused descriptor strands the triad and the family re-arms: firing every
 * enqueue is a refusal storm, and this backend has already paid for a wedged core.
 */
static bool residency_red_head_first_fires(const pn_tt_residency *residency)
{
    static bool fired = false;

    if (fired || !pn_tt_issue_red_head_first())
        return false;
    if (residency->issues < residency->queue.capacity)
        return false;
    fired = true;
    return true;
}

int pn_tt_residency_submit(pn_tt_residency *residency, uint32_t opcode,
                           const uint32_t *parameters, uint32_t timeout_ms)
{
    uint32_t repeat = 1u;

    if (residency && oncard_batch_wants(residency->name))
        repeat = oncard_batch();
    return pn_tt_residency_submit_repeat(residency, opcode, parameters, repeat, timeout_ms);
}

void pn_tt_residency_join_group(pn_tt_residency *residency, uint16_t x, uint16_t y,
                                uint32_t address, uint32_t member)
{
    if (!residency)
        return;
    /* REFUSED AFTER START rather than silently ignored: the binding is already in the
     * kernel's runtime args by then, so a late call would set a host field that describes a
     * card that is not doing it — which is the exact shape of the stale-runtime-arg defect
     * the cooperative role left behind. */
    if (residency->running) {
        fprintf(stderr, "pn_tt: %s is already running — a group binding set now would not "
                        "reach the kernel. IGNORED.\n", residency->name);
        return;
    }
    residency->group_x = x;
    residency->group_y = y;
    residency->group_address = address;
    residency->group_member = member;
}

/**
 * THE IDLE RETIRE: never send a descriptor into a triad that may be half gone.
 *
 * Every processor of a resident triad counts its OWN idle polls to `idle_spin_limit` and exits
 * alone, so a family that sits idle long enough — `gdn` and `topk` through a whole batched
 * prefill, every family in a `serve` between requests — passes through a window in which some
 * of its processors have exited and the rest are still polling. A descriptor that lands in that
 * window is fetched by the survivors, which block on circular buffers whose producer is gone;
 * the stop that follows finds a compute thread mid-descriptor, `did not retire`, and the core is
 * wedged (`instrn_buf_status = 0x66`, `recover --rung 3`). MEASURED: the first descriptor of
 * `gdn`, its members and `topk` after EVERY batched prefill was refused, and 2 of 3 runs at 16
 * matmul members wedged core 90 this way.
 *
 * So past `PN_TT_RESIDENCY_IDLE_RETIRE_MS` of silence the residency is ASKED to stop BEFORE
 * anything is written — every processor still alive is in its idle poll, which reads the ask,
 * so the triad exits whole — and the enqueue refuses. The family's own refusal path (await
 * fails on a stopped residency, `retired` reads true, re-arm, retry) does the rest, which is
 * the path every one of these families already runs on a refusal.
 *
 * The default must sit well below the FASTEST processor's idle exit; see the spec for the
 * measurement it is set from. `0` turns the retire off (the old behaviour, kept so a ladder's
 * control is the same binary).
 *
 * @return true when the residency was retired here and the enqueue must refuse
 */
long pn_tt_residency_idle_retire_limit_ms(void)
{
    static long limit_ms = -1;

    if (limit_ms < 0) {
        const char *value = getenv(PN_TT_RESIDENCY_IDLE_RETIRE_ENV);
        char *end = NULL;

        limit_ms = PN_TT_RESIDENCY_IDLE_RETIRE_DEFAULT_MS;
        if (value && *value) {
            const long parsed = strtol(value, &end, 10);

            if (end && *end == '\0' && parsed >= 0)
                limit_ms = parsed;
            else
                fprintf(stderr, "pn_tt: %s=%s is not a millisecond count — using %ld\n",
                        PN_TT_RESIDENCY_IDLE_RETIRE_ENV, value, limit_ms);
        }
    }
    return limit_ms;
}

/** DIAGNOSIS (`PN_TT_IDLE_RETIRE_ONLY=<name>`): only the residency of that name (or "matmul"
 *  for the decode group, "none" for nobody) is retired for idleness. Unsafe past the card's own
 *  idle exit (~8 s) — for short-gap bisects only. */
bool pn_tt_idle_retire_allowed(const char *name)
{
    const char *only = getenv("PN_TT_IDLE_RETIRE_ONLY");

    return !only || !*only || (name && strcmp(only, name) == 0);
}

static bool residency_idle_retire(pn_tt_residency *residency)
{
    const long limit_ms = pn_tt_residency_idle_retire_limit_ms();
    double idle;

    if (limit_ms == 0 || !residency || !residency->running || residency->last_alive_seconds <= 0.0 ||
        !pn_tt_idle_retire_allowed(residency->name))
        return false;
    idle = pn_tt_residency_now() - residency->last_alive_seconds;
    if (idle * 1000.0 < (double)limit_ms)
        return false;
    fprintf(stderr, "pn_tt: %s idle %.1f s (limit %ld ms) — retiring it by ASKING before a "
                    "descriptor can reach a half-exited triad\n",
            residency->name, idle, limit_ms);
    pn_tt_residency_stop(residency);
    residency->idle_retires++;
    return true;
}

int pn_tt_residency_write_param_extension(pn_tt_residency *residency, uint32_t offset,
                                          const uint32_t *words)
{
    const ft_tt_noc_access *payload;
    int rc;

    if (!residency || !residency->running || !words || offset == 0u)
        return -1;
    payload = pn_tt_issue_access(residency->runtime, residency->issue_shape);
    rc = ft_tt_queue_write_op_parameters(payload, &residency->queue,
                                         PN_TT_RESIDENCY_PARAM_SLOT + residency->param_slot +
                                             offset,
                                         words, NULL);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: %s parameter extension refused: %s\n", residency->name,
                ft_tt_error_string(rc));
        return -1;
    }
    return 0;
}

int pn_tt_residency_enqueue(pn_tt_residency *residency, uint32_t opcode,
                            const uint32_t *parameters, uint32_t repeat)
{
    return pn_tt_residency_enqueue_group(residency, opcode, parameters, repeat, 0u, NULL);
}

int pn_tt_residency_enqueue_group(pn_tt_residency *residency, uint32_t opcode,
                                  const uint32_t *parameters, uint32_t repeat,
                                  uint32_t sequence, uint32_t *out_token)
{
    ft_tt_queue_descriptor descriptor[PN_TT_RESIDENCY_QUEUE_CAPACITY];
    const ft_tt_noc_access *payload;
    uint32_t address = 0u;
    uint32_t slot;
    uint32_t index;
    double at;
    int rc;

    if (!residency || !residency->running || !parameters || repeat == 0u)
        return -1;
    if (repeat > residency->queue.capacity)
        repeat = residency->queue.capacity;
    /* PM 8: A DEFERRED RESIDENCY QUEUES SEVERAL DESCRIPTORS BEFORE ONE AWAIT. A full ring
     * drains first — ft_tt_queue_stage() does not check occupancy — and every queued
     * descriptor keeps its own parameter block until it runs (slot 0 would be overwritten by
     * the next enqueue's). A caller that picked a slot itself (`param_slot`) keeps it. */
    slot = residency->param_slot;
    if (residency->defer_await) {
        const uint32_t depth = residency->queue.capacity < PN_TT_RESIDENCY_DEFER_SLOTS
                                   ? residency->queue.capacity
                                   : PN_TT_RESIDENCY_DEFER_SLOTS;

        if (residency->unawaited + repeat > depth &&
            pn_tt_residency_settle(residency, PN_TT_RESIDENCY_DEFER_TIMEOUT_MS) != 0)
            return -1;
        if (slot == 0u)
            slot = 1u + (residency->queue.issued % PN_TT_RESIDENCY_DEFER_SLOTS);
    }

    /* THE PAYLOAD AND THE HEAD BUMP GO THROUGH DIFFERENT ACCESSORS ON THE `wide` SHAPE, and
     * the head bump is on `noc` on BOTH — a four-byte write at a word address cannot come
     * through the 16-byte L1 accessor at all (pn_tt_issue.h). */
    payload = pn_tt_issue_access(residency->runtime, residency->issue_shape);

    at = pn_tt_residency_now();
    rc = ft_tt_queue_write_op_parameters(payload, &residency->queue,
                                         PN_TT_RESIDENCY_PARAM_SLOT + slot, parameters,
                                         &address);
    residency->issue_param_seconds += pn_tt_residency_now() - at;
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: %s parameter block refused: %s\n", residency->name,
                ft_tt_error_string(rc));
        return -1;
    }

    if (residency_idle_retire(residency))
        return -1;
    memset(descriptor, 0, sizeof(descriptor));
    for (index = 0; index < repeat; index++) {
        descriptor[index].opcode = opcode;
        descriptor[index].a0 = address;
        /* `a5` = 0 is "post nothing", and `ftq_group_post()` returns on it before touching
         * the NoC. So the descriptor a caller that never asked for a group sends is byte for
         * byte the one this function has always sent. */
        descriptor[index].a5 = sequence;
    }
    /* READ BEFORE `ft_tt_queue_stage()` MOVES `issued`. The token the group check must match
     * is the one stamped on the LAST descriptor of this enqueue, which is
     * `token_at(issued + repeat - 1)` taken from the counter as it stands now. */
    if (out_token)
        *out_token = ft_tt_queue_token_at(&residency->queue,
                                          residency->queue.issued + repeat - 1u);

    if (residency_red_head_first_fires(residency)) {
        fprintf(stderr, "pn_tt: RED PROOF issue_head_first — %s publishes head %u BEFORE the "
                        "descriptor, after exactly one lap of the ring. The card will run slot "
                        "%u's PREVIOUS occupant: a legal descriptor, a matching count and no "
                        "kernel error. Only the TOKEN can catch it.\n",
                residency->name, residency->queue.head + repeat,
                residency->queue.head & (residency->queue.capacity - 1u));
        (void)ft_tt_queue_write_head(&residency->runtime->noc, &residency->queue,
                                     residency->queue.head + repeat);
        /* AND THEN WAIT, WHICH IS WHAT MAKES THE ARM DETERMINISTIC RATHER THAN A COIN FLIP.
         *
         * MEASURED: without this, the head-first arm was caught on the `wide` shape and NOT on
         * `word` — the ring write follows the head bump by ~130 ns through an already-aimed
         * window, and the kernel's poll simply did not land in that gap. A red proof whose
         * firing depends on a 130 ns race proves nothing on the run where it misses. Half a
         * millisecond is longer than any poll period on the card and shorter than any timeout
         * on the host, so the card is GUARANTEED to consume the stale slot. */
        {
            struct timespec settle = { 0, 500000L };

            (void)nanosleep(&settle, NULL);
        }
    }

    at = pn_tt_residency_now();
    rc = ft_tt_queue_stage(payload, &residency->queue, descriptor, repeat);
    residency->issue_ring_seconds += pn_tt_residency_now() - at;
    if (rc == FT_TT_OK) {
        at = pn_tt_residency_now();
        rc = ft_tt_queue_publish(&residency->runtime->noc, &residency->queue);
        residency->issue_head_seconds += pn_tt_residency_now() - at;
    }
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: %s enqueue failed: %s\n", residency->name,
                ft_tt_error_string(rc));
        return -1;
    }
    residency->issues++;
    residency->unawaited += repeat;
    residency->issue_accesses += pn_tt_issue_accesses(
        residency->issue_shape,
        FT_TT_QUEUE_PARAM_WORDS * 4u + repeat * FT_TT_QUEUE_DESCRIPTOR_BYTES);
    residency->ops += repeat;
    return 0;
}

int pn_tt_residency_await(pn_tt_residency *residency, uint32_t timeout_ms)
{
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    const char *reason = NULL;
    uint32_t expected, checked;
    int rc;

    if (!residency || !residency->running || timeout_ms == 0u)
        return -1;
    if (residency->defer_await) {
        residency->await_pending = true;
        return 0;
    }
    residency->await_pending = false;

    expected = residency->queue.issued;
    /* The POLL waits for what was issued; the CHECK is what the red arm moves. */
    checked = residency_red_await_early() ? expected + 1u : expected;
    residency->awaits++;
    memset(fields, 0, sizeof(fields));
    rc = pn_tt_residency_wait(residency, expected, timeout_ms, fields);
    /* THE COMPLETION WINDOW, NOT THE CONTROL BLOCK. Words 5..9 carry all four of rule 1's
     * agreements — COUNT 5, TOKEN 6, STATE 7, ERROR 9 — so this is the same verdict on the
     * same accessor for FIVE word reads instead of sixteen. `line` already has its answer out
     * of the poll and reads nothing more; `control` is the shipped shape, kept so a ladder's
     * control can be re-taken on this binary. */
    if (rc == FT_TT_OK && residency->await_shape == PN_TT_AWAIT_WINDOW) {
        rc = ft_tt_queue_read_completion(&residency->runtime->noc, &residency->queue, fields);
        residency->await_settle_transactions += FT_TT_QUEUE_COMPLETION_FIELD_COUNT;
    } else if (rc == FT_TT_OK && residency->await_shape == PN_TT_AWAIT_CONTROL) {
        rc = ft_tt_queue_read_control(&residency->runtime->noc, &residency->queue, fields);
        residency->await_settle_transactions += FT_TT_QUEUE_FIELD_COUNT;
    }
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: %s descriptor %u did not complete: %s\n", residency->name,
                expected, ft_tt_error_string(rc));
        return -1;
    }
    /* Completion is four independent agreements, not a count: the state is not the host's
     * poison, the count is exact, the TOKEN is this run's nonce for this descriptor, and the
     * kernel reports no error. Each of them fails a different lie about work that did not
     * happen (ft_tt_queue.h rule 1). The fast path holds THREE of them and asks the check
     * that does not look at the fourth — a separate function, so a line buffer whose `error`
     * word is a memset zero cannot pass the four-agreement check as KERR_NONE. */
    if ((residency->await_shape == PN_TT_AWAIT_LINE
             ? ft_tt_queue_check_completion_line(&residency->queue, fields, checked, &reason)
             : ft_tt_queue_check_completion(&residency->queue, fields, checked, &reason)) !=
        FT_TT_OK) {
        /* The FULL window on the way to a failure, and it is the only place the dropped
         * assertion has to be recovered: a caller reporting a refusal must be able to NAME
         * `error`, and five more reads cost nothing on a path that is already returning -1.
         * The verdict is NOT re-taken from it — `reason` stands. */
        if (residency->await_shape == PN_TT_AWAIT_LINE)
            (void)ft_tt_queue_read_completion(&residency->runtime->noc, &residency->queue,
                                              fields);
        fprintf(stderr, "pn_tt: %s completion refused: %s (state %s, error %s)\n",
                residency->name, reason ? reason : "?",
                ft_tt_queue_state_string(fields[FT_TT_QUEUE_FIELD_STATE]),
                ft_tt_queue_kernel_error_string(fields[FT_TT_QUEUE_FIELD_ERROR]));
        return -1;
    }
    residency->last_alive_seconds = pn_tt_residency_now();
    residency->unawaited = 0u;
    return 0;
}

/**
 * RED PROOF ONLY. `PN_TT_RED=residency_group_early` asks the GROUP check about one sequence
 * MORE than was issued, leaving the poll alone — the group-block twin of
 * `residency_await_early`, and there for the same reason. A ladder that swaps N per-member
 * awaits for one group read is only evidence if the group read's verdict is live; a group
 * await that returned OK on any slot array would measure the card's timing and nothing else.
 * Every member must be refused, naming the slot that did not reach the sequence.
 */
static bool residency_red_group_early(void)
{
    static int cached = -1;

    if (cached < 0)
        cached = pn_tt_name_list_selects(getenv("PN_TT_RED"), "residency_group_early") ? 1 : 0;
    return cached != 0;
}

int pn_tt_residency_settle(pn_tt_residency *residency, uint32_t timeout_ms)
{
    bool deferring;
    int rc;

    if (!residency || !residency->await_pending)
        return 0;
    deferring = residency->defer_await;
    residency->defer_await = false;
    rc = pn_tt_residency_await(residency, timeout_ms);
    residency->defer_await = deferring;
    return rc;
}

int pn_tt_residency_await_group(pn_tt_residency *const *set, const uint32_t *token,
                                uint32_t members, uint32_t sequence, uint32_t timeout_ms)
{
    const double deadline = pn_tt_residency_now() + (double)timeout_ms / 1000.0;
    ft_tt_queue_group_slot slots[FT_TT_QUEUE_GROUP_MAX_MEMBERS];
    const uint32_t want = sequence + (residency_red_group_early() ? 1u : 0u);
    uint32_t expected;
    uint32_t index;

    if (!set || !token || members == 0u || members > FT_TT_QUEUE_GROUP_MAX_MEMBERS ||
        sequence == 0u || timeout_ms == 0u)
        return -1;
    for (index = 0; index < members; index++)
        if (!set[index] || !set[index]->running)
            return -1;
    if (set[0]->group_address == 0u) {
        fprintf(stderr, "pn_tt: %s owns no group block — a group await here would poll an "
                        "address nothing posts to\n", set[0]->name);
        return -1;
    }
    expected = set[0]->queue.issued;

    /* STAGE 1 — MEMBER 0's OWN COUNT, one four-byte read, exactly the single-core poll. It is
     * the cheapest thing that can say "the set is probably done": member 0 is the slice this
     * family gives the largest share of the work to, and the record has it at 113.1 us a
     * descriptor against 3.3-10.6 for members 1..7. */
    for (;;) {
        uint32_t value = 0;

        set[0]->awaits++;
        set[0]->await_poll_transactions += 1u;
        if (ft_tt_queue_read_field(&set[0]->runtime->noc, &set[0]->queue,
                                   FT_TT_QUEUE_FIELD_COMPLETED_COUNT, &value) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: %s completion count is unreadable\n", set[0]->name);
            return -1;
        }
        if (value >= expected)
            break;
        if (pn_tt_residency_now() > deadline) {
            fprintf(stderr, "pn_tt: %s did not reach %u descriptors in %u ms\n", set[0]->name,
                    expected, timeout_ms);
            return -1;
        }
    }

    /* STAGE 2 — THE WHOLE SLOT ARRAY IN ONE BLOCK16 TRANSACTION, which is the entire point:
     * `ceil(8N/16)` 16-byte units for N members, against N independent poll loops. Two
     * assertions per member, each failing a different lie — the SEQUENCE says how far that
     * member got, the TOKEN says it ran THIS descriptor and not a previous occupant of the
     * slot. */
    for (;;) {
        uint32_t bad = members;

        if (ft_tt_queue_group_read_slots(&set[0]->runtime->bulk_l1, set[0]->queue.x,
                                         set[0]->queue.y, set[0]->group_address, members,
                                         slots) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: the %s group completion block did not read back\n",
                    set[0]->name);
            return -1;
        }
        set[0]->await_settle_transactions += ((uint64_t)members * 8u + 15u) / 16u;
        for (index = 0; index < members; index++)
            if (slots[index].sequence < want || slots[index].token != token[index])
                break;
        if (index == members) {
            const double now = pn_tt_residency_now();

            for (index = 0; index < members; index++)
                set[index]->last_alive_seconds = now;
            return 0;
        }
        bad = index;
        if (pn_tt_residency_now() > deadline) {
            fprintf(stderr, "pn_tt: the %s group did not reach sequence %u in %u ms — member "
                            "%u posted sequence %u token 0x%08x, wanted %u / 0x%08x\n",
                    set[0]->name, want, timeout_ms, bad, slots[bad].sequence, slots[bad].token,
                    want, token[bad]);
            return -1;
        }
    }
}

int pn_tt_residency_submit_repeat(pn_tt_residency *residency, uint32_t opcode,
                                  const uint32_t *parameters, uint32_t repeat,
                                  uint32_t timeout_ms)
{
    double started;
    int rc;

    if (!residency)
        return -1;
    started = pn_tt_residency_now();
    rc = pn_tt_residency_enqueue(residency, opcode, parameters, repeat);
    if (rc != 0)
        return -1;
    rc = pn_tt_residency_await(residency, timeout_ms);
    if (rc != 0)
        return -1;
    residency->seconds += pn_tt_residency_now() - started;
    return 0;
}

/* ================================================================
 * LIFECYCLE CLOCK (PM 5): wall seconds spent booting, starting and stopping residencies —
 * the idle retire + re-arm cost a request pays, which no op counter sees.
 * ================================================================ */

static double g_lifecycle_seconds;

int pn_tt_residency_bind_core(pn_tt_residency *residency, uint32_t core_index)
{
    const double started = pn_tt_residency_now();
    const int rc = residency_bind_core(residency, core_index);

    g_lifecycle_seconds += pn_tt_residency_now() - started;
    return rc;
}

static void residency_register(pn_tt_residency *residency);
static void residency_unregister(const pn_tt_residency *residency);

int pn_tt_residency_start(pn_tt_residency *residency)
{
    const double started = pn_tt_residency_now();
    const int rc = residency_start(residency);

    g_lifecycle_seconds += pn_tt_residency_now() - started;
    if (rc == 0) {
        residency->unawaited = 0u;
        residency_register(residency);
    }
    return rc;
}

void pn_tt_residency_stop(pn_tt_residency *residency)
{
    const double started = pn_tt_residency_now();

    residency_stop(residency);
    g_lifecycle_seconds += pn_tt_residency_now() - started;
    if (residency && !residency->running)
        residency_unregister(residency);
}

double pn_tt_residency_lifecycle_seconds(void)
{
    return g_lifecycle_seconds;
}

void pn_tt_residency_lifecycle_add(double seconds)
{
    g_lifecycle_seconds += seconds;
}

/* ================================================================
 * KEEPALIVE (PM 5) — the live-residency registry and the NOP sweep
 * ================================================================ */

#define PN_TT_RESIDENCY_LIVE_MAX 256u
static pn_tt_residency *g_live[PN_TT_RESIDENCY_LIVE_MAX];

/** Remember a residency that just started (idempotent). Full registry = not kept alive. */
static void residency_register(pn_tt_residency *residency)
{
    uint32_t index, empty = PN_TT_RESIDENCY_LIVE_MAX;

    for (index = 0; index < PN_TT_RESIDENCY_LIVE_MAX; index++) {
        if (g_live[index] == residency)
            return;
        if (!g_live[index] && empty == PN_TT_RESIDENCY_LIVE_MAX)
            empty = index;
    }
    if (empty < PN_TT_RESIDENCY_LIVE_MAX)
        g_live[empty] = residency;
}

static void residency_unregister(const pn_tt_residency *residency)
{
    uint32_t index;

    for (index = 0; index < PN_TT_RESIDENCY_LIVE_MAX; index++)
        if (g_live[index] == residency)
            g_live[index] = NULL;
}

static bool residency_wants_keepalive(const pn_tt_residency *residency, double now,
                                      double min_idle_seconds)
{
    const double limit = (double)pn_tt_residency_idle_retire_limit_ms() / 1000.0;
    double idle;

    if (!residency->running || residency->unawaited != 0u || residency->last_alive_seconds <= 0.0)
        return false;
    idle = now - residency->last_alive_seconds;
    /* Past the limit the next enqueue retires it anyway: leave that to the family's own path. */
    return idle >= min_idle_seconds && (limit <= 0.0 || idle < limit - 0.5);
}

int pn_tt_residency_keepalive_all(double min_idle_seconds)
{
    static const uint32_t zeros[FT_TT_QUEUE_PARAM_WORDS];
    bool sent[PN_TT_RESIDENCY_LIVE_MAX];
    const double now = pn_tt_residency_now();
    uint32_t index;
    int count = 0, failed = 0;

    for (index = 0; index < PN_TT_RESIDENCY_LIVE_MAX; index++) {
        pn_tt_residency *residency = g_live[index];

        sent[index] = false;
        if (!residency || !residency_wants_keepalive(residency, now, min_idle_seconds))
            continue;
        residency->param_slot = 0u;
        if (pn_tt_residency_enqueue(residency, FT_TT_QUEUE_OP_NOP, zeros, 1u) != 0) {
            failed = 1;
            continue;
        }
        sent[index] = true;
        count++;
    }
    for (index = 0; index < PN_TT_RESIDENCY_LIVE_MAX; index++)
        if (sent[index] && pn_tt_residency_await(g_live[index], PN_TT_RESIDENCY_KEEPALIVE_TIMEOUT_MS) != 0)
            failed = 1;
    return failed ? -1 : count;
}

#endif /* USE_TT */
