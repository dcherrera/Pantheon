/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_residency.h
 * @brief ONE resident Tensix program on ONE core: read its images, boot that core with the
 *        firmware they were linked against, place its circular buffers, kick it, and from
 *        then on submit descriptors to it.
 *
 *        This is the part every resident op family does identically. What differs per
 *        family — which circular buffers, how many pages, what the sixteen parameter words
 *        mean — stays with the family's own driver in foundry/tt, and this file never looks
 *        at it: the caller hands over an already-encoded CB descriptor array and an already
 *        encoded parameter block.
 *
 * ---------------------------------------------------------------------------
 * WHY ONE CORE PER FAMILY, RATHER THAN ONE CORE DOING EVERYTHING
 * ---------------------------------------------------------------------------
 * A resident kernel's text is linked at a fixed per-processor slab, so ONE program can be
 * resident on a core at a time (foundry/tt/ft_tt_op_vector.h). A transformer block needs a
 * matmul, a reduce (the RMS norms) and an elementwise (the SwiGLU gate, the residual adds),
 * and swapping programs between them would re-upload five kernel images per switch — the
 * 36.2 µs/kB cost the resident queue exists to remove. So each family gets its own core and
 * they hand data to each other through card DRAM, which the host never touches.
 *
 * That is the whole mechanism by which the readback goes away. It is not a faster transfer;
 * it is a transfer that does not happen, because the activation never leaves the card.
 *
 * ---------------------------------------------------------------------------
 * THREE HAZARDS, ALL PAID FOR IN CORES ALREADY
 * ---------------------------------------------------------------------------
 *  1. **A kernel and its firmware are ONE build output.** A kernel's text is linked at its
 *     own firmware's `__fw_export_text_end`; booting one directory's firmware and loading
 *     another's kernels is a wedged core and a card-level reset. So this file boots from the
 *     SAME directory it loads the program from, always, with no way to say otherwise.
 *  2. **Never park a resident triad, and never reset one out of a blocked wait.** Shutdown
 *     is a written word and a poll. There is deliberately no reset escalation here.
 *  3. **A refused descriptor STRANDS the triad** — the reader sets an error and leaves its
 *     loop, `tail` never advances and the other four wait forever on work that will not
 *     arrive. So the wait polls the ERROR word as well as the count, which turns a 60 s
 *     timeout into a ~1 ms return, and the caller is told to re-arm rather than to wait.
 *
 * @status Used by the reduce (RMS norm) and elementwise (SwiGLU, residual add) residencies
 *         of the on-card forward.
 * @issues None
 * @todo None
 */

#ifndef PN_TT_RESIDENCY_H
#define PN_TT_RESIDENCY_H

#ifdef USE_TT

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "pn_tt_issue.h"

#include "tt/ft_tt_queue.h"
#include "tt/ft_tt_runtime.h"

/** Ring slots, the MAXIMUM a residency may arm (`PN_TT_QUEUE_CAPACITY`, a power of two
 *  2..64; default PN_TT_RESIDENCY_QUEUE_CAPACITY_DEFAULT). 64 x 32 B ends at 0x900, far below
 *  the parameter slots at 0x9000. The kernel indexes with the mask it is given, so the depth
 *  is a host decision. A caller that enqueues without awaiting must stay within
 *  `queue.capacity` itself — ft_tt_queue_stage() does not check occupancy (the S2c deferred
 *  GDN rows overran an 8-slot ring at 10 rows and DIFFERED silently). */
#define PN_TT_RESIDENCY_QUEUE_CAPACITY 64u
#define PN_TT_RESIDENCY_QUEUE_CAPACITY_DEFAULT 8u

/** The ring depth residencies arm with: `PN_TT_QUEUE_CAPACITY`, else the default. */
uint32_t pn_tt_residency_queue_capacity(void);

/**
 * Milliseconds a stop will WAIT, one poll at a time, after its tight loop gives up.
 *
 * The thing being waited for is a compute thread spinning out its CB-wait bound
 * (100,000,000 iterations) after a refused descriptor — that is the family's design, not a
 * fault — and it takes seconds. The tight loop covers 60 ms.
 *
 * **THE BOOT GUARD DELIBERATELY DOES NOT SHARE THIS NUMBER**, and the reason is measured:
 * a stop is issued straight after a refusal, so the CB-wait bound is what it waits out,
 * while `pn_tt_core_boot_guard.c` waits out whatever an EARLIER PROCESS left — which
 * includes the 400,000,000-iteration IDLE bound, recorded at over 40 s. One number for both
 * would be right for one of them. See PN_TT_BOOT_GUARD_WAIT_MS.
 */
#define PN_TT_RESIDENCY_STOP_GRACE_MS 20000u

/** Which parameter-block slot every descriptor points at. One is enough while the host
 *  waits for each descriptor before issuing the next. */
#define PN_TT_RESIDENCY_PARAM_SLOT 0u
/** PM 8: a deferred residency's queued descriptors rotate over parameter slots 1..this (more
 *  than any ring depth in use), and its flow-control settle waits this long. */
#define PN_TT_RESIDENCY_DEFER_SLOTS 16u
#define PN_TT_RESIDENCY_DEFER_TIMEOUT_MS 60000u

/** Runtime args a data mover may take past the queue's ten (the arg block holds 32). */
#define PN_TT_RESIDENCY_TAIL_ARGS 22u

/** Longest program name held; for logs only. */
#define PN_TT_RESIDENCY_NAME_BYTES 32u

/** One resident program, its core, and its queue. Caller-owned storage. */
/**
 * How an await reads completion — the three shapes of `PN_TT_RESIDENCY_AWAIT`, and the whole
 * per-descriptor host cost of every family except the matmul. See `pn_tt_residency.c` for the
 * transaction arithmetic and why the matmul's recorded 6.9 µs is a different number.
 */
typedef enum pn_tt_await_shape {
    PN_TT_AWAIT_CONTROL = 0, /**< poll + the whole 64-byte control block — 17 PCIe reads */
    PN_TT_AWAIT_WINDOW,      /**< poll + words 5..9 — 6 reads, the same four agreements */
    PN_TT_AWAIT_LINE         /**< ONE BLOCK16 read of words 4..7; the poll IS the check */
} pn_tt_await_shape;

typedef struct pn_tt_residency {
    ft_tt_runtime *runtime; /**< NOT owned; the device's */
    ft_tt_runtime_images firmware;
    ft_tt_runtime_images program;
    ft_tt_launch_config config;
    ft_tt_queue queue;
    /** S2c: the parameter slot the next enqueue writes (0 = the shipped single slot). A
     *  caller queueing several descriptors before one await gives each its own slot. */
    uint32_t param_slot;
    unsigned char cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
    uint64_t cb_mask;

    char name[PN_TT_RESIDENCY_NAME_BYTES];
    /** The processors this program ENABLES, as passed to pn_tt_residency_load(). The
     *  retirement wait polls exactly these: a family with no compute thread (the memory
     *  family is DM0|DM1) leaves its TRISCs in the firmware GO wait, and they never write
     *  EXIT because they never ran this program. */
    uint32_t processor_mask;
    uint32_t core_index;
    uint16_t core_x, core_y;

    bool images_loaded; /**< the five firmware and five kernel ELFs are in memory */
    bool core_booted;   /**< the core is running this program's firmware */
    bool running;       /**< a residency is live on the core RIGHT NOW */
    /** Did the LAST `pn_tt_residency_stop()` see every enabled processor say EXIT?
     *
     *  IT IS THE ONE THING THAT SEPARATES A RE-ARM FROM A WEDGE. Booting firmware onto a core
     *  whose compute thread is still spinning out its 400,000,000-iteration bound leaves
     *  `instrn_buf_status = 0x66` and needs `ft_tt_core recover --rung 3` and sudo — measured
     *  on core 90, silicon 2026-08-21, on exactly that sequence. True until a stop says
     *  otherwise. */
    bool retired;

    /** When this residency last proved itself alive: armed, or a descriptor completed.
     *  Read by `pn_tt_residency_enqueue_group()` to refuse a descriptor to a residency that
     *  may already be idling out — see `PN_TT_RESIDENCY_IDLE_RETIRE_MS`. */
    double last_alive_seconds;
    uint64_t idle_retires; /**< times the idle retire fired on this handle */
    /** Descriptors enqueued since the last successful await (PM 5 keepalive: a residency
     *  with work outstanding is never sent a NOP — its parameter slot may be unread). */
    uint32_t unawaited;
    /** PM 7 ITEM 7: while set, pn_tt_residency_await() records and returns 0; the caller owes
     *  a pn_tt_residency_settle() before anything consumes the output (streamed, that is where
     *  the puller's AWAIT lands). A refusal surfaces at the settle, without the family's
     *  replay. */
    bool defer_await;
    bool await_pending;

    /** How this residency's awaits read completion. Decided ONCE, at arm time, from the env
     *  knob AND `ft_tt_queue_completion_line_is_addressable()` — not per descriptor, because
     *  a per-descriptor decision is a per-descriptor `getenv` on the hot path, and not from
     *  the knob alone, because a queue the BLOCK16 accessor cannot reach would turn every
     *  await into a completion that never completes. */
    pn_tt_await_shape await_shape;

    /** Did the kernel binding this residency was launched with arm the GROUP POST?
     *
     *  It is a READING off the binding that was actually built, not a constant, and it is
     *  the single fact the `line` completion shape rests on. `ftq_group_post()`'s refused
     *  inline write is the ONLY fault `resident_queue_protocol.h` raises after
     *  `completed_count` has been published — every other error path abandons the descriptor
     *  and leaves the count short (rule 5), where the poll and the 128th-poll error read
     *  catch it. So a residency that never arms the post gives up nothing on the line, and
     *  one that does must keep the window. `pn_tt_matmul_group.c` makes the identical
     *  exclusion for `PN_TT_FAST_COMPLETION`; this is that argument on this path, as code. */
    bool posts_group;

    /** WHERE THIS RESIDENCY POSTS ITS GROUP COMPLETION, set by the caller BEFORE
     *  `pn_tt_residency_start()` and copied straight into the kernel binding there.
     *
     *  Zero — the value one `memset` gives every residency — is "post nothing", which is
     *  what every family on this path did until the chunkwise GDN split needed the other
     *  shape. THE POINT IS THE AWAIT, NOT THE POST: a split of N members costs N independent
     *  blocking poll loops IN SERIES, and a group block collapses them into member 0's count
     *  plus ONE BLOCK16 read of the whole slot array. That is the shape
     *  `pn_tt_matmul_group.c` has run as its default since 2026-08-20.
     *
     *  Arming it is not free on the completion side: `ftq_group_post()`'s refused inline
     *  write is the one fault raised AFTER `completed_count` is published, so
     *  `pn_tt_residency_start()` demotes the `line` shape to `window` for exactly these
     *  residencies. That demotion is why a group await has to be MEASURED against the
     *  per-member one rather than assumed to win. */
    uint16_t group_x, group_y;
    uint32_t group_address; /**< 0 = post nothing */
    uint32_t group_member;  /**< this residency's slot in the block */

    /** RUNTIME-ARG TAILS past the queue's ten, for the two data movers only (index 0 = DM0,
     *  the reader; 1 = DM1, the writer): the resident matmul's cooperative + MoE tail
     *  (resident_matmul_multicast_protocol.h, resident_matmul_moe_protocol.h) and the router
     *  writer's routing tail. Set by the caller after pn_tt_residency_load() and before
     *  pn_tt_residency_start(); `tail_count` 0 plants exactly the ten, as every family did. */
    uint32_t tail_args[2][PN_TT_RESIDENCY_TAIL_ARGS];
    uint32_t tail_count[2];

    uint64_t ops;     /**< descriptors this residency has executed */
    double seconds;   /**< wall time inside pn_tt_residency_submit() */

    /** PCIe LOAD UNITS this residency's awaits have issued, SPLIT BY WHAT THEY ARE.
     *
     *  **POOLING THESE TWO MAKES THE SHAPE INVISIBLE, AND THAT WAS MEASURED RATHER THAN
     *  REASONED.** One pooled counter read **49.0 and 54.5 reads an await** on two builds
     *  that were BOTH on the six-read `window` — because a residency waiting on real card
     *  work SPINS, and at ~44 spin polls an await the completion read is 5 of 49. A ladder
     *  whose acceptance criterion is "17.0 / 6.0 / 1.0" would have been unreadable against
     *  that, or read as a failed arm. So they are counted apart:
     *
     *    `poll`   every read inside the wait loop — the count poll and the 128th-poll error
     *             word. CARD-BOUND: it is how the host waits, and no shape removes it.
     *    `settle` the block read issued AFTER the wait returned — 16 on `control`, 5 on
     *             `window`, **0 on `line`, where the last poll IS the check**. This is the
     *             term the shapes move and the only one a ladder may be scored on. */
    uint64_t await_poll_transactions;
    uint64_t await_settle_transactions;
    uint64_t awaits;

    /** How this residency's enqueues write the parameter block and the ring slot. Decided
     *  ONCE at arm time from `PN_TT_ISSUE` AND the queue's placement — see pn_tt_issue.h. */
    pn_tt_issue_shape issue_shape;

    /** THE ISSUE HALF, in the same currency the await is counted in — PCIe ACCESSES.
     *
     *  A byte counter could not price this link (64 bytes is sixteen accesses at WORD32 and
     *  four at BLOCK16) and a call counter could not either (three calls on both shapes). So
     *  the accesses are counted at the unit of the accessor that was ACTUALLY used, and the
     *  three calls are timed apart — which is what solves the per-call fixed cost against the
     *  per-access one instead of asserting a split between them. */
    uint64_t issues;          /**< enqueues, i.e. descriptors ISSUED (not repeats) */
    uint64_t issue_accesses;  /**< PCIe accesses those enqueues cost */
    double issue_param_seconds; /**< inside ft_tt_queue_write_op_parameters() */
    double issue_ring_seconds;  /**< inside ft_tt_queue_stage() */
    double issue_head_seconds;  /**< inside ft_tt_queue_publish() */
} pn_tt_residency;

/**
 * Resolve `program_name`'s firmware and kernel ELFs. No device access.
 *
 * BY NAME, out of the binary's own program blob, because a program root is a property of one
 * machine and a program name is a property of the model. `override_root` is the deliberate
 * escape and is never a fallback: NULL means embedded, non-NULL means
 * `<override_root>/<program_name>` off disk or a refusal (`ft_tt_program_resolve.h`).
 *
 * Failing here is not fatal to a serve: a card whose program is missing runs the host-side
 * version of that op, which is what the caller does with the `false` return.
 *
 * @param runtime        the open card; borrowed, must outlive the residency
 * @param program_name   a build_tensix_program.sh output directory's basename, e.g. "reduce"
 * @param override_root  NULL for the embedded blob; a directory to read instead
 * @param name           short label for logs
 * @param mask           processors the program enables, e.g. all five of a triad
 * @return               0, or -1 with the reason on stderr
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_residency_load(pn_tt_residency *residency, ft_tt_runtime *runtime,
                         const char *program_name, const char *override_root, const char *name,
                         uint32_t mask);

/**
 * Resolve `core_index` to a TRANSLATED coordinate and boot THIS program's firmware on it.
 *
 * The coordinate comes from ft_tt_coord, never from a guess — a wrong NoC endpoint hangs
 * the chip and costs a walk to the machine.
 *
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_residency_bind_core(pn_tt_residency *residency, uint32_t core_index);

/**
 * Load the kernel images, initialise the queue, plant the runtime args, write the caller's
 * circular-buffer descriptor array and GO.
 *
 * The caller must have filled `residency->cb_buffer` and `residency->cb_mask` with its
 * family's placement (ft_tt_op_*_place_cbs) before calling. That split is deliberate: CB
 * geometry is the op family's business and the core is this file's.
 *
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_residency_start(pn_tt_residency *residency);

/**
 * S2: zero a queue's WAIT SIDECAR and its reader's precondition words (control + 0x900 ..
 * 0xE20, foundry kernels/resident_precondition_protocol.h) before its kernel launches.
 *
 * Every arm and re-arm does it, whatever `PN_TT_ONCARD_COMPLETION` says: the region survives
 * the process, a re-armed queue counts its descriptors from 0 again, and an entry an earlier
 * run left at the same index would make the new reader wait on a producer that is gone
 * (refused only at its spin bound). Zeroed, every slot parses as "no entry".
 *
 * @return FT_TT_OK or the accessor's error
 *
 * Thread-safe: no (the caller owns the core)
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_tt_residency_clear_precondition(const ft_tt_noc_access *noc, uint16_t x, uint16_t y);

/**
 * Retire the residency by ASKING, and wait for all five processors to say EXIT.
 *
 * Never a reset and never a park. The control block's own `state` cannot answer this — it
 * has one writer — so every processor's own status slot is polled instead. Safe to call on
 * a residency that is not running.
 */
void pn_tt_residency_stop(pn_tt_residency *residency);

/** Did the last stop RETIRE it — every enabled processor saying EXIT? A caller about to boot
 *  firmware onto that core must ask: a `false` here is the wedge signature's precondition. */
bool pn_tt_residency_retired(const pn_tt_residency *residency);

/** Free the ELF images. Stops the residency first if it is still running. */
void pn_tt_residency_release(pn_tt_residency *residency);

/**
 * Write sixteen parameter words, enqueue one descriptor, and wait for the card to report it
 * EXECUTED — with the four independent agreements, not a count.
 *
 * The wait polls `completed_count` every iteration and the ERROR word every 128th, because
 * a descriptor the kernel REFUSES never advances the count and would otherwise cost the
 * whole timeout for something the card knew in microseconds.
 *
 * @param opcode      an ft_tt_queue_opcode this program implements
 * @param parameters  FT_TT_QUEUE_PARAM_WORDS words, already encoded by the family's driver
 * @param timeout_ms  never 0 — there is no untimed path to the card
 * `PN_TT_ONCARD_BATCH=K` turns this into the queue-depth instrument below, and
 * `PN_TT_ONCARD_BATCH_FAMILY=a,b` restricts it to the named residencies. The filter is what
 * makes the arm answer-preserving: `eltwise`, `render-widen` and `render-narrow` compute a
 * pure function of operands they do not touch, so K copies leave what one leaves, while the
 * GDN recurrence and the IN-PLACE per-head L2 norm do not. Unset means every family, which
 * is the shape every recorded batch arm was taken in.
 *
 * @return            0, or -1 with the reason on stderr. A -1 leaves the residency
 *                    STRANDED, and the caller must stop and re-arm it before reusing it
 */
int pn_tt_residency_submit(pn_tt_residency *residency, uint32_t opcode,
                           const uint32_t *parameters, uint32_t timeout_ms);

/**
 * `pn_tt_residency_submit()` with `repeat` COPIES of the same descriptor enqueued before the
 * host waits once — the queue-depth instrument, and the only thing in this stack that can
 * price a host round trip apart from the card's own work.
 *
 * The shipped path submits ONE descriptor and waits for it, so every descriptor pays a
 * parameter write, a ring write, a head bump and the first poll's PCIe latency. Those are
 * FIXED per host wait, not per descriptor, so `repeat` of them cost what one costs:
 *
 *     seconds(repeat) = host_fixed + repeat * on_card_serial
 *
 * Two repeat counts therefore solve for both terms, which is exactly the question "does
 * taking the host out of the dispatch loop help THIS op" — measured rather than argued.
 * `repeat` is CLAMPED to the ring capacity; the ring is empty on entry because the previous
 * call waited for it to drain.
 *
 * **THIS IS AN INSTRUMENT, NOT A FAST PATH.** Every copy runs the SAME parameter block, so
 * for a stateful family — the GDN recurrence reads and writes its carried state — the second
 * and later copies compute a value the model does not want. The timing is valid; the answer
 * is not. Nothing calls this unless a measurement knob asks for it.
 *
 * @param repeat  copies to enqueue, 1..PN_TT_RESIDENCY_QUEUE_CAPACITY; 0 is refused
 * @return        0, or -1 with the reason on stderr, same stranding contract as submit()
 */
int pn_tt_residency_submit_repeat(pn_tt_residency *residency, uint32_t opcode,
                                  const uint32_t *parameters, uint32_t repeat,
                                  uint32_t timeout_ms);

/**
 * The two halves of `pn_tt_residency_submit_repeat()` apart, so N residencies can be started
 * before ANY of them is waited on.
 *
 * That is the only way to ask whether an op scales across cores. Submitting to each core in
 * turn and waiting each time measures N serial descriptors however many cores are involved,
 * which is the same number the one-core arm prints and says nothing.
 *
 * `enqueue` counts the descriptors into `ops` and does NOT touch `seconds` — the caller owns
 * the clock when it owns the overlap. A caller that enqueues must await before enqueuing
 * again on the same residency: the ring holds one batch.
 */
int pn_tt_residency_enqueue(pn_tt_residency *residency, uint32_t opcode,
                            const uint32_t *parameters, uint32_t repeat);
int pn_tt_residency_await(pn_tt_residency *residency, uint32_t timeout_ms);

/**
 * Enqueue ONE descriptor whose parameter block is already on the card (a parameter-cache
 * block the caller wrote with ft_tt_queue_write_param_cache(), which never moves): the ring
 * slot and the head bump only, so a replayed token carries two entries for it instead of
 * three. `a1` is the reader's word (the MoE LIST word), `a2` the writer's destination row
 * pitch, `a3` / `a4` the matmul writer's PULL words (0 = no pull), `a5` the group sequence
 * the writer records as its done-sequence (0 = none).
 * @return 0, or -1 with the reason on stderr (the idle retire may have stopped it: re-arm)
 */
int pn_tt_residency_enqueue_cached(pn_tt_residency *residency, uint32_t opcode,
                                   uint32_t param_address, uint32_t a1, uint32_t a2, uint32_t a3,
                                   uint32_t a4, uint32_t a5);

/**
 * The same with the sixteen parameter words written first into this residency's rotating
 * slot `param_slot` (descriptors the parameter cache must not hold, e.g. a batched prefill's,
 * whose words move with the pass and would crowd out the decode blocks).
 */
int pn_tt_residency_enqueue_words(pn_tt_residency *residency, uint32_t opcode,
                                  const uint32_t *parameters, uint32_t a1, uint32_t a2,
                                  uint32_t a3, uint32_t a4, uint32_t a5);

/**
 * Record that everything issued to this residency completed, proven by ANOTHER core's
 * completion (a PULL: member 0 posted only after this one reached its done-sequence). Clears
 * the unawaited count and restarts the idle clock as a host await would.
 */
void pn_tt_residency_mark_settled(pn_tt_residency *residency);

/** The await a `defer_await` window skipped, now (0 when none is pending). */
int pn_tt_residency_settle(pn_tt_residency *residency, uint32_t timeout_ms);

/** Await everything issued NOW, whether or not an await was recorded or `defer_await` is on (a
 *  settle with nothing recorded returns at once — the deep attention queue's flow control must
 *  not depend on that bookkeeping; pn_tt_attn_deep.h). */
int pn_tt_residency_await_now(pn_tt_residency *residency, uint32_t timeout_ms);

/**
 * Write sixteen words into the parameter slot `offset` slots AFTER this residency's current
 * one — an EXTENSION block a family's descriptor reads beside its own (the gdn multi-row
 * DELTA's per-row state addresses). Call it before the enqueue that uses it. Goes through the
 * same accessor as the descriptor's own block, so the streamed puller captures it in order.
 */
int pn_tt_residency_write_param_extension(pn_tt_residency *residency, uint32_t offset,
                                          const uint32_t *words);

/** Wall seconds this process has spent in pn_tt_residency_bind_core / _start / _stop — the
 *  boots and retires behind every idle re-arm (PM 5 instrument). */
double pn_tt_residency_lifecycle_seconds(void);
/** Add a re-arm that does not go through the three calls above (the decode matmul group). */
void pn_tt_residency_lifecycle_add(double seconds);

/**
 * KEEPALIVE (PM 5): send one NOP descriptor to every LIVE residency that has been idle at
 * least `min_idle_seconds` but not yet past the idle-retire limit, has nothing outstanding,
 * and is not in `skip` (NULL-terminated, may be NULL); then await them all. The card's idle
 * exit and the host's retire both restart, so a family a long prefill or a pause leaves idle
 * is not retired and re-armed on its next use. The bounded-residency rule is unchanged: with
 * no host, the card still exits on its own bound.
 * @return NOPs sent, or -1 if one failed (that residency then retires on its next use)
 */
int pn_tt_residency_keepalive_all(double min_idle_seconds);

/**
 * The same enqueue, with a GROUP SEQUENCE stamped into every descriptor's `a5`.
 *
 * A non-zero sequence makes the kernel post `{sequence, token}` into this core's slot of the
 * group block after it publishes its own completion — which is what lets ONE reader learn
 * that N members finished, instead of N poll loops learning it one at a time. Zero is "post
 * nothing" and is exactly `pn_tt_residency_enqueue()`, which is now this function.
 *
 * The sequence is the CALLER's counter, not the queue's: it is the group's notion of "item
 * N", and every member of one issue must carry the same value or the slot array can never be
 * satisfied at once.
 *
 * `out_token` receives the token this enqueue stamped on the LAST descriptor, because the
 * group check needs it per member and `ft_tt_queue_token_at()` needs `issued` from BEFORE the
 * next enqueue moves it.
 */
int pn_tt_residency_enqueue_group(pn_tt_residency *residency, uint32_t opcode,
                                  const uint32_t *parameters, uint32_t repeat,
                                  uint32_t sequence, uint32_t *out_token);

/**
 * Would an enqueue go through right now? Runs the enqueue's own idle retire first (asking a
 * residency idle past `PN_TT_RESIDENCY_IDLE_RETIRE_MS` to stop), so a caller can learn that a
 * set will refuse BEFORE it sends any of it — the S1 group pull must not send member 0 a pull
 * of a member whose enqueue is about to fail.
 * @return true when the residency is running and was not retired here
 */
bool pn_tt_residency_enqueue_ready(pn_tt_residency *residency);

/**
 * One descriptor with its a3 / a4 / a5 set: the S1 GROUP PULL of a split vector family
 * (pn_tt_group_pull.h). On member 0 `a3` is the pulled member count and `a4` the sequence
 * every pulled member must reach; on every member `a5` is the group sequence its writer stores
 * as its done-sequence. All three 0 is exactly `pn_tt_residency_enqueue()` with `repeat` 1.
 * Honours `defer_await` and its rotating parameter slots, as that enqueue does.
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_residency_enqueue_pull(pn_tt_residency *residency, uint32_t opcode,
                                 const uint32_t *parameters, uint32_t a3, uint32_t a4,
                                 uint32_t a5);

/**
 * Bind this residency to a group completion block. Call BEFORE `pn_tt_residency_start()`;
 * after it, the binding is already in the kernel's runtime args and this does nothing.
 *
 * @param address  the block's L1 address on core (x, y); 0 clears the binding
 */
void pn_tt_residency_join_group(pn_tt_residency *residency, uint16_t x, uint16_t y,
                                uint32_t address, uint32_t member);

/**
 * WAIT FOR EVERY MEMBER OF ONE GROUP ISSUE, in one place, the way the matmul group does.
 *
 * Stage 1 polls MEMBER 0's `completed_count` — one four-byte read, exactly the single-core
 * poll, and the member that finishes last is the one that decides when this returns. Stage 2
 * reads the WHOLE slot array in one BLOCK16 transaction and requires, per member, that the
 * posted sequence has reached `sequence` AND that the posted token is the one this run
 * stamped on that member's own descriptor. Two assertions per member, each failing a
 * different lie: the sequence says how far it got, the token says it ran THIS descriptor.
 *
 * @param set       members[0] owns the block and is the one polled in stage 1
 * @param token     per member, from `pn_tt_residency_enqueue_group()`
 * @return          0, or -1 with the reason on stderr
 */
int pn_tt_residency_await_group(pn_tt_residency *const *set, const uint32_t *token,
                                uint32_t members, uint32_t sequence, uint32_t timeout_ms);

/**
 * The idle limit in milliseconds past which a residency (or the matmul group) is retired by
 * ASKING before it is sent a descriptor — `PN_TT_RESIDENCY_IDLE_RETIRE_MS`, 0 = off. Shared so
 * every family with its own queue applies the same bound. See residency_idle_retire().
 */
long pn_tt_residency_idle_retire_limit_ms(void);

#endif /* USE_TT */
#endif /* PN_TT_RESIDENCY_H */
