/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_matmul_group.h
 * @brief The GROUP of Tensix cores one projection is split across, and the load-time bank
 *        shard that makes the split worth having.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS IS, AND WHAT IT IS NOT
 * ---------------------------------------------------------------------------
 * Until now `pn_tt_matvec` drove ONE core: 169 projections a token, one after another, on
 * Tensix 71. This spreads each projection's OUTPUT COLUMNS over N cores — the SPLIT arm of
 * foundry/tt/ft_tt_op_matmul_cooperative.h — and keeps the resident work queue, so the
 * per-op cost stays a parameter block and a 4-byte head bump per member rather than five
 * kernel images per member.
 *
 * **It is SPLIT, not COOPERATIVE, and the reason is structural rather than a preference.**
 * COOPERATIVE multicasts the shared A block from a leader, and that rendezvous lives in
 * `cooperative_matmul_reader.c`, a ONE-SHOT program. The resident matmul triad this backend
 * dispatches through (`rtriad_bfp8`) has no multicast, no rectangle and no per-block
 * barrier. Running the cooperative reader instead would put the per-op kernel-image upload
 * back — 36.2 µs/kB x 5,384 B = 218 µs, per member, per op, i.e. 147 ms a token at N=4
 * against a 231 ms token. So the multicast is left for a resident cooperative triad, and
 * what is wired here is the arm that needs no new kernel at all.
 *
 * ---------------------------------------------------------------------------
 * WHY NO KERNEL CHANGED
 * ---------------------------------------------------------------------------
 * The artifact stores every matmul weight N-MAJOR (tile index `nt * Kt + kt`), and the
 * resident reader indexes it exactly that way (`resident_matmul_reader.c:157`). So member
 * c's column slice is an ALREADY CONTIGUOUS RUN OF BYTES, and "member c computes columns
 * [c*cc, (c+1)*cc)" is expressed by handing that member
 *
 *     B_BASE = its own panel's base        Nt = cc        C_BASE = output + c*cc*2048
 *
 * in the descriptor words that already exist. The writer streams `Mt*Nt` tiles from
 * `C_BASE` (`resident_matmul_writer.c:63-74`), so at Mt = 1 the N panels CONCATENATE into
 * exactly the single-core output row, tile for tile and byte for byte. That is what makes
 * the correctness gate a byte comparison against N = 1 rather than a correlation.
 *
 * ---------------------------------------------------------------------------
 * THE MEASURED CONSTRAINTS THIS RESPECTS
 * ---------------------------------------------------------------------------
 * (plan Phase 6.7 / foundry/tt/ft_tt_dram_stripe.h — these are design inputs, not trivia)
 *
 *   ENDPOINT ~45 GB/s, NoC ROW ~60, BANK ~64, CARD 502. `ft_tt_coord_dram_bank()` is port 0
 *   of three, so N members left on one operand's own coordinate share ONE door.
 *     -> the weight is SHARDED over N banks at load, and each member's DOOR into its bank
 *        comes from `ft_tt_dram_stripe_choose_ports()`, never from `port = c % 3`, which
 *        collides the moment the bank list holds a b and a b+4.
 *     -> the activation is one strip in one bank; its N readers get N DOORS of that bank
 *        from the same chooser, which is what separates ~45 GB/s from ~64.
 *
 *   OUTSTANDING READS ARE PER ENDPOINT AND MUST BE BALANCED. The chooser is greedy in
 *   reader order and differs by at most one stream between doors; `busiest`/`quietest` are
 *   reported at load so an unbalanced map is visible rather than inferred.
 *
 *   THE 128 KiB ADDRESS PERIOD. A shard moves a panel's BASE, never its stride: the reader
 *   still walks `(nt * Kt + kt) * tile_bytes` inside one member's panel, so the stride is
 *   `Kt * 1088` exactly as it is today and no shard can manufacture a 128 KiB row stride
 *   that was not already there.
 *
 *   NEVER MORE THAN 4 DESTINATIONS PER MULTICAST. Nothing here multicasts.
 *
 * ---------------------------------------------------------------------------
 * WHAT A MEMBER COUNT COSTS WHEN IT DOES NOT DIVIDE
 * ---------------------------------------------------------------------------
 * `Nt % N != 0` used to be REFUSED for that tensor, which cost it its split entirely — and
 * that refusal was COOPERATIVE's, whose reason is the per-block multicast rendezvous.
 * **SPLIT has no rendezvous**, so it now takes a RAGGED split: the remainder goes one column
 * each to the first members, widths differ by at most one, and `shard_column_first` is the
 * running sum. MEASURED 2026-08-16: under the old rule N=5 left 3,864 of 3,887 projections
 * on one core and N=7 left ALL of them, which read as a scaling ceiling for a whole round.
 *
 * A weight NARROWER than the member count still stays whole — a member with zero columns is
 * a descriptor the card refuses, and a refusal there strands a triad. It is counted
 * (`solo_ops`, `unsharded_weights`) and printed, because a run that quietly left a quarter
 * of its projections on one core would report a disappointing speedup and no reason.
 *
 * @status Wired into pn_tt_matvec_dispatch() and pn_tt_matvec_enqueue(). The cooperative
 *   multicast path (PN_TT_MATMUL_COOPERATIVE) is byte-exact at N=4 and N=8, saves 1.06-1.14x
 *   of card time at prefetch depth 16, and is OFF by default because it moves the member
 *   cores onto a rectangle and the N=8 teardown wedge is unresolved.
 *   PN_TT_MATMUL_CORES=1 is byte-for-byte the single-core path and is the default; N = 1..8,
 *   12 and 16 are byte-identical to it on silicon.
 * @issues The card-side completion post wedges the POSTING core at its next boot, so
 *   `PN_TT_GROUP_COMPLETION` is off and gated behind an acknowledgement. Kernel work.
 * @todo None
 */

#ifndef PN_TT_MATMUL_GROUP_H
#define PN_TT_MATMUL_GROUP_H

#ifdef USE_TT

#include "pn_tt_grid_marks.h"
#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_column_runs.h"
#include "pn_tt_device.h"
#include "pn_tt_issue.h"
#include "pn_tt_matmul_pull.h"

#include "tt/ft_tt_dram_stripe.h"
#include "tt/ft_tt_fttw.h"
#include "tt/ft_tt_launch.h"
#include "tt/ft_tt_op_matmul_resident_multicast.h"
#include "tt/ft_tt_queue.h"

/** One member of the group: a core, its resident work queue, and the launch state that
 *  staged it. Member 0's queue is the device's own, so this struct holds a POINTER rather
 *  than the queue — one residency has one owner and it is still pn_tt_device.c. */
typedef struct pn_tt_matmul_member {
    uint32_t core_index;
    uint16_t core_x, core_y;
    ft_tt_queue *queue;    /**< member 0: &device->queue; the rest: &own_queue */
    ft_tt_queue own_queue; /**< storage for members 1..N-1 */
    ft_tt_launch_config config;
    unsigned char cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
    bool booted;  /**< firmware is up on this core */
    bool running; /**< a residency is live on it RIGHT NOW */

    /**
     * THE PARAMETER CACHE, host side: the sixteen words of every block already written into
     * this member's `FT_TT_QUEUE_PARAM_CACHE` region, in slot order.
     *
     * Heap, allocated at arm and freed at retire, because 320 blocks x 64 B x 32 members is
     * 640 KiB and this struct is embedded in `pn_tt_device`.
     *
     * `cache_slot[h]` is an open-addressed index into it, so a lookup is one FNV-1a over 64
     * bytes plus ONE memcmp — not a scan. The memcmp is not an optimisation, it is the
     * correctness claim: a hit means the block at that address holds exactly these words,
     * and a hash collision therefore costs a miss and never a wrong descriptor.
     */
    uint32_t (*cache_words)[FT_TT_QUEUE_PARAM_WORDS];
    uint16_t *cache_slot;   /**< PN_TT_PARAM_CACHE_BUCKETS entries; 0xFFFF = empty */
    uint32_t cache_used;    /**< blocks written so far; never decreases within a residency */
    uint64_t cache_hits;    /**< projections that reused a block, i.e. writes NOT made */
    uint64_t cache_misses;  /**< projections that had to write one */
    uint64_t cache_full;    /**< projections that fell back to the rotating slots */
} pn_tt_matmul_member;

/** The group, and what the striping rules said about the map it produced. */
typedef struct pn_tt_matmul_group {
    /** 1 = today's single-core path, and every branch here is then a no-op. */
    uint32_t member_count;
    pn_tt_matmul_member member[PN_TT_MATMUL_MAX_CORES];

    /**
     * The DOOR each member reads an activation strip through, per bank the strip may live
     * in. Precomputed because the strip's bank moves per call (the on-card forward
     * round-robins strips across banks) while the assignment for a given bank does not.
     */
    ft_tt_dram_endpoint activation_door[FT_TT_MAX_DRAM_BANKS][PN_TT_MATMUL_MAX_CORES];
    uint32_t activation_bank_count;
    /**
     * WEIGHT BANK MAP (tt-place lane, `PN_TT_WEIGHT_BANKS`): 0 = the rotation (bank
     * `(cursor + m) % 8`, the shipped placement), 1 = `near` (each member's panels in the
     * bank and door nearest it on its reader NoC, 2 members a bank, ft_tt_dram_nearest.h),
     * 2 = `nearbank` (nearest banks, doors by the stripe chooser). Same bytes either way.
     */
    uint32_t weight_bank_mode;
    bool weight_near_ready;
    uint32_t weight_near_bank[PN_TT_MATMUL_MAX_CORES];
    uint32_t weight_near_port[PN_TT_MATMUL_MAX_CORES];
    bool activation_doors; /**< PN_TT_ACTIVATION_DOORS; the control arm, default on */

    /** Reported, never refused: pairs of doors in one plan that share a NoC row, worst over
     *  the weights, and the balance the chooser achieved on the widest shard map. */
    uint32_t worst_weight_row_collisions;
    uint32_t activation_row_collisions;
    uint32_t sharded_weights;   /**< weights split over member_count banks */
    uint32_t unsharded_weights; /**< weights too narrow to give every member a column */
    uint32_t ragged_weights;    /**< sharded weights whose members are not all one width */

    uint64_t split_ops; /**< projections dispatched across every member */
    uint64_t solo_ops;  /**< projections that ran on member 0 alone */

    /* ------------------------------------------------------------------
     * THE SHARED ACTIVATION, MULTICAST INSTEAD OF RE-READ
     * ------------------------------------------------------------------
     * At Mt = 1 every member re-reads the whole A column once per output column, so A is
     * 1.76 GB of the 2.70 GB a token reads — 65% — and all of it comes from the ONE bank
     * holding that strip. That is why four cores buy 3.44x of card time and sixteen buy
     * 4.80x, and why door rotation alone is worth 1.27x at N=8: a gain that GROWS with N
     * is what an endpoint ceiling looks like. With cooperation the LEADER reads A once and
     * multicasts it as ceil(receivers / 4) packets.
     *
     * It is OPT-IN and it MOVES THE CORES, which is the reason it is not simply a better
     * default. A multicast writes every node of its bounding box, and the default member
     * set {71, 75, 76, 77} is an arithmetic run whose box also holds 72, 73 and 74 — the
     * reduce, eltwise and sample residencies. So a cooperating group is placed as a
     * RECTANGLE from `device->core_index`, and that placement has to be chosen rather than
     * inherited.
     */
    /** PN_TT_MATMUL_COOPERATIVE. Off by default. */
    bool cooperative_requested;
    /** PN_TT_MATMUL_RECTANGLE, or implied by the above. Placing the members as a rectangle
     *  WITHOUT cooperating is the control arm: it moves the cores exactly where a
     *  cooperating group puts them and leaves the multicast as the only variable. */
    bool rectangle_requested;
    /** RED PROOF, `PN_TT_RED_COOP=stale`: the leader skips the payload multicast. See the
     *  .c for the two claims one arm settles. */
    bool red_coop_stale;
    /** The members are a rectangle and the broadcast partition built. Independent of
     *  whether the WEIGHTS allow it, which is only known once they are all placed. */
    bool cooperative_placed;
    /** ARMED: placed, more than one member, and every weight splits evenly. Cooperation is
     *  ALL-OR-NOTHING for a residency — a group that cooperates on some descriptors and
     *  not others drifts its members' CB_IN0 write pointers apart, and every later
     *  multicast then lands in the wrong half of the receivers' buffers with every kernel
     *  completing and every status word clean. */
    bool cooperative;
    /** Columns of the member rectangle; rows are member_count / columns. */
    uint32_t cooperative_columns;
    /** Weights whose Nt does not divide by the member count. Any at all disarms
     *  cooperation for the whole residency, and the count is what says why. */
    uint32_t uneven_weights;
    /** The roles, the geometry and the three rendezvous addresses. Valid only while
     *  `cooperative_placed`. */
    ft_tt_op_matmul_resident_multicast multicast;

    /* ------------------------------------------------------------------
     * COMPLETION THE CARD AGGREGATES, so the host polls ONE core, not N
     * ------------------------------------------------------------------
     * Every member's WRITER already posts {sequence, token} into its own slot of a block
     * on member 0 (ft_tt_queue.h, the group completion block) — the mechanism shipped with
     * the resident triad and nothing here used it. Asking each member separately cost a
     * five-word control read PER MEMBER PER PROJECTION, ~21 us, i.e. 21N us an op and
     * 426 ms a token at N = 120. The slots are CONTIGUOUS in one core's L1, so the same
     * question costs `ceil(8N / 16)` device accesses of ONE core instead: reads on this
     * link are transaction-count bound, and that is the difference between a term linear
     * in the member count and one that is nearly flat in it.
     *
     * **AND IT IS OFF, because it costs a core.** The post leaves the POSTING core
     * unbootable at its next boot — measured twice, 2026-08-16; the .c has the evidence and
     * the acknowledgement that arms it. What actually brought the per-member term down was
     * the WIDTH of the read, not moving where the question is asked.
     */
    /** Monotone, stamped into descriptor word a5. Never 0 — 0 means "post nothing". */
    uint32_t sequence;
    /** The sequence and token each member was last given, and whether it owes anything. */
    uint32_t pending_sequence[PN_TT_MATMUL_MAX_CORES];
    uint32_t pending_token[PN_TT_MATMUL_MAX_CORES];
    bool pending[PN_TT_MATMUL_MAX_CORES];
    /** Projections enqueued since the last flush. Bounded by the ring, not by taste. */
    uint32_t chained;
    /** **OFF unless asked for by its full spelling** — the post wedges the POSTING core at
     *  its next boot (see the .c). Off, the host polls every member's own control block,
     *  which is 2 transactions each through the BLOCK16 L1 accessor. */
    bool group_completion;
    /** L1 address of the block on member 0, or 0 before it is written. */
    uint32_t group_address;

    /**
     * THE DESTINATION ARM. `PN_TT_GROUP_POST_PROBE`, default OFF.
     *
     * The two recorded wedges took a REMOTE poster and left the LOCAL one (member 0,
     * posting into its own L1 under identical code) PASS both times. That is 2 of 2 and it
     * makes "remote destination" the discriminator the data supports — but the two arms
     * that produced it also differed in which member they were, so it is evidence and not
     * proof. This separates the destination from everything else: the SAME
     * `ftq_group_post()` on the SAME processor of the SAME members, with only the NoC
     * coordinates it writes to changing.
     *
     * `self` points every member's post at its OWN coordinates; `remote` points every
     * member's post at member 0, which is the recorded configuration. Both leave the HOST
     * on the ordinary per-member completion wait, so both arms are byte-exact runs and
     * `cmp` means something on each — the probe is about the card-side write, and the host
     * read direction cannot wedge a Tensix.
     *
     * NOT the same object as `group_completion`: that one changes what the HOST reads and
     * needs every member's post to land in one block. This changes only where the card
     * writes, and is mutually exclusive with it for that reason.
     *
     * **RUN 2026-08-20: `self` WEDGED cores 71 and 75 of 71/75/76/77, after a disarmed
     * control on the same four probed 4 PASS.** Core 71 posts locally under both arms, so
     * the configuration it survived twice is the one that broke it — the wedge does NOT
     * follow the remote destination, and aggregating local to the writer is not a way out.
     * See the .c.
     */
    enum {
        PN_TT_GROUP_POST_PROBE_OFF = 0,
        PN_TT_GROUP_POST_PROBE_SELF,
        PN_TT_GROUP_POST_PROBE_REMOTE
    } post_probe;
    /** RED PROOF, `PN_TT_RED_GROUP=early`: stop waiting one descriptor short. See the .c. */
    bool red_wait_early;

    /**
     * WHERE THE DISPATCH PHASE ACTUALLY GOES, split at the only seam that separates the
     * host from the card: `issue` is N parameter blocks, N descriptors and N head bumps,
     * and nothing in it waits for the card; `wait` is everything after the last head bump.
     *
     * `member_wait` is the same clock charged to whichever member was still running when
     * its turn to be awaited came. Members are awaited in index order and run
     * concurrently, so a balanced group puts nearly all of it on member 0 and nearly none
     * on the rest — and a STRAGGLER is the one whose share is not nearly none. That is the
     * only way to tell "the split is uneven" from "every core is slower".
     */
    double issue_seconds;
    double wait_seconds;

    /* FLUSH CENSUS (M1): one row per distinct byte total a flush covers. `span` runs from the
     * chain's FIRST issue to its completion, so it holds the card's whole work for the chain
     * whatever the issue overlapped; `wait` is the host's await alone. Fit across rows:
     * span = fixed + bytes / rate separates a per-flush fixed cost from the streaming rate. */
    double chain_started;
    double chain_issue_end;
    uint64_t chain_bytes_from;
    uint32_t chain_projections;
    uint32_t flush_census_rows;
    struct {
        uint64_t bytes;
        uint64_t flushes;
        uint64_t projections;
        double span;
        double wait;
        double issue;       /**< first issue -> last issue of the chain (the STAGGER) */
        double gap;         /**< last issue -> flush entered (host work in between) */
        double done_await;  /**< awaits that found the member ALREADY done (one poll) */
        uint64_t done_count;
        double busy_await;  /**< awaits that had to poll more than once */
        uint64_t busy_count;
        uint64_t polls;
        /* CLASS CLOCK (tt-place, `PN_TT_MATMUL_CLASS_CLOCK=1`, host-issued decode only): the
         * members' own card-busy counters (writer word 13) read after every flush, so a
         * flush's card time per member is a difference of two card clocks — the host
         * span above is not a card clock under dispatch. Units: 16 AICLK cycles. */
        uint64_t card_busy_max;
        uint64_t card_busy_sum;
        uint64_t card_pull_wait; /**< member 0's pull wait (writer word 61), same units */
        uint64_t card_busy_member[PN_TT_MATMUL_MAX_CORES];
    } flush_census[32];
    uint32_t class_clock_last[PN_TT_MATMUL_MAX_CORES];
    uint32_t class_clock_pull_last;
    int class_clock; /**< -1 unread, 0 off, 1 on */
    uint64_t flush_census_dropped;
    double member_wait_seconds[PN_TT_MATMUL_MAX_CORES];
    uint64_t poll_reads; /**< 4-byte control reads the completion polls spent */
    uint64_t group_reads; /**< wide reads of the slot array — `ceil(8N/16)` accesses each */

    /**
     * THE TRANSACTION LEDGER — what the per-member host term is actually MADE OF.
     *
     * `wall(N) = 0.3885 + 0.00242*N + card(N)` fits all eight member counts, but a FITTED
     * coefficient cannot say which device accesses it is. These counters can: every host
     * access this file makes is counted here at the point of issue, split by phase and by
     * direction, with its byte count beside it. The link's own rates then convert them —
     * writes are BANDWIDTH bound (~53 MB/s through the UC path) and reads are
     * TRANSACTION-COUNT bound (~3.4 us each, 16 bytes or 4, DEV_NOTES.md §6) — so
     *
     *     per-member host seconds = issue_bytes / 53e6  +  wait_transactions * 3.4e-6
     *
     * is a PREDICTION of the fitted coefficient from counted quantities, and the two either
     * agree or one of them is wrong. That is the instrument this round is missing: without
     * it, "host ISSUE is 2.42 ms/token per member" names a seam and not a mechanism.
     *
     * `wait_transactions` counts 16-byte units for the BLOCK16 accessor and one per 4-byte
     * word for the UC one, because that is what the link charges for.
     */
    uint64_t issue_writes;       /**< host->card writes in pn_tt_group_issue() */
    uint64_t issue_write_bytes;  /**< and their bytes */
    uint64_t wait_reads;         /**< card->host reads in the flush path */
    uint64_t wait_read_bytes;    /**< and their bytes */
    uint64_t wait_transactions;  /**< what the LINK charges for those reads */
    uint64_t flushes;            /**< how many times the chain was drained */
    uint64_t projections;        /**< how many projections were issued */
    /** THE DENOMINATOR, and it is NOT `projections * member_count`. A weight whose `Nt` the
     *  member count could not divide stays whole in one bank and is issued to member 0
     *  ALONE — 48 of Ornith's 249 — so the nominal product over-counts the pairs that
     *  actually cost a device access by ~20% and every per-member figure derived from it is
     *  low by the same factor. These count the pairs that happened. */
    uint64_t issue_pairs;        /**< (projection, member) pairs actually issued */
    uint64_t wait_pairs;         /**< (flush, member) pairs actually awaited */

    /** THE ISSUE IN PCIe ACCESSES, which is what `wait_transactions` already is for the read
     *  direction. `issue_write_bytes` cannot answer this: 64 bytes is SIXTEEN accesses through
     *  the UC/WORD32 accessor and FOUR through the BLOCK16 one, and a cache HIT writes no
     *  bytes at all and still costs the ring slot and the head bump. */
    uint64_t issue_accesses;

    /** Which accessor this group's payload writes go through — `PN_TT_ISSUE`, decided ONCE
     *  when a member's queue is armed. See pn_tt_issue.h for the WC/UC store-ordering seam. */
    pn_tt_issue_shape issue_shape;

    /** **THE 64-BYTE PARAMETER WRITE, MADE ONCE INSTEAD OF EVERY PROJECTION**,
     *  `PN_TT_PARAM_CACHE`, **DEFAULT ON since 2026-08-20**. The sixteen words of a
     *  projection are a function of `(weight, input strip, output strip, member)` and all
     *  four are constants of a decode run, so the same block is written ~24 times a token.
     *  Cached, ISSUE falls from three accesses per member per projection to two — and it is
     *  the COUNT that pays, not the bytes: **`b` 0.00242 -> 0.00199, 1.21x, not the 2.4x
     *  the bandwidth model predicted.** See the .c. */
    bool param_cache;

    /** **ONE TRANSACTION PER MEMBER PER FLUSH instead of three**, `PN_TT_FAST_COMPLETION`,
     *  **DEFAULT ON since 2026-08-20**. The poll and the completion check become the same
     *  16-byte read of the control block's second line — which carries `completed_count`,
     *  `completed_token` and `state` but NOT `error`. Worth **6.6-7.2 us per awaited member
     *  per flush, flat from 2 to 32 members**, i.e. matmul card work 0.0611 -> 0.0475 at the
     *  shipped 8. See the .c for what it drops, why that is recoverable, and why the WC
     *  tight loop is safe. */
    bool fast_completion;
    /** M2 PULL COMPLETION: member 0's writer NoC-reads every member's done-sequence before
     *  posting, so a flush whose chain ENDED in a split projection awaits member 0 alone.
     *  `PN_TT_MATMUL_PULL`; `pull_table_ok` is set only once the table read back exact. */
    bool pull;
    bool pull_table_ok;
    bool red_pull_skip;
    bool chain_last_pulled;
    uint32_t chain_pull_members;
    uint32_t chain_pull_target;
    uint64_t pulled_flushes;
    /** L2 (pn_tt_matmul_pull.h): `PN_TT_MATMUL_PULL_DEFER` holds member 0's descriptor until
     *  the next projection of its chain (published with no pull) or the flush (published with
     *  the pull); `PN_TT_MATMUL_PULL_SWEEP=0` asks the writer for the one-at-a-time walk. */
    bool pull_defer;
    bool pull_serial;
    bool m0_held;
    ft_tt_queue_descriptor m0_held_descriptor;
    uint64_t pulls_deferred; /**< member-0 descriptors published without the pull they had */
    /** PHASE G0: the descriptor word a1 every issue carries (0 = the shipped path; bit 31 =
     *  B stash at L1 address (a1 & 0xFFFFFF) << 4, bit 30 = RED skip the stash load) and the
     *  stash CB's placed address (`PN_TT_G0_STASH_TILES` > 0 at arm). */
    uint32_t g0_a1;
    uint32_t stash_address;
    /** S3b: while non-NULL, pn_tt_group_issue() writes FUSED gate/up descriptors — the up
     *  weight's shard in a1/a2 and `swiglu_bits` over word 15 (pn_tt_swiglu_fuse.h). Set and
     *  cleared around one issue by pn_tt_matmul_group_enqueue_swiglu() only. */
    const pn_tt_weight *swiglu_up;
    uint32_t swiglu_bits;
    uint64_t swiglu_issues; /**< fused projections issued (each one replaces gate + up + silu_mul) */
    /** S3a: while non-NULL, pn_tt_group_issue() writes RESADD descriptors — this residual at
     *  the member's first column in a1/a2 and `resadd_bits` over word 15
     *  (pn_tt_resadd_fuse.h). Set and cleared around one issue by
     *  pn_tt_matmul_group_enqueue_resadd() only. */
    const pn_tt_activation *resadd_residual;
    uint32_t resadd_bits;
    uint64_t resadd_issues; /**< projections issued with the residual add fused */

    /* ------------------------------------------------------------------
     * PHASE G — the PREFILL GRID: `PN_TT_PREFILL_GRID` resident matmul triads (default 0 =
     * off) on cores `PN_TT_PREFILL_GRID_FIRST`.., booted once, started at a prefill's first
     * multi-row projection and retired at its end. Each decode member's weight shard is cut
     * into `grid_pieces` column pieces, one per grid core, read ONCE into that core's L1
     * stash when it fits (`grid_stash_tiles`); A comes from the strip's mirrors.
     * ------------------------------------------------------------------ */
    pn_tt_matmul_member grid[64];
    uint32_t grid_count;
    uint32_t grid_pieces;
    uint32_t grid_stash_tiles;
    uint64_t grid_msplit_pieces; /**< row-split pieces issued (pn_tt_grid_msplit.h) */
    uint64_t grid_unsplit_projections; /**< unsplit weights cut by rows (O.0) */
    pn_tt_grid_marks grid_marks;       /**< G.2: a mark per detached projection */
    uint32_t grid_last_mark;           /**< the mark of the last detached projection */
    uint64_t grid_mark_waits, grid_mark_full; /**< waits by mark / that fell back to a full wait */
    bool grid_unsplit_ok; /**< pn_tt_matmul_grid_unsplit_allow() */
    uint32_t grid_stash_address;
    uint32_t grid_tile_bytes, grid_depth;
    bool grid_booted;
    bool grid_running;
    /** L1a.3: decode members 1.. serve as grid cores past `grid_lent_base` for this prompt
     *  (pn_tt_matmul_grid_extend); the decode group refuses to arm while set. */
    bool grid_lent;
    uint32_t grid_lent_base;
    /** L1a: decode members 1.. are parked and their cores lent to another role for this
     *  prompt (pn_tt_matmul_members_lend); same refusals as `grid_lent`. */
    bool members_lent;
    bool grid_pending[64];
    uint32_t grid_detached;   /**< PHASE W2: detached projections outstanding on the grid */
    uint32_t grid_ring;       /**< descriptors the busiest grid core holds since the last grid
                                   flush (stash chunks: more than one a projection) */
    uint64_t grid_projections, grid_stashed, grid_fallbacks;
    double grid_seconds;

    /* ------------------------------------------------------------------
     * THE DRAM READS THE CARD ACTUALLY ISSUES, counted from the descriptors
     * ------------------------------------------------------------------
     * Not an estimate. `resident_matmul_reader.c` walks `Mt * ceil(Nt/NB) * (Kt / depth)`
     * passes and issues exactly ONE A read of `depth * 2048 B` and `columns` B reads of
     * `depth * tile_bytes` in each, so these totals are arithmetic over the shape words this
     * file writes into the descriptor. They are kept because the recorded "29.4 GB/s
     * aggregate" was computed from the WEIGHT bytes alone, and the weight is the SMALLER half.
     *
     * A IS RE-READ ONCE PER PASS OVER THE COLUMNS. At Mt = 1 the A offset is `(mt*Kt + kt)`,
     * which does not depend on `nt` — so at NB = 1 the same Kt tiles are fetched again for
     * every one of the member's Nt columns and A's bytes are `2048/1088 = 1.882x` the
     * weight's, whatever the shape, all of it out of the ONE bank the activation strip lives
     * in. `column_block` below is the divisor.
     *
     * FALSIFIABLE, and that is what makes it worth printing: `b_read_bytes` per token must
     * reproduce the artifact's own matmul weight bytes, which the load path reports
     * independently, AND IT MUST NOT MOVE WITH `column_block` — blocking `nt` changes when B
     * is read, never how much. If either fails this counter is wrong and nothing derived
     * from it holds.
     */
    uint64_t a_read_bytes;
    uint64_t b_read_bytes;
    uint64_t read_blocks;
    uint64_t bank_read_bytes[FT_TT_MAX_DRAM_BANKS];
    /**
     * BYTES CENSUS (printed under `PN_TT_READ_CENSUS=1`; counting is free and always on).
     * `a_read_bytes` books every A tile at 2048 B; `wire_a_bytes` is what the reader really
     * fetches (a row-0 half tile is 1024 B). Multi-row descriptors (Mt > 1: the batched
     * prefill) are kept apart, because the report divides by DECODE forwards. Per weight
     * shape (Kt, whole Nt): projections issued, bytes booked, and the weight's own bytes.
     */
    uint64_t wire_a_bytes;
    uint64_t multi_row_a_bytes;
    uint64_t multi_row_b_bytes;
    struct {
        uint32_t kt, nt, tile_bytes;
        uint64_t calls, multi_row_calls, a_bytes, wire_a_bytes, b_bytes;
    } read_census[48];
    uint32_t read_census_rows;
    /** PARAM-CACHE CENSUS (printed with the read census): per weight shape (Kt, member Nt),
     *  misses (a block written into the cache) and fallbacks (cache full, rotating slot). */
    struct {
        uint32_t kt, nt;
        uint64_t misses, fallbacks;
    } param_census[48];
    uint32_t param_census_rows;
    /** `PN_TT_PARAM_CENSUS_DIFF=1`: on a fallback, the one word in which it differs from a
     *  cached block (17 = differs in more than one word / nothing close). */
    uint64_t param_diff_word[18];

    /** Output columns the reader supplies per pass, `PN_TT_MATMUL_BLOCK`, default 1 (off).
     *  A PER-RESIDENCY CONSTANT: CB_IN1's chunk is `depth * column_block` pages and the
     *  kernel's grid guard refuses a chunk size that CHANGES through a buffer, so the same
     *  value goes into every descriptor and sizes every CB. Capped at 4, which is the DST
     *  section at `dst_accum_mode 1`. It divides A's 65.3% of the token's DRAM reads by
     *  itself and leaves B's bytes and B's request size untouched. */
    uint32_t column_block;

    /** Output ROWS the reader supplies per pass, `PN_TT_MATMUL_ROWS`, default 1 (off), and
     *  the mirror of `column_block` in every respect. It divides B's bytes — the WEIGHT
     *  column is swept once for `row_block` output rows instead of once per row — and leaves
     *  A's alone, exactly as `column_block` does the other way round.
     *
     *  THE TWO SHARE THE DST SECTION AND THAT IS THE CEILING: `row_block * column_block <= 4`
     *  at `dst_accum_mode 1` (kernels/resident_matmul_block_protocol.h carries the upstream
     *  assert it comes from).
     *
     *  **AND ON A Q8_0 MODEL IT IS WORTH TWO PER CENT.** The traffic is
     *  `Mt*T*(a/NB + b/MB)`: NB divides the operand whose tile is the bf16 ACTIVATION
     *  (2048 B) and MB divides the one whose tile is the WEIGHT (1088 B at Bfp8_b). So the
     *  two blocks are symmetric in TILES and not in BYTES, and the budget belongs to NB. At
     *  Bfp8_b the lattice reads 1600 / 1568 / 2320 bytes a unit for (1,4) / (2,2) / (4,1) —
     *  the shipped setting is already within 2% of the best, and at Bfp4_b it IS the best.
     *  MB=4/NB=1 serves 128 prompt positions a weight sweep and reads **45% MORE**, because
     *  the A side grows by more than the B side shrank.
     *
     *  The law was validated against silicon with no fitted constant: it predicts
     *  `A/B = a/(4b) = 0.4706` at the shipped shape, and Ornith measures 0.4718.
     *
     *  **THE LEVER THAT WOULD MATTER IS A BIGGER DST** — `dst_section_tiles` below. At
     *  `MB*NB <= 8` the Bfp8_b optimum is MB=2/NB=4 at 1056 bytes a unit (**1.52x**), but
     *  that point needs `Mt >= 2` and NO DESCRIPTOR IN THIS TREE HAS ONE, so what is
     *  reachable there is MB=1/NB=8 at 1344 bytes (**1.19x**).
     *
     *  L1 is nowhere near binding: at depth 8 the CBs are `MB*depth*2*2048 +
     *  depth*NB*2*tile + 4096` of a 1,351,680 B arena, which admits MB near 39.
     *
     *  IT ONLY BITES AT `Mt > 1`, which is a batch of more than 32 prompt positions. At the
     *  Mt = 1 every descriptor in the tree carries today it changes the CB sizing and the
     *  chunk the reader pushes and NOTHING ELSE — which is what makes `PN_TT_MATMUL_ROWS=2`
     *  a `cmp`-identity arm for the new loops rather than a speed arm. `tile_rows` is
     *  written from a LITERAL 1 in `pn_tt_group_write_parameters()` and in the request
     *  builder, on the decode path and the batched-prefill path alike, so `Mt >= 2` is a
     *  host-shape change and not a matmul one. */
    uint32_t row_block;

    /** The DST section the ARMED KERNEL was built for, in 32x32 tiles: 4 at
     *  `DstSync::SyncHalf` (the embedded blob) and 8 at `SyncFull`
     *  (kernels/resident_matmul_dst_sync.h). It is the ceiling on `row_block *
     *  column_block`.
     *
     *  IT IS A PROPERTY OF THE KERNEL AND THE HOST CANNOT SEE IT. One Pantheon binary drives
     *  whichever matmul images are resolved, and a section the host thinks is bigger than
     *  the kernel's is not a refusal — DST has no bounds check, so the pass writes over the
     *  half the packer is reading and every kernel retires clean. That is why 8 is refused
     *  unless a program-directory override (`PN_TT_PROGRAM_DIR`) is in force: the embedded
     *  blob's matmul is SyncHalf by construction, so the only way an 8-tile kernel can be
     *  running is off a directory the caller named. */
    uint32_t dst_section_tiles;

    /** RED PROOF, `PN_TT_RED=matmul_a_spread`: member c reads its A block through a door of
     *  bank `(activation bank + c) % banks` instead of the activation's own. The BYTES, the
     *  request size and the block count are untouched and only the SOURCE BANK moves, so it
     *  isolates "one bank cannot supply A" from every other candidate for the N=4 knee.
     *  **It reads the wrong bytes and the answer is garbage** — a timing arm, nothing else. */
    bool red_a_spread;
    /** RED PROOF, `PN_TT_RED=matmul_a_offset`: same bank, addresses 4 MiB apart. The PAIR
     *  separates a per-channel ceiling from N members hammering ONE 16 KiB window, which
     *  `matmul_a_spread` alone cannot because it moves both at once. Also garbage. */
    bool red_a_offset;

    /** THE IDLE RETIRE, as `pn_tt_residency` has it (pn_tt_residency.c). The group keeps its
     *  own queues, so the residency layer's guard never saw it: a serve idle past a member's
     *  idle bound sent its first projection into a half-exited triad. Set at arm and after
     *  every completed flush; checked before an enqueue with nothing outstanding. */
    double last_alive_seconds;
    /** PER MEMBER (2026-10-27): when each member last completed work. `last_alive_seconds`
     *  is the OLDEST running member's: a flush that ran only member 0 (the solo alpha / beta)
     *  no longer reads as the whole group alive. With the out-projections on the grid, members
     *  1..15 sat idle through a whole 14.6k prefill while member 0 kept the group "alive", went
     *  past their own idle exit, and the tail forward's first projection hung. */
    double member_alive_seconds[PN_TT_MATMUL_MAX_CORES];
    uint64_t idle_retires;
} pn_tt_matmul_group;

/** Which named way `PN_TT_RED_SPLIT` breaks member 1. See the .c for what each proves. */
typedef enum pn_tt_red_split {
    PN_TT_RED_SPLIT_NONE = 0,
    PN_TT_RED_SPLIT_BASE,
    PN_TT_RED_SPLIT_OUT,
    PN_TT_RED_SPLIT_NARROW
} pn_tt_red_split;

/** `PN_TT_RED_SPLIT`'s value, or NONE when unset or unrecognised (which it says). */
pn_tt_red_split pn_tt_matmul_group_red_mode(void);

/**
 * Read `PN_TT_MATMUL_CORES`, resolve members 1..N-1, and boot their firmware.
 *
 * Member 0 is whatever core pn_tt_bring_up_core() already claimed, so a one-member group
 * touches no core the pre-group tree did not.
 *
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_matmul_group_configure(pn_tt_device *device);

/** Arm a resident triad on every member, or retire the ones that came up and fail. */
int pn_tt_matmul_group_arm(pn_tt_device *device, uint32_t weight_tile_bytes, uint32_t depth);

/** Retire every member's residency by ASKING. Never parks and never resets. */
void pn_tt_matmul_group_retire(pn_tt_device *device);

/** PM 5 keepalive: NOP an idle decode group's members (see pn_tt_matmul_group.c). */
int pn_tt_matmul_group_keepalive(pn_tt_device *device, double min_idle_seconds);

/** Idle past `PN_TT_RESIDENCY_IDLE_RETIRE_MS` with nothing outstanding? See the .c. */
bool pn_tt_matmul_group_idle_expired(const pn_tt_device *device, double *out_idle_seconds);

/**
 * Place one weight in DRAM: sharded over `member_count` banks when the member count
 * divides its `tile_cols`, whole in one bank otherwise.
 *
 * @param entry        the artifact directory entry, already known MATMUL_NMAJOR
 * @param bank_cursor  advanced by one bank per tensor, so no bank is every tensor's first
 * @param weight       shape fields (`tile_rows`, `tile_cols`, `tile_bytes`) filled by the
 *                     caller; every placement field is written here
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_matmul_group_place_weight(pn_tt_device *device, const ft_tt_artifact_entry *entry,
                                    uint32_t *bank_cursor, pn_tt_weight *weight);

/**
 * The same, over a PERMUTATION of the source tensor's output columns.
 *
 * This is how a joint projection is de-interleaved: `runs` names, in destination order, the
 * runs of source column tiles the new weight is built from, so the file is read once per run
 * per shard and nothing is permuted at run time. `runs` NULL with `run_count` 0 is the
 * identity and is byte for byte what `pn_tt_matmul_group_place_weight()` does — the two share
 * this body precisely so an ordinary weight cannot take a different path from a permuted one.
 *
 * REFUSES a run list that leaves the source or uses a source column twice, because
 * downstream it is only a byte offset: the failure mode is a weight with one column
 * duplicated and another missing, which is a smoothly wrong projection nothing else can see.
 *
 * @param runs       destination-ordered runs of SOURCE column tiles, or NULL for identity
 * @param run_count  how many
 * @param weight     `tile_rows`, `tile_bytes` and the DESTINATION `tile_cols` filled by the
 *                   caller; every placement field is written here
 * @return 0, or -1 with the reason on stderr
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_matmul_group_place_weight_columns(pn_tt_device *device,
                                            const ft_tt_artifact_entry *entry,
                                            const pn_tt_column_run *runs, uint32_t run_count,
                                            uint32_t *bank_cursor, pn_tt_weight *weight);

/** One projection across the group, issued AND awaited. @return FT_TT_OK, or a foundry
 *  error code. Exactly `pn_tt_matmul_group_enqueue()` followed by
 *  `pn_tt_matmul_group_flush()`, and the only caller that wants both is one whose next
 *  read is of this projection's own output. */
int pn_tt_matmul_group_dispatch(pn_tt_device *device, const pn_tt_weight *weight,
                                const pn_tt_activation *input, const pn_tt_activation *output);

/**
 * ISSUE one projection and DO NOT WAIT for it.
 *
 * The resident queue was built so the host enqueues and bumps a counter — *"do not fuse ops
 * to save dispatch, there is no dispatch to save"* (ft_tt_queue.h) — and the forward then
 * waited on all 169 of them anyway. Within a layer **q/k/v are independent** and **gate/up
 * are independent**, so their descriptors can sit in the ring together: the card runs them
 * back to back at its own 0.39-0.45 us op-to-op instead of one host round trip apiece, and
 * the host's issue of the second is overlapped with the card's execution of the first.
 *
 * The caller owes a `pn_tt_matmul_group_flush()` before it reads ANY of the outputs, and
 * before any op on another core consumes them. Chained descriptors must be independent:
 * the members run concurrently, so a second projection reading the first's output would
 * race even though one core's own queue is ordered.
 *
 * @return FT_TT_OK, or a foundry error code
 */
int pn_tt_matmul_group_enqueue(pn_tt_device *device, const pn_tt_weight *weight,
                               const pn_tt_activation *input, const pn_tt_activation *output);

/** Wait for every projection enqueued since the last flush. A no-op when nothing is
 *  pending, so a caller may flush defensively. @return FT_TT_OK, or a foundry error code. */
/** PHASE G0 MEASURE (`PN_TT_G0`): one projection over `PN_TT_G0_ROWS` (default 51) tile-rows
 *  of synthetic activation — 1 chunked (2 rows a pass), 2 full M, 3 full M + B stash,
 *  4 full M + RED stash — timed, with an FNV-1a of the output for cross-arm comparison. */
int pn_tt_matmul_group_g0(pn_tt_device *device, const pn_tt_weight *weight);

/** PHASE G: start the prefill grid (`PN_TT_PREFILL_GRID` cores; 0 = off, a no-op) for a
 *  prefill, and retire it after. Multi-row projections of sharded weights then run on it. */
int pn_tt_matmul_grid_begin(pn_tt_device *device);
void pn_tt_matmul_grid_end(pn_tt_device *device);

/**
 * L1a.3 (prefill rework spec §4): grow the RUNNING grid by the decode group's members 1..N-1
 * (stopped; same firmware and program, so no boot) plus `extra_core` (a free core: the grid's
 * firmware is booted on it) — 48 + 15 + 1 = 64 cores, 4 column pieces a shard. Exact: an N
 * split only. While extended the decode group refuses to arm and to split a projection.
 * Call with nothing issued to the grid since it started.
 * @return grid cores added (0: nothing changed)
 */
uint32_t pn_tt_matmul_grid_extend(pn_tt_device *device, uint32_t extra_core);

/**
 * L1a: stop decode members 1..N-1 and hold their RISCs in reset, so another role's firmware can
 * boot on their cores for this prompt. While lent the decode group refuses to arm or split.
 * @return cores written to `cores` (0: nothing lent; a member that did not exit stops it)
 */
uint32_t pn_tt_matmul_members_lend(pn_tt_device *device, uint32_t *cores, uint32_t max);

/**
 * L1a: boot the matmul firmware on every lent member core again (skipping `stuck` ones, whose
 * borrower did not exit) and re-arm the decode group whole.
 * @return 0, or -1 (named) when a core did not boot or the group did not re-arm
 */
int pn_tt_matmul_members_return(pn_tt_device *device, const uint32_t *stuck, uint32_t stuck_count);

/**
 * L1a.3: shrink the grid back (flush, stop the added cores, park `extra_core`) and re-arm the
 * decode group whole. `*extra_parked` (may be NULL) says whether the extra core exited and is
 * held in reset — false means nothing may boot on it.
 * @return 0, or -1 (named) when a core did not exit or the decode group did not re-arm
 */
int pn_tt_matmul_grid_retract(pn_tt_device *device, bool *extra_parked);

/**
 * C: flush and stop the running grid, park every core, and
 * write their indexes to `cores` so another program may borrow them. pn_tt_matmul_grid_begin()
 * boots the grid again once the borrower has parked them.
 *
 * @return the cores lent (the whole grid), or 0 when the grid is not running or not every core
 *         parked (then nothing may boot on any of them; grid_begin restores the grid)
 */
uint32_t pn_tt_matmul_grid_lend(pn_tt_device *device, uint32_t *cores, uint32_t max);
void pn_tt_matmul_grid_close(pn_tt_device *device);

/** PHASE W2: issue a grid projection OUTSIDE the matvec chain (the chain's flushes never wait
 *  for it). Needs the grid running, a sharded weight and a mirrored input. */
int pn_tt_matmul_grid_issue_detached(pn_tt_device *device, const pn_tt_weight *weight,
                                     const pn_tt_activation *input,
                                     const pn_tt_activation *output);
/** One non-blocking look: 1 every detached projection is done (and checked), 0 still running,
 *  -1 an error. */
int pn_tt_matmul_grid_detached_poll(pn_tt_device *device);
/** Wait for every detached projection. FT_TT_OK or an error code. */
int pn_tt_matmul_grid_detached_wait(pn_tt_device *device);
/** PHASE W wavefront: where the decode group's member 0 counts EXECUTED descriptors, and how
 *  many have been issued to it — a reader gate for a solo projection (Nt below the member
 *  count runs on member 0 alone). */
bool pn_tt_matmul_group_member0_completion(const pn_tt_device *device, uint16_t *out_x,
                                           uint16_t *out_y, uint32_t *out_control,
                                           uint32_t *out_issued);
/** Is the prefill grid running (a detached issue would be accepted)? */
bool pn_tt_matmul_grid_is_running(const pn_tt_device *device);
/**
 * O.0: may a multi-row projection of an UNSPLIT weight go to the grid cut by rows? Off by
 * default; the caller turns it on only where nothing gates on the decode member's completion
 * count (the wavefront's alpha/beta reader gate is member 0's) — then back off.
 */
void pn_tt_matmul_grid_unsplit_allow(pn_tt_device *device, bool allow);
/** G.2: the mark of the last detached grid projection (PN_TT_GRID_MARK_NONE before any). */
uint32_t pn_tt_matmul_grid_last_mark(const pn_tt_device *device);
/**
 * G.2: wait for the detached grid projection marked `mark` (and everything issued before it),
 * not for the projections issued after it. Streamed: puller AWAITs on each grid core's count at
 * the mark. Unstreamed, or a mark the FIFO no longer holds: a full detached wait.
 * `PN_TT_GRID_MARKS=0` always waits for everything. RED `grid_mark_early`: waits for the
 * previous mark — must DIFFER.
 */
int pn_tt_matmul_grid_wait_mark(pn_tt_device *device, uint32_t mark);

int pn_tt_matmul_group_flush(pn_tt_device *device);

/**
 * S3b: ISSUE one fused SwiGLU projection — one descriptor per member carrying `gate`'s and
 * `up`'s panels of the member's columns, writing silu(gate) * up into `output` (and its
 * mirrors, through the members' replica table) — and DO NOT WAIT, as
 * pn_tt_matmul_group_enqueue().
 *
 * Only the shape the fused kernel serves is accepted: the fused program armed
 * (PN_TT_FUSE_SWIGLU), gate and up split on one shard map over the group, one tile row, the
 * pipelined reader, no cooperation, no G0 stash, MB = 1, and an output whose mirrors the
 * members' table covers. Anything else is refused BEFORE anything is written, and the caller
 * runs the unfused gate, up and silu_mul.
 *
 * @return FT_TT_OK issued; FT_TT_EINVAL refused (nothing issued); another foundry error code
 *         if the issue itself failed
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_matmul_group_enqueue_swiglu(pn_tt_device *device, const pn_tt_weight *gate,
                                      const pn_tt_weight *up, const pn_tt_activation *input,
                                      const pn_tt_activation *output);

/**
 * S3a: ISSUE one projection with the residual add fused — one descriptor per member writing
 * `residual` + bf16(`input` . `weight`) into `output` at the member's columns — and DO NOT
 * WAIT, as pn_tt_matmul_group_enqueue().
 *
 * Only the shape the RESADD kernel serves is accepted: the fused program armed with
 * PN_TT_FUSE_RESADD, `weight` split over the group, one tile row, the pipelined reader, no
 * cooperation, no G0 stash, MB = 1, and `residual` / `output` distinct, unmirrored and wide
 * enough (pn_tt_resadd_strips_pair()). Anything else is refused BEFORE anything is written,
 * and the caller runs the projection and pn_tt_add() unfused.
 *
 * @return FT_TT_OK issued; FT_TT_EINVAL refused (nothing issued); another foundry error code
 *         if the issue itself failed
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_matmul_group_enqueue_resadd(pn_tt_device *device, const pn_tt_weight *weight,
                                      const pn_tt_activation *input,
                                      const pn_tt_activation *residual,
                                      const pn_tt_activation *output);

/** Whether the one-shot escape can read this weight at all — false once it is sharded. */
bool pn_tt_matmul_group_can_fall_back(const pn_tt_device *device, const pn_tt_weight *weight);

/** One line about the map the striping rules produced. Silent for a one-member group. */
void pn_tt_matmul_group_report(const pn_tt_device *device);

/** Projections that went across the group, and those that ran on member 0 alone. */
void pn_tt_matmul_group_op_counts(const pn_tt_device *device, uint64_t *out_split,
                                  uint64_t *out_solo);

/** One line splitting the dispatch phase into host issue, card wait and per-member wait.
 *  Silent for a one-member group only in the per-member half; the split itself is printed
 *  either way, because "is the host issuing or is the card computing" is a question the
 *  single-core path never answered either. */
void pn_tt_matmul_group_time_report(const pn_tt_device *device);

/**
 * Two lines saying what the matmul reads from DRAM, and from which bank.
 *
 * The bytes are counted from the descriptors rather than measured, because the reader's
 * loop makes them exact (see `a_read_bytes`). The point of printing them is that the
 * recorded "29.4 GB/s aggregate" divided the WEIGHT bytes by the card time, and the weight
 * is 35% of what the card reads — the activation, re-read once per output column and served
 * by ONE bank, is the other 65%.
 *
 * @param tokens forwards this run measured, so the totals are per token and can be checked
 *               against the artifact's own weight bytes
 */
void pn_tt_matmul_group_read_report(const pn_tt_device *device, double tokens);

/**
 * STEP 3a — forget every member's cached parameter blocks (host side only; the card's blocks
 * are simply overwritten later). Called once a batched prefill has flushed: the prefill's
 * blocks (no row-0 bit, word 13) otherwise fill all 320 slots and every decode block then
 * falls back to a rotating-slot write (census: 1,632 fallbacks a token, all differing in
 * word 13 alone). `PN_TT_PARAM_CACHE_RESET` default 1, 0 = keep (the old behaviour).
 * The caller guarantees nothing is in flight.
 */
void pn_tt_matmul_group_param_cache_reset(pn_tt_device *device);

/** Diagnostics: members whose completion count is short of what was issued to them now. */
uint32_t pn_tt_matmul_group_incomplete(pn_tt_device *device);

#endif /* USE_TT */
#endif /* PN_TT_MATMUL_GROUP_H */
