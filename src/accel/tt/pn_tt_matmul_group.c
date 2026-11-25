/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_matmul_group.c
 * @brief Implementation of the multi-core matmul group (see pn_tt_matmul_group.h) — the
 *        resident triads, the load-time bank shard, and the split dispatch.
 *
 *        Everything about ONE residency lives here too, including the single-member case:
 *        a group of one is the pre-group path with the same core, the same queue and the
 *        same descriptor words, which is what makes `PN_TT_MATMUL_CORES=1` a control rather
 *        than a second implementation.
 *
 * @status Green at N=1..8 on MiniCPM5-1B; N=1 is byte-identical to the pre-group tree, and
 *   so are the cooperative arms at N=4 and N=8.
 * @issues The COOPERATIVE multicast is byte-exact and its trade DEPENDS ON THE PREFETCH
 *   DEPTH, because the rendezvous is paid once per block: at depth 8 it costs 1.24x more
 *   card time at N=4, at depth 16 it SAVES 1.059x at N=4 and 1.137x at N=8. It is off by
 *   default because it also moves the member cores onto a rectangle, and because the N=8
 *   arm has left one member of the second rectangle unbootable at its next boot in five of
 *   five pairs, with the split control on the same cores clean immediately before each.
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_matmul_group.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_attention_grid.h"
#include "pn_tt_core_boot_guard.h"
#include "pn_tt_dispatch.h"
#include "pn_tt_grid_marks.h"
#include "pn_tt_grid_idle.h"

#include "pn_tt_core_list.h"
#include "pn_tt_grid_msplit.h"
#include "pn_tt_matmul_rows.h"
#include "pn_tt_unmirrored_probe.h"
#include "pn_tt_grid_probe.h"
#include "pn_tt_weight_banks.h"
#include "pn_tt_replica_map.h"
#include "pn_tt_resadd_fuse.h"

#include "tt/ft_tt_coord.h"
#include "tt/ft_tt_op_dispatch.h"
#include "tt/ft_tt_op_l1_layout.h"
#include "tt/ft_tt_op_matmul_cooperative.h"
#include "tt/ft_tt_group_shared_operand.h"
#include "tt/ft_tt_op_matmul_resident_multicast.h"
#include "tt/ft_tt_program_resolve.h"
#include "tt/kernels/resident_matmul_block_protocol.h"
#include "tt/kernels/resident_matmul_swiglu_protocol.h"
#include "tt/kernels/resident_matmul_resadd_protocol.h"

/** How many Tensix one projection is split across. `PN_TT_MATMUL_CORES=1` is the pre-group
 *  path and the control arm. */
#define PN_TT_MATMUL_CORES_ENV "PN_TT_MATMUL_CORES"

/**
 * EIGHT SINCE 2026-08-19, and it is the MEASURED WALL OPTIMUM rather than a preference.
 *
 * It was 1 — "multi-core dispatch is asked for, never inferred" — which meant a serve with
 * no env at all ran the single-core path at 1.28 tok/s while the serve config's own
 * `PN_TT_MATMUL_CORES=4` got 1.94. That gap was a configuration, not a finding.
 *
 * Swept with the hot A strips mirrored across banks, matmul CARD work and the WALL, every
 * arm `cmp` IDENTICAL and `company=0`:
 *
 *     members       1      2      4      8     12     16     24     32
 *     card work  .3949  .2001  .1045  .0611  .0559  .0594  .0850  .1114  s/token
 *     WALL       .7858  .5932  .5035  .4687  .4743  .4860  .5318  .5764  s/token
 *
 * **The card bottoms at 12 and the WALL bottoms at 8, and the difference between them is
 * HOST ISSUE.** Issue is 2.42 ms/token per member — fitted across the whole sweep, and
 * `wall(N) = 0.3885 + 0.00242*N + card(N)` reproduces all eight measurements to 1 ms. So
 * every member past 8 buys less card time than it spends on the host writing its
 * descriptor, and 32 members spend 0.077 s/token of pure host issue, which alone exceeds
 * the entire matmul's card work.
 *
 * Members run upward from PN_TT_MATMUL_CORE_FIRST (75), so eight of them occupy 75-81 plus
 * member 0 at 71 — clear of the reduce/eltwise/sample residencies at 72/73/74, of the GDN
 * at 90 and its head-split members from 92, and of the head norms at 91.
 *
 * ---------------------------------------------------------------------------
 * RE-SWEPT 2026-08-20 WITH THE TWO HOST MECHANISMS ON, AND 8 STAYS
 * ---------------------------------------------------------------------------
 *     members        1      2      4      8     12     16     24     32
 *     card work  .3939  .1969  .0984  .0498  .0379  .0329  .0405  .0524  s/token
 *     WALL       .7830  .5901  .4996  .4540  .4503  .4593  .4774  .5047  s/token
 *     tok/s       1.28   1.69   2.00   2.20   2.22   2.18   2.09   1.98
 *
 * **The card knee moved 12 -> 16 and the wall knee 8 -> 12** — the whole of the control
 * curve's 12-to-16 turn-up was the host's own per-member completion read inside WAIT, and
 * removing two of its three transactions removed it. `b` is 0.00199 now, so 32 members
 * still spend 0.064 s/token on host issue.
 *
 * **THE DEFAULT STAYS 8 ANYWAY, and the reason is placement, not speed.** Twelve members
 * run 75..85, which lands ON the attention family's rope/memory/attention residencies at
 * **82/83/84** — `pn_tt_prepare_family()` refuses a residency on a member's core and logs
 * it, so a 12-member default would silently drop Phase 5's attention back to the host the
 * day it ships. The measured gain for taking that risk is 2.20 -> 2.22 tok/s, which is
 * inside the 2.14/2.13 drift between this session's own opening and closing controls.
 * **Moving to 12 means moving `PN_TT_MATMUL_CORE_FIRST` first and re-measuring, because
 * the placement changes the NoC distances.**
 *
 * ---------------------------------------------------------------------------
 * DONE 2026-08-28 ON PREFILL, AND THE ANSWER IS THE OPPOSITE OF THE DECODE SWEEP'S
 * ---------------------------------------------------------------------------
 * **"92..102 is eleven free cores" WAS WRONG ON BOTH ENDS and cost two rounds.** 95 is the
 * gdnc residency, 92-94 are the GDN head-split members, and the RESIDENCY TABLE does not
 * list every occupant — `head-norm-2` at 103 and the eltwise member at 106 have no row,
 * because their clocks are the recorded twenty-eighth vacuous-gate class. The DEFAULTS in
 * the source are the only complete map. What places clean today:
 *
 *     PN_TT_MATMUL_CORE_FIRST=96 PN_TT_GDNC_MEMBER_CORE_FIRST=114
 *     PN_TT_CORE_HEAD_NORM_2=111 PN_TT_ELTWISE_MEMBER_CORE_FIRST=112
 *
 * On a 1,629-token prefill, `cmp` IDENTICAL at every rung against a **0.02%** control drift:
 *
 *     members   8 (75..81)   8 (96..102)   12 (96..106)   16 (96..110)
 *     tok/s        194.77       194.62         204.16         205.78
 *     batched      3.3286       3.3322         2.9317         2.7070   s
 *     GB/s          194.8        194.9          249.5          310.1
 *     decode         7.74            —           8.36           8.74   tok/s
 *
 * **Moving the same eight members is FREE (-0.08%, inside drift), so NoC distance is not a
 * term here.** Prefill +5.7% and DECODE +13.0% at 16 — where the recorded decode sweep above
 * has 16 WORSE than 8, because on decode the per-member host term ate the card saving and on
 * prefill a descriptor carries 32 positions.
 *
 * **AND THE HOST TERM'S IDENTITY IS TLB WINDOW RE-AIMS, MEASURED:** window aims go
 * **829.2 -> 1,939.9 -> 7,566.1 a token** at N = 8/12/16 for **0.0090 -> 0.0192 -> 0.0700
 * s/token**, as the bulk bank's hit rate falls **99.74% -> 99.50% -> 98.20% on 32 windows**
 * — and **95-98% of every aim is the verification READBACK**. The three writes per member
 * are unchanged in count and shape at every N. At 16 the aims are **7.1% of the wall**, so
 * the knee between 12 and 16 is the BANK, not the descriptors:
 * `FT_TT_UDRV_NOC_BANK_MAX_WINDOWS` is a compile-time 32.
 */
/* SIXTEEN SINCE 2026-10 (PX7+PX8 flipped as ONE change, its own confirmation round): with
 * the placement below and a 64-window bank, prefill 195.4 -> 213.4 and decode 7.73 -> 8.83.
 * Neither half is worth much alone. */
#define PN_TT_MATMUL_CORES_DEFAULT 16u

/** Where members 1..N-1 sit. Member 0 keeps `PN_TT_CORE` (71). 96 is the PX7 placement:
 *  sixteen members run 96..110, clear of 72/73/74, attention at 82..84, the GDN at 90..95,
 *  head norm 91, and of head-norm-2 (111), the eltwise member (112) and gdnc (114..120),
 *  which moved WITH it. Was 75, which caps the group at 8 below attention. */
#define PN_TT_MATMUL_CORE_FIRST_ENV     "PN_TT_MATMUL_CORE_FIRST"
#define PN_TT_MATMUL_CORE_FIRST_DEFAULT 96u

/** OR an explicit list for members 1..N-1, "96-110,0-15" (pn_tt_core_list.h), which wins over
 *  PN_TT_MATMUL_CORE_FIRST. Unset by default. Decode frontier F1.1: a 32-member arm on the
 *  sixteen PX7 cores plus sixteen idle prefill-grid cores, which no single run reaches without
 *  crossing the GDN / gdnc / head-norm families. Exact by construction: the members' columns
 *  are the same N split whatever cores hold them. */
#define PN_TT_MATMUL_CORE_LIST_ENV "PN_TT_MATMUL_CORE_LIST"
/** The largest index the list may name (140 Tensix on a p150); a harvested part's missing
 *  cores are refused later, by the coordinate lookup, as for PN_TT_MATMUL_CORE_FIRST. */
#define PN_TT_MATMUL_CORE_INDEX_MAX 139u

/**
 * THE SHARED ACTIVATION, MULTICAST ONCE INSTEAD OF READ N TIMES. Off by default.
 *
 * At Mt = 1 every member re-reads the whole A column once per output column, so A is
 * 1.76 GB of the 2.70 GB a token reads — 65% — and every one of those reads goes to the
 * ONE DRAM bank holding that strip. Four cores buy 3.44x of card time and SIXTEEN buy
 * 4.80x; door rotation alone is worth 1.27x at N=8, and a gain that grows with N is what
 * an endpoint ceiling looks like. With this on, the leader reads A once and multicasts it.
 *
 * **IT MOVES THE MEMBER CORES, which is why it is opt-in rather than simply better.** A
 * multicast writes every node of its bounding box, and the default member set
 * {71, 75, 76, 77} is an arithmetic run whose box also contains 72, 73 and 74 — the
 * reduce, eltwise and sample residencies. So the members are placed as a RECTANGLE from
 * `PN_TT_CORE`, and `PN_TT_MATMUL_CORE_FIRST` is not consulted. Choose `PN_TT_CORE` so
 * the rectangle clears those three: on a 14-column part, 80 puts a 1x4 group at
 * translated x 13..16 of row 7 and a 2x4 group at rows 7-8, both clear of everything.
 *
 * `PN_TT_MATMUL_COLUMNS` overrides the width; by default it is the largest divisor of the
 * member count that is at most four, so N=4 is 1x4, N=8 is 2x4 and N=6 is 2x3. Four is the
 * standing hazard limit on destinations in ONE packet, and a row is one packet.
 */
#define PN_TT_MATMUL_COOPERATIVE_ENV "PN_TT_MATMUL_COOPERATIVE"
#define PN_TT_MATMUL_COLUMNS_ENV     "PN_TT_MATMUL_COLUMNS"

/**
 * PLACE the members as a rectangle WITHOUT cooperating.
 *
 * It exists to make the A/B honest. Cooperation moves the member cores, and "the cores
 * moved" and "the leader multicasts A" would otherwise be one change — so a speedup could
 * be either. With this set the cores are exactly where a cooperating group puts them and
 * every member still reads A from DRAM, which leaves the multicast as the only variable.
 * `PN_TT_MATMUL_COOPERATIVE=1` implies it.
 */
#define PN_TT_MATMUL_RECTANGLE_ENV "PN_TT_MATMUL_RECTANGLE"

/**
 * RED PROOF ONLY. `PN_TT_RED_COOP=stale` makes the LEADER skip the payload multicast while
 * still telling every receiver the block arrived, so each receiver computes from the
 * PREVIOUS block's bytes.
 *
 * One arm, two claims, which is why it is the cooperative red proof worth having:
 *  - the byte comparison can still go red on the cooperative path — the same question
 *    PN_TT_RED_SPLIT answers for the split path;
 *  - the multicast is the ONLY source of A on a receiver. A receiver that were still
 *    reading A from DRAM — a leftover read, a role that did not take, a branch that did
 *    not fold — would make this arm produce IDENTICAL output, and identical output under a
 *    deliberate break is exactly what a vacuous gate looks like.
 */
#define PN_TT_RED_COOP_ENV "PN_TT_RED_COOP"

/** Spins ONE rendezvous wait takes before abandoning. Bounded for the same reason every
 *  other wait here is: a wait that cannot expire is a wait only `assert_reset` can end,
 *  and asserting reset on a blocked triad is how cores were wedged. Sized well above a
 *  block's real cost and well below the idle bound, so a group that has fallen out of
 *  lockstep is reported in milliseconds rather than seconds. */
#define PN_TT_COOP_SPIN_LIMIT 20000000u

/**
 * RED PROOF ONLY. `PN_TT_RED_SPLIT` breaks the split on member 1 in one named way, on the
 * SHIPPING path, so the byte comparison against N=1 is proven able to go red on silicon:
 *
 *   base    member 1 reads member 0's weight panel — the "two units that share a resource
 *           will silently share a BASE ADDRESS too" defect, which produced a valid panel of
 *           the same matmul from a live endpoint with every status word clean
 *   out     member 1 writes over member 0's output panel — the overwrite that byte
 *           comparison exists to catch and that PCC scored as fine
 *   narrow  member 1 computes one column fewer, so exactly 32 output values of one op are
 *           stale. One element out of thousands, on one member of N, is the resolution a
 *           multicore gate has to have
 *
 * A knob rather than a test build, for the same reason `PN_TT_RED_BAD_DESCRIPTOR` is one:
 * the claim is about the shipping path, and a separate binary proves it about a different
 * program.
 */
#define PN_TT_RED_SPLIT_ENV "PN_TT_RED_SPLIT"

/** Microseconds a completion poll sleeps between device reads. 0 — the default and what
 *  every recorded figure was taken with — is the tightest loop. It exists to ask whether
 *  watching the core costs the core; see pn_tt_group_await(). */
#define PN_TT_POLL_BACKOFF_ENV "PN_TT_POLL_BACKOFF_US"

/** `PN_TT_ACTIVATION_DOORS=0` leaves every member on the activation bank's OWN endpoint
 *  (port 0), which is what a caller got before striping existed. It is the control that
 *  separates "the shared A operand is the binding read" from "N cores are N cores": the
 *  bytes, the addresses and the kernels are identical in both arms and only the door
 *  moves, so a difference can only be the endpoint ceiling. Default on. */
#define PN_TT_ACTIVATION_DOORS_ENV "PN_TT_ACTIVATION_DOORS"

/**
 * `PN_TT_MATMUL_BLOCK=N` makes the reader supply N output columns per pass instead of one.
 *
 * WHY IT EXISTS. At Mt = 1 the reader's A offset is `(mt*Kt + kt)` and does not depend on
 * `nt`, so the unblocked loop re-reads the whole A column ONCE PER OUTPUT COLUMN. Counted
 * from the descriptors that is **15.871 of the 24.303 GB a token reads, 65.3%**, and 74% of
 * it comes out of the single bank the `normed` strip lives in at 43.1 of that bank's 59.2
 * GB/s ceiling while the card as a whole sits at 16% of its 510.8. At N columns per pass A
 * costs 1/N of that: 12.400 GB/token at N = 4.
 *
 * WHY 4 IS THE CEILING. The block's N results live in DST for the whole K loop, and a DST
 * SECTION is `DEST_REGISTER_HALF_SIZE >> 1` rows with `dst_accum_mode 1` — **4 tiles of
 * 32x32** (ckernel.h `get_dest_max_tiles`, SyncHalf + ACCUM_MODE). The matmul CB table has
 * `dst_accum_mode 1` and it is worth 13-35x accuracy, so 8 is not available by dropping it.
 * A larger value would write outside the half with nothing refusing anywhere.
 *
 * IT IS A PER-RESIDENCY CONSTANT. CB_IN1's chunk becomes `depth * N` pages, and
 * `ft_tensix_cb_block_grid_is_safe()` refuses a chunk size that CHANGES through a buffer —
 * the write pointer wraps only on an exact hit of the limit. So the same value goes into
 * every descriptor and the CBs are sized for it; a weight whose Nt is not a multiple of N
 * takes a SHORT last block, which shortens the fill and the tile count but never the chunk.
 *
 * **DEFAULT 4, ON, since 2026-08-19.** Gated on the 48-token oracle and nothing softer: three
 * separate NB=4 runs and one NB=2 run are `cmp` IDENTICAL (189 B, md5 `80fc6259`) beside two
 * NB=1 controls, every one of them with zero residency re-arms and zero contention. Measured
 * at the shipped `PN_TT_MATMUL_CORES=4`, depth 8: **24.303 -> 12.409 GB/token, matmul card
 * work 0.2969 -> 0.1158 s (2.56x), wall 1.43 -> 1.92 tok/s.** `PN_TT_MATMUL_BLOCK=1` is the
 * control arm.
 */
#define PN_TT_MATMUL_BLOCK_ENV "PN_TT_MATMUL_BLOCK"

/**
 * `PN_TT_MATMUL_DST_TILES=8` says the ARMED matmul kernel was built at `DstSync::SyncFull`,
 * so `MB*NB` may reach eight instead of four (kernels/resident_matmul_dst_sync.h).
 *
 * IT IS AN ASSERTION ABOUT THE KERNEL, NOT A REQUEST OF IT, and that is why it is refused
 * unless `PN_TT_PROGRAM_DIR` is set. The embedded blob's matmul images are SyncHalf by
 * construction; against those, eight tiles is not a refusal anywhere — DST has no bounds
 * check, the pass writes over the half the packer is reading, and every kernel retires
 * clean with a smoothly wrong answer. The override is the only way an eight-tile kernel can
 * be on the card, so it is the only state in which the claim can be true.
 *
 * WHAT IT BUYS AT Mt=1, WHICH IS EVERY DESCRIPTOR: NB=8, `bytes = a/8 + b = 1344` against
 * the shipped `a/4 + b = 1600`, i.e. **1.19x on the matmul's DRAM bytes**. MB=2/NB=4's
 * 1.52x needs `Mt >= 2` and is a host-shape question, not this one.
 *
 * MEASURED ON THE SHIPPING SHAPE 2026-08-27, one binary, two program roots differing ONLY in
 * this define (their gdn images are byte-identical, checked): **A 4.051 -> 2.050 GB/token
 * (1.976x) with B UNMOVED at 8.585, total 12.635 -> 10.635 = 1.1881x**, decode
 * **0.1327/0.1325 -> 0.1295/0.1293 s/token, +2.47% (7.545 -> 7.725 tok/s)** against a 0.15%
 * control-vs-control drift, and the matmul's own WAIT 0.0575 -> 0.0542 s/token (-5.7%).
 * Prefill **166.35/166.49 -> 168.86 tok/s (+1.51%)** at a 0.084% drift, and the gain is
 * ENTIRELY in the `batched` term (3.4662 -> 3.3265, -4.03%) while sequential and transfer
 * move 0.10% and 0.07%. `cmp` IDENTICAL at 189 B / 80fc6259 on every arm; the two 256-token
 * soaks are byte-identical at 587 B / abca1fa9.
 *
 * AND THE SERIALISATION IS PRICED ON ITS OWN, ON THE SAME KERNEL. A SyncFull image held at
 * `PN_TT_MATMUL_BLOCK=4` reads the IDENTICAL 12.635 GB/token and runs 0.1331 against
 * SyncHalf's 0.1326 — **the MATH/PACK double buffer is worth 0.38% of decode wall**, against
 * the recorded HALF_SERIAL 0.52%. So the byte win alone is ~2.85% and 0.38 pp goes back.
 */
#define PN_TT_MATMUL_DST_TILES_ENV "PN_TT_MATMUL_DST_TILES"

/**
 * The shipped block: THE WHOLE DST SECTION, whatever the armed image's section is.
 *
 * It is not a sweep — NB=2 measures 0.1767 s of matmul card work against NB=4's 0.1158 and
 * NB=1's 0.2969, and the section is what caps it. What changed on 2026-08-27 is where the
 * section comes from. It used to be a CLAIM the operator made with `PN_TT_MATMUL_DST_TILES`,
 * unverifiable from the host, so the block deliberately did NOT follow it: naming the section
 * alone must not be able to move the shipped shape. It is now a FACT read off the armed
 * compute image's own recorded `defines=`, so following it is following the kernel that is on
 * the card. `PN_TT_MATMUL_BLOCK=4` remains the control arm on a SyncFull binary, and NB=8 is
 * worth A 4.051 -> 2.050 GB/token with B unmoved, 1.1881x on the matmul's DRAM bytes.
 */
#define PN_TT_MATMUL_BLOCK_DEFAULT 0u

/**
 * `PN_TT_MATMUL_ROWS=M` makes the reader hold M A tile-rows and stream B past ALL of them,
 * so the WEIGHT column is swept once for M output rows instead of once per row.
 *
 * IT IS THE MIRROR OF `PN_TT_MATMUL_BLOCK` AND THEY SHARE ONE CEILING. Over a descriptor:
 *
 *     A tiles = Mt * ceil(Nt/NB) * Kt        B tiles = ceil(Mt/MB) * Nt * Kt
 *     bytes   = Mt * T * (a/NB + b/MB)       T = sum of Kt*Nt
 *
 * and both blocks are live in DST across the whole K loop, so `MB * NB <= 4` — upstream's
 * own assert, `dst_index + ct_dim*rt_dim - 1 < get_dest_max_tiles<SyncHalf, ACCUM, Tile32x32>`
 * (llk_math_matmul_api.h), and `dst_accum_mode` is worth 13-35x accuracy so 8 is not
 * available by dropping it.
 *
 * **AND ON THIS MODEL IT IS WORTH TWO PER CENT.** The two blocks are symmetric in TILES and
 * NOT in BYTES: NB divides the operand whose tile is the bf16 ACTIVATION (a = 2048 B) and MB
 * divides the one whose tile is the WEIGHT (b = 1088 B at Bfp8_b, which is what Ornith's
 * Q8_0 loads as). At Bfp8_b the lattice reads 1600 / 1568 / 2320 bytes a unit of T for
 * (MB,NB) = (1,4) / (2,2) / (4,1), so the SHIPPED MB=1/NB=4 is already within 2.0% of the
 * best pair — and at Bfp4_b it IS the best pair and MB only loses. **MB=4/NB=1 serves 128
 * prompt positions a weight sweep and reads 45% MORE than today**, because the A side grows
 * by more than the B side shrank. `tt/tests/test_matmul_block.c` pins the tile law, the byte
 * law and the optimum at each of the three weight formats.
 *
 * THE LAW WAS VALIDATED AGAINST SILICON WITH NO FITTED CONSTANT: it predicts
 * `A/B = a/(4b) = 2048/4352 = 0.4706` at the shipped shape, and this file's own read report
 * measures A 3.978 GB / B 8.432 GB = 0.4718 on Ornith.
 *
 * **THE LEVER THAT WOULD MATTER IS A BIGGER DST, NOT A BIGGER MB.** At `MB*NB <= 8` the
 * Bfp8_b optimum is MB=2/NB=4 at 1056 bytes a unit — **1.52x** — and eight tiles is what
 * `DstSync::SyncFull` gives at the same `dst_accum_mode 1`. That is a compute-stack change
 * (three `ft_tensix_*` entry points encode SyncHalf) and it costs the MATH/PACK double
 * buffer. MB is its PREREQUISITE rather than its rival: at DST 8 the optimum needs both.
 *
 * L1 was never the binding constraint — at depth 8 the three CBs are
 * `MB*depth*2*2048 + depth*NB*2*tile + 4096` of a 1,351,680 B arena, which admits MB near
 * 39, nine times what DST allows.
 *
 * IT ONLY BITES AT `Mt > 1`. Every descriptor this file writes today carries `Mt = 1`
 * (`FT_TT_QUEUE_MATMUL_TILE_ROWS` below), which is a batch of 32 positions, and at Mt = 1 a
 * row block changes the CB sizing and the chunk the reader pushes and NOTHING ELSE. That is
 * what makes `PN_TT_MATMUL_ROWS=2` an INERTNESS arm for the new loops rather than a speed
 * arm: it must be `cmp` IDENTICAL.
 *
 * DEFAULT 1, OFF.
 */
#define PN_TT_MATMUL_ROWS_ENV "PN_TT_MATMUL_ROWS"
#define PN_TT_MATMUL_ROWS_DEFAULT 1u

/**
 * THE CARD-AGGREGATED COMPLETION, AND WHY IT IS OFF.
 *
 * The design is right and the mechanism already exists: every member's WRITER posts
 * {sequence, token} into its own slot of a block on member 0 (ft_tt_queue.h, the group
 * completion block; resident_matmul_writer.c calls ftq_group_post), so "has every member
 * finished item N" is ONE read of ONE core instead of N reads of N. That is what turns a
 * term linear in the member count into a nearly flat one.
 *
 * **IT WEDGES THE POSTING CORE, and that is measured, not suspected.** MEASURED
 * 2026-08-16, twice, on the shipping path: an N=2 decode with the post armed produced
 * byte-identical output, retired every residency cleanly by asking (state EXIT, stop=1,
 * head == tail == completed), and left **core 75 — the POSTER, not the destination —
 * unbootable at its NEXT boot**, `soft reset 0x00047000`, BRISC out of reset and the
 * subordinates never released. Against that: EIGHT arms at N = 1..8 with the post disarmed,
 * every one byte-identical and every core PASS on the probe afterwards.
 *
 * That is the `noc_inline_dw_write` failure family of DEV_NOTES.md §1 — clean retirement,
 * dead on the next boot — except the destination here is legal, 4-byte-aligned Tensix L1
 * and `ft_tensix_noc_inline_dword_write_l1()` refuses anything else. So the destination
 * class is not the whole story and diagnosing it is a KERNEL round: the suspects are the
 * shared per-(processor, NoC) inline staging slot and the two sent-counter drains
 * interacting with the matmul writer's own outstanding non-posted writes.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE PLACEMENT SAYS ABOUT THOSE TWO ARMS, 2026-08-19
 * ---------------------------------------------------------------------------
 * **Core 75 is not the leader and never was, and reading the recorded arms with that in
 * hand sharpens "two shapes, proof of neither" considerably.** Member 0 is
 * `device->core_index` (`PN_TT_CORE`, default **71**) and members 1..N-1 run upward from
 * `PN_TT_MATMUL_CORE_FIRST` (**75**) — see `pn_tt_matmul_group_configure()` below. So:
 *
 *   - N=2 is cores **71** (member 0, and where the group block lives) and **75** (member 1).
 *   - N=4 is **71, 75, 76, 77** — which is exactly the core set the post-DISARMED
 *     chat-parity wedge took, whole. That set is a fingerprint, and it identifies those
 *     numbers as this group rather than as an arbitrary four.
 *
 * **Both members post. Only one of them wedged, twice of two.** With the post armed, every
 * member's DM1 calls `ftq_group_post()`; member 0's destination is its OWN L1 and members
 * 1..N-1's is a REMOTE Tensix. The two armed wedges took core 75 — a member whose post
 * crosses the NoC — and left core 71, which posts to itself, PASS on the probe both times.
 *
 * So the discriminator the recorded data appeared to support was **remote vs local
 * destination**, not "the inline write" as a primitive.
 *
 * ---------------------------------------------------------------------------
 * THAT ARM WAS RUN ON 2026-08-20 AND IT REFUTED THE REMOTENESS READING
 * ---------------------------------------------------------------------------
 * `PN_TT_GROUP_POST_PROBE=self` at N=4 — every member posting into its OWN L1, the host
 * left on the per-member wait — ran byte-exact (`cmp` IDENTICAL, 0 refusals, 0 restarts,
 * 2.02 tok/s against a 2.02 disarmed control taken minutes earlier on the same four cores,
 * which probed 4 PASS) and then left **cores 71 AND 75 unbootable**: `soft reset
 * 0x00047000`, BRISC running with all four subordinates held, `instrn_buf_status = 0x66`,
 * 0/4 probe runs. Cores 76 and 77, local posters in the same arm, were PASS.
 *
 * **Core 71 is the refutation.** Member 0's destination is its own L1 under `remote` too,
 * so core 71's configuration in this arm is IDENTICAL to the one it survived twice — and
 * this time it wedged. "Local posting is safe, 2 of 2" was a 2-sample null, and the third
 * sample killed it. The wedge does NOT follow the destination.
 *
 * What survives: the post is an ACCELERANT of the resident queue's own "retired clean,
 * will not boot next" defect (DEV_NOTES.md §1), it hits some cores and not others in the
 * same arm, and neither remoteness nor member index predicts which. So **aggregating on a
 * core local to the writer is not a way out**, and the card-aggregated completion has no
 * cheap placement fix left. `PN_TT_FAST_COMPLETION`, which buys most of the same
 * transaction count with no cross-core write at all, is what shipped instead.
 *
 * It also says what the post is NOT: it is not the reason a group wedges at all. The
 * disarmed N=4 arm took all four cores after ~46x the work, so the base rate is not zero
 * and the post is at most an accelerant of a defect the resident queue already has
 * (DEV_NOTES.md §1, "RETIRED CLEAN, WILL NOT BOOT NEXT").
 *
 * **AND THE WHOLE POINT IS AVOIDABLE.** What the post buys is transaction count, and
 * `PN_TT_FAST_COMPLETION` below buys most of the same thing with NO cross-core write at
 * all: 3 transactions per member per flush become 1, against the post's `1 + ceil(8N/16)`
 * for the whole group. At N=8 that is 8 against 5 — the post is still better, and it is not
 * 4.8x better, which is what it has to be to be worth a core.
 *
 * So it is not a knob, it is a hazard behind its own spelling — the same shape as
 * `ft_tt_core probe --i-know-this-must-be-a-vector-add-triad`. `1` is REFUSED, because a
 * `1` is what someone types when they have not read this.
 */
#define PN_TT_GROUP_COMPLETION_ENV "PN_TT_GROUP_COMPLETION"
#define PN_TT_GROUP_COMPLETION_ACK "i-know-this-wedges-the-posting-core"

/**
 * THE DESTINATION ARM — is the wedge the REMOTENESS or the CALL?
 * `PN_TT_GROUP_POST_PROBE=self-i-know-this-may-wedge-a-core` or `remote-...`, default OFF.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE RECORDED DATA CAN AND CANNOT SAY
 * ---------------------------------------------------------------------------
 * With the post armed EVERY member's DM1 calls `ftq_group_post()`. Member 0's destination
 * is its own L1; members 1..N-1 write across the NoC. The two armed wedges took core 75, a
 * remote poster, and core 71 was PASS both times under identical code, the same shared
 * inline staging slot, the same two sent-counter drains and the same barrier.
 *
 * **2 of 2, and confounded.** Core 71 is also member 0, also the device's own core, also
 * the one the host polls first, and also the one whose queue the group block lives in. Any
 * of those could carry the difference. So the recorded arms are consistent with
 * "remoteness" and do not isolate it.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS ISOLATES
 * ---------------------------------------------------------------------------
 * ONE variable: the coordinates in the inline write.
 *
 *   `self`    every member, member 0 included, posts into its OWN L1 at the same offset.
 *   `remote`  every member posts to member 0 — the recorded configuration.
 *
 * Same kernel image, same processor, same descriptor stream, same member set, same slot
 * arithmetic, same barrier. If `self` leaves every core bootable and `remote` wedges a
 * remote poster, the destination IS the mechanism and a card-aggregated completion has a
 * path: aggregate on the core that is local to the writer, or do not aggregate at all.
 * If `self` wedges too, the destination is NOT the mechanism, the inline write is, and the
 * post is dead as a design rather than as a placement.
 *
 * ---------------------------------------------------------------------------
 * WHY THE HOST DOES NOT MOVE WITH IT
 * ---------------------------------------------------------------------------
 * Both arms leave the host on the per-member completion wait. Under `self` the slots are
 * scattered across N cores and an aggregated read could not work at all; under `remote` it
 * could, but then the two arms would differ in two places instead of one. Keeping the host
 * fixed also makes both arms byte-exact runs, so `cmp` against the oracle is a real check
 * on each — which the aggregated-read arms never had for the members' own answers.
 *
 * The cost of being wrong is a Tensix until the next rung-3 reset, which is why the value
 * is a sentence and not a `1`, and why the base rate matters: the post-DISARMED N=4 arm
 * took all four cores after ~46x the work, so the resident queue has a non-zero wedge rate
 * of its own and ONE clean `self` arm is not proof of safety.
 *
 * ---------------------------------------------------------------------------
 * THE ANSWER, 2026-08-20: `self` WEDGED. IT IS THE CALL, NOT THE DESTINATION.
 * ---------------------------------------------------------------------------
 * One `self` arm at N=4 was enough. The disarmed control on the same four cores minutes
 * earlier probed 4 PASS; the `self` arm ran byte-exact and left cores 71 and 75 at
 * `0x00047000` / `instrn_buf_status 0x66`, 0/4 probe runs, with 76 and 77 PASS. **Core 71
 * posts to its own L1 under BOTH arms, so its configuration was unchanged from the two it
 * survived — and it wedged.** The remote arm was never run: it would have cost a third
 * core to confirm a discriminator that had just been refuted.
 *
 * This knob stays, because the next question about the post is which of the inline write's
 * pieces does it (the shared per-(processor, NoC) staging slot, the sent-counter drains,
 * the barrier against the writer's own outstanding non-posted writes), and that question is
 * asked by arms of exactly this shape. It is NOT a placement knob any more.
 */
#define PN_TT_GROUP_POST_PROBE_ENV        "PN_TT_GROUP_POST_PROBE"
#define PN_TT_GROUP_POST_PROBE_SELF_ACK   "self-i-know-this-may-wedge-a-core"
#define PN_TT_GROUP_POST_PROBE_REMOTE_ACK "remote-i-know-this-wedges-the-posting-core"

/**
 * RED PROOF ONLY. `PN_TT_RED_GROUP=early` accepts the sequence BEFORE the one just issued,
 * so the host stops waiting one descriptor short.
 *
 * It exists because byte-exactness at N > 1 is only evidence if the wait is real. A gate
 * that would be green with NO wait at all is measuring the card's timing, not the protocol
 * — and this backend has caught four vacuous gates. One descriptor early is bounded (the
 * ring still holds the chain, nothing is overrun) and it is exactly the question: does the
 * dump of a projection the card has not finished differ from the one it has?
 */
#define PN_TT_RED_GROUP_ENV "PN_TT_RED_GROUP"

/**
 * THE PER-MEMBER WAIT IN ONE TRANSACTION INSTEAD OF THREE. `PN_TT_FAST_COMPLETION=1`,
 * DEFAULT OFF.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE THREE ARE
 * ---------------------------------------------------------------------------
 * `pn_tt_group_await()` asks a member two questions in three device accesses:
 *
 *   1. `read_field(COMPLETED_COUNT)`      one 4-byte UC read, the poll
 *   2. `read_completion_wide()`           words 4..11, 32 bytes, TWO BLOCK16 transactions
 *
 * and for members 1..N-1 the poll almost always succeeds on its FIRST iteration — they run
 * concurrently with member 0 and member 0 is awaited first, so by the time their turn comes
 * they are already finished. Their three transactions are therefore ~10 us of host time
 * that overlaps NO card work, once per member per flush. That is the per-member term.
 *
 * ---------------------------------------------------------------------------
 * WHY ONE IS THE FLOOR WITHOUT A CROSS-CORE MECHANISM
 * ---------------------------------------------------------------------------
 * N members' control blocks are in N different cores' L1, so N transactions is the minimum
 * any host-side scheme can reach. `ft_tt_queue_read_completion_line()` reaches it: words
 * 4..7 are ONE 16-byte L1 line and carry `completed_count`, `completed_token` and `state`,
 * so the poll IS the check and there is no second read. The card-aggregated post is the
 * only thing that goes below N, and it costs a core (above).
 *
 * ---------------------------------------------------------------------------
 * WHAT IT GIVES UP, AND WHY THAT IS RECOVERABLE
 * ---------------------------------------------------------------------------
 * `error` is word 9, in the NEXT line. It is the fourth of rule 1's four agreements and
 * this path does not read it on the success side. It is still read:
 *
 *   - every 128th poll iteration, by `pn_tt_group_check_errors()`, which is unchanged;
 *   - on the timeout path;
 *   - and a REFUSAL cannot hide behind it, because a kernel that refuses a descriptor never
 *     publishes completion — `completed_count` stalls, the poll never succeeds, and the
 *     128th iteration names the error. The refusal path is the one this assertion exists
 *     for and the refusal path still runs through it.
 *
 * The state in which `error` is set AND completion is correct is a fault raised AFTER
 * `completed_count` was published, and the protocol has exactly one:
 * `ftq_group_post()`'s own refusal. So this path is REFUSED when the group post is armed —
 * the two are mutually exclusive by construction, not by discipline.
 *
 * ---------------------------------------------------------------------------
 * MEASURED ON SILICON 2026-08-20, AND IT IS NOW ON BY DEFAULT
 * ---------------------------------------------------------------------------
 * Sixteen arms — this flag and this flag plus the parameter cache, at N = 1, 2, 4, 8, 12,
 * 16, 24, 32 — every one `cmp` IDENTICAL to the 48-token oracle, 0 refusals, 0 residency
 * restarts, `company=0` on each. It moves **matmul card work, not `b`**: `b` is the ISSUE
 * coefficient and this is a WAIT mechanism, so the refit reads 0.00242 control against
 * 0.00245 fast. What it moves is `card(N)` — 0.0611 -> 0.0475 at 8 members, 0.0559 ->
 * 0.0348 at 12, 0.0595 -> 0.0307 at 16 — and the wall, 2.13 -> 2.19 tok/s at the shipped
 * member count.
 *
 * **The saving is 6.6-7.2 us per awaited member per flush, flat across 2..32 members**,
 * which is two transactions at ~3.5 us each and is the direct measurement of a link read.
 *
 * THE WC/BLOCK16 TIGHT LOOP IS SAFE, and this is what settles it. The fast poll reads
 * through the `bulk_l1` accessor (WC, BLOCK16) where the old one used UC/WORD32, and a
 * stale WC read costs an extra iteration rather than a wrong answer — but that had never
 * been done in a TIGHT loop here. The N=1 arm spins **1,048 poll iterations per flush,
 * 6.49 million reads in one run**, and is byte-identical. A stale line would have shown up
 * as a hang or a wrong answer at that rate and did not.
 */
#define PN_TT_FAST_COMPLETION_ENV "PN_TT_FAST_COMPLETION"
#define PN_TT_FAST_COMPLETION_DEFAULT 1u

/**
 * THE 64-BYTE PARAMETER BLOCK, WRITTEN ONCE INSTEAD OF EVERY PROJECTION.
 * `PN_TT_PARAM_CACHE`, **DEFAULT ON since 2026-08-20** — the header has said so since it
 * flipped and this line still said OFF, which is the kind of stale comment that gets a
 * shipped default measured as an arm.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE ISSUE SIDE COSTS AND WHERE IT GOES
 * ---------------------------------------------------------------------------
 * Issuing one projection to one member is three host writes: 64 bytes of parameters, a
 * 32-byte descriptor, a 4-byte head bump. The write direction is BANDWIDTH bound — ~53 MB/s
 * through the UC path — so it is the 100 bytes that convert, and the parameters are 64 of
 * them.
 *
 * `pn_tt_group_parameters()` derives those sixteen words from `(weight, input strip, output
 * strip, member)` plus the residency's own depth, column block and cooperative role. **Every
 * one of those is a constant of a decode run.** Weights are placed once at load and never
 * move; the activation strips are resident at fixed addresses and each weight is used with
 * exactly one (input, output) pair in the forward; depth and column block are per-residency
 * by construction. So the host writes the same 64 bytes to the same member ~24 times a
 * token, once per layer.
 *
 * ---------------------------------------------------------------------------
 * WHY IT IS CONTENT-ADDRESSED AND NOT KEYED ON THE WEIGHT
 * ---------------------------------------------------------------------------
 * "Each weight is used with one strip pair" is TRUE of this forward and is not enforced
 * anywhere, and a red arm perturbs individual words on purpose (`PN_TT_RED_SPLIT`,
 * `matmul_a_spread`, `matmul_a_offset`, the malformed-descriptor proof). A cache keyed on
 * the weight would serve a stale block under any of those, silently, and the failure would
 * look like a wrong answer with every status word clean — the exact class this backend uses
 * byte comparison to catch.
 *
 * Keyed on the sixteen WORDS, a hit means the block at that address holds exactly those
 * words. The hash is a bucket index; the memcmp is the claim. A collision costs a miss.
 *
 * ---------------------------------------------------------------------------
 * WHY A CACHED BLOCK IS SAFE AGAINST AN IN-FLIGHT DESCRIPTOR
 * ---------------------------------------------------------------------------
 * The rotating slots are indexed `descriptor index mod 8` precisely so a chained projection
 * cannot overwrite parameters the card has not read yet. A cached block is **never
 * rewritten and never evicted**, so that hazard does not exist for it: the block a
 * descriptor points at holds the same words for the life of the residency. The cache is
 * cleared at ARM, where no descriptor is in flight by construction.
 *
 * When the region fills (320 blocks; Ornith's whole matmul weight set is 249) a projection
 * falls back to the rotating slots and is exactly as expensive as it was. `cache_full`
 * counts that, so "it stopped paying" is visible rather than inferred.
 *
 * ---------------------------------------------------------------------------
 * MEASURED ON SILICON 2026-08-20 — AND THE PREDICTION WAS WRONG ABOUT WHY
 * ---------------------------------------------------------------------------
 * **`b` 0.00242 -> 0.00199, which is 1.21x and not the predicted 2.4x**, over eight member
 * counts each `cmp` IDENTICAL. 249 of 320 blocks used, 0 fell back at any member count,
 * 304,560 hits against 6,480 misses at 32 members — one miss per projection per member,
 * i.e. the first token and nothing after it. The mechanism works exactly as designed.
 *
 * **The prediction was wrong because the WRITE direction is TRANSACTION bound, not
 * bandwidth bound.** The ledger measured the two directly: 0.928 s of issue at 8 members
 * against a 53 MB/s bandwidth model's 0.150 s — a 6x miss — and 3.94 us per write against
 * a link that would have to be reading 25 B/us for the bandwidth model to hold. So a
 * projection's three writes cost by their COUNT, and the one this cache removes is the
 * CHEAP one: the 64-byte parameter block measures **2.08 us** against **4.84 us** for the
 * 32-byte descriptor and the 4-byte head bump. Taking 100 B to 36 B took the bytes by 2.8x
 * and the time by 1.21x, and the difference between those two numbers is the whole finding.
 *
 * ON BY DEFAULT because 1.21x of `b` is real, free and byte-exact — but the honest scoping
 * of the ISSUE side from here is a TRANSACTION-count one: the descriptor and its head bump
 * are 4.84 us each and there are two of them, which is where the remaining 0.00199 lives.
 */
#define PN_TT_PARAM_CACHE_ENV "PN_TT_PARAM_CACHE"
#define PN_TT_PARAM_CACHE_DEFAULT 1u

/** Open-addressed buckets. A power of two, and comfortably over the slot count so the
 *  table stays sparse — a full table would degrade the probe, not the answer. */
#define PN_TT_PARAM_CACHE_BUCKETS 1024u
#define PN_TT_PARAM_CACHE_EMPTY   0xFFFFu

/**
 * FNV-1a over the sixteen words. It is a BUCKET INDEX and nothing rests on it: the memcmp
 * behind it is what decides a hit, so a bad hash costs throughput and never correctness.
 *
 * RED PROOF `PN_TT_RED=param_cache_hash` collapses it to a constant, so every block lands
 * in one bucket and the probe walks them all. **That arm must be `cmp` IDENTICAL**, and
 * that is the whole claim of the previous paragraph stated in a form that can fail: if a
 * degenerate hash changes the ANSWER, then the hash was load-bearing and the design is
 * wrong. Its partner `param_cache_blind` removes the memcmp instead and must DIFFER.
 */
static uint32_t pn_tt_param_hash(const uint32_t *words, bool red_constant)
{
    const unsigned char *bytes = (const unsigned char *)words;
    uint32_t hash = 2166136261u;
    size_t index;

    if (red_constant)
        return 0u;
    for (index = 0; index < FT_TT_QUEUE_PARAM_WORDS * sizeof(uint32_t); index++) {
        hash ^= bytes[index];
        hash *= 16777619u;
    }
    return hash;
}

/** Release a member's cache. Idempotent — retire runs on partly-armed groups. */
static void pn_tt_param_cache_free(pn_tt_matmul_member *slot)
{
    free(slot->cache_words);
    free(slot->cache_slot);
    slot->cache_words = NULL;
    slot->cache_slot = NULL;
    slot->cache_used = 0u;
}

/** Allocate a member's cache. A failure is NOT fatal: the caller runs uncached. */
static void pn_tt_param_cache_alloc(pn_tt_matmul_member *slot)
{
    uint32_t bucket;

    pn_tt_param_cache_free(slot);
    slot->cache_words = calloc(FT_TT_QUEUE_PARAM_CACHE_SLOTS,
                               sizeof(uint32_t) * FT_TT_QUEUE_PARAM_WORDS);
    slot->cache_slot = malloc(sizeof(uint16_t) * PN_TT_PARAM_CACHE_BUCKETS);
    if (!slot->cache_words || !slot->cache_slot) {
        pn_tt_param_cache_free(slot);
        return;
    }
    for (bucket = 0; bucket < PN_TT_PARAM_CACHE_BUCKETS; bucket++)
        slot->cache_slot[bucket] = PN_TT_PARAM_CACHE_EMPTY;
}

/**
 * The address this member's card-side block for `parameters` is at, writing it first if it
 * is not there yet.
 *
 * @param out_wrote  set true when a 64-byte write was made, false on a hit
 * @return the address, or 0 when the caller must fall back to a rotating slot
 */
static void pn_tt_param_census(pn_tt_device *device, const uint32_t *parameters, bool full)
{
    pn_tt_matmul_group *group = &device->group;
    const uint32_t kt = parameters[FT_TT_QUEUE_MATMUL_TILE_INNER];
    const uint32_t nt = parameters[FT_TT_QUEUE_MATMUL_TILE_COLS];
    uint32_t row;

    for (row = 0; row < group->param_census_rows; row++)
        if (group->param_census[row].kt == kt && group->param_census[row].nt == nt)
            break;
    if (row == group->param_census_rows) {
        if (row >= 48u)
            return;
        memset(&group->param_census[row], 0, sizeof(group->param_census[row]));
        group->param_census[row].kt = kt;
        group->param_census[row].nt = nt;
        group->param_census_rows++;
    }
    if (full)
        group->param_census[row].fallbacks++;
    else
        group->param_census[row].misses++;
}

static uint32_t pn_tt_param_cache_address(pn_tt_device *device, pn_tt_matmul_member *slot,
                                          const uint32_t *parameters, bool *out_wrote)
{
    uint32_t bucket;
    uint32_t probe;
    uint32_t address = 0u;

    *out_wrote = false;
    if (!slot->cache_words || !slot->cache_slot)
        return 0u;

    bucket = pn_tt_param_hash(parameters, device->red_param_cache_hash) &
             (PN_TT_PARAM_CACHE_BUCKETS - 1u);
    for (probe = 0; probe < PN_TT_PARAM_CACHE_BUCKETS; probe++) {
        const uint32_t index = (bucket + probe) & (PN_TT_PARAM_CACHE_BUCKETS - 1u);
        const uint16_t entry = slot->cache_slot[index];

        if (entry == PN_TT_PARAM_CACHE_EMPTY) {
            /* A MISS. Take the next block if there is one; otherwise the caller uses a
             * rotating slot and pays what it paid before. */
            if (slot->cache_used >= FT_TT_QUEUE_PARAM_CACHE_SLOTS) {
                slot->cache_full++;
                pn_tt_param_census(device, parameters, true);
                if (pn_tt_env_u32("PN_TT_PARAM_CENSUS_DIFF", 0u) != 0u) {
                    uint32_t best = 17u, e, w;

                    for (e = 0; e < slot->cache_used && best == 17u; e++) {
                        uint32_t differ = 0u, at = 0u;

                        for (w = 0; w < FT_TT_QUEUE_PARAM_WORDS; w++)
                            if (slot->cache_words[e][w] != parameters[w]) {
                                differ++;
                                at = w;
                            }
                        if (differ == 1u)
                            best = at;
                    }
                    device->group.param_diff_word[best]++;
                }
                return 0u;
            }
            pn_tt_param_census(device, parameters, false);
            if (ft_tt_queue_write_param_cache(pn_tt_issue_access(&device->runtime,
                                                                 device->group.issue_shape),
                                              slot->queue, slot->cache_used, parameters,
                                              &address) != FT_TT_OK)
                return 0u;
            memcpy(slot->cache_words[slot->cache_used], parameters,
                   sizeof(uint32_t) * FT_TT_QUEUE_PARAM_WORDS);
            slot->cache_slot[index] = (uint16_t)slot->cache_used;
            slot->cache_used++;
            slot->cache_misses++;
            *out_wrote = true;
            return address;
        }
        /* THE MEMCMP IS THE CLAIM, not the hash. RED PROOF
         * `PN_TT_RED=param_cache_blind` removes it — paired with the constant hash every
         * projection then reuses block 0, so the arm MUST produce a `cmp` that DIFFERS. A
         * blind cache that still matched the oracle would mean the sixteen words never
         * varied, i.e. that this whole mechanism was measuring nothing. */
        if (device->red_param_cache_blind ||
            memcmp(slot->cache_words[entry], parameters,
                   sizeof(uint32_t) * FT_TT_QUEUE_PARAM_WORDS) == 0) {
            slot->cache_hits++;
            return FT_TT_QUEUE_PARAM_CACHE_ADDRESS(slot->queue->control_address, entry);
        }
    }
    return 0u;
}

void pn_tt_matmul_group_param_cache_reset(pn_tt_device *device)
{
    pn_tt_matmul_group *group;
    uint32_t member, bucket;

    if (!device || pn_tt_env_u32("PN_TT_PARAM_CACHE_RESET", 1u) == 0u)
        return;
    group = &device->group;
    for (member = 0; member < group->member_count; member++) {
        pn_tt_matmul_member *slot = &group->member[member];

        if (!slot->cache_slot)
            continue;
        for (bucket = 0; bucket < PN_TT_PARAM_CACHE_BUCKETS; bucket++)
            slot->cache_slot[bucket] = PN_TT_PARAM_CACHE_EMPTY;
        slot->cache_used = 0u;
    }
}

/** Ring slots per member. */
#define PN_TT_QUEUE_CAPACITY 8u

/** Parameter-block slots a descriptor may point at, and how many projections may be in
 *  flight before a flush is forced.
 *
 * ONE SLOT WAS ENOUGH ONLY WHILE THE HOST WAITED FOR EVERY MEMBER BEFORE ISSUING THE NEXT
 * PROJECTION. Chaining two descriptors through one parameter block would have the host
 * overwrite the first's parameters while the card was still reading them — a wrong matmul
 * with every status word clean, which is the failure class this backend uses byte
 * comparison against. The slot is the descriptor's own index modulo the count, so a block
 * is reused only after the ring itself has wrapped.
 */
#define PN_TT_QUEUE_PARAM_SLOTS 8u

/** Bounded residency (ft_tt_queue.h rule 3); neither may be 0. */
#define PN_TT_IDLE_SPIN_LIMIT    FT_TT_QUEUE_DEFAULT_IDLE_SPINS
#define PN_TT_CB_WAIT_SPIN_LIMIT 100000000u

/** Every processor of the triad. */
#define PN_TT_TRIAD_MASK                                                                  \
    ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1) | (1u << FT_TT_PROC_MATH0) |         \
     (1u << FT_TT_PROC_MATH1) | (1u << FT_TT_PROC_MATH2))

pn_tt_red_split pn_tt_matmul_group_red_mode(void)
{
    const char *value = getenv(PN_TT_RED_SPLIT_ENV);

    if (!value || !*value)
        return PN_TT_RED_SPLIT_NONE;
    if (strcmp(value, "base") == 0)
        return PN_TT_RED_SPLIT_BASE;
    if (strcmp(value, "out") == 0)
        return PN_TT_RED_SPLIT_OUT;
    if (strcmp(value, "narrow") == 0)
        return PN_TT_RED_SPLIT_NARROW;
    fprintf(stderr, "pn_tt: %s=%s is not base|out|narrow — ignored\n", PN_TT_RED_SPLIT_ENV,
            value);
    return PN_TT_RED_SPLIT_NONE;
}

/* ================================================================
 * Bring-up
 * ================================================================ */

/**
 * Which DOOR of a bank each member reads an activation strip through.
 *
 * Every member reads the SAME bytes at the SAME address, so this changes no data and no
 * kernel — it is the ~45 GB/s endpoint ceiling against the bank's ~64 and nothing else
 * (ft_tt_dram_stripe.h). `ft_tt_dram_stripe_choose_ports()` rather than `member % 3`
 * because the chooser is the one that knows the NoC-row cost order.
 */
static int pn_tt_group_activation_doors(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t banks[PN_TT_MATMUL_MAX_CORES];
    uint32_t bank, member;

    group->activation_bank_count = device->dram_bank_count;
    for (bank = 0; bank < device->dram_bank_count; bank++) {
        for (member = 0; member < group->member_count; member++)
            banks[member] = bank;
        if (ft_tt_dram_stripe_choose_ports(&device->runtime.grid, banks, group->member_count,
                                           FT_TT_COORD_TRANSLATED,
                                           group->activation_door[bank]) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: could not choose activation doors for bank %u\n", bank);
            return -1;
        }
        if (bank == 0)
            group->activation_row_collisions = ft_tt_dram_stripe_shared_rows(
                group->activation_door[bank], group->member_count, NULL, NULL);
    }
    return 0;
}

/**
 * The widest rectangle a group of `member_count` can be laid out in without any ROW
 * exceeding the four-destination limit: the largest divisor of the count that is at most
 * four.
 *
 * A row is one multicast packet, so the width IS the destination count of every packet
 * except the leader's own row. Choosing a divisor rather than `min(count, 4)` is what
 * keeps the layout a rectangle at counts like 6, where 1x6 would be one 5-destination
 * packet and 2x3 is two 3-destination ones.
 */
static uint32_t pn_tt_group_rectangle_columns(uint32_t member_count)
{
    uint32_t columns;

    for (columns = FT_TT_GROUP_BROADCAST_HAZARD_LIMIT; columns > 1u; columns--)
        if (member_count % columns == 0u)
            return columns;
    return 1u;
}

/**
 * Place the members of a COOPERATING group as a rectangle from member 0's core, and build
 * the broadcast partition over it.
 *
 * Every refusal below leaves `cooperative_placed` false and the caller falls back to the
 * ordinary arithmetic placement, because a group that cannot cooperate is still a perfectly
 * good SPLIT group — the multicast is a speed lever, not a correctness one. What must never
 * happen is cooperating over a set that is not a rectangle: the bounding box then contains
 * cores this group does not own, and a multicast writes every node of its box.
 */
static int pn_tt_group_place_rectangle(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    ft_tt_noc_coord members[FT_TT_RESIDENT_MULTICAST_MAX_MEMBERS];
    uint32_t columns = pn_tt_env_u32(PN_TT_MATMUL_COLUMNS_ENV, 0u);
    uint32_t column_count;
    uint32_t rows;
    uint32_t member;
    int rc;

    if (columns == 0u)
        columns = pn_tt_group_rectangle_columns(group->member_count);
    if (columns > FT_TT_GROUP_BROADCAST_HAZARD_LIMIT || group->member_count % columns != 0u) {
        fprintf(stderr, "pn_tt: %s=%u does not divide %u members into rows of at most %u\n",
                PN_TT_MATMUL_COLUMNS_ENV, columns, group->member_count,
                FT_TT_GROUP_BROADCAST_HAZARD_LIMIT);
        return -1;
    }
    rows = group->member_count / columns;
    column_count = ft_tt_coord_tensix_column_count(&device->runtime.grid);
    if (column_count == 0u) {
        fprintf(stderr, "pn_tt: the live grid reports no Tensix columns\n");
        return -1;
    }
    /* The live grid decides whether this block is a rectangle in NoC space, which it is
     * NOT wherever it would straddle the columns the ARC spine and the east DRAM column
     * occupy. That refusal is ft_tt_group_rectangle_coordinates()'s and it is the reason
     * the coordinates are asked for rather than computed here. */
    rc = ft_tt_group_rectangle_coordinates(&device->runtime.grid, device->core_index, rows,
                                           columns, members,
                                           FT_TT_RESIDENT_MULTICAST_MAX_MEMBERS);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: a %ux%u block at core %u is not a multicast rectangle "
                        "(rc=%d) — cooperation is off, the split still runs\n",
                rows, columns, device->core_index, rc);
        return -1;
    }
    rc = ft_tt_op_matmul_resident_multicast_plan(members, group->member_count, 0u,
                                                 PN_TT_COOP_SPIN_LIMIT, &group->multicast);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the broadcast partition over that block was refused "
                        "(rc=%d) — cooperation is off, the split still runs\n", rc);
        return -1;
    }

    for (member = 0u; member < group->member_count; member++) {
        pn_tt_matmul_member *slot = &group->member[member];

        slot->core_index = device->core_index + (member / columns) * column_count +
                           (member % columns);
        slot->core_x = (uint16_t)members[member].x;
        slot->core_y = (uint16_t)members[member].y;
    }
    group->cooperative_columns = columns;
    group->cooperative_placed = true;
    fprintf(stderr, "pn_tt: cooperative placement %ux%u from core %u — %u broadcast "
                    "packet(s) reach %u receivers\n",
            rows, columns, device->core_index, group->multicast.partition.rectangle_count,
            group->multicast.receiver_count);
    return 0;
}

/** The `-D` a SyncFull matmul compute image records in its own provenance. Spelled once. */
#define PN_TT_MATMUL_SYNC_FULL_DEFINE "-DFT_TT_MATMUL_DST_SYNC=1"

/**
 * Was the matmul compute image THIS PROCESS WILL ARM built at `DstSync::SyncFull`?
 *
 * ---------------------------------------------------------------------------
 * THE INTERLOCK INVERTED, AND WHY IT HAD TO BE
 * ---------------------------------------------------------------------------
 * This used to key on `PN_TT_PROGRAM_DIR` being set. That was a sound proxy for exactly one
 * reason — "the embedded blob's matmul is SyncHalf, so an eight-tile kernel can only be on
 * the card if you named a directory" — and it is a reason about the BLOB, not about DST. The
 * moment the blob is regenerated at SyncFull the argument REVERSES: the embedded default IS
 * the eight-tile kernel and a SyncHalf OVERRIDE DIRECTORY becomes the unsafe case, which the
 * old test cannot see at all, because it reads "a directory was named" as "safe to go wide".
 * Flipping its polarity would only move the hole; no version of "which directory did you
 * name" survives a regeneration.
 *
 * SO IT ASKS THE IMAGE. `ft_tt_program_recorded_defines()` reads the `defines=` field of the
 * armed program's own provenance record — embedded or override, whichever this process will
 * actually load — and that answer travels with the IMAGE instead of with the operator. A
 * record that predates the field, or a program with no record at all, is NOT SyncFull here:
 * it is a record that cannot answer, and reading "cannot answer" as permission is precisely
 * the silent failure this guard exists for.
 *
 * IT READS THE COMPUTE IMAGE because that is where the mode lives —
 * `kernels/resident_matmul_dst_sync.h` is included by `resident_matmul_compute.c` and by
 * nothing else in the tree. The one-shot `matmul_*` programs build `matmul_compute.c`, which
 * does not include it, so they can never be SyncFull; and they never consult
 * `dst_section_tiles`, which is what makes the first fact harmless rather than a second thing
 * to guard.
 */
static bool pn_tt_matmul_group_image_is_sync_full(const char **out_where)
{
    const char *root = pn_tt_program_override_root();
    char defines[512];
    int rc;

    *out_where = pn_tt_program_where(root);
    rc = ft_tt_program_recorded_defines(PN_TT_RESIDENT_PROGRAM, root, FT_TT_PROC_MATH0,
                                        defines, sizeof(defines));
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: %s carries no readable build record in the %s programs "
                        "(rc=%d) — reading that as DstSync::SyncHalf, because a record that "
                        "cannot answer must never be read as permission\n",
                PN_TT_RESIDENT_PROGRAM, *out_where, rc);
        return false;
    }
    return ft_tt_program_defines_contain(defines, PN_TT_MATMUL_SYNC_FULL_DEFINE) != 0;
}

/**
 * The DST section the armed matmul kernel was built for, in 32x32 tiles.
 *
 * THE IMAGE DECIDES THE DEFAULT AND THE KNOB CAN ONLY NARROW IT. Unset, this is whatever the
 * armed compute image RECORDS — eight at SyncFull, four otherwise — so shipping the wider
 * section is a property of the blob that is on the card rather than of a variable somebody
 * remembered to export. `=4` is always honoured, because four tiles fit any section, and that
 * is what makes a SyncHalf-shaped CONTROL ARM available on a SyncFull binary. `=8` against an
 * image that does not record SyncFull is REFUSED BY NAME and clamped to four.
 */
static uint32_t pn_tt_matmul_group_dst_section_tiles(void)
{
    const char *where = "none";
    const bool full = pn_tt_matmul_group_image_is_sync_full(&where);
    const uint32_t available =
        full ? FT_TT_MATMUL_DST_SECTION_TILES_FULL : FT_TT_MATMUL_DST_SECTION_TILES_HALF;
    const uint32_t asked = pn_tt_env_u32(PN_TT_MATMUL_DST_TILES_ENV, available);

    if (asked == FT_TT_MATMUL_DST_SECTION_TILES_HALF)
        return FT_TT_MATMUL_DST_SECTION_TILES_HALF;
    if (asked != FT_TT_MATMUL_DST_SECTION_TILES_FULL) {
        fprintf(stderr, "pn_tt: %s=%u is not %u (DstSync::SyncHalf) or %u (SyncFull) — using "
                        "%u\n", PN_TT_MATMUL_DST_TILES_ENV, asked,
                FT_TT_MATMUL_DST_SECTION_TILES_HALF, FT_TT_MATMUL_DST_SECTION_TILES_FULL,
                FT_TT_MATMUL_DST_SECTION_TILES_HALF);
        return FT_TT_MATMUL_DST_SECTION_TILES_HALF;
    }
    if (!full) {
        fprintf(stderr, "pn_tt: %s=%u REFUSED — the %s %s compute image does not record %s, "
                        "so eight tiles would write past the section with nothing refusing "
                        "anywhere. Using %u\n",
                PN_TT_MATMUL_DST_TILES_ENV, asked, where, PN_TT_RESIDENT_PROGRAM,
                PN_TT_MATMUL_SYNC_FULL_DEFINE, FT_TT_MATMUL_DST_SECTION_TILES_HALF);
        return FT_TT_MATMUL_DST_SECTION_TILES_HALF;
    }
    fprintf(stderr, "pn_tt: matmul DST section %u tiles (DstSync::SyncFull) — the %s %s "
                    "compute image records %s, so MB*NB may reach %u and the MATH/PACK double "
                    "buffer is GONE\n",
            FT_TT_MATMUL_DST_SECTION_TILES_FULL, where, PN_TT_RESIDENT_PROGRAM,
            PN_TT_MATMUL_SYNC_FULL_DEFINE, FT_TT_MATMUL_DST_SECTION_TILES_FULL);
    return FT_TT_MATMUL_DST_SECTION_TILES_FULL;
}

int pn_tt_matmul_group_configure(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t listed_cores[PN_TT_MATMUL_MAX_CORES];
    uint32_t first, member;
    int listed;

    memset(group, 0, sizeof(*group));
    group->class_clock = -1;
    group->member_count = pn_tt_env_u32(PN_TT_MATMUL_CORES_ENV, PN_TT_MATMUL_CORES_DEFAULT);
    device->poll_backoff_us = pn_tt_env_u32(PN_TT_POLL_BACKOFF_ENV, 0u);
    group->activation_doors = pn_tt_env_u32(PN_TT_ACTIVATION_DOORS_ENV, 1u) != 0u;
    group->dst_section_tiles = pn_tt_matmul_group_dst_section_tiles();
    /* 0 is not a legal block, so it is the sentinel for "the default, which is the section
     * the armed image records". Spelling it as a sentinel rather than as a constant is what
     * keeps ONE statement of the default — the alternative is a second copy of the SyncHalf
     * four that a SyncFull blob would silently leave behind. */
    group->column_block = pn_tt_env_u32(PN_TT_MATMUL_BLOCK_ENV, PN_TT_MATMUL_BLOCK_DEFAULT);
    if (group->column_block == 0u)
        group->column_block = group->dst_section_tiles;
    if (group->column_block == 0u || group->column_block > group->dst_section_tiles ||
        (group->column_block & (group->column_block - 1u)) != 0u) {
        fprintf(stderr, "pn_tt: %s=%u is not a power of two in 1..%u (the DST section at "
                        "dst_accum_mode 1) — using 1\n", PN_TT_MATMUL_BLOCK_ENV,
                group->column_block, group->dst_section_tiles);
        group->column_block = 1u;
    }
    if (group->column_block > 1u)
        fprintf(stderr, "pn_tt: matmul column block %u — the reader supplies %u output "
                        "columns a pass, so A is fetched Nt/%u times instead of Nt\n",
                group->column_block, group->column_block, group->column_block);
    group->row_block = pn_tt_env_u32(PN_TT_MATMUL_ROWS_ENV, PN_TT_MATMUL_ROWS_DEFAULT);
    if (group->row_block == 0u || (group->row_block & (group->row_block - 1u)) != 0u ||
        !ftmm_block_fits_section(group->row_block, group->column_block,
                                 group->dst_section_tiles)) {
        /* THE PRODUCT IS THE CEILING, not either factor: MB*NB output tiles are live in DST
         * across the whole K loop and DST has no bounds check. Refused here rather than
         * left to the kernel, because the host refusal is a message and the kernel's is a
         * stranded triad — but the kernel checks it too, for the descriptor this host did
         * not write. */
        fprintf(stderr, "pn_tt: %s=%u is not a power of two with %s=%u inside the %u-tile "
                        "DST section (dst_accum_mode 1) — using 1\n", PN_TT_MATMUL_ROWS_ENV,
                group->row_block, PN_TT_MATMUL_BLOCK_ENV, group->column_block,
                group->dst_section_tiles);
        group->row_block = 1u;
    }
    if (group->row_block > 1u)
        fprintf(stderr, "pn_tt: matmul ROW block %u — the reader holds %u A tile-rows and "
                        "sweeps B past all of them, so B is fetched Mt/%u times instead of "
                        "Mt. INERT at Mt=1, which is every descriptor today: it must be cmp "
                        "IDENTICAL.\n",
                group->row_block, group->row_block, group->row_block);
    group->red_a_spread = pn_tt_red("matmul_a_spread");
    group->red_a_offset = pn_tt_red("matmul_a_offset");
    if (group->red_a_spread)
        fprintf(stderr, "pn_tt: RED PROOF matmul_a_spread — every member's A block is read "
                        "from a DIFFERENT BANK at the same offset. THE ANSWER IS GARBAGE; "
                        "this arm times the A supply and nothing else.\n");
    if (group->red_a_offset)
        fprintf(stderr, "pn_tt: RED PROOF matmul_a_offset — every member's A block is read "
                        "from the SAME BANK 4 MiB apart. THE ANSWER IS GARBAGE; it separates "
                        "a channel ceiling from N readers on one 16 KiB window.\n");
    group->cooperative_requested = pn_tt_env_u32(PN_TT_MATMUL_COOPERATIVE_ENV, 0u) != 0u;
    group->rectangle_requested = group->cooperative_requested ||
                                 pn_tt_env_u32(PN_TT_MATMUL_RECTANGLE_ENV, 0u) != 0u;
    {
        const char *red = getenv(PN_TT_RED_COOP_ENV);

        group->red_coop_stale = red && strcmp(red, "stale") == 0;
        if (red && !group->red_coop_stale)
            fprintf(stderr, "pn_tt: %s=%s is not `stale` — ignored\n", PN_TT_RED_COOP_ENV, red);
        if (group->red_coop_stale)
            fprintf(stderr, "pn_tt: RED PROOF — the leader SKIPS the payload multicast; "
                            "every receiver computes from the previous block\n");
    }
    {
        const char *ask = getenv(PN_TT_GROUP_COMPLETION_ENV);

        group->group_completion = ask && strcmp(ask, PN_TT_GROUP_COMPLETION_ACK) == 0;
        if (ask && !group->group_completion && strcmp(ask, "0") != 0)
            fprintf(stderr, "pn_tt: %s=%s REFUSED — the card-side post leaves the POSTING "
                            "core unbootable at its next boot (measured twice, 2026-08-16); "
                            "the only accepted value is `%s`\n",
                    PN_TT_GROUP_COMPLETION_ENV, ask, PN_TT_GROUP_COMPLETION_ACK);
    }
    {
        const char *ask = getenv(PN_TT_GROUP_POST_PROBE_ENV);

        group->post_probe = PN_TT_GROUP_POST_PROBE_OFF;
        if (ask && strcmp(ask, PN_TT_GROUP_POST_PROBE_SELF_ACK) == 0)
            group->post_probe = PN_TT_GROUP_POST_PROBE_SELF;
        else if (ask && strcmp(ask, PN_TT_GROUP_POST_PROBE_REMOTE_ACK) == 0)
            group->post_probe = PN_TT_GROUP_POST_PROBE_REMOTE;
        else if (ask && strcmp(ask, "0") != 0)
            fprintf(stderr, "pn_tt: %s=%s REFUSED — the card-side post can leave the POSTING "
                            "core unbootable at its next boot; the only accepted values are "
                            "`%s` and `%s`\n",
                    PN_TT_GROUP_POST_PROBE_ENV, ask, PN_TT_GROUP_POST_PROBE_SELF_ACK,
                    PN_TT_GROUP_POST_PROBE_REMOTE_ACK);
    }
    /* Two mechanisms cannot both own where DM1 posts. The aggregated read needs every slot
     * in one block; the probe's whole point is that they are not. Dropping BOTH, for the
     * same reason the fast/aggregated pair drops both: a contradictory pair must not
     * resolve to the arm that costs a core to be wrong about. */
    if (group->post_probe != PN_TT_GROUP_POST_PROBE_OFF && group->group_completion) {
        fprintf(stderr, "pn_tt: %s and %s are mutually exclusive — both decide where DM1 "
                        "posts, and the aggregated READ requires every slot in one block. "
                        "BOTH are OFF; ask for one.\n",
                PN_TT_GROUP_POST_PROBE_ENV, PN_TT_GROUP_COMPLETION_ENV);
        group->post_probe = PN_TT_GROUP_POST_PROBE_OFF;
        group->group_completion = false;
    }
    if (group->post_probe != PN_TT_GROUP_POST_PROBE_OFF && group->member_count == 1u) {
        fprintf(stderr, "pn_tt: %s is meaningless at one member — there is no remote arm to "
                        "compare against. OFF.\n", PN_TT_GROUP_POST_PROBE_ENV);
        group->post_probe = PN_TT_GROUP_POST_PROBE_OFF;
    }
    group->fast_completion =
        pn_tt_env_u32(PN_TT_FAST_COMPLETION_ENV, PN_TT_FAST_COMPLETION_DEFAULT) != 0u;
    /* MUTUALLY EXCLUSIVE BY CONSTRUCTION, AND BOTH ARE DROPPED. The line read cannot see
     * `error`, and the group post is the ONE fault the protocol raises after completion is
     * already published — so arming both would silently drop the only assertion that could
     * catch it.
     *
     * BOTH, not "the fast one loses": the post costs a core, so a contradictory pair must
     * not resolve to the arm that is expensive to be wrong about. Silently keeping the post
     * armed here would make this very refusal a way to wedge a Tensix. */
    if (group->fast_completion && group->group_completion) {
        fprintf(stderr, "pn_tt: %s and %s are mutually exclusive — the one-line completion "
                        "read does not carry `error`, and the group post is the only fault "
                        "raised AFTER completion is published. BOTH are OFF; ask for one.\n",
                PN_TT_FAST_COMPLETION_ENV, PN_TT_GROUP_COMPLETION_ENV);
        group->fast_completion = false;
        group->group_completion = false;
    }
    group->param_cache =
        pn_tt_env_u32(PN_TT_PARAM_CACHE_ENV, PN_TT_PARAM_CACHE_DEFAULT) != 0u;
    /* THE ISSUE SHAPE, before any member arms. Each member then DEMOTES it if its own queue
     * is not 16-byte addressable, so the group reports one shape and it is the one every
     * member could actually use. */
    group->issue_shape = pn_tt_issue_shape_selected();
    device->red_param_cache_blind = pn_tt_red("param_cache_blind");
    /* `blind` implies the constant hash: on its own it would almost never take a wrong
     * entry, and an arm that fires "almost never" is not a red proof. */
    device->red_param_cache_hash = device->red_param_cache_blind ||
                                   pn_tt_red("param_cache_hash");
    if (device->red_param_cache_hash && !device->red_param_cache_blind)
        fprintf(stderr, "pn_tt: RED PROOF param_cache_hash — the cache hash is a CONSTANT, "
                        "so every block shares one bucket and the memcmp does all the work. "
                        "This arm must be `cmp` IDENTICAL.\n");
    if (device->red_param_cache_blind)
        fprintf(stderr, "pn_tt: RED PROOF param_cache_blind — constant hash AND no memcmp, "
                        "so every projection reuses block 0. THE ANSWER IS GARBAGE; this arm "
                        "must be `cmp` DIFFERS.\n");
    device->red_fast_check = pn_tt_red("fast_check");
    if (device->red_fast_check)
        fprintf(stderr, "pn_tt: RED PROOF fast_check — the one-line completion check is asked "
                        "about one descriptor MORE than the poll waited for. It must REFUSE "
                        "on the first flush; a run that completes proves the check is inert.\n");
    /* The two "armed" lines are AFTER the guard on purpose: a log that says a mechanism is
     * armed and then says it is off is a log nobody can grep. */
    if (group->group_completion)
        fprintf(stderr, "pn_tt: card-aggregated completion armed — THIS COSTS A CORE. "
                        "Recover with `ft_tt_core recover <n> --rung 3 --write "
                        "--then-probe` and retrain the link afterwards.\n");
    if (group->post_probe != PN_TT_GROUP_POST_PROBE_OFF)
        fprintf(stderr, "pn_tt: group post probe armed, destination %s — THIS MAY COST A "
                        "CORE. Every member's DM1 calls ftq_group_post(); the host stays on "
                        "the per-member wait, so the run is byte-exact either way. Recover a "
                        "wedged core with `ft_tt_core recover <n> --rung 3 --driver udrv "
                        "--write --then-probe` and retrain the link afterwards.\n",
                group->post_probe == PN_TT_GROUP_POST_PROBE_SELF ? "SELF (own L1)"
                                                                 : "REMOTE (member 0)");
    if (group->fast_completion)
        fprintf(stderr, "pn_tt: fast completion armed — ONE 16-byte transaction per member "
                        "per flush instead of three. `error` is checked every 128th poll "
                        "and on the timeout path, not on the success path.\n");
    if (group->param_cache)
        fprintf(stderr, "pn_tt: parameter cache armed — a projection's sixteen words are "
                        "written to a member ONCE and pointed at afterwards, %u blocks a "
                        "member. ISSUE falls from 100 B in three accesses to 36 B in two.\n",
                FT_TT_QUEUE_PARAM_CACHE_SLOTS);
    group->group_address = FT_TT_OP_QUEUE_BASE + FT_TT_QUEUE_GROUP_OFFSET;
    {
        const char *red = getenv(PN_TT_RED_GROUP_ENV);

        group->red_wait_early = red && strcmp(red, "early") == 0;
        if (red && !group->red_wait_early)
            fprintf(stderr, "pn_tt: %s=%s is not `early` — ignored\n", PN_TT_RED_GROUP_ENV, red);
        if (group->red_wait_early)
            fprintf(stderr, "pn_tt: RED PROOF — the group wait accepts one descriptor EARLY\n");
    }
    if (group->member_count == 0u || group->member_count > PN_TT_MATMUL_MAX_CORES) {
        fprintf(stderr, "pn_tt: %s=%u is outside 1..%u\n", PN_TT_MATMUL_CORES_ENV,
                group->member_count, PN_TT_MATMUL_MAX_CORES);
        return -1;
    }
    first = pn_tt_env_u32(PN_TT_MATMUL_CORE_FIRST_ENV, PN_TT_MATMUL_CORE_FIRST_DEFAULT);
    {
        const char *list = getenv(PN_TT_MATMUL_CORE_LIST_ENV);

        listed = pn_tt_core_list_parse(list, listed_cores, PN_TT_MATMUL_MAX_CORES,
                                       PN_TT_MATMUL_CORE_INDEX_MAX);
        if (listed < 0 || (listed > 0 && (uint32_t)listed < group->member_count - 1u)) {
            fprintf(stderr, "pn_tt: %s=%s is not a list of %u distinct cores for members "
                            "1..%u\n", PN_TT_MATMUL_CORE_LIST_ENV, list,
                    group->member_count - 1u, group->member_count - 1u);
            return -1;
        }
        if (listed > 0)
            fprintf(stderr, "pn_tt: matmul members 1..%u from %s=%s\n",
                    group->member_count - 1u, PN_TT_MATMUL_CORE_LIST_ENV, list);
    }

    /* Member 0 is the core pn_tt_bring_up_core() already resolved and booted, and its queue
     * is the device's own. Nothing about it changes when the group is one member wide. */
    group->member[0].core_index = device->core_index;
    group->member[0].core_x = device->core_x;
    group->member[0].core_y = device->core_y;
    group->member[0].queue = &device->queue;
    group->member[0].booted = true;

    /* THE PLACEMENT IS THE COOPERATION DECISION, and it is taken before anything is
     * booted. A cooperating group must be a rectangle; an ordinary split group is an
     * arithmetic run, which is cheaper to place and is what every recorded number was
     * taken with. A rectangle that cannot be built here is reported and falls back — the
     * multicast is a speed lever, not a correctness one. */
    if (group->rectangle_requested && group->member_count > 1u &&
        pn_tt_group_place_rectangle(device) != 0)
        fprintf(stderr, "pn_tt: a rectangle was asked for but could not be placed; running "
                        "as an ordinary split\n");

    for (member = 1u; member < group->member_count; member++) {
        pn_tt_matmul_member *slot = &group->member[member];
        ft_tt_noc_coord core;

        slot->queue = &slot->own_queue;
        if (!group->cooperative_placed)
            slot->core_index = listed > 0 ? listed_cores[member - 1u] : first + member - 1u;
        if (ft_tt_coord_tensix_by_index(&device->runtime.grid, slot->core_index,
                                        FT_TT_COORD_TRANSLATED, &core) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: Tensix index %u is not on this part\n", slot->core_index);
            return -1;
        }
        /* Cross-checked rather than assigned when the rectangle placed them: the
         * coordinates came from ft_tt_group_rectangle_coordinates() and the INDEX from
         * this file's own arithmetic, so a disagreement means one of the two is wrong
         * about the live grid — and the multicast rectangle is built from the
         * coordinates. */
        if (group->cooperative_placed &&
            (slot->core_x != (uint16_t)core.x || slot->core_y != (uint16_t)core.y)) {
            fprintf(stderr, "pn_tt: member %u core %u is (%u,%u) by index and (%u,%u) by "
                            "rectangle\n", member, slot->core_index, core.x, core.y,
                    slot->core_x, slot->core_y);
            return -1;
        }
        slot->core_x = (uint16_t)core.x;
        slot->core_y = (uint16_t)core.y;
        /* THE CARD IS ASKED WHETHER THIS CORE IS FREE, BEFORE THE BOOT. The group's members
         * are the one set that MOVES with a knob, so they land on cores other families held
         * in an earlier run — and a compute thread left spinning by that run outlives the
         * process that started it. `pn_tt_core_boot_guard.h` is the whole argument. */
        if (!pn_tt_core_boot_is_permitted(&device->runtime, slot->core_index,
                                          "a matmul group member",
                                          PN_TT_MATMUL_CORE_FIRST_ENV))
            return -1;
        /* SEQUENTIALLY, from this one process. Booting firmware on many cores from
         * independent processes returns `device I/O failed` from 12-way up and leaves cores
         * mid-handshake (foundry/tt/DEV_NOTES.md §1); sequential boots are the safe case. */
        if (ft_tt_runtime_boot_core(&device->runtime.noc, slot->core_x, slot->core_y,
                                    &device->firmware) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: matmul member %u core %u (%u,%u) firmware boot failed\n",
                    member, slot->core_index, core.x, core.y);
            return -1;
        }
        slot->booted = true;
        fprintf(stderr, "pn_tt: matmul member %u = core %u at translated (%u,%u)\n", member,
                slot->core_index, core.x, core.y);
    }

    if (group->member_count > 1u && pn_tt_group_activation_doors(device) != 0)
        return -1;
    return 0;
}

/* ================================================================
 * The residencies
 * ================================================================ */

/** Place the triad's three circular buffers: two blocks of `depth` pages on each input so
 *  the reader fills one while the compute thread drains the other, two pages out.
 *
 *  The page COUNT is `2 * depth` for a reason that outlives this function: the kernel's own
 *  guard demands `depth` divide the page count AND the write pointer stay on the block
 *  grid, and with one constant depth through a buffer of exactly two blocks both hold for
 *  every descriptor, forever, with no state to carry between them. */
/* M4 (2026-10-04): the pipelined reader and 3-chunk matmul CBs, a MODEL preference
 * (pn_tt_device_prefer_matmul_pipe; Qwen3.5 sets it) with env overrides `PN_TT_MATMUL_PIPE`
 * and `PN_TT_MATMUL_CB_DEEP`. Read once; the residency is armed once per process shape. */
static uint32_t g_matmul_pipe_preferred = 0u;

void pn_tt_device_prefer_matmul_pipe(pn_tt_device *device, bool pipe)
{
    (void)device;
    g_matmul_pipe_preferred = pipe ? 1u : 0u;
}

static int pn_tt_matmul_pipe(void)
{
    return (int)pn_tt_env_u32("PN_TT_MATMUL_PIPE", g_matmul_pipe_preferred);
}

static uint32_t pn_tt_matmul_cb_deep(void)
{
    return pn_tt_env_u32("PN_TT_MATMUL_CB_DEEP", g_matmul_pipe_preferred ? 3u : 2u) >= 3u ? 3u
                                                                                         : 2u;
}

/* M5 DIAGNOSIS: `PN_TT_MATMUL_ROW0_AB=N` runs the first N single-row projections twice —
 * row-0 forced OFF, then ON — and prints the two outputs' row-0 difference. -1 = no override. */
static int g_row0_override = -1;

static uint32_t g_stash_address;

/**
 * BYTES A-REUSE (`PN_TT_A_REUSE=<tiles>`, default 128, 0 off): CB_IN0's ring is sized to exactly that many
 * pages on the decode members, and descriptor word 13 bit 18 asks the reader to skip the A
 * read on every pass over a row after its first. The KERNEL decides per descriptor (ring ==
 * Kt and MB == 1), so a projection of another Kt reads A as before. 128 = Ornith's hidden Kt:
 * gate/up (3 passes a member), qkv (2) and the LM head (61) read A once.
 */
/* BYTES A-REUSE ring (pages): the kernel skips a pass's A read only when the ring is EXACTLY
 * the descriptor's Kt, so the ring is a MODEL preference — its hidden Kt, the K of most of its
 * bytes (pn_tt_device_prefer_a_reuse_tiles; Qwen3.5 sets n_embd / 32). Default 128 (Ornith's
 * hidden Kt); `PN_TT_A_REUSE` overrides, 0 off. */
static uint32_t g_a_reuse_preferred = 0u;

void pn_tt_device_prefer_a_reuse_tiles(pn_tt_device *device, uint32_t tiles)
{
    (void)device;
    g_a_reuse_preferred = tiles;
}

static uint32_t pn_tt_a_reuse_tiles(void)
{
    static int env = -2; /* -2 unread, -1 unset, else the override */

    if (env == -2)
        env = getenv("PN_TT_A_REUSE") ? (int)pn_tt_env_u32("PN_TT_A_REUSE", 0u) : -1;
    return pn_tt_a_reuse_ring(env >= 0, env >= 0 ? (uint32_t)env : 0u, g_a_reuse_preferred);
}

static int pn_tt_group_place_cbs(uint32_t weight_tile_bytes, uint32_t depth,
                                 uint32_t column_block, uint32_t row_block,
                                 unsigned char *cb_buffer, size_t cb_bytes,
                                 uint64_t *cb_mask, uint32_t stash_tiles)
{
    ft_tt_op_cb_request requests[7];
    ft_tt_launch_cb_entry placed[7];
    uint32_t count;
    int rc;

    /* THE ACTIVATION CB CARRIES THE WHOLE ROW BLOCK, exactly as CB_IN1 carries the whole
     * column block and for the same grid reason: `depth * MB` pages a pass, two passes deep,
     * and the kernel's own guard refuses any descriptor whose `depth * MB` does not divide
     * this count. At MB = 1 it is the `depth * 2` it has always been. */
    requests[0].index = 0u;
    requests[0].page_bytes = PN_TT_TILE_BF16;
    /* M4: `PN_TT_MATMUL_CB_DEEP` chunks (default 2) — 3 lets the reader run two blocks ahead
     * of the compute thread (the pipelined reader needs it to keep one block BUFFERED while
     * two are in flight). */
    /* PHASE G option 3: a GRID core (stash_tiles != 0) may run shallower CBs
     * (`PN_TT_PREFILL_GRID_DEEP`, default the decode depth) — its B comes from the local
     * stash, and two chunks instead of three leave ffn_down's 1,152-tile stash 8 KB to spare. */
    const uint32_t deep = stash_tiles != 0u
                              ? (pn_tt_env_u32("PN_TT_PREFILL_GRID_DEEP", 2u) >= 3u ? 3u : 2u)
                              : pn_tt_matmul_cb_deep();

    requests[0].page_count = depth * row_block * deep;
    /* A-REUSE: the ring is Kt pages, never less than the pipelined minimum, and a whole
     * number of chunks (the kernel's push-grid guard). Not on a grid core (stash_tiles). */
    if (pn_tt_a_reuse_tiles() != 0u && stash_tiles == 0u && row_block == 1u &&
        pn_tt_a_reuse_tiles() >= requests[0].page_count &&
        pn_tt_a_reuse_tiles() % (depth * row_block) == 0u)
        requests[0].page_count = pn_tt_a_reuse_tiles();
    /* THE WEIGHT CB CARRIES THE WHOLE COLUMN BLOCK: `depth * NB` pages a pass, two passes
     * deep. The kernel's own grid guard reads this page count and refuses any descriptor
     * whose `depth * NB` does not divide it, so a mismatch between this line and the
     * descriptor word is a loud refusal rather than a write off the end of the buffer. */
    requests[1].index = 1u;
    requests[1].page_bytes = weight_tile_bytes;
    requests[1].page_count = depth * column_block * deep;
    requests[2].index = 16u;
    requests[2].page_bytes = PN_TT_TILE_BF16;
    requests[2].page_count = 2u;
    /* PHASE G0: the B STASH, a CB no kernel names (index 7), so the placement reserves it. */
    requests[3].index = 7u;
    requests[3].page_bytes = weight_tile_bytes;
    requests[3].page_count = stash_tiles;
    count = stash_tiles ? 4u : 3u;
    /* S3b: the fused SwiGLU program's STAGE and PANEL CBs (resident_matmul_swiglu_protocol.h),
     * on a decode member only — a grid core never runs a fused descriptor and its stash keeps
     * the L1. Placed after every other CB, so cb 0/1/16 land where they always have. */
    if (pn_tt_fused_program_armed() && stash_tiles == 0u) {
        requests[count].index = FT_TT_MATMUL_SWIGLU_CB_STAGE;
        requests[count].page_bytes = PN_TT_TILE_BF16;
        requests[count].page_count = FT_TT_MATMUL_SWIGLU_STAGE_PAGES;
        count++;
        requests[count].index = FT_TT_MATMUL_SWIGLU_CB_PANEL;
        requests[count].page_bytes = PN_TT_TILE_BF16;
        requests[count].page_count = ftsw_panel_pages(column_block);
        count++;
    }
    /* S3a: the RESIDUAL CB the RESADD reader lands a column block's residual tiles in, two
     * blocks deep (resident_matmul_resadd_protocol.h); decode members only, after the rest. */
    if (pn_tt_fuse_resadd_mode() && stash_tiles == 0u) {
        requests[count].index = FT_TT_MATMUL_RESADD_CB_RESIDUAL;
        requests[count].page_bytes = PN_TT_TILE_BF16;
        requests[count].page_count = ftra_residual_cb_pages(column_block);
        count++;
    }
    rc = ft_tt_op_cb_place(requests, count, placed, cb_buffer, cb_bytes, cb_mask);
    if (rc == FT_TT_OK && stash_tiles)
        g_stash_address = placed[3].address;
    return rc;
}

/** Stage and KICK one member: five images, the CB descriptor array, five runtime-arg
 *  blocks, the launch message, GO — and then the host walks away. */
static int pn_tt_slot_start(pn_tt_device *device, pn_tt_matmul_member *slot, uint32_t member,
                            uint32_t weight_tile_bytes, uint32_t depth, bool grid);
/** Members that completed work (all of them when `all`) are alive at `now`; the group's
 *  `last_alive_seconds` is then its OLDEST running member's (see the header). */
static void pn_tt_group_mark_alive(pn_tt_matmul_group *group, double now, bool all)
{
    double oldest = now;
    uint32_t member;

    for (member = 0; member < group->member_count; member++) {
        if (all || group->pending[member])
            group->member_alive_seconds[member] = now;
        if (group->member[member].running && group->member_alive_seconds[member] < oldest)
            oldest = group->member_alive_seconds[member];
    }
    group->last_alive_seconds = oldest;
}

static int pn_tt_grid_flush(pn_tt_device *device);
static void pn_tt_grid_stop(pn_tt_device *device);

static int pn_tt_group_member_start(pn_tt_device *device, uint32_t member,
                                    uint32_t weight_tile_bytes, uint32_t depth)
{
    return pn_tt_slot_start(device, &device->group.member[member], member, weight_tile_bytes,
                            depth, false);
}

/** Stage and KICK one resident matmul triad on `slot`. `grid` = a PHASE G prefill grid core:
 *  no group completion, no post probe, never cooperative, and a B stash CB. */
/** A grid core's own idle bound (q38-tt-grid-abort): `PN_TT_GRID_IDLE_SPINS` (default the
 *  decode members' 400M, ~8 s). A gate knob: a stalled host outlasts the default and not
 *  2,000,000,000 (~40 s), which says the cores' own exit is what the stall reached. */
static uint32_t pn_tt_grid_idle_spins(void)
{
    static uint32_t spins;

    if (spins == 0u) {
        spins = pn_tt_env_u32("PN_TT_GRID_IDLE_SPINS", PN_TT_IDLE_SPIN_LIMIT);
        if (spins == 0u)
            spins = PN_TT_IDLE_SPIN_LIMIT;
        if (spins != PN_TT_IDLE_SPIN_LIMIT)
            fprintf(stderr, "pn_tt: prefill grid idle bound %u spins (PN_TT_GRID_IDLE_SPINS)\n",
                    spins);
    }
    return spins;
}

static int pn_tt_slot_start(pn_tt_device *device, pn_tt_matmul_member *slot, uint32_t member,
                            uint32_t weight_tile_bytes, uint32_t depth, bool grid)
{
    pn_tt_matmul_group *group = &device->group;
    static const unsigned processors[5] = { FT_TT_PROC_DM0, FT_TT_PROC_DM1, FT_TT_PROC_MATH0,
                                            FT_TT_PROC_MATH1, FT_TT_PROC_MATH2 };
    /* kArgCoopCount words, not FT_TT_QUEUE_KERNEL_ARG_COUNT: the reader takes the queue's
     * ten plus the cooperative tail. The other four processors are still HANDED ten — the
     * tail is DM0's alone, and `ftq_context_init()` reads exactly the ten it always did, so
     * the other four resident families' images are untouched by its existing. */
    uint32_t args[5][kArgCoopCount];
    ft_tt_op_args blocks[5];
    unsigned index;
    int rc;

    /* STEP 3 B: members 1.. share ONE nonce (`PN_TT_DISPATCH_MCAST`, default 1), so their ring words are
     * byte-identical and the puller can multicast them. A token still protects each queue
     * against its own stale slots; it never compared across cores. */
    {
        static uint32_t shared_nonce;
        const char *mcast = getenv("PN_TT_DISPATCH_MCAST");
        uint32_t nonce = ft_tt_queue_make_nonce();

        if (!grid && member >= 1u && (!mcast || atoi(mcast) > 0)) {
            if (member == 1u || shared_nonce == 0u)
                shared_nonce = nonce;
            nonce = shared_nonce;
        }
        rc = ft_tt_queue_init(&device->runtime.noc, slot->core_x, slot->core_y,
                              FT_TT_OP_QUEUE_BASE, PN_TT_QUEUE_CAPACITY, nonce, slot->queue);
    }
    if (rc != FT_TT_OK)
        return rc;
    /* M2: the done-sequence word (control byte 0xE0) is NOT in the 16 words queue_init
     * clears, and a stale value from the previous arm would satisfy member 0's pull at a
     * sequence that restarts at 1. Zeroed BEFORE the kernel launches. */
    {
        uint32_t zeros[4] = {0u, 0u, 0u, 0u};

        rc = device->runtime.noc.write(device->runtime.noc.user, slot->core_x, slot->core_y,
                                       FT_TT_OP_QUEUE_BASE + 0xE0u, zeros, sizeof(zeros));
        if (rc != FT_TT_OK)
            return rc;
    }
    /* S2: the wait sidecar, which a re-armed queue's indices would otherwise meet again. */
    rc = pn_tt_residency_clear_precondition(&device->runtime.noc, slot->core_x, slot->core_y);
    if (rc != FT_TT_OK)
        return rc;
    /* THE ISSUE SHAPE, decided once per armed member and DEMOTED rather than refused if this
     * queue's blocks are not 16-byte addressable. Every member is placed at the same
     * `FT_TT_OP_QUEUE_BASE`, so they agree; taking the minimum makes that a fact rather than
     * an assumption, and the group's counters then price what actually happened. */
    if (pn_tt_issue_shape_for(&device->runtime, slot->queue) != PN_TT_ISSUE_WIDE)
        device->group.issue_shape = PN_TT_ISSUE_WORD;

    rc = ft_tt_op_program_load(&device->runtime.noc, slot->core_x, slot->core_y,
                               &device->program_resident.program, NULL, &slot->config);
    if (rc != FT_TT_OK)
        return rc;
    rc = pn_tt_group_place_cbs(weight_tile_bytes, depth, group->column_block, group->row_block,
                               slot->cb_buffer, sizeof(slot->cb_buffer),
                               &slot->config.local_cb_mask,
                               grid ? group->grid_stash_tiles
                                    : pn_tt_env_u32("PN_TT_G0_STASH_TILES", 0u));
    if (grid && rc == FT_TT_OK)
        group->grid_stash_address = g_stash_address;
    if (rc != FT_TT_OK)
        return rc;
    slot->config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    /* The reader is on BRISC and was built with NOC_INDEX=0; the firmware re-reads this from
     * the launch message at run time, so the two must agree. */
    slot->config.brisc_noc_id = 0u;

    for (index = 0; index < 5u; index++) {
        ft_tt_queue_kernel_binding binding;

        memset(&binding, 0, sizeof(binding));
        binding.idle_spin_limit = grid ? pn_tt_grid_idle_spins() : PN_TT_IDLE_SPIN_LIMIT;
        binding.cb_wait_spin_limit = PN_TT_CB_WAIT_SPIN_LIMIT;
        /* Per-processor status. "The triad is stuck" is not a diagnosis; "NCRISC is in
         * CB_WAIT_FRONT at descriptor 12" is, and only these slots can say it. */
        binding.status_address =
            FT_TT_QUEUE_STATUS_ADDRESS(slot->queue->control_address, processors[index]);
        /* WHERE THIS MEMBER POSTS ITS COMPLETION, and ONLY on the processor that publishes
         * it. The writer is DM1 (resident_matmul_writer.c is the one that calls
         * ftq_group_post), so DM1 is the single writer of this member's slot — a second
         * poster would give one slot two writers, which is the one property a slot array
         * exists to have. The block itself lives on member 0, which is why the host can ask
         * "has every member finished item N" with one read of one core.
         *
         * A SPLIT group is still N independent residencies with no barrier between them:
         * this changes nothing about the WORK, only about who the host has to ask. */
        if (!grid && group->group_completion && group->member_count > 1u &&
            processors[index] == FT_TT_PROC_DM1) {
            binding.group_x = group->member[0].core_x;
            binding.group_y = group->member[0].core_y;
            binding.group_address = group->group_address;
            binding.group_member = member;
        }
        /* THE DESTINATION ARM. Same processor, same slot arithmetic, same address — the
         * offset is a constant of the queue layout, so it names this member's own unused
         * group block under `self` and member 0's under `remote`. ONE variable moves. */
        if (!grid && group->post_probe != PN_TT_GROUP_POST_PROBE_OFF &&
            processors[index] == FT_TT_PROC_DM1) {
            const bool self = group->post_probe == PN_TT_GROUP_POST_PROBE_SELF;

            binding.group_x = self ? slot->core_x : group->member[0].core_x;
            binding.group_y = self ? slot->core_y : group->member[0].core_y;
            binding.group_address = group->group_address;
            /* The SLOT INDEX stays `member` under both arms. Collapsing it to 0 under
             * `self` would move a second variable — the address the inline write names —
             * for no gain: a member's own block has a slot for it either way. */
            binding.group_member = member;
        }
        memset(args[index], 0, sizeof(args[index]));
        rc = ft_tt_queue_kernel_args(slot->queue, &binding, args[index]);
        if (rc != FT_TT_OK)
            return rc;
        blocks[index].processor = processors[index];
        blocks[index].values = args[index];
        blocks[index].count = FT_TT_QUEUE_KERNEL_ARG_COUNT;
        /* THE COOPERATIVE TAIL, ON THE READER ONLY — AND IT IS PLANTED EVEN WHEN THIS
         * GROUP DOES NOT COOPERATE. `ft_tt_launch_plant_runtime_args()` writes exactly the
         * words it is given and NOTHING CLEARS L1 BETWEEN RESIDENCIES, so a member that
         * held a LEADER or RECEIVER role in a previous run would still be reading it here
         * — and the kernel would then refuse every solo descriptor with
         * kErrCoopRoleMismatch. Measured exactly that way: an N=8 SPLIT arm on cores a
         * cooperative N=4 arm had just used refused its first projection, re-armed, and
         * refused again. A zero role is SOLO, which is what the memset above leaves. */
        if (processors[index] == FT_TT_PROC_DM0) {
            if (!grid && group->cooperative) {
                rc = ft_tt_op_matmul_resident_multicast_args(&group->multicast, member,
                                                             args[index]);
                if (rc != FT_TT_OK)
                    return rc;
            }
            blocks[index].count = kArgCoopCount;
        }
    }

    rc = ft_tt_op_stage(&device->runtime.noc, slot->core_x, slot->core_y, &slot->config,
                        slot->cb_buffer, sizeof(slot->cb_buffer), blocks, 5u);
    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_launch_send_go(&device->runtime.noc, slot->core_x, slot->core_y);
    if (rc != FT_TT_OK)
        return rc;
    slot->running = true;
    return FT_TT_OK;
}

/**
 * Retire ONE member by ASKING, and wait for all five processors to say EXIT.
 *
 * The control block's own `state` cannot answer this: it has ONE writer (the reader), so a
 * reader that exited while the compute thread was still spinning out a bound would read
 * EXIT and be a lie. There is deliberately no reset escalation — `assert_reset` on a
 * blocked resident kernel is what wedged Tensix 0 for a whole host uptime.
 */
static void pn_tt_group_member_stop(pn_tt_device *device, pn_tt_matmul_member *slot)
{
    const uint32_t mask = PN_TT_TRIAD_MASK;
    uint32_t pending = mask;
    unsigned attempt;

    if (!slot->running)
        return;
    slot->running = false;
    if (ft_tt_queue_request_stop(&device->runtime.noc, slot->queue) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: could not ask core %u's residency to stop\n", slot->core_index);
        return;
    }
    /* Each poll is a device read of ~3.5 µs; 20000 of them is comfortably longer than the
     * longest descriptor this path issues and still bounded. There is no untimed wait. */
    for (attempt = 0; attempt < 20000u; attempt++) {
        if (ft_tt_queue_all_processors_exited(&device->runtime.noc, slot->queue, mask,
                                              &pending) == FT_TT_OK)
            return;
    }
    fprintf(stderr, "pn_tt: core %u's residency did not retire; processors still running: "
                    "0x%x (the core is left alone, NOT parked and NOT reset)\n",
            slot->core_index, pending);
}

int pn_tt_matmul_group_arm(pn_tt_device *device, uint32_t weight_tile_bytes, uint32_t depth)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t member;

    /* L1a.3: members 1.. are grid cores for this prompt; arming them would start a second
     * residency under a live grid triad. Refused; pn_tt_matmul_grid_retract() re-arms. */
    if (group->grid_lent || group->members_lent) {
        fprintf(stderr, "pn_tt: the decode group will not arm while its members are lent to "
                        "the prefill\n");
        return FT_TT_EINVAL;
    }

    group->sequence = 0u;
    group->chained = 0u;
    memset(group->pending, 0, sizeof(group->pending));
    /* The sequences restart at 0: a pull target from the previous residency is unreachable.
     * RED `grid_chain_stale` keeps it (and skips the grid path's chain begin) — the pre-fix
     * state, which must abort a long prompt after a pause. */
    if (!pn_tt_red("grid_chain_stale")) {
        group->chain_pull_members = 0u;
        group->chain_pull_target = 0u;
        group->chain_last_pulled = false;
    }

    /* THE ARM IS WHERE COOPERATION IS DECIDED, because it is the first moment every weight
     * has been placed and the answer is a property of the WHOLE weight set. It is
     * all-or-nothing: a residency that cooperates on some descriptors and not others
     * leaves its members' CB_IN0 write pointers at different offsets, and every later
     * multicast then lands in the wrong half of the receivers' buffers with every kernel
     * completing, every status word clean and only a byte comparison able to see it. */
    group->cooperative = group->cooperative_requested && group->cooperative_placed &&
                         group->member_count > 1u && group->uneven_weights == 0u &&
                         /* THE ROW BLOCK CHANGES WHAT AN A BLOCK IS, and the rendezvous
                          * counts A blocks: the leader would multicast `depth * MB` tiles
                          * where the receivers expect `depth`. Two mechanisms that both
                          * move A's supply, agreed separately, is a wrong answer with every
                          * status word clean — the same reason the column block disarms it,
                          * one operand over. The kernel refuses the pair as well. */
                         group->row_block <= 1u;
    if (group->cooperative_requested && group->cooperative_placed && !group->cooperative)
        fprintf(stderr, "pn_tt: cooperation is OFF — %u of the placed weights do not split "
                        "evenly across %u members, and the rendezvous needs every member to "
                        "walk the same number of blocks\n",
                group->uneven_weights, group->member_count);
    if (group->cooperative) {
        /* Staged BEFORE any member is kicked: a member whose arrival word still held the
         * previous residency's block counter would sail through its first wait with
         * nothing having arrived. */
        int rc = ft_tt_op_matmul_resident_multicast_reset(&device->runtime.noc,
                                                          &group->multicast);

        if (rc != FT_TT_OK) {
            fprintf(stderr, "pn_tt: could not stage the rendezvous words: %s\n",
                    ft_tt_error_string(rc));
            return rc;
        }
    }

    /* THE BLOCK IS WRITTEN BEFORE ANY MEMBER IS KICKED, so no writer can post into an
     * un-magicked block, and the sequences a previous residency left cannot be read as
     * this one's. It lives on member 0 above the parameter blocks — ft_tt_queue.c pins
     * the layout with static asserts, so nothing here re-derives the offsets. */
    if (group->group_completion && group->member_count > 1u) {
        int rc = ft_tt_queue_group_init(&device->runtime.noc, group->member[0].core_x,
                                        group->member[0].core_y, group->group_address,
                                        group->member_count);

        if (rc != FT_TT_OK) {
            fprintf(stderr, "pn_tt: could not write the group completion block: %s\n",
                    ft_tt_error_string(rc));
            return rc;
        }
    }
    /* THE PROBE'S BLOCKS, for the same reason and by the same call. `self` posts land in
     * N different cores' L1, so all N are written; `remote` writes only member 0's, which
     * is what the aggregated path does. The card side never checks the magic — this is so
     * a stale block cannot be MISREAD afterwards when the wedge is being attributed. */
    if (group->post_probe != PN_TT_GROUP_POST_PROBE_OFF) {
        const uint32_t blocks =
            group->post_probe == PN_TT_GROUP_POST_PROBE_SELF ? group->member_count : 1u;
        uint32_t index;

        for (index = 0; index < blocks; index++) {
            int rc = ft_tt_queue_group_init(&device->runtime.noc, group->member[index].core_x,
                                            group->member[index].core_y, group->group_address,
                                            group->member_count);

            if (rc != FT_TT_OK) {
                fprintf(stderr, "pn_tt: could not write the probe's group block on member "
                                "%u (core %u): %s\n", index, group->member[index].core_index,
                        ft_tt_error_string(rc));
                return rc;
            }
        }
    }

    for (member = 0; member < group->member_count; member++) {
        int rc;

        /* CLEARED AT ARM, WHICH IS THE ONLY MOMENT IT IS SAFE. Nothing clears a core's L1
         * between residencies, so a table carried across an arm would hand a descriptor an
         * address holding the PREVIOUS residency's words — the same class of defect as the
         * cooperative role a stale runtime-arg word left behind. No descriptor is in flight
         * here by construction. */
        if (group->param_cache)
            pn_tt_param_cache_alloc(&group->member[member]);
        rc = pn_tt_group_member_start(device, member, weight_tile_bytes, depth);

        if (rc != FT_TT_OK) {
            fprintf(stderr, "pn_tt: matmul member %u (core %u) did not arm: %s\n", member,
                    group->member[member].core_index, ft_tt_error_string(rc));
            /* Partly armed is not a state anything may run in: a projection would be split
             * across members that are not all listening, and the missing member's columns
             * would be whatever the output strip held last. */
            pn_tt_matmul_group_retire(device);
            return rc;
        }
    }
    /* THE MAGIC IS CHECKED ONCE, HERE, and never again — re-reading sixteen header words
     * on every poll is exactly the cost the slots-only read exists to remove. What it
     * catches is the wrong address, which is a constant of the run, not a per-poll risk. */
    if (group->group_completion && group->member_count > 1u) {
        ft_tt_queue_group_slot slots[PN_TT_MATMUL_MAX_CORES];
        int rc = ft_tt_queue_group_read(&device->runtime.noc, group->member[0].core_x,
                                        group->member[0].core_y, group->group_address,
                                        group->member_count, slots);

        if (rc != FT_TT_OK) {
            fprintf(stderr, "pn_tt: the group completion block does not read back with its "
                            "magic: %s\n", ft_tt_error_string(rc));
            pn_tt_matmul_group_retire(device);
            return rc;
        }
    }
    group->pull_table_ok = false;
    group->pull = pn_tt_env_u32("PN_TT_MATMUL_PULL", 1u) != 0u;
    group->red_pull_skip = pn_tt_red("matmul_pull_skip");
    group->pull_defer = pn_tt_env_u32("PN_TT_MATMUL_PULL_DEFER", 0u) != 0u;
    group->pull_serial = pn_tt_env_u32("PN_TT_MATMUL_PULL_SWEEP", 1u) == 0u;
    group->m0_held = false;
    if (group->pull && group->member_count > 1u && group->member_count <= 64u &&
        !group->cooperative) {
        uint32_t table[2u + 64u];
        uint32_t back[2u + 64u];
        const uint32_t words = 2u + group->member_count;
        const uint32_t address = FT_TT_OP_QUEUE_BASE + 0xF900u;
        uint32_t m;

        table[0] = 0x46545150u; /* "FTQP", kPullMagic */
        table[1] = group->member_count;
        for (m = 0; m < group->member_count; m++)
            table[2u + m] = (uint32_t)group->member[m].core_x |
                            ((uint32_t)group->member[m].core_y << 16);
        if (device->runtime.noc.write(device->runtime.noc.user, group->member[0].core_x,
                                      group->member[0].core_y, address, table,
                                      words * 4u) == FT_TT_OK &&
            device->runtime.noc.read(device->runtime.noc.user, group->member[0].core_x,
                                     group->member[0].core_y, address, back,
                                     words * 4u) == FT_TT_OK &&
            memcmp(table, back, words * 4u) == 0)
            group->pull_table_ok = true;
        fprintf(stderr, "pn_tt: matmul PULL completion %s — member 0 (core %u) reads %u "
                        "members' done-sequence before posting; %s, %s sweep%s\n",
                group->pull_table_ok ? "ARMED" : "REFUSED (the table did not read back)",
                group->member[0].core_index, group->member_count,
                group->pull_defer ? "at the chain's end only" : "every split projection",
                group->pull_serial ? "one-at-a-time" : "parallel",
                group->red_pull_skip ? "; RED PROOF: the pull is SKIPPED" : "");
    }
    device->resident_tile_bytes = weight_tile_bytes;
    device->resident_depth = depth;
    device->resident_running = true;
    pn_tt_group_mark_alive(group, pn_tt_now_seconds(), true);
    /* S3b: RE-PLANT the members' replica table on every arm (a no-op with the knob off). A
     * failure is not the arm's: the fused path is refused and the FFN runs unfused. */
    if (pn_tt_replica_map_publish_fused(device) != 0)
        fprintf(stderr, "pn_tt: the fused SwiGLU replica table did not land on the matmul "
                        "members — the FFN runs unfused\n");
    return FT_TT_OK;
}

void pn_tt_matmul_group_retire(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t member;

    device->resident_running = false;
    /* A held member-0 descriptor belongs to a chain nobody will flush on this residency. */
    group->m0_held = false;
    /* NOT the prefill grid: an idle refresh retires and re-arms the decode group at a
     * prefill's first projection, and the grid it would kill is the one about to be used.
     * The grid is retired by pn_tt_matmul_grid_end() and at device close. */
    for (member = 0; member < group->member_count; member++) {
        pn_tt_group_member_stop(device, &group->member[member]);
        /* AFTER the stop, not before: the card must not be executing a descriptor that
         * points into a region this host copy is the only record of. */
        pn_tt_param_cache_free(&group->member[member]);
    }
}

/* ================================================================
 * The load-time shard
 * ================================================================ */

/**
 * Split one N-major weight over `member_count` banks, or say why it stayed whole.
 *
 * The split is a byte-run relocation and nothing else: an N-major tile grid indexes
 * `nt * Kt + kt`, so columns [c*cc, (c+1)*cc) are the contiguous run
 * `[c*cc*Kt*tile_bytes, (c+1)*cc*Kt*tile_bytes)` of the canonical payload. Member c then
 * reads a grid `cc` columns wide from column 0 of its own panel, which is why the reader
 * kernel cannot tell a sharded weight from an unsharded one.
 *
 * The cooperative planner is what REFUSES a shape here — a `tile_cols` the member count
 * does not divide, a zero dimension, a shard list that does not cover every member — and
 * what counts the doors that would share a NoC row. Its column arithmetic is then checked
 * against this function's own, so the two cannot drift apart silently.
 *
 * @return 0 (sharded or deliberately left whole), or -1 with the reason on stderr
 */
/**
 * Upload destination columns `[first, first+count)` of `entry` into one panel.
 *
 * `runs` NULL is the IDENTITY and is one `ft_tt_fttw_upload_range()` — byte for byte what
 * this path always did, which is the property that lets every ordinary weight go through the
 * permuting code with nothing about it changing. A run list issues one range per run the
 * panel intersects, because an N-major artifact lays column tile `c` as a contiguous
 * `column_bytes` run and a permutation of columns is therefore a permutation of byte ranges.
 *
 * @return 0, or -1 with the reason on stderr
 */
static int pn_tt_upload_panel_columns(pn_tt_device *device, const ft_tt_artifact_entry *entry,
                                      const pn_tt_column_run *runs, uint32_t run_count,
                                      uint64_t column_bytes, uint32_t first, uint32_t count,
                                      uint16_t door_x, uint16_t door_y, uint32_t address)
{
    uint32_t written = 0u;

    if (!runs) {
        return ft_tt_fttw_upload_range(&device->artifact, entry,
                                       (uint64_t)first * column_bytes,
                                       (uint64_t)count * column_bytes,
                                       &device->runtime.bulk_dram, door_x, door_y, address,
                                       0u) == FT_TT_OK
                   ? 0
                   : -1;
    }
    while (written < count) {
        uint32_t source = 0u, contiguous = 0u, piece;

        if (!pn_tt_column_runs_map(runs, run_count, first + written, &source, &contiguous)) {
            fprintf(stderr, "pn_tt: %s destination column %u is past the run list\n",
                    entry->name, first + written);
            return -1;
        }
        piece = contiguous < count - written ? contiguous : count - written;
        if (ft_tt_fttw_upload_range(&device->artifact, entry, (uint64_t)source * column_bytes,
                                    (uint64_t)piece * column_bytes, &device->runtime.bulk_dram,
                                    door_x, door_y,
                                    address + (uint32_t)((uint64_t)written * column_bytes),
                                    0u) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: upload failed for %s columns %u..%u\n", entry->name,
                    source, source + piece);
            return -1;
        }
        written += piece;
    }
    return 0;
}

/** One bank, one buffer, one upload — the pre-group placement, kept as the arm a sharded
 *  weight is compared against and the only one a `Nt % N != 0` tensor can take. */
static int pn_tt_group_place_whole(pn_tt_device *device, const ft_tt_artifact_entry *entry,
                                   const pn_tt_column_run *runs, uint32_t run_count,
                                   uint32_t bank, pn_tt_weight *weight)
{
    const uint64_t column_bytes = (uint64_t)weight->tile_rows * weight->tile_bytes;
    const uint64_t bytes = (uint64_t)weight->tile_cols * column_bytes;
    ft_tt_buffer buffer;

    if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, bank, bytes, &buffer) !=
        FT_TT_OK) {
        fprintf(stderr, "pn_tt: DRAM full at %s\n", entry->name);
        return -1;
    }
    if (pn_tt_upload_panel_columns(device, entry, runs, run_count, column_bytes, 0u,
                                   weight->tile_cols,
                                   (uint16_t)device->dram[buffer.logical_bank].x,
                                   (uint16_t)device->dram[buffer.logical_bank].y,
                                   (uint32_t)buffer.address) != 0)
        return -1;
    weight->bank = buffer.logical_bank;
    weight->address = (uint32_t)buffer.address;
    weight->shard_count = 1u;
    weight->shard_tile_cols[0] = weight->tile_cols;
    weight->shard_column_first[0] = 0u;
    weight->shard_address[0] = weight->address;
    weight->shard_x[0] = (uint16_t)device->dram[buffer.logical_bank].x;
    weight->shard_y[0] = (uint16_t)device->dram[buffer.logical_bank].y;
    weight->shard_bank[0] = buffer.logical_bank;
    device->bytes_written += bytes;
    return 0;
}

int pn_tt_matmul_group_place_weight(pn_tt_device *device, const ft_tt_artifact_entry *entry,
                                    uint32_t *bank_cursor, pn_tt_weight *weight)
{
    return pn_tt_matmul_group_place_weight_columns(device, entry, NULL, 0u, bank_cursor,
                                                   weight);
}

int pn_tt_matmul_group_place_weight_columns(pn_tt_device *device,
                                            const ft_tt_artifact_entry *entry,
                                            const pn_tt_column_run *runs, uint32_t run_count,
                                            uint32_t *bank_cursor, pn_tt_weight *weight)
{
    pn_tt_matmul_group *group = &device->group;
    ft_tt_op_matmul_cooperative_request request;
    ft_tt_op_matmul_cooperative_plan plan;
    ft_tt_dram_endpoint doors[PN_TT_MATMUL_MAX_CORES];
    uint32_t banks[PN_TT_MATMUL_MAX_CORES];
    ft_tt_buffer buffer[PN_TT_MATMUL_MAX_CORES];
    uint32_t members = group->member_count;
    uint64_t column_bytes = (uint64_t)weight->tile_rows * weight->tile_bytes;
    uint32_t columns[PN_TT_MATMUL_MAX_CORES];
    uint32_t first[PN_TT_MATMUL_MAX_CORES];
    uint64_t panel_bytes[PN_TT_MATMUL_MAX_CORES];
    uint32_t base_columns, remainder, running;
    uint32_t member;

    /* THE PERMUTATION IS REFUSED HERE OR NOWHERE. Downstream this is a byte offset, so a run
     * list that leaves the tensor, or uses one source column twice, is a plausible-looking
     * weight with one column duplicated and another missing — a smoothly wrong projection
     * that neither the planner nor the card can see. pn_tt_column_runs.h states it. */
    if (runs && (pn_tt_column_runs_total(runs, run_count) != weight->tile_cols ||
                 !pn_tt_column_runs_are_injective(runs, run_count, entry->tile_cols))) {
        fprintf(stderr, "pn_tt: the column permutation for %s does not produce %u columns "
                        "injectively out of %u\n", entry->name, weight->tile_cols,
                entry->tile_cols);
        return -1;
    }

    weight->probe_selected = pn_tt_weight_probe_selected(entry->name);

    /* THE REMAINDER GOES TO THE FIRST MEMBERS, one column each, so the widths differ by at
     * most one and every column of the weight is owned exactly once. `first` is the running
     * sum, which is what makes the panels concatenate at Mt = 1 when they are not all the
     * same width. */
    base_columns = members > 0u ? weight->tile_cols / members : 0u;
    remainder = members > 0u ? weight->tile_cols % members : 0u;
    running = 0u;
    for (member = 0; member < members; member++) {
        columns[member] = base_columns + (member < remainder ? 1u : 0u);
        first[member] = running;
        panel_bytes[member] = (uint64_t)columns[member] * column_bytes;
        running += columns[member];
    }

    /* The tile grid and the payload must agree before a byte offset into the payload means
     * anything. They always have; this is the check that says so rather than assumes it.
     * A member with ZERO columns is a descriptor the resident reader refuses, and a refusal
     * on the card strands a triad — so too few columns for the group keeps the weight whole
     * rather than sending one. */
    /* The SOURCE grid and the payload, not the destination's: a permuted placement is a
     * subset, so holding the destination width against the file's byte count would call
     * every de-interleaved half a mismatch. */
    if (members < 2u || base_columns == 0u ||
        (uint64_t)entry->tile_cols * column_bytes != entry->byte_count) {
        if (members >= 2u && base_columns != 0u)
            fprintf(stderr, "pn_tt: %s is %llu B but Kt*Nt*tile is %llu — not sharding it\n",
                    entry->name, (unsigned long long)entry->byte_count,
                    (unsigned long long)((uint64_t)entry->tile_cols * column_bytes));
        group->unsharded_weights++;
        /* A weight that is not split at all leaves the members walking different block
         * counts, which is the same desynchronisation a ragged split causes and is fatal
         * to the rendezvous for exactly the same reason. */
        group->uneven_weights++;
        if (pn_tt_group_place_whole(device, entry, runs, run_count,
                                    *bank_cursor % device->dram_bank_count, weight) != 0)
            return -1;
        *bank_cursor = (*bank_cursor + 1u) % device->dram_bank_count;
        return 0;
    }
    if (remainder != 0u) {
        group->ragged_weights++;
        /* COOPERATION CANNOT SURVIVE A RAGGED SPLIT, and this is where that is decided.
         * The per-block rendezvous is correct only because every member walks the same
         * number of blocks in the same order, and blocks are Mt * Nt_local * Kt/depth — so
         * members of different widths fall out of step on the first descriptor and stay
         * out of step. SPLIT has no rendezvous and takes the ragged shape happily, which is
         * why this counts rather than refuses. */
        group->uneven_weights++;
    }

    /* WHICH BANKS: the rotation, or a fixed nearest-bank map (PN_TT_WEIGHT_BANKS,
     * pn_tt_weight_banks.h). Either way the same bytes go up; only (bank, address, door)
     * moves. */
    {
        const int chosen = pn_tt_weight_banks_choose(device, members, banks, doors);

        if (chosen < 0)
            return -1;
        if (chosen == 0) {
            for (member = 0; member < members; member++)
                banks[member] = (*bank_cursor + member) % device->dram_bank_count;
            if (ft_tt_dram_stripe_choose_ports(&device->runtime.grid, banks, members,
                                               FT_TT_COORD_TRANSLATED, doors) != FT_TT_OK) {
                fprintf(stderr, "pn_tt: could not choose weight doors for %s\n",
                        entry->name);
                return -1;
            }
        }
    }

    memset(&request, 0, sizeof(request));
    request.tile_rows = 1u; /* Mt: a token is one row, and the panels concatenate only here */
    request.tile_inner = weight->tile_rows;
    request.tile_cols = weight->tile_cols;
    request.prefetch_tiles = device->resident_depth ? device->resident_depth : 1u;
    request.weight_n_major = 1u;
    request.mode = FT_TT_OP_MATMUL_SPLIT;
    request.spin_limit = 1u; /* unread outside COOPERATIVE; never 0, which is unbounded */
    request.activation.x = (uint16_t)device->dram[0].x;
    request.activation.y = (uint16_t)device->dram[0].y;
    request.activation.tile_bytes = PN_TT_TILE_BF16;
    request.weight.tile_bytes = weight->tile_bytes;
    request.output.tile_bytes = PN_TT_TILE_BF16;
    request.weight_shard_count = members;

    for (member = 0; member < members; member++) {
        if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, banks[member],
                               (size_t)panel_bytes[member], &buffer[member]) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: DRAM full at %s shard %u (bank %u)\n", entry->name, member,
                    banks[member]);
            return -1;
        }
        request.weight_shards[member].endpoint = doors[member];
        request.weight_shards[member].address = (uint32_t)buffer[member].address;
    }

    if (ft_tt_op_matmul_cooperative_plan_build(NULL, members, 0u, &request, &plan) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: the cooperative planner refused %s over %u members\n",
                entry->name, members);
        return -1;
    }
    if (plan.row_collisions > group->worst_weight_row_collisions)
        group->worst_weight_row_collisions = plan.row_collisions;

    for (member = 0; member < members; member++) {
        const ft_tt_op_matmul_cooperative_member *slice = &plan.member[member];

        /* The planner's split and this function's byte arithmetic must be the SAME split.
         * They are derived independently — one from the planner's own remainder rule, one
         * from a payload offset — so disagreeing is a wrong answer with no error anywhere,
         * and a RAGGED split is exactly where two independent derivations can drift. */
        if (slice->column_count != columns[member] || slice->column_first != first[member]) {
            fprintf(stderr, "pn_tt: %s shard %u disagrees with the plan (%u..%u vs %u..%u)\n",
                    entry->name, member, slice->column_first,
                    slice->column_first + slice->column_count, first[member],
                    first[member] + columns[member]);
            return -1;
        }
        /* RED PROOF `PN_TT_RED=weight_place_offset`: member 1's panel of the FIRST sharded
         * weight is uploaded ONE TILE past the address its descriptor reads, so every tile
         * of that panel is read one tile early. A placement change that moved bytes wrong
         * looks exactly like this, and it must DIFFER. */
        static bool red_offset_used = false;
        uint32_t upload_address = (uint32_t)buffer[member].address;

        if (member == 1u && !red_offset_used && pn_tt_red("weight_place_offset")) {
            red_offset_used = true;
            upload_address += weight->tile_bytes;
            fprintf(stderr, "pn_tt: RED PROOF — %s member 1 panel uploaded one tile late\n",
                    entry->name);
        }
        if (pn_tt_upload_panel_columns(device, entry, runs, run_count, column_bytes,
                                       first[member], columns[member],
                                       (uint16_t)doors[member].coord.x,
                                       (uint16_t)doors[member].coord.y,
                                       upload_address) != 0) {
            fprintf(stderr, "pn_tt: upload failed for %s shard %u\n", entry->name, member);
            return -1;
        }
        weight->shard_address[member] = (uint32_t)buffer[member].address;
        weight->shard_x[member] = (uint16_t)doors[member].coord.x;
        weight->shard_y[member] = (uint16_t)doors[member].coord.y;
        weight->shard_tile_cols[member] = columns[member];
        weight->shard_column_first[member] = first[member];
        weight->shard_bank[member] = banks[member];
        device->bytes_written += panel_bytes[member];
    }

    weight->bank = buffer[0].logical_bank;
    weight->address = (uint32_t)buffer[0].address;
    weight->shard_count = members;
    group->sharded_weights++;
    /* Offset the next tensor's bank run by ONE, not by `members`: successive projections
     * are serial, so what matters is that no bank is the first of every tensor, and a
     * stride of one visits every bank however many members there are. */
    *bank_cursor = (*bank_cursor + 1u) % device->dram_bank_count;
    return 0;
}

/* ================================================================
 * Dispatch
 * ================================================================ */

/**
 * Is this shape one the resident reader will execute rather than REFUSE?
 *
 * A host mirror of `descriptor_is_executable()` in resident_matmul_reader.c, and it exists
 * because the two answers have very different costs: refused here is a return value, and
 * refused on the card stops that member's reader mid-residency. `columns` is the MEMBER's
 * Nt, not the weight's — a member with zero columns is the refusal this catches.
 */
static bool pn_tt_group_descriptor_is_executable(const pn_tt_device *device,
                                                 const pn_tt_weight *weight, uint32_t columns)
{
    const uint32_t block = device->group.column_block ? device->group.column_block : 1u;
    const uint32_t rows = device->group.row_block ? device->group.row_block : 1u;
    const uint32_t deep = pn_tt_matmul_cb_deep();
    uint32_t pages = device->resident_depth * rows * deep;
    uint32_t weight_pages = device->resident_depth * block * deep;
    uint32_t chunk = device->resident_depth * block;
    uint32_t a_chunk = device->resident_depth * rows;

    if (weight->tile_bytes != device->resident_tile_bytes)
        return false; /* the residency is bound to ONE weight tile size */
    if (weight->tile_rows == 0u || columns == 0u)
        return false;
    /* MB * NB OUTPUT TILES ARE LIVE IN DST AT ONCE. The kernel refuses this too; asked here
     * so a mis-set knob is a message rather than a stranded triad. */
    if (!ftmm_block_fits_section(rows, block, device->group.dst_section_tiles))
        return false;
    /* The kernel silently collapses a depth that does not divide Kt to 1 — which is correct
     * but would push single pages through a buffer sized in blocks of `depth`, leaving the
     * write pointer off the block grid for every LATER descriptor. Refuse instead. */
    if (device->resident_depth == 0u || weight->tile_rows % device->resident_depth != 0u)
        return false;
    /* CB_IN0's chunk is the WHOLE row block whatever this descriptor's Mt is, the mirror of
     * the CB_IN1 condition below. `Mt < MB` is legal — it takes a short block. */
    if (a_chunk == 0u || a_chunk > pages || pages % a_chunk != 0u)
        return false;
    /* CB_IN1's chunk is the WHOLE column block whatever this weight's Nt is, so the grid
     * condition is on `depth * NB` and not on the columns. `columns < NB` is legal and
     * common — 48 of Ornith's 249 weights are Nt = 1. */
    if (chunk > weight_pages || weight_pages % chunk != 0u)
        return false;
    /* The MOP's SrcA stride is `depth * page` in a 16-bit SETDMAREG halfword; the compute
     * thread's page unit is 16 B, so this is the only shape that could truncate silently. */
    if (block > 1u && (uint64_t)chunk * device->resident_tile_bytes / 16u > 0xFFFFu)
        return false;
    return true;
}

/**
 * Which MIRROR of the A strip member `m` reads.
 *
 * `m % replica_count`, and that is the whole descriptor side of the bank spread — dispatch
 * already writes a separate parameter block per member, so naming a different address in
 * each is an index and not a protocol change.
 *
 * IT IS ROUND-ROBIN AND NOT `m / (members / replicas)` DELIBERATELY. The two agree when the
 * replica count divides the member count and disagree when it does not, and the modulo is
 * the one that spreads a RAGGED count: at 6 members over 4 mirrors it gives 2/2/1/1 readers
 * a bank, where the division gives 4/2/0/0 and leaves half the banks idle.
 *
 * A strip with one mirror returns 0 for every member, which is the pre-replication
 * descriptor word for word.
 */
static uint32_t pn_tt_group_replica_index(const pn_tt_activation *input, uint32_t member)
{
    if (!input || input->replica_count <= 1u)
        return 0u;
    return member % input->replica_count;
}

/** The bank member `m` really reads A from, red arms included. ONE expression, because it is
 *  needed twice — in the descriptor and in the per-bank read accounting — and two copies of
 *  it would print a bank table describing a run that did not happen. */
static uint32_t pn_tt_group_activation_bank(const pn_tt_matmul_group *group,
                                            const pn_tt_activation *input, uint32_t member)
{
    const uint32_t bank = input->replica_bank[pn_tt_group_replica_index(input, member)];

    if (group->red_a_spread && group->activation_doors && group->member_count > 1u &&
        bank < group->activation_bank_count)
        return (bank + member) % group->activation_bank_count;
    return bank;
}

/**
 * `Mt` for this projection: how many 32-row bands of A one descriptor sweeps the weight past.
 *
 * ONE EXPRESSION, because it is needed in the parameter block AND in the read accounting, and
 * two copies of it would print a byte table describing a descriptor that was not issued.
 */
static uint32_t pn_tt_group_tile_rows(const pn_tt_activation *input,
                                      const pn_tt_activation *output)
{
    uint32_t rows = input && input->active_rows ? input->active_rows : 1u;

    if (output && output->active_rows && output->active_rows < rows)
        rows = output->active_rows;
    return rows;
}

/**
 * CLASS CLOCK: charge each member's card-busy delta since the last flush to census `row`.
 * Host-issued decode only (the host awaits every flush there); one 4-byte read a member.
 */
static void pn_tt_group_class_clock(pn_tt_device *device, uint32_t row)
{
    pn_tt_matmul_group *group = &device->group;
    uint64_t max_delta = 0u, sum = 0u;
    uint32_t member;

    if (group->class_clock < 0)
        group->class_clock = pn_tt_env_u32("PN_TT_MATMUL_CLASS_CLOCK", 0u) != 0u ? 1 : 0;
    if (group->class_clock == 0)
        return;
    for (member = 0; member < group->member_count; member++) {
        uint32_t busy = 0u, delta;

        if (!group->member[member].queue ||
            ft_tt_queue_read_field(&device->runtime.noc, group->member[member].queue,
                                   (ft_tt_queue_control_field)13, &busy) != FT_TT_OK)
            continue;
        /* A re-arm restarts the counter: then the whole new count is this flush's. */
        delta = busy >= group->class_clock_last[member] ? busy - group->class_clock_last[member]
                                                        : busy;
        group->class_clock_last[member] = busy;
        sum += delta;
        group->flush_census[row].card_busy_member[member] += delta;
        if (delta > max_delta)
            max_delta = delta;
    }
    /* Member 0's PULL WAIT (writer word 61): the part of its busy spent waiting for the
     * others, so its own work is busy - pull. */
    if (group->member_count > 0u && group->member[0].queue) {
        uint32_t pull = 0u;

        if (device->runtime.noc.read(device->runtime.noc.user, group->member[0].queue->x,
                                     group->member[0].queue->y,
                                     group->member[0].queue->control_address + 61u * 4u,
                                     &pull, sizeof(pull)) == FT_TT_OK) {
            group->flush_census[row].card_pull_wait +=
                pull >= group->class_clock_pull_last ? pull - group->class_clock_pull_last : pull;
            group->class_clock_pull_last = pull;
        }
    }
    group->flush_census[row].card_busy_max += max_delta;
    group->flush_census[row].card_busy_sum += sum;
}

/** The sixteen descriptor words for ONE member of ONE projection. */
static void pn_tt_group_parameters(pn_tt_device *device, const pn_tt_weight *weight,
                                   const pn_tt_activation *input,
                                   const pn_tt_activation *output, uint32_t member,
                                   bool split, uint32_t *parameters)
{
    const pn_tt_matmul_group *group = &device->group;
    uint32_t columns = split ? weight->shard_tile_cols[member] : weight->tile_cols;
    uint32_t column_first = split ? weight->shard_column_first[member] : 0u;
    uint32_t weight_address = split ? weight->shard_address[member] : weight->address;
    uint16_t weight_x = split ? weight->shard_x[member]
                              : (uint16_t)device->dram[weight->bank].x;
    uint16_t weight_y = split ? weight->shard_y[member]
                              : (uint16_t)device->dram[weight->bank].y;
    /* WHICH MIRROR, AND THEREFORE WHICH BANK. A strip with one mirror resolves to the
     * strip's own bank and address, so an unreplicated run is the pre-replication
     * descriptor word for word. */
    const uint32_t replica = pn_tt_group_replica_index(input, member);
    const uint32_t activation_bank = pn_tt_group_activation_bank(group, input, member);
    uint32_t activation_address = input->replica_address[replica];
    uint16_t activation_x = (uint16_t)device->dram[activation_bank].x;
    uint16_t activation_y = (uint16_t)device->dram[activation_bank].y;

    /* A DOOR, not a different memory: members that landed on the SAME bank read the same
     * bytes at the same address through a different NoC port of it. It is the cheapest
     * bandwidth lever there is and it moves nothing (ft_tt_dram_stripe.h) — but it is a
     * lever against the ~45 GB/s ENDPOINT ceiling and not against the ~59 GB/s BANK, which
     * is what the mirrors above are for. Left at the operand's own coordinate when the
     * group is one member wide, so N=1 is the pre-group descriptor word for word. */
    if (group->activation_doors && group->member_count > 1u &&
        activation_bank < group->activation_bank_count) {
        activation_x = (uint16_t)group->activation_door[activation_bank][member].coord.x;
        activation_y = (uint16_t)group->activation_door[activation_bank][member].coord.y;
    }
    /* RED PROOF `PN_TT_RED=matmul_a_spread` moved the BANK above and leaves the address
     * word alone, so the member reads whatever that bank holds at the strip's offset —
     * garbage, deliberately. It is the timing arm the production mirrors replace, kept
     * because it is the one that isolates the bank from everything else. */

    memset(parameters, 0, FT_TT_QUEUE_PARAM_WORDS * sizeof(parameters[0]));
    /* Mt. A decode token is one tile-row and that is the literal this used to be; a batched
     * prefill pass is `active_rows` bands of 32 positions, and the weight is swept ONCE for
     * `ceil(Mt/MB)` of them rather than once for each.
     *
     * THE SPAN IS THE NARROWER OF THE TWO STRIPS, not the input's. The reader walks A at
     * `(mt + row) * tile_inner + kt` and the writer places C at `(mt + row) * a2 + nt +
     * column`, so an Mt the OUTPUT cannot hold is a write past the end of the destination
     * buffer with every kernel retiring clean and DRAM having no bounds check. The clamp is
     * the one place that can see both. */
    parameters[FT_TT_QUEUE_MATMUL_TILE_ROWS] =
        pn_tt_group_tile_rows(input, output);
    parameters[FT_TT_QUEUE_MATMUL_TILE_INNER] = weight->tile_rows;
    parameters[FT_TT_QUEUE_MATMUL_TILE_COLS] = columns;
    parameters[FT_TT_QUEUE_MATMUL_A_X] = activation_x;
    parameters[FT_TT_QUEUE_MATMUL_A_Y] = activation_y;
    /* RED PROOF `PN_TT_RED=matmul_a_offset`: same BANK, different ADDRESS. It is the other
     * half of the pair. `matmul_a_spread` moves the bank AND the address range at once, so
     * on its own it cannot tell a channel ceiling from the fact that N members are hammering
     * ONE 16 KiB window — the 59.2 GB/s per-bank figure in the spec was measured with each
     * core on a DISJOINT 256 MiB region, which is the opposite access pattern. This arm
     * keeps the bank and disjoins the windows, so the two together separate them.
     * 4 MiB is well inside a bank and well past any strip. **The answer is garbage.** */
    if (group->red_a_offset)
        activation_address += member * (4u << 20);
    parameters[FT_TT_QUEUE_MATMUL_A_BASE] = activation_address;
    parameters[FT_TT_QUEUE_MATMUL_B_X] = weight_x;
    parameters[FT_TT_QUEUE_MATMUL_B_Y] = weight_y;
    parameters[FT_TT_QUEUE_MATMUL_B_BASE] = weight_address;
    parameters[FT_TT_QUEUE_MATMUL_C_X] = (uint16_t)device->dram[output->bank].x;
    parameters[FT_TT_QUEUE_MATMUL_C_Y] = (uint16_t)device->dram[output->bank].y;
    /* The panels CONCATENATE, which is the whole correctness claim: at Mt=1 member c's
     * `columns` output tiles land exactly where the single-core writer would have put
     * columns [column_first, column_first + columns). The offset is the RUNNING SUM and not
     * `member * columns` — those agree only while every member is the same width, and a
     * ragged split is precisely where they stop agreeing. */
    parameters[FT_TT_QUEUE_MATMUL_C_BASE] =
        output->address + column_first * PN_TT_TILE_BF16;
    parameters[FT_TT_QUEUE_MATMUL_DEPTH] = device->resident_depth;
    /* A STATEMENT about DRAM, not a request: the artifact's MATMUL_NMAJOR layout IS the
     * N-major tile grid, and a wrong value here is a wrong answer with no error anywhere. */
    parameters[FT_TT_QUEUE_MATMUL_B_N_MAJOR] = 1u;
    /* M4 PIPELINED READER (bit 8; resident_matmul_reader.c read_matmul_pipelined):
     * `PN_TT_MATMUL_PIPE`; `PN_TT_RED=matmul_pipe_no_wait` (bit 9) never waits and must DIFFER. */
    {
        const int pipe = pn_tt_matmul_pipe();

        if (pipe != 0)
            parameters[FT_TT_QUEUE_MATMUL_B_N_MAJOR] |= 1u << 8;
        if (pipe != 0 && pn_tt_red("matmul_pipe_no_wait"))
            parameters[FT_TT_QUEUE_MATMUL_B_N_MAJOR] |= 1u << 9;
        /* M5 ROW-0 (bit 10): a single-token forward's matmuls read A faces 0-1 only and
         * multiply in0 as a 16x32 half tile. Needs the pipelined reader. `PN_TT_MATMUL_ROW0`
         * 0 off, 1 honour the single-row promise. */
        {
            static int row0 = -1;

            if (row0 < 0)
                row0 = (int)pn_tt_env_u32("PN_TT_MATMUL_ROW0", 1u);
            if (g_row0_override >= 0 ? (g_row0_override != 0 && pipe != 0)
                                     : (row0 != 0 && pipe != 0 &&
                                        (device->single_row ||
                                         (device->verify_rows && device->verify_rows <= 16u)))) {
                parameters[FT_TT_QUEUE_MATMUL_B_N_MAJOR] |= 1u << 10;
                if (row0 == 2) /* DIAGNOSTIC: reader row-0, compute full (bit 17) */
                    parameters[FT_TT_QUEUE_MATMUL_B_N_MAJOR] |= 1u << 17;
                if (pn_tt_red("matmul_row0_face0"))
                    parameters[FT_TT_QUEUE_MATMUL_B_N_MAJOR] |= 1u << 11;
                if (pn_tt_env_u32("PN_TT_MATMUL_ROW0_DIAG", 0u) != 0u)
                    parameters[FT_TT_QUEUE_MATMUL_B_N_MAJOR] |= 1u << 16;
            }
        }
        /* BYTES A-REUSE (bit 18; bit 19 RED = skip regardless of the ring). */
        if (pipe != 0 && pn_tt_a_reuse_tiles() != 0u)
            parameters[FT_TT_QUEUE_MATMUL_B_N_MAJOR] |= 1u << 18;
        if (pipe != 0 && pn_tt_red("matmul_a_reuse"))
            parameters[FT_TT_QUEUE_MATMUL_B_N_MAJOR] |= 1u << 19;
        /* M5: `PN_TT_MATMUL_PROBE` = word-13 bits 12-15 (1 no read, 2 no compute, 4 no write,
         * 8 FLUSH writer) — resident_queue_protocol.h kMatmulProbe*. */
        if (weight->probe_selected || !getenv(PN_TT_MATMUL_PROBE_ONLY_ENV) ||
            !*getenv(PN_TT_MATMUL_PROBE_ONLY_ENV))
            parameters[FT_TT_QUEUE_MATMUL_B_N_MAJOR] |=
                (pn_tt_env_u32("PN_TT_MATMUL_PROBE", 0u) & 0xFu) << 12;
    }
    /* THE SAME VALUE ON EVERY DESCRIPTOR OF THE RESIDENCY, because CB_IN1's chunk is
     * `depth * NB` and a chunk size that changes walks the write pointer off the grid the
     * next descriptor needs. A weight narrower than NB is not an exception — it takes a
     * short last block, which shortens the fill and not the chunk.
     *
     * ONE WORD, TWO BLOCKS. The row block rides in bits 16..23 of the same word and obeys
     * the same per-residency rule for the same reason, one CB over. At `row_block == 1` the
     * encoder returns the pre-encoding word BIT FOR BIT, which is what makes a decode `cmp`
     * evidence rather than a coincidence (tt/kernels/resident_matmul_block_protocol.h). */
    parameters[FT_TT_QUEUE_MATMUL_COLUMN_BLOCK] =
        ftmm_block_word(group->row_block, group->column_block);
    /* ALL-OR-NOTHING, and it is `split` that decides it rather than the weight: a
     * cooperating residency's members must agree on every descriptor, and the kernel
     * REFUSES one whose word disagrees with its role. A weight that did not split would
     * therefore have to stop the whole residency cooperating, which is what
     * `group->uneven_weights` does at arm time — by the time a descriptor is written,
     * `group->cooperative` already means "every weight splits". */
    parameters[FT_TT_QUEUE_MATMUL_COOPERATIVE] =
        (split && device->group.cooperative)
            ? (device->group.red_coop_stale ? FT_TT_RESIDENT_COOP_DESCRIPTOR_RED_SKIP_PAYLOAD
                                            : FT_TT_RESIDENT_COOP_DESCRIPTOR_COOPERATE)
            : FT_TT_RESIDENT_COOP_DESCRIPTOR_SOLO;

    if (split && member == 1u && device->red_split != PN_TT_RED_SPLIT_NONE) {
        switch (device->red_split) {
        case PN_TT_RED_SPLIT_BASE:
            parameters[FT_TT_QUEUE_MATMUL_B_BASE] = weight->shard_address[0];
            parameters[FT_TT_QUEUE_MATMUL_B_X] = weight->shard_x[0];
            parameters[FT_TT_QUEUE_MATMUL_B_Y] = weight->shard_y[0];
            break;
        case PN_TT_RED_SPLIT_OUT:
            parameters[FT_TT_QUEUE_MATMUL_C_BASE] = output->address;
            break;
        case PN_TT_RED_SPLIT_NARROW:
            if (columns > 1u)
                parameters[FT_TT_QUEUE_MATMUL_TILE_COLS] = columns - 1u;
            break;
        default:
            break;
        }
    }
}

/**
 * Wait for `expected` descriptors on one member — or for its kernel to say it refused one,
 * which is a different thing and must not be waited out.
 *
 * `ft_tt_queue_wait_completed()` polls the count alone, so a refusal costs the whole
 * timeout before anyone looks at the error word. The count is polled every iteration and
 * the error word every 128th, which finds a refusal inside a millisecond for 0.8% more
 * reads.
 */
static int pn_tt_group_await(pn_tt_device *device, pn_tt_matmul_member *slot,
                             uint32_t expected)
{
    const double deadline = pn_tt_now_seconds() + (double)PN_TT_OP_TIMEOUT_MS / 1000.0;
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    const char *reason = NULL;
    unsigned poll = 0;

    /* THE ONE-TRANSACTION PATH. The poll and the check are the same 16-byte read, so a
     * member that is already finished — which is every member but the straggler — costs ONE
     * transaction instead of three. `error` is not in the line; see PN_TT_FAST_COMPLETION_ENV
     * for the four places it is still read and why a refusal cannot hide behind it. */
    if (device->group.fast_completion) {
        for (;;) {
            int rc = ft_tt_queue_read_completion_line(&device->runtime.bulk_l1, slot->queue,
                                                      fields);

            device->group.poll_reads++;
            device->group.wait_reads++;
            device->group.wait_read_bytes += 16u;
            device->group.wait_transactions += 1u;
            if (rc != FT_TT_OK)
                return rc;
            if (fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] >= expected)
                break;
            if (device->poll_backoff_us)
                usleep(device->poll_backoff_us);
            if ((++poll & 127u) == 0u) {
                uint32_t error = 0;

                rc = ft_tt_queue_read_field(&device->runtime.noc, slot->queue,
                                            FT_TT_QUEUE_FIELD_ERROR, &error);
                device->group.wait_reads++;
                device->group.wait_read_bytes += 4u;
                device->group.wait_transactions += 1u;
                if (rc != FT_TT_OK)
                    return rc;
                if (error != FT_TT_QUEUE_KERR_NONE)
                    return FT_TT_ERROR;
                if (pn_tt_now_seconds() > deadline)
                    return FT_TT_EIO;
            }
        }
        /* RED PROOF `PN_TT_RED=fast_check`. The POLL is left alone and the CHECK is asked
         * about one descriptor further on, so the wait still ends when the card really is
         * finished and only the four-agreement check is under test. Green means the check
         * on this path is INERT — the same failure class as the four vacuous gates this
         * backend has already found — and every arm of the ladder is meaningless. It must
         * refuse, on the first flush, naming `completed_count`. */
        if (ft_tt_queue_check_completion_line(slot->queue, fields,
                                              device->red_fast_check ? expected + 1u : expected,
                                              &reason) != FT_TT_OK) {
            /* The FULL block on the way to a failure, and it is the ONLY place the dropped
             * assertion has to be recovered: a caller that reports a refusal must be able to
             * name `error`, and one more read costs nothing here. */
            (void)ft_tt_queue_read_control(&device->runtime.noc, slot->queue, fields);
            fprintf(stderr, "pn_tt: core %u completion refused: %s (state %s, error %s)\n",
                    slot->core_index, reason ? reason : "?",
                    ft_tt_queue_state_string(fields[FT_TT_QUEUE_FIELD_STATE]),
                    ft_tt_queue_kernel_error_string(fields[FT_TT_QUEUE_FIELD_ERROR]));
            return FT_TT_ERROR;
        }
        return FT_TT_OK;
    }

    for (;;) {
        uint32_t value = 0;
        int rc = ft_tt_queue_read_field(&device->runtime.noc, slot->queue,
                                        FT_TT_QUEUE_FIELD_COMPLETED_COUNT, &value);

        device->group.poll_reads++;
        device->group.wait_reads++;
        device->group.wait_read_bytes += 4u;
        device->group.wait_transactions += 1u;
        if (rc != FT_TT_OK)
            return rc;
        if (value >= expected)
            break;
        /* A COMPLETION POLL IS A NoC READ INTO THE POLLED CORE'S OWN L1, and that L1 is
         * what its reader is filling and its writer is draining. Backing the poll off is
         * therefore an experiment about CONTENTION, not about latency: if the op gets
         * faster while being watched less often, the watching was costing the op.
         * 0 (the default) is the tightest loop and what every recorded figure was taken
         * with. */
        if (device->poll_backoff_us)
            usleep(device->poll_backoff_us);
        if ((++poll & 127u) == 0u) {
            uint32_t error = 0;

            rc = ft_tt_queue_read_field(&device->runtime.noc, slot->queue,
                                        FT_TT_QUEUE_FIELD_ERROR, &error);
            if (rc != FT_TT_OK)
                return rc;
            device->group.wait_reads++;
            device->group.wait_read_bytes += 4u;
            device->group.wait_transactions += 1u;
            if (error != FT_TT_QUEUE_KERR_NONE)
                return FT_TT_ERROR;
            if (pn_tt_now_seconds() > deadline)
                return FT_TT_EIO;
        }
    }
    /* Completion is four independent agreements, not a count: the state is not the host's
     * poison, the count is exact, the TOKEN is this run's nonce for this descriptor, and the
     * kernel reports no error. Each of them fails a different lie about work that did not
     * happen (ft_tt_queue.h rule 1).
     *
     * THE WIDE READ, and it is what the per-member term costs. This read is paid ONCE PER
     * MEMBER PER PROJECTION, so its TRANSACTION COUNT is the slope of the whole design in
     * the core count. Three widths have been measured on it:
     *
     *   64 B through the UC/WORD32 accessor   16 transactions   ~53 us   (the original)
     *   the five-word window, WORD32           5 transactions   ~21 us
     *   words 4..11, BLOCK16 over L1           2 transactions   ~7 us    (this)
     *
     * The bytes went UP from 20 to 32 and the cost went DOWN, which is the whole lesson of
     * this link: reads are transaction-count bound, so the fix for a hot-path device read is
     * to make it WIDER, not smaller. Every one of the four assertions is still inside it. */
    device->group.wait_reads++;
    device->group.wait_read_bytes += 32u;
    device->group.wait_transactions += 2u;
    if (ft_tt_queue_read_completion_wide(&device->runtime.bulk_l1, slot->queue, fields) !=
        FT_TT_OK)
        return FT_TT_ERROR;
    if (ft_tt_queue_check_completion(slot->queue, fields, expected, &reason) != FT_TT_OK) {
        /* The FULL block only on the way to a failure, where HEAD, TAIL and the idle spins
         * are what a human needs and one more read costs nothing. */
        (void)ft_tt_queue_read_control(&device->runtime.noc, slot->queue, fields);
        fprintf(stderr, "pn_tt: core %u completion refused: %s (state %s, error %s)\n",
                slot->core_index, reason ? reason : "?",
                ft_tt_queue_state_string(fields[FT_TT_QUEUE_FIELD_STATE]),
                ft_tt_queue_kernel_error_string(fields[FT_TT_QUEUE_FIELD_ERROR]));
        return FT_TT_ERROR;
    }
    return FT_TT_OK;
}

/**
 * ASK EVERY MEMBER AT ONCE, by reading the block they post into.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS A DIFFERENT COST AND NOT THE SAME COST REARRANGED
 * ---------------------------------------------------------------------------
 * Reads on this link are TRANSACTION-COUNT bound (DEV_NOTES.md §6: 8-byte loads read
 * 2.3 MB/s, 16-byte loads 4.7). Asking members one at a time costs a five-word control
 * read EACH — five transactions a member, ~21 us, 169 times a token. The members' slots
 * are CONTIGUOUS in member 0's L1, so the same question is 8N bytes of one core, and
 * through the BLOCK16 L1 accessor that is `ceil(8N / 16)` transactions instead of 5N.
 * At N=8 it is 4 against 40.
 *
 * IT MUST NOT BE THE POLL, THOUGH. The block costs N/2 transactions and a single control
 * word costs one, so a loop that read the block every iteration would be N/2 times more
 * expensive than the loop it replaces. So the wait is in two stages:
 *
 *   1. poll ONE word — member 0's own `completed_count`. Member 0 is the straggler by
 *      construction: it is handed the widest panel of a ragged split, and the recorded
 *      per-member wait puts ~90% of the group's on it. This is the same poll, at the same
 *      cost, as the single-core path.
 *   2. read the WHOLE slot array once, wide, and check every member.
 *
 * If a member is somehow still behind, stage 2 simply repeats — which is correct and rare,
 * and the shape it degrades to is the one that was cheap in the first place.
 *
 * TWO ASSERTIONS PER MEMBER, and each fails a different lie. The SEQUENCE says how far that
 * member got; the TOKEN is the nonce this run stamped on that member's own descriptor and
 * says it executed THIS one. The token is compared per member against that member's OWN
 * queue rather than one group-wide value, because a solo projection advances member 0's
 * issue counter and nobody else's — after the first weight too narrow to split, the members
 * are permanently out of step and a single expected token could never match again.
 */
static bool pn_tt_group_slots_satisfied(const pn_tt_matmul_group *group,
                                        const ft_tt_queue_group_slot *slots)
{
    uint32_t member;

    for (member = 0; member < group->member_count; member++) {
        uint32_t want = group->pending_sequence[member];

        if (!group->pending[member])
            continue;
        if (group->red_wait_early) {
            /* One SHORT — and the token assertion goes with it, because the token names the
             * exact descriptor and leaving it in would make the arm wait the full amount,
             * which would prove nothing. */
            if (want > 1u && slots[member].sequence >= want - 1u)
                continue;
        } else if (slots[member].sequence >= want &&
                   slots[member].token == group->pending_token[member]) {
            continue;
        }
        return false;
    }
    return true;
}

/** Has any member's kernel REFUSED a descriptor? A refusal posts nothing, so a sequence
 *  that never advances is all the block would ever show and the host would pay the whole
 *  timeout for something the card knew in microseconds. */
static int pn_tt_group_check_errors(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t member;

    for (member = 0; member < group->member_count; member++) {
        uint32_t error = 0;
        int rc;

        if (!group->pending[member])
            continue;
        rc = ft_tt_queue_read_field(&device->runtime.noc, group->member[member].queue,
                                    FT_TT_QUEUE_FIELD_ERROR, &error);
        if (rc != FT_TT_OK)
            return rc;
        if (error != FT_TT_QUEUE_KERR_NONE) {
            fprintf(stderr, "pn_tt: core %u refused a descriptor: %s\n",
                    group->member[member].core_index,
                    ft_tt_queue_kernel_error_string(error));
            return FT_TT_ERROR;
        }
    }
    return FT_TT_OK;
}

static int pn_tt_group_await_all(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    const double deadline = pn_tt_now_seconds() + (double)PN_TT_OP_TIMEOUT_MS / 1000.0;
    ft_tt_queue_group_slot slots[PN_TT_MATMUL_MAX_CORES];
    const uint32_t expected = group->member[0].queue->issued -
                              (group->red_wait_early && group->member[0].queue->issued > 0u ? 1u
                                                                                            : 0u);
    unsigned poll = 0;

    /* Stage 1 — ONE word, member 0, exactly the single-core poll. */
    for (;;) {
        uint32_t value = 0;
        int rc = ft_tt_queue_read_field(&device->runtime.noc, group->member[0].queue,
                                        FT_TT_QUEUE_FIELD_COMPLETED_COUNT, &value);

        group->poll_reads++;
        group->wait_reads++;
        group->wait_read_bytes += 4u;
        group->wait_transactions += 1u;
        if (rc != FT_TT_OK)
            return rc;
        if (value >= expected)
            break;
        if (device->poll_backoff_us)
            usleep(device->poll_backoff_us);
        if ((++poll & 127u) == 0u) {
            rc = pn_tt_group_check_errors(device);
            if (rc != FT_TT_OK)
                return rc;
            if (pn_tt_now_seconds() > deadline) {
                fprintf(stderr, "pn_tt: member 0 did not reach %u descriptors in %u ms\n",
                        expected, (unsigned)PN_TT_OP_TIMEOUT_MS);
                return FT_TT_EIO;
            }
        }
    }

    /* Stage 2 — the whole slot array, wide, through the BLOCK16 L1 accessor. */
    for (;;) {
        int rc = ft_tt_queue_group_read_slots(&device->runtime.bulk_l1, group->member[0].core_x,
                                              group->member[0].core_y, group->group_address,
                                              group->member_count, slots);

        group->group_reads++;
        group->wait_reads++;
        /* `ceil(8N/16)` 16-byte units, which is the whole reason the block exists. */
        group->wait_read_bytes += ((uint64_t)group->member_count * 8u + 15u) & ~(uint64_t)15u;
        group->wait_transactions += ((uint64_t)group->member_count * 8u + 15u) / 16u;
        if (rc != FT_TT_OK) {
            fprintf(stderr, "pn_tt: could not read the group completion block: %s\n",
                    ft_tt_error_string(rc));
            return rc;
        }
        if (pn_tt_group_slots_satisfied(group, slots))
            return FT_TT_OK;
        rc = pn_tt_group_check_errors(device);
        if (rc != FT_TT_OK)
            return rc;
        if (pn_tt_now_seconds() > deadline) {
            fprintf(stderr, "pn_tt: the group did not reach sequence %u in %u ms\n",
                    group->sequence, (unsigned)PN_TT_OP_TIMEOUT_MS);
            return FT_TT_EIO;
        }
    }
}

/**
 * Count the DRAM reads this descriptor will make the card issue, from the descriptor itself.
 *
 * `resident_matmul_reader.c` is three nested loops over `mt < Mt`, `nt < Nt step NB`,
 * `kt < Kt step depth`, and in every one of those passes it issues ONE A read of
 * `depth * 2048 B` and `columns` B reads of `depth * tile_bytes` each. So the byte totals
 * are a function of the shape words, the depth and the column block, with nothing to measure
 * and nothing to estimate — and the DEPTH DROPS OUT of both, because a bigger block is
 * proportionally fewer of them.
 *
 * The kernel collapses a depth that does not divide Kt to 1, and that collapse is mirrored
 * here rather than assumed away: it changes the request SIZE, which is the thing this
 * accounting exists to report, even though it leaves the totals alone.
 *
 * A IS COUNTED Nt/NB TIMES ON PURPOSE. Its offset does not depend on `nt`, so at Mt = 1 the
 * same Kt tiles are re-fetched for every PASS the member makes over its columns. At NB = 1
 * that is every column and A is 65.3% of the token; NB is the lever that divides it.
 */
static void pn_tt_group_count_reads(pn_tt_device *device, const pn_tt_weight *weight,
                                    const uint32_t *parameters, uint32_t a_bank,
                                    uint32_t b_bank, bool first_member)
{
    pn_tt_matmul_group *group = &device->group;
    const uint64_t tile_rows = parameters[FT_TT_QUEUE_MATMUL_TILE_ROWS];
    const uint64_t tile_cols = parameters[FT_TT_QUEUE_MATMUL_TILE_COLS];
    const uint64_t tile_inner = parameters[FT_TT_QUEUE_MATMUL_TILE_INNER];
    const uint32_t asked = parameters[FT_TT_QUEUE_MATMUL_DEPTH];
    const uint64_t depth = (asked > 0u && tile_inner % asked == 0u) ? asked : 1u;
    const uint32_t block_word = parameters[FT_TT_QUEUE_MATMUL_COLUMN_BLOCK];
    const uint64_t block = ftmm_column_block(block_word);
    const uint64_t rows = ftmm_row_block(block_word);
    uint64_t column_blocks, row_blocks, blocks, a_bytes, b_bytes;

    if (tile_rows == 0u || tile_cols == 0u || tile_inner == 0u)
        return;
    /* A IS COUNTED ONCE PER ROW PER COLUMN BLOCK, B ONCE PER COLUMN PER ROW BLOCK — that
     * symmetry IS the lever, and it is why the two divisors go into DIFFERENT counters. The
     * last block of either axis is short, which costs A nothing (the pass happens either
     * way) and costs B nothing (only the real columns are read).
     *
     * FALSIFIABLE: `a_bytes` must NOT move with MB and `b_bytes` must NOT move with NB. An
     * arm reporting a saving the knob never armed is visible as those two numbers. */
    column_blocks = (tile_cols + block - 1u) / block;
    row_blocks = (tile_rows + rows - 1u) / rows;
    blocks = row_blocks * column_blocks * (tile_inner / depth);
    a_bytes = tile_rows * column_blocks * tile_inner * (uint64_t)PN_TT_TILE_BF16;
    /* A-REUSE: the kernel's own condition (ring == Kt, MB == 1) — A once per row. */
    if ((parameters[FT_TT_QUEUE_MATMUL_B_N_MAJOR] & (1u << 18)) != 0u &&
        pn_tt_a_reuse_fires(pn_tt_a_reuse_tiles(), tile_inner, rows))
        a_bytes = tile_rows * tile_inner * (uint64_t)PN_TT_TILE_BF16;
    b_bytes = row_blocks * tile_cols * tile_inner * (uint64_t)device->resident_tile_bytes;

    group->read_blocks += blocks;
    group->a_read_bytes += a_bytes;
    group->b_read_bytes += b_bytes;
    {
        /* The reader's own a_half test (resident_matmul_reader.c): row-0 bit, Mt 1, MB 1. */
        const uint32_t flags = parameters[FT_TT_QUEUE_MATMUL_B_N_MAJOR];
        const bool half = (flags & (1u << 10)) != 0u && tile_rows == 1u && rows == 1u;
        const uint64_t wire_a = half ? a_bytes / 2u : a_bytes;
        uint32_t row;

        group->wire_a_bytes += wire_a;
        if (tile_rows > 1u) {
            group->multi_row_a_bytes += a_bytes;
            group->multi_row_b_bytes += b_bytes;
        }
        for (row = 0; row < group->read_census_rows; row++)
            if (group->read_census[row].kt == weight->tile_rows &&
                group->read_census[row].nt == weight->tile_cols)
                break;
        if (row == group->read_census_rows && row < 48u) {
            memset(&group->read_census[row], 0, sizeof(group->read_census[row]));
            group->read_census[row].kt = weight->tile_rows;
            group->read_census[row].nt = weight->tile_cols;
            group->read_census[row].tile_bytes = device->resident_tile_bytes;
            group->read_census_rows++;
        }
        if (row < group->read_census_rows) {
            if (first_member) {
                group->read_census[row].calls++;
                if (tile_rows > 1u)
                    group->read_census[row].multi_row_calls++;
            }
            group->read_census[row].a_bytes += a_bytes;
            group->read_census[row].wire_a_bytes += wire_a;
            group->read_census[row].b_bytes += b_bytes;
        }
    }
    if (a_bank < FT_TT_MAX_DRAM_BANKS)
        group->bank_read_bytes[a_bank] += a_bytes;
    if (b_bank < FT_TT_MAX_DRAM_BANKS)
        group->bank_read_bytes[b_bank] += b_bytes;
}

/**
 * ISSUE one projection: N parameter blocks, N descriptors, N head bumps, and no wait.
 *
 * ISSUE EVERYTHING BEFORE WAITING FOR ANYTHING. The members are independent residencies, so
 * a member that is waited on before the last one has been issued would have the other
 * members' issue cost serialised behind its own compute.
 */
/**
 * A NEW CHAIN starts with no member-0 PULL owed. Called by BOTH issue paths: a chain that
 * began on the prefill GRID used to skip this, so its first decode-group projection (a solo
 * `ssm_alpha` after the grid's split `qkv` and `z`) inherited the PREVIOUS chain's pull —
 * 16 members up to a sequence of the last decode — and after an idle re-arm (sequences back
 * at 0) member 0 waited on a sequence no member would reach: a long prompt after a pause
 * aborted. The arm clears it too. RED `grid_chain_stale` skips both and must abort that shape
 * (skipping only one is IDENTICAL: either reset alone closes it — measured).
 */
static void pn_tt_group_chain_begin(pn_tt_matmul_group *group, double now, bool grid)
{
    if (group->chained != 0u || (grid && pn_tt_red("grid_chain_stale")))
        return;
    group->chain_pull_members = 0u;
    group->chain_last_pulled = false;
    group->chain_started = now;
    group->chain_bytes_from = group->a_read_bytes + group->b_read_bytes;
    group->chain_projections = 0u;
}

/**
 * Publish member 0's held descriptor (pn_tt_matmul_pull.h). `last`: the flush awaits it, so it
 * keeps the pull it was written with; otherwise another projection of the chain follows and it
 * goes without one. A no-op when nothing is held.
 */
static int pn_tt_group_release_m0(pn_tt_device *device, bool last)
{
    pn_tt_matmul_group *group = &device->group;
    ft_tt_queue_descriptor descriptor = group->m0_held_descriptor;
    int rc;

    if (!group->m0_held)
        return FT_TT_OK;
    group->m0_held = false;
    if (!last && descriptor.a3 != 0u) {
        descriptor.a3 = 0u;
        descriptor.a4 = 0u;
        group->pulls_deferred++;
    }
    rc = ft_tt_queue_stage(pn_tt_issue_access(&device->runtime, group->issue_shape),
                           group->member[0].queue, &descriptor, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_publish(&device->runtime.noc, group->member[0].queue);
    return rc;
}

static int pn_tt_group_issue(pn_tt_device *device, const pn_tt_weight *weight,
                             const pn_tt_activation *input, const pn_tt_activation *output,
                             bool split, uint32_t members)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t parameters[FT_TT_QUEUE_PARAM_WORDS];
    double issued_from = pn_tt_now_seconds();
    /* BEFORE this projection's member 0 reads `issued` for its parameter slot and token. */
    int held_rc = pn_tt_group_release_m0(device, false);

    if (held_rc != FT_TT_OK)
        return held_rc;

    pn_tt_group_chain_begin(group, issued_from, false);
    device->group.chain_projections++;
    uint32_t member;

    /* 0 means "post nothing" to the kernel, so the sequence skips it on the wrap rather
     * than silently disarming every member's post for one projection. */
    if (++group->sequence == 0u)
        group->sequence = 1u;

    for (member = 0; member < members; member++) {
        pn_tt_matmul_member *slot = &group->member[member];
        ft_tt_queue_descriptor descriptor;
        uint32_t address = 0u;
        uint32_t index;
        int rc;

        pn_tt_group_parameters(device, weight, input, output, member, split, parameters);
        /* S3b: a FUSED descriptor is the gate member's own plus the fused bits over word 15
         * (whose low 24 bits stay the block word). */
        if (group->swiglu_up)
            parameters[FT_TT_QUEUE_MATMUL_COLUMN_BLOCK] |= group->swiglu_bits;
        /* S3a: a RESADD descriptor is the projection's own plus the RESADD bits. */
        if (group->resadd_residual)
            parameters[FT_TT_QUEUE_MATMUL_COLUMN_BLOCK] |= group->resadd_bits;

        /* RED PROOF. Nt = 0 is the one malformation that is unambiguously a REFUSAL and not
         * a wrong answer. Injected AFTER the host check on purpose — the point is to run the
         * KERNEL's refusal on real silicon. */
        if (member == 0u && device->red_bad_descriptor != 0u &&
            device->matvec_count + 1u == device->red_bad_descriptor) {
            fprintf(stderr, "pn_tt: RED PROOF — sending a malformed descriptor (Nt=0) as "
                            "matvec %llu\n", (unsigned long long)device->red_bad_descriptor);
            parameters[FT_TT_QUEUE_MATMUL_TILE_COLS] = 0u;
            device->red_bad_descriptor = 0u;
        }

        /* The parameter slot is THIS descriptor's own index modulo the slot count, so a
         * chained projection cannot overwrite the parameters of one the card has not read
         * yet. It is read before the enqueue, which is what makes it that descriptor's. */
        index = slot->queue->issued;
        /* THE CACHED BLOCK FIRST. A hit costs no device access at all; a miss costs the
         * same 64-byte write as the rotating path and never happens again for those words.
         * A 0 address means "not cached and not cacheable right now" — fall through. */
        address = 0u;
        if (group->param_cache) {
            bool wrote = false;

            address = pn_tt_param_cache_address(device, slot, parameters, &wrote);
            if (address != 0u && wrote) {
                group->issue_writes++;
                group->issue_write_bytes += FT_TT_QUEUE_MATMUL_PARAM_BYTES;
                group->issue_accesses += FT_TT_QUEUE_MATMUL_PARAM_BYTES /
                                         pn_tt_issue_unit_bytes(group->issue_shape);
            }
        }
        if (address == 0u) {
            rc = ft_tt_queue_write_op_parameters(pn_tt_issue_access(&device->runtime,
                                                                    group->issue_shape),
                                                 slot->queue,
                                                 index & (PN_TT_QUEUE_PARAM_SLOTS - 1u),
                                                 parameters, &address);
            /* THE LEDGER, at the point of issue. Three writes per member per projection —
             * 64 B of parameters, a 32 B descriptor and a 4 B head bump — and the write
             * direction is BANDWIDTH bound, so it is the BYTES that convert, not the count.
             * `ft_tt_queue_enqueue()` is the other two and is counted below it. */
            group->issue_writes++;
            group->issue_write_bytes += FT_TT_QUEUE_MATMUL_PARAM_BYTES;
            group->issue_accesses +=
                FT_TT_QUEUE_MATMUL_PARAM_BYTES / pn_tt_issue_unit_bytes(group->issue_shape);
            if (rc != FT_TT_OK)
                return rc;
        }

        memset(&descriptor, 0, sizeof(descriptor));
        descriptor.opcode = FT_TT_QUEUE_OP_MATMUL;
        descriptor.a0 = address;
        /* a1 is the DEBUG starve count — trailing K-blocks the reader deliberately withholds
         * so the compute thread is left in a real cb_wait_front. 0 for every real request. */
        descriptor.a1 = (split && members > 1u) ? group->g0_a1 : 0u;
        /* a3/a4 = M2 PULL: member 0 waits on the card for the members of the chain's last
         * split projection — that projection's own, or a split before a SOLO one (v2).
         * Written as if this descriptor ends the chain; a held one that does not is
         * published without it (pn_tt_matmul_pull.h). */
        if (member == 0u && group->pull_table_ok) {
            pn_tt_pull_chain chain = { group->chain_pull_members, group->chain_pull_target };
            pn_tt_pull_words words;

            pn_tt_pull_chain_note(&chain, split, members, group->sequence);
            words = pn_tt_pull_words_for(&chain, true, group->red_pull_skip,
                                         group->pull_serial);
            group->chain_pull_members = chain.members;
            group->chain_pull_target = chain.target;
            descriptor.a3 = words.a3;
            descriptor.a4 = words.a4;
            group->chain_last_pulled = words.a3 != 0u;
        } else if (member == 0u) {
            group->chain_last_pulled = false;
        }
        /* a2 is the WRITER'S DESTINATION ROW PITCH in tiles, and it is the weight's FULL Nt
         * rather than this member's slice: at Mt > 1 the panels concatenate along columns
         * INSIDE each output row, so the pitch is the whole row. **Left at 0 while Mt is 1**,
         * which is every descriptor today — 0 means `tile_cols` in the kernel and a single
         * tile-row's destinations are the running counter either way, so the wire bytes do
         * not move and a decode `cmp` is about the loops rather than about this word. */
        descriptor.a2 = weight->tile_rows > 0u && parameters[FT_TT_QUEUE_MATMUL_TILE_ROWS] > 1u
                            ? weight->tile_cols
                            : 0u;
        /* S3b: a fused descriptor is Mt = 1 with no stash, so a1/a2 carry the UP shard. */
        if (group->swiglu_up)
            pn_tt_swiglu_up_words(group->swiglu_up, member, &descriptor.a1, &descriptor.a2);
        /* S3a: a RESADD descriptor is Mt = 1 with no stash, so a1/a2 carry the RESIDUAL at
         * the member's first column. */
        if (group->resadd_residual)
            pn_tt_resadd_words(group->resadd_residual,
                               split ? weight->shard_column_first[member] : 0u,
                               (uint16_t)device->dram[group->resadd_residual->bank].x,
                               (uint16_t)device->dram[group->resadd_residual->bank].y,
                               &descriptor.a1, &descriptor.a2);
        /* a5 is the GROUP SEQUENCE, and it is what the writer posts beside the token. */
        descriptor.a5 = group->sequence;
        /* STAGE THEN PUBLISH, NOT `enqueue` — the two halves go through DIFFERENT accessors
         * on the wide shape, and the head bump is on `noc` on both because a four-byte write
         * cannot come through the 16-byte L1 accessor at all (pn_tt_issue.h). Member 0's is
         * HELD when pulls are deferred; nothing else stages on its queue before the release,
         * so it still lands at `index`. */
        if (member == 0u && group->pull_table_ok && group->pull_defer) {
            group->m0_held_descriptor = descriptor;
            group->m0_held = true;
            rc = FT_TT_OK;
        } else {
            rc = ft_tt_queue_stage(pn_tt_issue_access(&device->runtime, group->issue_shape),
                                   slot->queue, &descriptor, 1u);
            if (rc == FT_TT_OK)
                rc = ft_tt_queue_publish(&device->runtime.noc, slot->queue);
        }
        /* Two writes: the ring slot, then the head bump that publishes it. The head is a
         * separate write BY REQUIREMENT — it is what makes the descriptor visible, so it
         * cannot be merged with the payload it announces. */
        group->issue_writes += 2u;
        group->issue_write_bytes += FT_TT_QUEUE_DESCRIPTOR_BYTES + 4u;
        /* THE ACCESSES, which the bytes cannot say: a 32 B ring slot is eight WORD32 accesses
         * or two BLOCK16 ones, and the head bump is one word on both shapes. The parameter
         * block is added at its own call site above, because a cache HIT makes none. */
        group->issue_accesses +=
            FT_TT_QUEUE_DESCRIPTOR_BYTES / pn_tt_issue_unit_bytes(group->issue_shape) + 1u;
        group->issue_pairs++;
        if (rc != FT_TT_OK)
            return rc;
        group->pending_sequence[member] = group->sequence;
        group->pending_token[member] = ft_tt_queue_token_at(slot->queue, index);
        group->pending[member] = true;
        /* The bank A is REALLY read from, which is neither `input->bank` once the strip has
         * mirrors nor under the spread arm. Attributing those bytes to the nominal bank
         * would print a per-bank line showing one bank at several times its own ceiling,
         * and the per-bank table is the ONLY evidence that the spread worked at all — a
         * matmul that got faster for some other reason looks identical in every other
         * counter. ONE expression, shared with `pn_tt_group_parameters`. */
        pn_tt_group_count_reads(device, weight, parameters,
                                pn_tt_group_activation_bank(group, input, member),
                                split ? weight->shard_bank[member] : weight->bank,
                                member == 0u);
        /* The up panel's B bytes (and the A re-reads its passes make) are the same shape. */
        if (group->swiglu_up)
            pn_tt_group_count_reads(device, group->swiglu_up, parameters,
                                    pn_tt_group_activation_bank(group, input, member),
                                    group->swiglu_up->shard_bank[member], false);
    }

    group->chain_issue_end = pn_tt_now_seconds();
    group->issue_seconds += group->chain_issue_end - issued_from;
    group->projections++;
    group->chained++;
    device->resident_ops++;
    if (split)
        group->split_ops++;
    else
        group->solo_ops++;
    return FT_TT_OK;
}

int pn_tt_matmul_group_flush(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    double waiting_from, mark;
    double census_done_await = 0.0, census_busy_await = 0.0;
    uint64_t census_done_count = 0u, census_busy_count = 0u, census_polls = 0u;
    uint32_t member;
    int rc = FT_TT_OK;

    if (group->chained == 0u)
        return FT_TT_OK;
    if (group->grid_running) {
        int grc = pn_tt_grid_flush(device);

        if (grc != FT_TT_OK)
            return grc;
    }
    group->flushes++;
    /* The chain's last member-0 descriptor goes out WITH its pull, before anything waits. */
    rc = pn_tt_group_release_m0(device, true);
    if (rc != FT_TT_OK)
        return rc;
    waiting_from = pn_tt_now_seconds();

    if (group->pull_table_ok && group->chain_last_pulled && group->pending[0]) {
        /* M2: member 0 posted only after every member reached its sequence, and every
         * member runs its descriptors in order, so member 0's count covers the chain. */
        group->wait_pairs++;
        rc = pn_tt_group_await(device, &group->member[0], group->member[0].queue->issued);
        mark = pn_tt_now_seconds();
        group->member_wait_seconds[0] += mark - waiting_from;
        group->pulled_flushes++;
        census_busy_await = mark - waiting_from;
        census_busy_count = 1u;
    } else if (group->group_completion && group->member_count > 1u) {
        rc = pn_tt_group_await_all(device);
        mark = pn_tt_now_seconds();
        /* The whole wait is charged to member 0, because that is the core the host read —
         * a per-member split is not observable through an aggregated poll, and inventing
         * one would report a number nothing measured. */
        group->member_wait_seconds[0] += mark - waiting_from;
    } else {
        mark = waiting_from;
        census_done_await = census_busy_await = 0.0;
        census_done_count = census_busy_count = census_polls = 0u;
        for (member = 0; member < group->member_count; member++) {
            double member_from = mark;
            const uint64_t polls_from = group->poll_reads;

            if (!group->pending[member])
                continue;
            group->wait_pairs++;
            rc = pn_tt_group_await(device, &group->member[member],
                                   group->member[member].queue->issued);
            mark = pn_tt_now_seconds();
            group->member_wait_seconds[member] += mark - member_from;
            census_polls += group->poll_reads - polls_from;
            if (group->poll_reads - polls_from <= 1u) {
                census_done_await += mark - member_from;
                census_done_count++;
            } else {
                census_busy_await += mark - member_from;
                census_busy_count++;
            }
            if (rc != FT_TT_OK)
                break;
        }
        mark = pn_tt_now_seconds();
    }

    group->wait_seconds += mark - waiting_from;
    {
        const uint64_t bytes =
            group->a_read_bytes + group->b_read_bytes - group->chain_bytes_from;
        uint32_t row;

        for (row = 0; row < group->flush_census_rows; row++)
            if (group->flush_census[row].bytes == bytes)
                break;
        if (row == group->flush_census_rows && row < 32u) {
            memset(&group->flush_census[row], 0, sizeof(group->flush_census[row]));
            group->flush_census[row].bytes = bytes;
            group->flush_census_rows++;
        }
        if (row < group->flush_census_rows) {
            group->flush_census[row].flushes++;
            group->flush_census[row].projections += group->chain_projections;
            group->flush_census[row].span += mark - group->chain_started;
            group->flush_census[row].wait += mark - waiting_from;
            group->flush_census[row].issue += group->chain_issue_end - group->chain_started;
            group->flush_census[row].gap += waiting_from - group->chain_issue_end;
            group->flush_census[row].done_await += census_done_await;
            group->flush_census[row].done_count += census_done_count;
            group->flush_census[row].busy_await += census_busy_await;
            group->flush_census[row].busy_count += census_busy_count;
            group->flush_census[row].polls += census_polls;
            pn_tt_group_class_clock(device, row);
        } else {
            group->flush_census_dropped++;
        }
    }
    group->chained = 0u;
    if (rc == FT_TT_OK)
        pn_tt_group_mark_alive(group, mark, false);
    memset(group->pending, 0, sizeof(group->pending));
    return rc;
}

/**
 * Has the group sat idle past the shared limit with NOTHING outstanding? Asks nothing of the
 * card; the retire and re-arm are the caller's (`pn_tt_resident_idle_refresh()`, which owns
 * the arm). `PN_TT_RED=matmul_idle_retire_off` answers false always, for the arm that proves
 * an idle group wedges without the guard.
 */
bool pn_tt_matmul_group_idle_expired(const pn_tt_device *device, double *out_idle_seconds)
{
    const pn_tt_matmul_group *group = &device->group;
    const long limit_ms = pn_tt_residency_idle_retire_limit_ms();
    double idle;

    /* L1a.3: no idle re-arm while members serve the grid (the arm would refuse). */
    if (limit_ms == 0 || group->chained != 0u || group->last_alive_seconds <= 0.0 ||
        group->grid_lent || group->members_lent || pn_tt_red("matmul_idle_retire_off") ||
        !pn_tt_idle_retire_allowed("matmul"))
        return false;
    idle = pn_tt_now_seconds() - group->last_alive_seconds;
    if (out_idle_seconds)
        *out_idle_seconds = idle;
    return idle * 1000.0 >= (double)limit_ms;
}

/**
 * KEEPALIVE (PM 5): one NOP descriptor (a0 0, a5 0 = post nothing) to every running member of
 * an IDLE decode group — nothing chained, nothing pending, idle >= `min_idle_seconds` and not
 * yet past the retire limit — staged and published exactly as a projection is, then each
 * member awaited on its own count. The group's sequence, tokens and PULL table are untouched:
 * a NOP posts nothing and the next projection's expectations come from `queue->issued`.
 * @return members NOP'd, 0 if not due, -1 on a failure (the next projection then retires and
 *         re-arms the group as it always has)
 */
int pn_tt_matmul_group_keepalive(pn_tt_device *device, double min_idle_seconds)
{
    pn_tt_matmul_group *group = &device->group;
    const double limit = (double)pn_tt_residency_idle_retire_limit_ms() / 1000.0;
    double idle;
    uint32_t member;
    int sent = 0;

    if (!device->resident_running || group->chained != 0u || group->last_alive_seconds <= 0.0)
        return 0;
    for (member = 0; member < group->member_count; member++)
        if (group->pending[member])
            return 0;
    idle = pn_tt_now_seconds() - group->last_alive_seconds;
    if (idle < min_idle_seconds || (limit > 0.0 && idle >= limit - 0.5))
        return 0;
    for (member = 0; member < group->member_count; member++) {
        pn_tt_matmul_member *slot = &group->member[member];
        ft_tt_queue_descriptor descriptor;

        if (!slot->running)
            continue;
        memset(&descriptor, 0, sizeof(descriptor));
        descriptor.opcode = FT_TT_QUEUE_OP_NOP;
        if (ft_tt_queue_stage(pn_tt_issue_access(&device->runtime, group->issue_shape),
                              slot->queue, &descriptor, 1u) != FT_TT_OK ||
            ft_tt_queue_publish(&device->runtime.noc, slot->queue) != FT_TT_OK)
            return -1;
        sent++;
    }
    for (member = 0; member < group->member_count; member++)
        if (group->member[member].running &&
            pn_tt_group_await(device, &group->member[member],
                              group->member[member].queue->issued) != FT_TT_OK)
            return -1;
    pn_tt_group_mark_alive(group, pn_tt_now_seconds(), true);
    return sent;
}

/* ===========================================================================
 * PHASE G — THE PREFILL GRID
 * =========================================================================== */

/** Set only while the grid cycle gate runs: where each grid boot's host time goes (P-boot). */
static ft_tt_runtime_boot_timing *g_grid_boot_timing;

/** Boot the grid's firmware ONCE (a process's first prefill). Sequential, like the members. */
static int pn_tt_grid_boot(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    /* DEFAULT 48 (G option 3 ships with on-card dispatch: its decode tax is 0.1% on
     * replayed tokens, run_d_ceilings.sh). 0 = the decode group serves the prefill. */
    const uint32_t count = pn_tt_env_u32("PN_TT_PREFILL_GRID", 48u);
    const uint32_t first = pn_tt_env_u32("PN_TT_PREFILL_GRID_FIRST", 0u);
    uint32_t index;

    if (group->grid_booted)
        return 0;
    if (count == 0u)
        return -1;
    if (count > 64u || group->member_count < 2u || count % group->member_count != 0u) {
        fprintf(stderr, "pn_tt: PN_TT_PREFILL_GRID=%u must be a multiple of the %u decode "
                        "members, at most 64 — the grid stays off\n",
                count, group->member_count);
        return -1;
    }
    /* A RESIDENT FAMILY ON A GRID CORE IS A WEDGE, NOT A SHARE: residencies live for the
     * whole process (decode and prefill alike), and grid firmware booted under one leaves two
     * programs on one Tensix. Refused by name before any core boots; the grid stays off. */
    for (index = 0; index < count; index++) {
        const char *holder = pn_tt_core_holder(device, first + index, NULL);

        if (holder) {
            fprintf(stderr, "pn_tt: prefill grid core %u is held by %s — the grid stays off. "
                            "Move that family or set PN_TT_PREFILL_GRID_FIRST.\n",
                    first + index, holder);
            return -1;
        }
    }
    for (index = 0; index < count; index++) {
        pn_tt_matmul_member *slot = &group->grid[index];
        ft_tt_noc_coord core;

        memset(slot, 0, sizeof(*slot));
        slot->queue = &slot->own_queue;
        slot->core_index = first + index;
        if (ft_tt_coord_tensix_by_index(&device->runtime.grid, slot->core_index,
                                        FT_TT_COORD_TRANSLATED, &core) != FT_TT_OK)
            return -1;
        slot->core_x = (uint16_t)core.x;
        slot->core_y = (uint16_t)core.y;
        if (!pn_tt_core_boot_is_permitted(&device->runtime, slot->core_index,
                                          "a prefill grid core", "PN_TT_PREFILL_GRID_FIRST") ||
            ft_tt_runtime_boot_core_timed(&device->runtime.noc, slot->core_x, slot->core_y,
                                          &device->firmware, g_grid_boot_timing) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: prefill grid core %u could not be booted\n",
                    slot->core_index);
            return -1;
        }
        slot->booted = true;
    }
    group->grid_count = count;
    group->grid_pieces = count / group->member_count;
    group->grid_stash_tiles = pn_tt_env_u32("PN_TT_PREFILL_GRID_STASH", 1152u);
    group->grid_booted = true;
    fprintf(stderr, "pn_tt: prefill GRID — %u cores from %u, %u column pieces per decode "
                    "shard, B stash %u tiles\n",
            count, first, group->grid_pieces, group->grid_stash_tiles);
    return 0;
}

static void pn_tt_grid_stop(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t index;

    for (index = 0; index < group->grid_count; index++)
        pn_tt_group_member_stop(device, &group->grid[index]);
    group->grid_running = false;
    memset(group->grid_pending, 0, sizeof(group->grid_pending));
    group->grid_detached = 0u;
    group->grid_ring = 0u;
    pn_tt_grid_marks_reset(&group->grid_marks, group->grid_count);
    group->grid_last_mark = PN_TT_GRID_MARK_NONE;
}

/**
 * PHASE G option 3 — PARK THE GRID after the prefill (`PN_TT_PREFILL_GRID_PARK=1`). A grid
 * left booted (firmware idle in GO-wait) costs decode ~1.1% (G2 round 2). Every RISC of every
 * grid core whose residency EXITED is held in reset, and the next prefill boots the grid
 * again. A core whose residency did not exit is left alone, never reset under a kernel.
 * STANDING HAZARD: re-booting has wedged cores here (BRISC stalled in `device_setup()`), so
 * this is opt-in until its own wedge gate (`PN_TT_PREFILL_GRID_CYCLES`) passes.
 * @return how many cores were parked
 */
static uint32_t pn_tt_grid_park(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t index, parked = 0u;

    for (index = 0; index < group->grid_count; index++) {
        pn_tt_matmul_member *slot = &group->grid[index];
        uint32_t pending = PN_TT_TRIAD_MASK;

        if (!slot->booted)
            continue;
        if (ft_tt_queue_all_processors_exited(&device->runtime.noc, slot->queue,
                                              PN_TT_TRIAD_MASK, &pending) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: grid core %u has not exited (0x%x) — NOT parked\n",
                    slot->core_index, pending);
            continue;
        }
        if (ft_tt_launch_assert_reset(&device->runtime.noc, slot->core_x, slot->core_y,
                                      FT_TT_SOFT_RESET_ALL_RISCS) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: grid core %u could not be parked\n", slot->core_index);
            continue;
        }
        slot->booted = false;
        parked++;
    }
    if (parked == group->grid_count)
        group->grid_booted = false;
    return parked;
}

/** The grid's wedge gate: `PN_TT_PREFILL_GRID_CYCLES=N` boot -> start -> stop -> park the
 *  whole grid N times before the first real prefill, and say how many cycles went clean and
 *  where each one's time went (boot split by ft_tt_runtime_boot_core_timed; P-boot). */
static void pn_tt_grid_cycle_gate(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    const uint32_t cycles = pn_tt_env_u32("PN_TT_PREFILL_GRID_CYCLES", 0u);
    /* PROBE: `PN_TT_GRID_CYCLE_ATTN=1` also arms, retires
     * and parks the attention program on every grid core each cycle (pn_tt_attention_grid.h),
     * and the gate prints what each half of the swap cost. */
    const bool attention = pn_tt_attention_grid_cycle_wanted();
    double boot_s = 0.0, start_s = 0.0, stop_s = 0.0, park_s = 0.0, attn_up = 0.0, attn_down = 0.0;
    uint32_t cores[sizeof(group->grid) / sizeof(group->grid[0])];
    ft_tt_runtime_boot_timing boot_timing;
    static bool done;
    uint32_t cycle, index, clean = 0u;
    double mark;

    if (done || cycles == 0u)
        return;
    done = true;
    memset(&boot_timing, 0, sizeof(boot_timing));
    g_grid_boot_timing = &boot_timing;
    for (cycle = 0; cycle < cycles; cycle++) {
        bool ok;

        mark = pn_tt_now_seconds();
        ok = pn_tt_grid_boot(device) == 0;
        boot_s += pn_tt_now_seconds() - mark;
        mark = pn_tt_now_seconds();
        for (index = 0; ok && index < group->grid_count; index++) {
            if (pn_tt_slot_start(device, &group->grid[index], index,
                                 device->resident_tile_bytes, device->resident_depth,
                                 true) != FT_TT_OK) {
                fprintf(stderr, "pn_tt: GRID CYCLE %u: core %u did not start\n", cycle,
                        group->grid[index].core_index);
                ok = false;
                break;
            }
            group->grid[index].running = true;
        }
        start_s += pn_tt_now_seconds() - mark;
        mark = pn_tt_now_seconds();
        pn_tt_grid_stop(device);
        stop_s += pn_tt_now_seconds() - mark;
        mark = pn_tt_now_seconds();
        if (pn_tt_grid_park(device) != group->grid_count)
            ok = false;
        park_s += pn_tt_now_seconds() - mark;
        if (ok && attention) {
            pn_tt_attention_grid_cycle_time spent;

            for (index = 0; index < group->grid_count; index++)
                cores[index] = group->grid[index].core_index;
            memset(&spent, 0, sizeof(spent));
            if (pn_tt_attention_grid_cycle(device, cores, group->grid_count, &spent) != 0)
                ok = false;
            attn_up += spent.arm_seconds;
            attn_down += spent.park_seconds;
        }
        if (!ok) {
            fprintf(stderr, "pn_tt: GRID CYCLE %u FAILED — stopping the gate\n", cycle);
            break;
        }
        clean++;
    }
    g_grid_boot_timing = NULL;
    fprintf(stderr, "pn_tt: GRID CYCLE GATE — %u of %u boot/start/stop/park cycles clean%s\n",
            clean, cycles, attention ? " (with the attention program on every grid core)" : "");
    if (clean > 0u)
        fprintf(stderr, "pn_tt: GRID CYCLE TIMING (ms a cycle, %u cores) — boot %.2f (reset %.2f, "
                        "firmware images %.2f for %.0f KiB, handshake %.2f) start %.2f stop %.2f "
                        "park %.2f; attention arm %.2f retire+park %.2f\n",
                group->grid_count, boot_s * 1e3 / clean, boot_timing.reset_seconds * 1e3 / clean,
                boot_timing.images_seconds * 1e3 / clean,
                (double)boot_timing.image_bytes / 1024.0 / clean,
                boot_timing.handshake_seconds * 1e3 / clean, start_s * 1e3 / clean,
                stop_s * 1e3 / clean, park_s * 1e3 / clean, attn_up * 1e3 / clean,
                attn_down * 1e3 / clean);
}

int pn_tt_matmul_grid_begin(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t index;

    if (group->grid_running || !device->resident_running)
        return 0;
    pn_tt_grid_cycle_gate(device);
    if (pn_tt_grid_boot(device) != 0)
        return 0; /* off: the decode group serves the prefill as before */
    /* MEASURE (G option 3 tax bisect): `PN_TT_GRID_BISECT=1` boot then park, no start;
     * `=2` boot, start, stop, park — the grid then serves nothing (prefill as without G). */
    {
        const uint32_t bisect = pn_tt_env_u32("PN_TT_GRID_BISECT", 0u);

        if (bisect == 2u)
            for (index = 0; index < group->grid_count; index++)
                if (pn_tt_slot_start(device, &group->grid[index], index,
                                     device->resident_tile_bytes, device->resident_depth,
                                     true) == FT_TT_OK)
                    group->grid[index].running = true;
        if (bisect != 0u) {
            pn_tt_grid_stop(device);
            pn_tt_device_telemetry_log(device, "bisect booted");
            fprintf(stderr, "pn_tt: GRID BISECT %u — %u of %u cores parked, grid off\n",
                    bisect, pn_tt_grid_park(device), group->grid_count);
            return 0;
        }
    }
    for (index = 0; index < group->grid_count; index++) {
        if (pn_tt_slot_start(device, &group->grid[index], index, device->resident_tile_bytes,
                             device->resident_depth, true) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: prefill grid core %u did not start — grid off for this "
                            "prefill\n", group->grid[index].core_index);
            pn_tt_grid_stop(device);
            return 0;
        }
        group->grid[index].running = true;
    }
    group->grid_running = true;
    pn_tt_grid_marks_reset(&group->grid_marks, group->grid_count);
    group->grid_last_mark = PN_TT_GRID_MARK_NONE;
    /* The cores idle from their start: the guard counts from here. */
    group->grid_alive_seconds = pn_tt_now_seconds();
    pn_tt_device_telemetry_log(device, "grid started");
    return 0;
}

uint32_t pn_tt_matmul_grid_lend(pn_tt_device *device, uint32_t *cores, uint32_t max)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t index, parked;

    if (!group->grid_running || !cores || max < group->grid_count)
        return 0u;
    if (pn_tt_grid_flush(device) != 0)
        return 0u;
    pn_tt_grid_stop(device);
    parked = pn_tt_grid_park(device);
    if (parked != group->grid_count) {
        fprintf(stderr, "pn_tt: grid LEND — %u of %u cores parked; lending none\n", parked,
                group->grid_count);
        return 0u;
    }
    for (index = 0; index < group->grid_count; index++)
        cores[index] = group->grid[index].core_index;
    return group->grid_count;
}

/** Has `slot`'s triad exited (after a stop)? */
static bool pn_tt_slot_exited(pn_tt_device *device, pn_tt_matmul_member *slot)
{
    uint32_t pending = PN_TT_TRIAD_MASK;

    return ft_tt_queue_all_processors_exited(&device->runtime.noc, slot->queue,
                                             PN_TT_TRIAD_MASK, &pending) == FT_TT_OK;
}

/** Add one grid slot on `core` at index `g`; `boot` boots the grid firmware there first. */
static int pn_tt_grid_add_slot(pn_tt_device *device, uint32_t g, uint32_t core, bool boot)
{
    pn_tt_matmul_member *slot = &device->group.grid[g];
    ft_tt_noc_coord coord;

    memset(slot, 0, sizeof(*slot));
    slot->queue = &slot->own_queue;
    slot->core_index = core;
    if (ft_tt_coord_tensix_by_index(&device->runtime.grid, core, FT_TT_COORD_TRANSLATED,
                                    &coord) != FT_TT_OK)
        return FT_TT_EINVAL;
    slot->core_x = (uint16_t)coord.x;
    slot->core_y = (uint16_t)coord.y;
    if (boot) {
        if (!pn_tt_core_boot_is_permitted(&device->runtime, core, "a lent grid core",
                                          "PN_TT_PREFILL_LEND") ||
            ft_tt_runtime_boot_core(&device->runtime.noc, slot->core_x, slot->core_y,
                                    &device->firmware) != FT_TT_OK)
            return FT_TT_EIO;
        slot->booted = true;
    }
    if (pn_tt_slot_start(device, slot, g, device->resident_tile_bytes, device->resident_depth,
                         true) != FT_TT_OK)
        return FT_TT_EIO;
    slot->running = true;
    return FT_TT_OK;
}

/** The grid at `count` cores: pieces and marks follow. */
static void pn_tt_grid_set_count(pn_tt_matmul_group *group, uint32_t count)
{
    group->grid_count = count;
    group->grid_pieces = count / group->member_count;
    pn_tt_grid_marks_reset(&group->grid_marks, count);
    group->grid_last_mark = PN_TT_GRID_MARK_NONE;
}

uint32_t pn_tt_matmul_grid_extend(pn_tt_device *device, uint32_t extra_core)
{
    pn_tt_matmul_group *group = &device->group;
    const uint32_t base = group->grid_count, members = group->member_count;
    uint32_t m, g;

    if (!group->grid_running || group->grid_lent || group->members_lent || members < 2u ||
        extra_core == UINT32_MAX ||
        base + members > (uint32_t)(sizeof(group->grid) / sizeof(group->grid[0])) ||
        (base + members) % members != 0u || pn_tt_grid_flush(device) != 0)
        return 0u;
    for (m = 1; m < members; m++) {
        pn_tt_group_member_stop(device, &group->member[m]);
        if (!pn_tt_slot_exited(device, &group->member[m])) {
            fprintf(stderr, "pn_tt: grid EXTEND — decode member %u (core %u) did not exit; "
                            "the grid stays at %u\n", m, group->member[m].core_index, base);
            pn_tt_matmul_group_retire(device);
            (void)pn_tt_matmul_group_arm(device, device->resident_tile_bytes,
                                         device->resident_depth);
            return 0u;
        }
    }
    group->grid_lent = true;
    group->grid_lent_base = base;
    for (g = base, m = 1; m < members; m++, g++)
        if (pn_tt_grid_add_slot(device, g, group->member[m].core_index, false) != FT_TT_OK)
            break;
    if (m == members && pn_tt_grid_add_slot(device, g, extra_core, true) == FT_TT_OK)
        g++;
    pn_tt_grid_set_count(group, g);
    if (g != base + members) {
        fprintf(stderr, "pn_tt: grid EXTEND — core %u did not start; giving the cores back\n",
                g < base + members - 1u ? group->member[g - base + 1u].core_index : extra_core);
        (void)pn_tt_matmul_grid_retract(device, NULL);
        return 0u;
    }
    fprintf(stderr, "pn_tt: grid EXTEND — %u -> %u cores (decode members 1-%u + core %u), %u "
                    "pieces a shard\n", base, g, members - 1u, extra_core, group->grid_pieces);
    return members;
}

uint32_t pn_tt_matmul_members_lend(pn_tt_device *device, uint32_t *cores, uint32_t max)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t m, lent = 0u;

    if (!device->resident_running || group->grid_lent || group->members_lent ||
        group->member_count < 2u || group->chained != 0u)
        return 0u;
    for (m = 1; m < group->member_count && lent < max; m++) {
        pn_tt_matmul_member *slot = &group->member[m];

        pn_tt_group_member_stop(device, slot);
        group->members_lent = true; /* from the first stop: the group must not re-arm */
        if (!pn_tt_slot_exited(device, slot) ||
            ft_tt_launch_assert_reset(&device->runtime.noc, slot->core_x, slot->core_y,
                                      FT_TT_SOFT_RESET_ALL_RISCS) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: members LEND — decode member %u (core %u) did not exit; "
                            "%u lent\n", m, slot->core_index, lent);
            break;
        }
        cores[lent++] = slot->core_index;
    }
    return lent;
}

int pn_tt_matmul_members_return(pn_tt_device *device, const uint32_t *stuck, uint32_t stuck_count)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t m, s;
    int rc = 0;

    if (!group->members_lent)
        return 0;
    for (m = 1; m < group->member_count; m++) {
        pn_tt_matmul_member *slot = &group->member[m];
        bool is_stuck = false;

        for (s = 0; s < stuck_count; s++)
            is_stuck |= stuck[s] == slot->core_index;
        if (is_stuck || !pn_tt_core_boot_is_permitted(&device->runtime, slot->core_index,
                                                      "a returned matmul member",
                                                      "PN_TT_PREFILL_LEND") ||
            ft_tt_runtime_boot_core(&device->runtime.noc, slot->core_x, slot->core_y,
                                    &device->firmware) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: members RETURN — core %u did not boot%s\n", slot->core_index,
                    is_stuck ? " (its borrower did not exit)" : "");
            rc = -1;
        }
    }
    group->members_lent = false;
    if (rc != 0) {
        device->resident_running = false; /* a missing member: never arm a partial group */
        return rc;
    }
    pn_tt_matmul_group_retire(device);
    if (pn_tt_matmul_group_arm(device, device->resident_tile_bytes, device->resident_depth) !=
        FT_TT_OK) {
        fprintf(stderr, "pn_tt: members RETURN — the decode group did not re-arm\n");
        device->resident_running = false;
        rc = -1;
    }
    return rc;
}

int pn_tt_matmul_grid_retract(pn_tt_device *device, bool *extra_parked)
{
    pn_tt_matmul_group *group = &device->group;
    const uint32_t base = group->grid_lent_base;
    uint32_t g;
    int rc = 0;

    if (extra_parked)
        *extra_parked = false;
    if (!group->grid_lent)
        return 0;
    if (pn_tt_grid_flush(device) != 0)
        rc = -1;
    for (g = base; g < group->grid_count; g++) {
        pn_tt_matmul_member *slot = &group->grid[g];
        bool exited;

        pn_tt_group_member_stop(device, slot);
        exited = pn_tt_slot_exited(device, slot);
        if (!exited) {
            fprintf(stderr, "pn_tt: grid RETRACT — core %u did not exit\n", slot->core_index);
            rc = -1;
        }
        /* The extra core carries the grid's firmware: parked for its owner to boot again. */
        if (slot->booted && exited &&
            ft_tt_launch_assert_reset(&device->runtime.noc, slot->core_x, slot->core_y,
                                      FT_TT_SOFT_RESET_ALL_RISCS) == FT_TT_OK && extra_parked)
            *extra_parked = true;
        memset(slot, 0, sizeof(*slot));
    }
    pn_tt_grid_set_count(group, base);
    group->grid_lent = false;
    pn_tt_matmul_group_retire(device);
    if (pn_tt_matmul_group_arm(device, device->resident_tile_bytes, device->resident_depth) !=
        FT_TT_OK) {
        fprintf(stderr, "pn_tt: grid RETRACT — the decode group did not re-arm\n");
        device->resident_running = false;
        rc = -1;
    }
    return rc;
}

/** A grid flush failed: every grid core's control block and its five processors' status, so
 *  "a chained projection did not complete" names a core, an error and a place. */
static void pn_tt_grid_failure_dump(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t index, shown = 0u;

    for (index = 0; index < group->grid_count && shown < 48u; index++) {
        pn_tt_matmul_member *slot = &group->grid[index];
        uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
        uint32_t status[FT_TT_QUEUE_STATUS_WORDS];
        unsigned proc;

        if (ft_tt_queue_read_control(&device->runtime.noc, slot->queue, fields) != FT_TT_OK)
            continue;
        if (fields[FT_TT_QUEUE_FIELD_ERROR] == FT_TT_QUEUE_KERR_NONE &&
            fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] >= slot->queue->issued)
            continue;
        shown++;
        fprintf(stderr, "pn_tt: GRID FAIL core %u (%u,%u): state %s error %s head %u tail %u "
                        "completed %u issued %u last %s\n",
                slot->core_index, slot->core_x, slot->core_y,
                ft_tt_queue_state_string(fields[FT_TT_QUEUE_FIELD_STATE]),
                ft_tt_queue_kernel_error_string(fields[FT_TT_QUEUE_FIELD_ERROR]),
                fields[FT_TT_QUEUE_FIELD_HEAD], fields[FT_TT_QUEUE_FIELD_TAIL],
                fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT], slot->queue->issued,
                ft_tt_queue_opcode_string(fields[FT_TT_QUEUE_FIELD_LAST_OPCODE]));
        for (proc = 0; proc < 5u; proc++)
            if (ft_tt_queue_read_status(&device->runtime.noc, slot->queue, proc, status) ==
                FT_TT_OK)
                fprintf(stderr, "pn_tt:   proc %u: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                        proc, status[0], status[1], status[2], status[3], status[4], status[5],
                        status[6], status[7]);
    }
    fprintf(stderr, "pn_tt: GRID FAIL — %u of %u grid cores not complete or in error; the last "
                    "grid descriptor went out %.3f s ago, longest feed gap %.3f s (before "
                    "projection %llu, %s)\n",
            shown, group->grid_count,
            group->grid_alive_seconds > 0.0 ? pn_tt_now_seconds() - group->grid_alive_seconds
                                            : -1.0,
            group->grid_longest_gap, (unsigned long long)group->grid_longest_gap_at,
            group->grid_longest_gap_streamed ? "in a stream" : "quiet");
}

/** Each grid core's issued count (G.2 marks). */
static void pn_tt_grid_issued_counts(const pn_tt_matmul_group *group, uint32_t *issued)
{
    memset(issued, 0, PN_TT_GRID_MARK_CORES * sizeof(issued[0]));
    for (uint32_t i = 0; i < group->grid_count && i < PN_TT_GRID_MARK_CORES; i++)
        issued[i] = group->grid[i].queue ? group->grid[i].queue->issued : 0u;
}

/** G.2: mark everything issued so far as the last detached projection. */
static void pn_tt_grid_mark_push(pn_tt_matmul_group *group)
{
    uint32_t issued[PN_TT_GRID_MARK_CORES];

    pn_tt_grid_issued_counts(group, issued);
    group->grid_last_mark = pn_tt_grid_marks_push(&group->grid_marks, issued);
}

static int pn_tt_grid_flush(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    const double from = pn_tt_now_seconds();
    uint32_t index;
    int rc = FT_TT_OK;

    for (index = 0; index < group->grid_count; index++) {
        if (!group->grid_pending[index])
            continue;
        if (rc == FT_TT_OK)
            rc = pn_tt_group_await(device, &group->grid[index],
                                   group->grid[index].queue->issued);
        group->grid_pending[index] = false;
    }
    group->grid_ring = 0u;
    {
        uint32_t issued[PN_TT_GRID_MARK_CORES];

        pn_tt_grid_issued_counts(group, issued);
        pn_tt_grid_marks_all(&group->grid_marks, issued);
    }
    if (rc != FT_TT_OK)
        pn_tt_grid_failure_dump(device);
    group->grid_seconds += pn_tt_now_seconds() - from;
    return rc;
}

void pn_tt_matmul_grid_close(pn_tt_device *device)
{
    if (device->group.grid_running)
        pn_tt_grid_stop(device);
}

void pn_tt_matmul_grid_end(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;

    if (!group->grid_running)
        return;
    (void)pn_tt_grid_flush(device);
    pn_tt_grid_stop(device);
    pn_tt_device_telemetry_log(device, "grid end (before park)");
    /* DEFAULT OFF since the prefill lane's run_pg_park.sh (2026-10-22): kept booted, the next
     * prefill's grid-begin is 0.038 s instead of 0.100 (serve 1,630 warm TTFT 2,397 -> 2,333-
     * 2,342 ms) and decode after it moved -0.1..-0.4% (g256 38.47 vs 38.30-38.44), all texts
     * IDENTICAL. `PN_TT_PREFILL_GRID_PARK=1` parks as before. */
    if (pn_tt_env_u32("PN_TT_PREFILL_GRID_PARK", 0u) != 0u)
        fprintf(stderr, "pn_tt: prefill grid PARKED — %u of %u cores held in reset\n",
                pn_tt_grid_park(device), group->grid_count);
    fprintf(stderr, "pn_tt: prefill grid — %llu projections (%llu from the L1 stash, %llu read "
                    "B from DRAM, %llu row-split pieces, %llu unsplit weights by rows), %.3f s "
                    "awaiting\n",
            (unsigned long long)group->grid_projections,
            (unsigned long long)group->grid_stashed,
            (unsigned long long)group->grid_fallbacks,
            (unsigned long long)group->grid_msplit_pieces,
            (unsigned long long)group->grid_unsplit_projections, group->grid_seconds);
    fprintf(stderr, "pn_tt: prefill grid — longest feed gap %.3f s (before projection %llu, %s), "
                    "%u idle re-arm(s), %u keepalive NOP round(s)\n",
            group->grid_longest_gap, (unsigned long long)group->grid_longest_gap_at,
            group->grid_longest_gap_streamed ? "in a stream" : "quiet",
            group->grid_idle_rearms, group->grid_keepalive_rounds);
    group->grid_alive_seconds = 0.0;
    group->grid_longest_gap = 0.0;
    group->grid_longest_gap_at = 0u;
    group->grid_idle_rearms = 0u;
    group->grid_keepalive_rounds = 0u;
    if (group->grid_mark_waits || group->grid_mark_full)
        fprintf(stderr, "pn_tt: prefill grid — G.2 marks: %llu waits by step, %llu fell back to a "
                        "full wait\n",
                (unsigned long long)group->grid_mark_waits,
                (unsigned long long)group->grid_mark_full);
    pn_tt_dispatch_report_await_retries(device);
    pn_tt_grid_probe_report();
}

/** One grid descriptor: columns [col_first, col_first + cols) of shard `m` (local to the shard)
 *  over tile rows [row_first, row_first + rows) (`by_rows`: the M split's piece), on core `g`. */
typedef struct pn_tt_grid_piece {
    uint32_t g, m, col_first, cols, row_first, rows;
    bool by_rows;
    bool shift_b; /**< RED `grid_chunk_b_shift` on this descriptor */
    bool unsplit; /**< an UNSPLIT weight's row piece (O.0): B is the whole weight */
} pn_tt_grid_piece;

static int pn_tt_grid_issue_piece(pn_tt_device *device, const pn_tt_weight *weight,
                                  const pn_tt_activation *input,
                                  const pn_tt_activation *output, const pn_tt_grid_piece *piece)
{
    pn_tt_matmul_group *group = &device->group;
    const uint32_t kt = weight->tile_rows;
    const uint32_t g = piece->g;
    pn_tt_matmul_member *slot = &group->grid[g];
    const uint32_t replica = input->replica_count > 1u ? g % input->replica_count : 0u;
    const bool red = pn_tt_red("grid_piece_shift");
    uint32_t parameters[FT_TT_QUEUE_PARAM_WORDS];
    ft_tt_queue_descriptor descriptor;
    uint32_t address = 0u;
    int rc;

    pn_tt_group_parameters(device, weight, input, output, piece->m, !piece->unsplit, parameters);
    parameters[FT_TT_QUEUE_MATMUL_TILE_COLS] = piece->cols;
    /* RED PROOF `grid_piece_shift`: every piece reads its B one column late — must
     * DIFFER (the last piece of a shard reads past its slice). `grid_chunk_b_shift`: the
     * same for stash chunks past the first only. */
    parameters[FT_TT_QUEUE_MATMUL_B_BASE] +=
        (piece->col_first + (red || piece->shift_b ? 1u : 0u)) * kt * weight->tile_bytes;
    parameters[FT_TT_QUEUE_MATMUL_C_BASE] += piece->col_first * PN_TT_TILE_BF16;
    parameters[FT_TT_QUEUE_MATMUL_A_X] = (uint16_t)device->dram[input->replica_bank[replica]].x;
    parameters[FT_TT_QUEUE_MATMUL_A_Y] = (uint16_t)device->dram[input->replica_bank[replica]].y;
    parameters[FT_TT_QUEUE_MATMUL_A_BASE] = input->replica_address[replica];
    if (piece->by_rows) {
        /* RED PROOF `grid_msplit_a_shift`: pieces past the first read A one tile row
         * early (inside the strip) — must DIFFER. */
        const uint32_t a_row =
            piece->row_first - (pn_tt_red("grid_msplit_a_shift") && piece->row_first ? 1u : 0u);

        parameters[FT_TT_QUEUE_MATMUL_TILE_ROWS] = piece->rows;
        parameters[FT_TT_QUEUE_MATMUL_A_BASE] += a_row * kt * PN_TT_TILE_BF16;
        parameters[FT_TT_QUEUE_MATMUL_C_BASE] +=
            piece->row_first * weight->tile_cols * PN_TT_TILE_BF16;
        group->grid_msplit_pieces++;
    }
    /* TIMING PROBE `PN_TT_PROBE_GRID_ROWS=N` (P-nogrid, prefill rework spec §3): every grid
     * descriptor computes at most N tile rows — a legal, smaller descriptor with the same
     * count, marks and B reads, so the pass runs at ~no grid compute. Text garbage by design. */
    {
        const uint32_t probe_rows = pn_tt_env_u32("PN_TT_PROBE_GRID_ROWS", 0u);

        if (probe_rows != 0u && parameters[FT_TT_QUEUE_MATMUL_TILE_ROWS] > probe_rows)
            parameters[FT_TT_QUEUE_MATMUL_TILE_ROWS] = probe_rows;
    }
    /* DIAGNOSIS `PN_TT_GRID_TRACE=1`: grid core 0's and 23's words, every projection. */
    if ((g == 0u || g == 23u) && pn_tt_env_u32("PN_TT_GRID_TRACE", 0u) != 0u)
        fprintf(stderr, "pn_tt: GRID TRACE g%u issued %u slot %u: Mt %u Kt %u Nt %u A "
                        "(%u,%u)+%x B (%u,%u)+%x C (%u,%u)+%x depth %u flags %x block "
                        "%x coop %u stash %x\n",
                g, slot->queue->issued, slot->queue->issued & (PN_TT_QUEUE_PARAM_SLOTS - 1u),
                parameters[0], parameters[1], parameters[2], parameters[3], parameters[4],
                parameters[5], parameters[6], parameters[7], parameters[8], parameters[9],
                parameters[10], parameters[11], parameters[12], parameters[13], parameters[15],
                parameters[14], group->grid_stash_address);
    rc = ft_tt_queue_write_op_parameters(
        pn_tt_issue_access(&device->runtime, group->issue_shape), slot->queue,
        slot->queue->issued & (PN_TT_QUEUE_PARAM_SLOTS - 1u), parameters, &address);
    if (rc != FT_TT_OK)
        return rc;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_MATMUL;
    descriptor.a0 = address;
    if (group->grid_stash_address != 0u &&
        (uint64_t)piece->cols * kt <= group->grid_stash_tiles) {
        descriptor.a1 = 0x80000000u | (group->grid_stash_address >> 4);
        group->grid_stashed++;
    } else {
        group->grid_fallbacks++;
    }
    descriptor.a2 = parameters[FT_TT_QUEUE_MATMUL_TILE_ROWS] > 1u ? weight->tile_cols : 0u;
    rc = ft_tt_queue_stage(pn_tt_issue_access(&device->runtime, group->issue_shape),
                           slot->queue, &descriptor, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_publish(&device->runtime.noc, slot->queue);
    if (rc == FT_TT_OK)
        group->grid_pending[g] = true;
    return rc;
}

/** Shard `m`'s M split decision (pn_tt_grid_msplit.h): narrow shards cut by rows. */
static bool pn_tt_grid_shard_by_rows(const pn_tt_matmul_group *group, uint32_t cols, uint32_t kt,
                                     uint32_t mt)
{
    return group->grid_stash_address != 0u &&
           pn_tt_grid_msplit_applies(cols, kt, group->grid_stash_tiles, mt, group->grid_pieces,
                                     pn_tt_grid_msplit_max_columns());
}

/** Column piece `p` of a shard of `cols` columns when it is cut by columns. */
static uint32_t pn_tt_grid_piece_columns(uint32_t cols, uint32_t pieces, uint32_t p)
{
    return cols / pieces + (p < cols % pieces ? 1u : 0u);
}

/** Descriptors the busiest grid core takes for this projection (GRID STASH CHUNKS). */
static uint32_t pn_tt_grid_projection_depth(const pn_tt_device *device,
                                            const pn_tt_weight *weight,
                                            const pn_tt_activation *input,
                                            const pn_tt_activation *output)
{
    const pn_tt_matmul_group *group = &device->group;
    const uint32_t kt = weight->tile_rows, mt = pn_tt_group_tile_rows(input, output);
    const uint32_t stash = group->grid_stash_address != 0u ? group->grid_stash_tiles : 0u;
    uint32_t depth = 1u;

    for (uint32_t m = 0; m < weight->shard_count; m++) {
        const uint32_t cols = weight->shard_tile_cols[m];
        /* The widest piece is piece 0. */
        const uint32_t chunks =
            pn_tt_grid_shard_by_rows(group, cols, kt, mt)
                ? 1u
                : pn_tt_grid_stash_chunks(pn_tt_grid_piece_columns(cols, group->grid_pieces, 0u),
                                          kt, stash, pn_tt_grid_stash_chunks_enabled());

        depth = chunks > depth ? chunks : depth;
    }
    return depth;
}

/** G.2 marks on? (`PN_TT_GRID_MARKS=0` off.) */
static bool pn_tt_grid_marks_enabled(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *env = getenv("PN_TT_GRID_MARKS");

        cached = env && *env ? atoi(env) != 0 : 1;
    }
    return cached != 0;
}

/* ---- GRID IDLE GUARD (pn_tt_grid_idle.h) ------------------------------------------------- */

/** A grid descriptor just went out: book the gap since the last one. */
static void pn_tt_grid_fed(pn_tt_matmul_group *group)
{
    const double now = pn_tt_now_seconds();

    if (group->grid_alive_seconds > 0.0 && now - group->grid_alive_seconds > group->grid_longest_gap) {
        group->grid_longest_gap = now - group->grid_alive_seconds;
        group->grid_longest_gap_at = group->grid_projections;
        group->grid_longest_gap_streamed = !pn_tt_dispatch_quiet();
    }
    group->grid_alive_seconds = now;
}

static double pn_tt_grid_idle_seconds(const pn_tt_matmul_group *group)
{
    return group->grid_alive_seconds > 0.0 ? pn_tt_now_seconds() - group->grid_alive_seconds
                                           : -1.0;
}

static double pn_tt_grid_idle_limit_seconds(void)
{
    return pn_tt_red("grid_idle_guard_off")
               ? 0.0
               : (double)pn_tt_residency_idle_retire_limit_ms() / 1000.0;
}

/**
 * The grid sat idle near its cores' own exit: everything it was given is waited for, every
 * core is stopped by ASKING (one that already idled out has exited and answers at once) and
 * started fresh on its booted firmware, as pn_tt_matmul_grid_begin starts it. Quiet only (no
 * stream open). The same descriptors follow on the same cores: bit-identical.
 */
static int pn_tt_grid_idle_rearm(pn_tt_device *device, double idle)
{
    pn_tt_matmul_group *group = &device->group;
    const double from = pn_tt_now_seconds();
    int rc = pn_tt_grid_flush(device);

    group->grid_detached = 0u;
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: grid idle %.1f s — its outstanding work did not complete, so it "
                        "cannot be re-armed exactly\n", idle);
        return rc;
    }
    for (uint32_t i = 0; i < group->grid_count; i++) {
        pn_tt_matmul_member *slot = &group->grid[i];

        pn_tt_group_member_stop(device, slot);
        if (!pn_tt_slot_exited(device, slot) ||
            pn_tt_slot_start(device, slot, i, device->resident_tile_bytes,
                             device->resident_depth, true) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: grid idle %.1f s — core %u did not re-arm\n", idle,
                    slot->core_index);
            return FT_TT_EIO;
        }
        slot->running = true;
        pn_tt_dispatch_stream_queue_rearmed(slot->queue->x, slot->queue->y);
    }
    pn_tt_grid_marks_reset(&group->grid_marks, group->grid_count);
    group->grid_last_mark = PN_TT_GRID_MARK_NONE;
    group->grid_ring = 0u;
    group->grid_idle_rearms++;
    group->grid_alive_seconds = pn_tt_now_seconds();
    fprintf(stderr, "pn_tt: prefill grid idle %.1f s (limit %.0f ms) — %u cores stopped by ASKING "
                    "and re-armed in %.3f s, before a descriptor can reach a half-exited triad\n",
            idle, pn_tt_grid_idle_limit_seconds() * 1000.0, group->grid_count,
            pn_tt_now_seconds() - from);
    return FT_TT_OK;
}

/** The guard at an issue: re-arm a grid idle past the limit (quiet), else only say so. */
static int pn_tt_grid_idle_guard(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    const double idle = pn_tt_grid_idle_seconds(group);

    if (pn_tt_grid_idle_decide(idle, 0.0, pn_tt_grid_idle_limit_seconds(), false) !=
        PN_TT_GRID_IDLE_REARM)
        return FT_TT_OK;
    if (!pn_tt_dispatch_quiet() || group->chained != 0u) {
        fprintf(stderr, "pn_tt: prefill grid idle %.1f s inside a stream or a chain — not "
                        "re-armed here\n", idle);
        return FT_TT_OK;
    }
    return pn_tt_grid_idle_rearm(device, idle);
}

int pn_tt_matmul_grid_keepalive(pn_tt_device *device, double min_idle_seconds)
{
    pn_tt_matmul_group *group;
    double idle;
    uint32_t sent = 0u;

    if (!device)
        return 0;
    group = &device->group;
    if (!group->grid_running || !pn_tt_dispatch_quiet() || group->chained != 0u ||
        group->grid_detached != 0u)
        return 0;
    idle = pn_tt_grid_idle_seconds(group);
    switch (pn_tt_grid_idle_decide(idle, min_idle_seconds, pn_tt_grid_idle_limit_seconds(),
                                   true)) {
    case PN_TT_GRID_IDLE_REARM:
        return pn_tt_grid_idle_rearm(device, idle) == FT_TT_OK ? 0 : -1;
    case PN_TT_GRID_IDLE_NOP:
        break;
    default:
        return 0;
    }
    for (uint32_t i = 0; i < group->grid_count; i++) {
        pn_tt_matmul_member *slot = &group->grid[i];
        ft_tt_queue_descriptor descriptor;

        if (!slot->running)
            continue;
        memset(&descriptor, 0, sizeof(descriptor));
        descriptor.opcode = FT_TT_QUEUE_OP_NOP;
        if (ft_tt_queue_stage(pn_tt_issue_access(&device->runtime, group->issue_shape),
                              slot->queue, &descriptor, 1u) != FT_TT_OK ||
            ft_tt_queue_publish(&device->runtime.noc, slot->queue) != FT_TT_OK)
            return -1;
        group->grid_pending[i] = true;
        sent++;
    }
    pn_tt_grid_fed(group);
    group->grid_keepalive_rounds++;
    /* Waited like any grid work, so the marks and the ring stay exact. */
    return pn_tt_grid_flush(device) == FT_TT_OK ? (int)sent : -1;
}

/**
 * Room in the grid's rings for `depth` more descriptors a core. The rings hold 8; a chain longer
 * than 6 is flushed first. Detached projections (PHASE W2) count against the same ring, and so
 * does every stash chunk. G.2: with nothing on the matvec chain, wait only for the OLDEST
 * detached projections that make room (their marks) instead of draining every one.
 */
static int pn_tt_grid_room(pn_tt_device *device, uint32_t depth)
{
    pn_tt_matmul_group *group = &device->group;
    int rc = pn_tt_grid_idle_guard(device);

    if (rc != FT_TT_OK)
        return rc;
    if (group->chained + group->grid_detached < 6u && group->grid_ring + depth <= 6u)
        return FT_TT_OK;
    if (group->chained == 0u && pn_tt_grid_marks_enabled()) {
        const uint32_t id = pn_tt_grid_marks_room(&group->grid_marks, depth, 0u, 6u);

        if (id != PN_TT_GRID_MARK_FULL && id != PN_TT_GRID_MARK_NONE) {
            rc = pn_tt_matmul_grid_wait_mark(device, id);
            if (rc != FT_TT_OK)
                return rc;
            if (group->grid_detached < 6u && group->grid_ring + depth <= 6u)
                return FT_TT_OK;
        }
    }
    rc = group->chained ? pn_tt_matmul_group_flush(device) : FT_TT_OK;
    if (rc == FT_TT_OK && group->grid_detached)
        rc = pn_tt_matmul_grid_detached_wait(device);
    return rc;
}

uint32_t pn_tt_matmul_grid_last_mark(const pn_tt_device *device)
{
    return device ? device->group.grid_last_mark : PN_TT_GRID_MARK_NONE;
}

int pn_tt_matmul_grid_wait_mark(pn_tt_device *device, uint32_t mark)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t need[PN_TT_GRID_MARK_CORES];
    int n, rc = 1;

    if (mark == PN_TT_GRID_MARK_NONE)
        return FT_TT_OK;
    /* RED `grid_mark_early`: the wait stops one projection short — must DIFFER. */
    if (pn_tt_red("grid_mark_early") && mark > 0u)
        mark--;
    if (!pn_tt_grid_marks_enabled())
        return pn_tt_matmul_grid_detached_wait(device);
    n = pn_tt_grid_marks_take(&group->grid_marks, mark, need);
    for (uint32_t i = 0; n > 0 && rc == 1 && i < group->grid_count && i < PN_TT_GRID_MARK_CORES;
         i++)
        if (need[i])
            rc = pn_tt_dispatch_stream_await(group->grid[i].queue->x, group->grid[i].queue->y,
                                             need[i]);
    if (n < 0 || rc != 1) {
        /* Not streamed, or not expressible: wait for everything for real. */
        group->grid_mark_full++;
        rc = pn_tt_grid_flush(device);
        group->grid_detached = 0u;
        return rc;
    }
    group->grid_mark_waits++;
    group->grid_ring = pn_tt_grid_marks_ring(&group->grid_marks);
    group->grid_detached = pn_tt_grid_marks_detached(&group->grid_marks);
    return FT_TT_OK;
}

/** One multi-row projection over the grid: every decode shard cut into `grid_pieces` column
 *  pieces, piece p of shard m on grid core m * pieces + p; a piece wider than the B stash goes
 *  as stash-sized column chunks on its core (pn_tt_grid_stash_chunks). */
static int pn_tt_grid_issue_descriptors(pn_tt_device *device, const pn_tt_weight *weight,
                                        const pn_tt_activation *input,
                                        const pn_tt_activation *output)
{
    pn_tt_matmul_group *group = &device->group;
    const uint32_t kt = weight->tile_rows, pieces = group->grid_pieces;
    const uint32_t mt = pn_tt_group_tile_rows(input, output);
    const uint32_t stash = group->grid_stash_address != 0u ? group->grid_stash_tiles : 0u;
    const bool chunked = pn_tt_grid_stash_chunks_enabled();
    const bool red_chunk = pn_tt_red("grid_chunk_b_shift");

    for (uint32_t m = 0; m < weight->shard_count; m++) {
        const uint32_t cols = weight->shard_tile_cols[m];
        const bool by_rows = pn_tt_grid_shard_by_rows(group, cols, kt, mt);
        uint32_t first_local = 0u;

        for (uint32_t p = 0; p < pieces; p++) {
            const uint32_t pc = by_rows ? cols : pn_tt_grid_piece_columns(cols, pieces, p);
            const uint32_t chunks = by_rows ? 1u : pn_tt_grid_stash_chunks(pc, kt, stash, chunked);
            pn_tt_grid_piece piece = { m * pieces + p, m, first_local, pc, 0u, mt, by_rows, false,
                                       false };

            if (by_rows)
                pn_tt_grid_msplit_rows(mt, pieces, group->row_block, p, &piece.row_first,
                                       &piece.rows);
            if (pc == 0u || piece.rows == 0u)
                continue;
            for (uint32_t c = 0; c < chunks; c++) {
                uint32_t chunk_first = 0u;
                int rc;

                if (chunks > 1u) {
                    pn_tt_grid_stash_chunk_columns(pc, chunks, c, &chunk_first, &piece.cols);
                    piece.col_first = first_local + chunk_first;
                    piece.shift_b = red_chunk && c > 0u;
                }
                rc = pn_tt_grid_issue_piece(device, weight, input, output, &piece);
                if (rc != FT_TT_OK)
                    return rc;
            }
            if (!by_rows)
                first_local += pc;
        }
    }
    group->grid_projections++;
    return FT_TT_OK;
}

static int pn_tt_grid_issue(pn_tt_device *device, const pn_tt_weight *weight,
                            const pn_tt_activation *input, const pn_tt_activation *output)
{
    pn_tt_matmul_group *group = &device->group;
    int rc;

    /* The grid's rings hold 8; a chain longer than 6 is flushed first. Detached projections
     * (PHASE W2) count against the same ring, and so does every stash chunk. */
    const uint32_t depth = pn_tt_grid_projection_depth(device, weight, input, output);

    rc = pn_tt_grid_room(device, depth);
    if (rc != FT_TT_OK)
        return rc;
    pn_tt_group_chain_begin(group, pn_tt_now_seconds(), true);
    rc = pn_tt_grid_issue_descriptors(device, weight, input, output);
    if (rc == FT_TT_OK) {
        group->chained++;
        group->grid_ring += depth;
        pn_tt_grid_marks_issued(&group->grid_marks, depth, false);
        pn_tt_grid_fed(group);
    }
    return rc;
}

/**
 * GRID UNSPLIT ROWS (q38-gdn-chain O.0, pn_tt_grid_msplit.h): a multi-row projection of an
 * UNSPLIT weight cut by rows over `pieces` grid cores — piece p on grid core p, all the weight's
 * columns over its rows. Chained like any grid projection, so the flush that waited for the one
 * decode member now waits for these cores; or DETACHED (G.2, the overlap's wavefront under
 * table gates). RED `grid_unsplit_rows_skip`: the last piece is never issued (its rows keep
 * whatever they held) — must DIFFER.
 */
static int pn_tt_grid_issue_unsplit(pn_tt_device *device, const pn_tt_weight *weight,
                                    const pn_tt_activation *input,
                                    const pn_tt_activation *output, uint32_t pieces,
                                    bool detached)
{
    pn_tt_matmul_group *group = &device->group;
    const uint32_t mt = pn_tt_group_tile_rows(input, output);
    const uint32_t issued = pieces - (pn_tt_red("grid_unsplit_rows_skip") ? 1u : 0u);
    int rc;

    rc = pn_tt_grid_room(device, 1u);
    if (rc != FT_TT_OK)
        return rc;
    if (!detached)
        pn_tt_group_chain_begin(group, pn_tt_now_seconds(), true);
    for (uint32_t p = 0; p < issued; p++) {
        pn_tt_grid_piece piece = { p, 0u, 0u, weight->tile_cols, 0u, 0u, true, false, true };

        pn_tt_grid_msplit_rows(mt, pieces, group->row_block, p, &piece.row_first, &piece.rows);
        if (piece.rows == 0u)
            continue;
        rc = pn_tt_grid_issue_piece(device, weight, input, output, &piece);
        if (rc != FT_TT_OK)
            return rc;
    }
    group->grid_projections++;
    group->grid_unsplit_projections++;
    if (detached)
        group->grid_detached++;
    else
        group->chained++;
    group->grid_ring += 1u;
    pn_tt_grid_marks_issued(&group->grid_marks, 1u, detached);
    pn_tt_grid_fed(group);
    return FT_TT_OK;
}

/* ---- PHASE W2: DETACHED grid projections ---------------------------------------------------
 * A projection the prefill overlaps with the GDN chain: issued to the grid like any other but
 * NOT part of the matvec chain (no `chained`, no device `pending_count`), so the chain's own
 * ops, which flush the chain on entry, never wait for it. The caller polls it between chain
 * steps and waits for it before anything reads its output. Same descriptors, same cores, same
 * K order as the chained path: bit-identical by construction. */

int pn_tt_matmul_grid_issue_detached(pn_tt_device *device, const pn_tt_weight *weight,
                                     const pn_tt_activation *input,
                                     const pn_tt_activation *output)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t depth;
    int rc;

    if (!device || !weight || !input || !output || !group->grid_running ||
        input->replica_count < 2u)
        return FT_TT_EINVAL;
    /* G.2: an UNSPLIT weight (ssm_alpha / ssm_beta) detached, cut by rows, where allowed. */
    if (weight->shard_count != group->member_count) {
        const uint32_t mt = pn_tt_group_tile_rows(input, output);
        const uint32_t pieces =
            group->grid_unsplit_ok && mt > 1u
                ? pn_tt_grid_unsplit_pieces(weight->tile_cols, weight->tile_rows,
                                            group->grid_stash_address != 0u
                                                ? group->grid_stash_tiles : 0u,
                                            mt, group->row_block, group->grid_count,
                                            pn_tt_grid_unsplit_max_columns())
                : 0u;

        if (pieces == 0u)
            return FT_TT_EINVAL;
        rc = pn_tt_grid_issue_unsplit(device, weight, input, output, pieces, true);
        if (rc == FT_TT_OK)
            pn_tt_grid_mark_push(group);
        return rc;
    }
    depth = pn_tt_grid_projection_depth(device, weight, input, output);
    rc = pn_tt_grid_room(device, depth);
    if (rc != FT_TT_OK)
        return rc;
    rc = pn_tt_grid_issue_descriptors(device, weight, input, output);
    if (rc == FT_TT_OK) {
        group->grid_detached++;
        group->grid_ring += depth;
        pn_tt_grid_marks_issued(&group->grid_marks, depth, true);
        pn_tt_grid_mark_push(group);
        pn_tt_grid_fed(group);
    }
    return rc;
}

bool pn_tt_matmul_group_member0_completion(const pn_tt_device *device, uint16_t *out_x,
                                           uint16_t *out_y, uint32_t *out_control,
                                           uint32_t *out_issued)
{
    const ft_tt_queue *queue;

    if (!device || !device->resident_running || !device->group.member[0].queue)
        return false;
    queue = device->group.member[0].queue;
    *out_x = queue->x;
    *out_y = queue->y;
    *out_control = queue->control_address;
    *out_issued = queue->issued;
    return true;
}

bool pn_tt_matmul_grid_is_running(const pn_tt_device *device)
{
    return device && device->group.grid_running;
}

void pn_tt_matmul_grid_unsplit_allow(pn_tt_device *device, bool allow)
{
    if (device)
        device->group.grid_unsplit_ok = allow;
}

int pn_tt_matmul_grid_detached_poll(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    uint32_t index;

    if (group->grid_detached == 0u)
        return 1;
    for (index = 0; index < group->grid_count; index++) {
        if (!group->grid_pending[index])
            continue;
        if (ft_tt_queue_read_completion_line(&device->runtime.bulk_l1, group->grid[index].queue,
                                             fields) != FT_TT_OK)
            return -1;
        if (fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] < group->grid[index].queue->issued)
            return 0;
    }
    return pn_tt_matmul_grid_detached_wait(device) == FT_TT_OK ? 1 : -1;
}

int pn_tt_matmul_grid_detached_wait(pn_tt_device *device)
{
    pn_tt_matmul_group *group = &device->group;
    int rc;

    if (group->grid_detached == 0u)
        return FT_TT_OK;
    rc = pn_tt_grid_flush(device);
    group->grid_detached = 0u;
    return rc;
}

/* PREFILL GRID PROBE (`PN_TT_GRID_PROBE=1`, pn_tt_grid_probe.h): a multi-row projection
 * timed alone — flushed before, issued, flushed — and booked by shape. */
static int pn_tt_matmul_group_enqueue_probed(pn_tt_device *device, const pn_tt_weight *weight,
                                             const pn_tt_activation *input,
                                             const pn_tt_activation *output)
{
    static bool inside;
    const uint64_t grid_before = device->group.grid_projections;
    double t0, t1;
    int rc;

    if (inside || pn_tt_matmul_group_flush(device) != FT_TT_OK)
        return -2;
    inside = true;
    t0 = pn_tt_now_seconds();
    rc = pn_tt_matmul_group_enqueue(device, weight, input, output);
    t1 = pn_tt_now_seconds();
    if (rc == FT_TT_OK)
        rc = pn_tt_matmul_group_flush(device);
    inside = false;
    if (rc == FT_TT_OK)
        pn_tt_grid_probe_book(pn_tt_group_tile_rows(input, output), weight->tile_rows,
                              weight->tile_cols,
                              device->group.grid_projections != grid_before
                                  ? device->group.grid_count
                                  : (weight->shard_count > 1u ? weight->shard_count : 1u),
                              t1 - t0, pn_tt_now_seconds() - t1);
    return rc;
}

int pn_tt_matmul_group_enqueue(pn_tt_device *device, const pn_tt_weight *weight,
                               const pn_tt_activation *input, const pn_tt_activation *output)
{
    pn_tt_matmul_group *group = &device->group;
    /* A weight the member count could not divide stayed whole in ONE bank, so it can only
     * be read as one grid — by member 0. That is a real, reported cost of a group size, not
     * a silent degradation. */
    bool split = weight->shard_count > 1u;
    uint32_t members = split ? weight->shard_count : 1u;
    uint32_t member;
    pn_tt_activation probe_input;
    bool probe_spread = false;

    if (!device->resident_running)
        return FT_TT_ERROR;
    if (pn_tt_grid_probe_enabled() && pn_tt_group_tile_rows(input, output) > 1u) {
        const int probed = pn_tt_matmul_group_enqueue_probed(device, weight, input, output);

        if (probed != -2)
            return probed;
    }
    if (!split && (pn_tt_fuse_probe_bits() & PN_TT_FUSE_PROBE_SOLO))
        return FT_TT_OK;
    if (group->member_count < members)
        return FT_TT_EINVAL;
    /* PRICE PROBE (timing only): an unmirrored input read as if mirrored. */
    if (split && pn_tt_unmirrored_probe_view(input, pn_tt_group_tile_rows(input, output),
                                             device->dram_bank_count, &probe_input)) {
        input = &probe_input;
        probe_spread = true;
    }
    /* PER MEMBER, because a ragged split gives them different widths and the shape the
     * reader refuses is a MEMBER's Nt, not the weight's. */
    for (member = 0; member < members; member++) {
        uint32_t columns = split ? weight->shard_tile_cols[member] : weight->tile_cols;

        if (!pn_tt_group_descriptor_is_executable(device, weight, columns))
            return FT_TT_EINVAL;
    }
    /* PHASE G: a multi-row projection of a sharded weight goes to the prefill grid. */
    /* Only for MIRRORED inputs: 48 cores reading one bank are capped at its ~60 GB/s (G0),
     * slower than the decode group (`PN_TT_PREFILL_GRID_ANY=1` overrides). */
    if (group->grid_running && split && weight->shard_count == group->member_count &&
        pn_tt_group_tile_rows(input, output) > 1u &&
        !(probe_spread && pn_tt_unmirrored_probe_mode() == 2u) &&
        (input->replica_count > 1u || pn_tt_env_u32("PN_TT_PREFILL_GRID_ANY", 0u) != 0u))
        return pn_tt_grid_issue(device, weight, input, output);
    /* O.0: a multi-row projection of an UNSPLIT weight (ssm_alpha / ssm_beta) goes to the grid
     * cut by rows, instead of one decode member sweeping every row — where the caller allowed
     * it (nothing gates on member 0's count). Mirrored inputs only, as above. */
    if (group->grid_running && group->grid_unsplit_ok && !split &&
        pn_tt_group_tile_rows(input, output) > 1u &&
        (input->replica_count > 1u || pn_tt_env_u32("PN_TT_PREFILL_GRID_ANY", 0u) != 0u)) {
        const uint32_t pieces = pn_tt_grid_unsplit_pieces(
            weight->tile_cols, weight->tile_rows,
            group->grid_stash_address != 0u ? group->grid_stash_tiles : 0u,
            pn_tt_group_tile_rows(input, output), group->row_block, group->grid_count,
            pn_tt_grid_unsplit_max_columns());

        if (pieces != 0u)
            return pn_tt_grid_issue_unsplit(device, weight, input, output, pieces, false);
    }
    /* L1a.3: members 1.. are grid cores for this prompt — a split projection the grid did
     * not take has no decode members to run on. Refused by name, never issued. */
    if ((group->grid_lent || group->members_lent) && split) {
        fprintf(stderr, "pn_tt: a split projection Kt=%u Nt=%u reached the decode group while "
                        "its members serve the prefill grid — refused\n",
                weight->tile_rows, weight->tile_cols);
        return FT_TT_EINVAL;
    }
    /* THE RING IS THE BOUND, not the caller's judgement: ft_tt_queue_stage() does not check
     * occupancy, so a chain longer than the ring would write over descriptors the card has
     * not read. Flushing here keeps every caller correct without every caller counting. */
    if (group->chained >= pn_tt_matvec_chain_limit()) {
        int rc = pn_tt_matmul_group_flush(device);

        if (rc != FT_TT_OK)
            return rc;
    }
    {
        static int ab_left = -1;

        if (ab_left < 0)
            ab_left = (int)pn_tt_env_u32("PN_TT_MATMUL_ROW0_AB", 0u);
        if (ab_left > 0 && device->single_row) {
            const uint32_t n = weight->tile_cols * 32u;
            float *ref = calloc(n, sizeof(float));
            float *got = calloc(n, sizeof(float));
            double max_diff = 0.0, mean_ref = 0.0, dot = 0.0, rr = 0.0, gg = 0.0;
            uint32_t i, worst = 0u;
            int rc;

            ab_left--;
            if (!ref || !got) {
                free(ref);
                free(got);
                return FT_TT_ERROR;
            }
            rc = pn_tt_matmul_group_flush(device);
            g_row0_override = 0;
            if (rc == FT_TT_OK)
                rc = pn_tt_group_issue(device, weight, input, output, split, members);
            if (rc == FT_TT_OK)
                rc = pn_tt_matmul_group_flush(device);
            if (rc == FT_TT_OK && pn_tt_activation_download(device, output, ref, n) != 0)
                rc = FT_TT_ERROR;
            g_row0_override = 1;
            if (rc == FT_TT_OK)
                rc = pn_tt_group_issue(device, weight, input, output, split, members);
            if (rc == FT_TT_OK)
                rc = pn_tt_matmul_group_flush(device);
            if (rc == FT_TT_OK && pn_tt_activation_download(device, output, got, n) != 0)
                rc = FT_TT_ERROR;
            /* A SECOND row-0 run of the same projection, and the per-member split of the
             * first one: is a wrong answer the FIRST row-0 descriptor a member ever runs? */
            {
                float *again = calloc(n, sizeof(float));
                double second = 0.0;
                uint32_t m, col = 0u;

                if (again && rc == FT_TT_OK &&
                    pn_tt_group_issue(device, weight, input, output, split, members) == FT_TT_OK &&
                    pn_tt_matmul_group_flush(device) == FT_TT_OK &&
                    pn_tt_activation_download(device, output, again, n) == 0) {
                    for (i = 0; i < n; i++)
                        if (fabs((double)again[i] - ref[i]) > second)
                            second = fabs((double)again[i] - ref[i]);
                    fprintf(stderr, "pn_tt: ROW0 A/B — second row-0 run max |diff| %.6g; first "
                                    "run per member:", second);
                    for (m = 0; m < members; m++) {
                        const uint32_t cols = split ? weight->shard_tile_cols[m] : weight->tile_cols;
                        double md = 0.0;

                        for (i = col * 32u; i < (col + cols) * 32u && i < n; i++)
                            if (fabs((double)got[i] - ref[i]) > md)
                                md = fabs((double)got[i] - ref[i]);
                        fprintf(stderr, " m%u %.3g", m, md);
                        col += cols;
                    }
                    fprintf(stderr, "\npn_tt: ROW0 A/B — first run per tile (max |diff|, member 0 and 1):");
                    for (i = 0; i < 32u && i * 32u < n; i++) {
                        double td = 0.0;
                        uint32_t e;

                        for (e = i * 32u; e < (i + 1u) * 32u; e++)
                            if (fabs((double)got[e] - ref[e]) > td)
                                td = fabs((double)got[e] - ref[e]);
                        fprintf(stderr, " %.2g", td);
                    }
                    fprintf(stderr, "\npn_tt: ROW0 A/B — tile 1 cols: ");
                    for (i = 32u; i < 64u && i < n; i++)
                        fprintf(stderr, " %.3g/%.3g", (double)ref[i], (double)got[i]);
                    fprintf(stderr, "\n");
                }
                free(again);
            }
            g_row0_override = -1;
            for (i = 0; rc == FT_TT_OK && i < n; i++) {
                const double d = fabs((double)got[i] - (double)ref[i]);

                if (d > max_diff) {
                    max_diff = d;
                    worst = i;
                }
                mean_ref += fabs((double)ref[i]);
                dot += (double)got[i] * ref[i];
                rr += (double)ref[i] * ref[i];
                gg += (double)got[i] * got[i];
            }
            fprintf(stderr,
                    "pn_tt: ROW0 A/B — Kt %u Nt %u split %d: max |diff| %.6g at %u (ref %.6g "
                    "got %.6g), mean |ref| %.6g, got.ref/ref.ref %.6f, |got|/|ref| %.6f; ref[0..3] "
                    "%.5g %.5g %.5g %.5g got[0..3] %.5g %.5g %.5g %.5g; ref[16..17] %.5g %.5g "
                    "got[16..17] %.5g %.5g\n",
                    weight->tile_rows, weight->tile_cols, split ? 1 : 0, max_diff, worst,
                    (double)ref[worst], (double)got[worst], n ? mean_ref / n : 0.0,
                    rr > 0 ? dot / rr : 0.0, rr > 0 ? sqrt(gg / rr) : 0.0, (double)ref[0],
                    (double)ref[1], (double)ref[2], (double)ref[3], (double)got[0],
                    (double)got[1], (double)got[2], (double)got[3], n > 17 ? (double)ref[16] : 0.0,
                    n > 17 ? (double)ref[17] : 0.0, n > 17 ? (double)got[16] : 0.0,
                    n > 17 ? (double)got[17] : 0.0);
            free(ref);
            free(got);
            return rc;
        }
    }
    return pn_tt_group_issue(device, weight, input, output, split, members);
}

/** S3b: can THIS group run `gate` + `up` fused into `output` right now? (header contract). */
static bool pn_tt_group_swiglu_fits(const pn_tt_device *device, const pn_tt_weight *gate,
                                    const pn_tt_weight *up, const pn_tt_activation *input,
                                    const pn_tt_activation *output)
{
    const pn_tt_matmul_group *group = &device->group;
    uint32_t member;

    if (!pn_tt_fuse_swiglu_mode() || !device->resident_running || !input || !output ||
        !pn_tt_swiglu_shards_pair(gate, up) || group->member_count < gate->shard_count ||
        pn_tt_group_tile_rows(input, output) != 1u || pn_tt_matmul_pipe() == 0 ||
        group->cooperative || group->g0_a1 != 0u || group->row_block > 1u ||
        group->swiglu_up != NULL || pn_tt_env_u32("PN_TT_MATMUL_PROBE", 0u) != 0u ||
        !pn_tt_replica_map_fused_covers(device, output))
        return false;
    for (member = 0u; member < gate->shard_count; member++)
        if (!pn_tt_group_descriptor_is_executable(device, gate, gate->shard_tile_cols[member]))
            return false;
    return true;
}

int pn_tt_matmul_group_enqueue_swiglu(pn_tt_device *device, const pn_tt_weight *gate,
                                      const pn_tt_weight *up, const pn_tt_activation *input,
                                      const pn_tt_activation *output)
{
    pn_tt_matmul_group *group = &device->group;
    int rc;

    if (!device || !pn_tt_group_swiglu_fits(device, gate, up, input, output))
        return FT_TT_EINVAL;
    /* THE RING IS THE BOUND, as in pn_tt_matmul_group_enqueue(). */
    if (group->chained >= pn_tt_matvec_chain_limit()) {
        rc = pn_tt_matmul_group_flush(device);
        if (rc != FT_TT_OK)
            return rc;
    }
    group->swiglu_up = up;
    group->swiglu_bits = pn_tt_swiglu_block_bits(pn_tt_eltwise_silu_row0(device),
                                                 pn_tt_red("fuse_swiglu_no_stage"));
    rc = pn_tt_group_issue(device, gate, input, output, true, gate->shard_count);
    group->swiglu_up = NULL;
    group->swiglu_bits = 0u;
    if (rc == FT_TT_OK)
        group->swiglu_issues++;
    return rc;
}

/** S3a: can THIS group run `weight` with the residual add fused right now? (header contract). */
static bool pn_tt_group_resadd_fits(const pn_tt_device *device, const pn_tt_weight *weight,
                                    const pn_tt_activation *input,
                                    const pn_tt_activation *residual,
                                    const pn_tt_activation *output)
{
    const pn_tt_matmul_group *group = &device->group;
    uint32_t member;

    if (!pn_tt_fuse_resadd_mode() || !device->resident_running || !weight || !input ||
        !residual || !output || weight->shard_count < 2u ||
        group->member_count < weight->shard_count ||
        !pn_tt_resadd_strips_pair(residual, output, weight->tile_cols) ||
        pn_tt_group_tile_rows(input, output) != 1u || pn_tt_matmul_pipe() == 0 ||
        group->cooperative || group->g0_a1 != 0u || group->row_block > 1u ||
        group->swiglu_up != NULL || group->resadd_residual != NULL ||
        pn_tt_env_u32("PN_TT_MATMUL_PROBE", 0u) != 0u)
        return false;
    for (member = 0u; member < weight->shard_count; member++)
        if (!pn_tt_group_descriptor_is_executable(device, weight, weight->shard_tile_cols[member]))
            return false;
    return true;
}

int pn_tt_matmul_group_enqueue_resadd(pn_tt_device *device, const pn_tt_weight *weight,
                                      const pn_tt_activation *input,
                                      const pn_tt_activation *residual,
                                      const pn_tt_activation *output)
{
    pn_tt_matmul_group *group;
    int rc;

    if (!device || !pn_tt_group_resadd_fits(device, weight, input, residual, output))
        return FT_TT_EINVAL;
    group = &device->group;
    /* THE RING IS THE BOUND, as in pn_tt_matmul_group_enqueue(). */
    if (group->chained >= pn_tt_matvec_chain_limit()) {
        rc = pn_tt_matmul_group_flush(device);
        if (rc != FT_TT_OK)
            return rc;
    }
    group->resadd_residual = residual;
    group->resadd_bits = pn_tt_resadd_block_bits(pn_tt_red("fuse_resadd_skip"));
    rc = pn_tt_group_issue(device, weight, input, output, true, weight->shard_count);
    group->resadd_residual = NULL;
    group->resadd_bits = 0u;
    if (rc == FT_TT_OK)
        group->resadd_issues++;
    return rc;
}

int pn_tt_matmul_group_g0(pn_tt_device *device, const pn_tt_weight *weight)
{
    pn_tt_matmul_group *group = &device->group;
    const uint32_t mode = pn_tt_env_u32("PN_TT_G0", 0u);
    const uint32_t mt = pn_tt_env_u32("PN_TT_G0_ROWS", 51u);
    const uint32_t kt = weight->tile_rows;
    const uint32_t nt = weight->tile_cols;
    const size_t a_bytes = (size_t)mt * kt * PN_TT_TILE_BF16;
    const size_t c_bytes = (size_t)mt * nt * PN_TT_TILE_BF16;
    ft_tt_buffer a_buf, c_buf;
    pn_tt_activation in, out;
    uint16_t *host = NULL;
    uint64_t fnv = 1469598103934665603ull;
    uint32_t state = 12345u, row, member;
    double started, elapsed;
    size_t index;
    int rc = -1;

    if (mode == 0u || mt == 0u)
        return 0;
    if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, 3u, a_bytes, &a_buf) !=
            FT_TT_OK ||
        ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, 4u, c_bytes, &c_buf) !=
            FT_TT_OK) {
        fprintf(stderr, "pn_tt: G0 — DRAM allocation failed\n");
        return -1;
    }
    host = malloc(a_bytes > c_bytes ? a_bytes : c_bytes);
    if (!host)
        return -1;
    /* Deterministic bf16 in [-1, 1): exponent 126 or below, random mantissa and sign. */
    for (index = 0; index < a_bytes / 2u; index++) {
        state = state * 1664525u + 1013904223u;
        host[index] = (uint16_t)(((state >> 16) & 0x807Fu) | (0x3C00u + ((state >> 8) & 0x0380u)));
    }
    memset(&in, 0, sizeof(in));
    memset(&out, 0, sizeof(out));
    in.bank = 3u;
    in.address = (uint32_t)a_buf.address;
    in.tile_count = kt;
    in.replica_count = 1u;
    in.replica_bank[0] = in.bank;
    in.replica_address[0] = in.address;
    out.bank = 4u;
    out.address = (uint32_t)c_buf.address;
    out.tile_count = nt;
    out.replica_count = 1u;
    out.replica_bank[0] = out.bank;
    out.replica_address[0] = out.address;
    if (pn_tt_write_dram(device, in.bank, in.address, host, a_bytes) != 0 ||
        pn_tt_zero_dram(device, out.bank, out.address, c_bytes) != 0)
        goto done;
    /* A MIRRORED over `PN_TT_G0_REPLICAS` banks (default 8): 48 members on one bank's ~59 GB/s
     * is what bound the first G0 round. Member m reads mirror m % R (the decode mirrors' rule). */
    {
        const uint32_t replicas = pn_tt_env_u32("PN_TT_G0_REPLICAS", 8u);
        uint32_t r;

        for (r = 1u; r < replicas && r < PN_TT_ACTIVATION_MAX_REPLICAS; r++) {
            ft_tt_buffer mirror;
            const uint32_t bank = (3u + r) % device->dram_bank_count;

            if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, bank, a_bytes,
                                   &mirror) != FT_TT_OK ||
                pn_tt_write_dram(device, bank, (uint32_t)mirror.address, host, a_bytes) != 0)
                break;
            in.replica_bank[r] = bank;
            in.replica_address[r] = (uint32_t)mirror.address;
            in.replica_count = r + 1u;
        }
    }
    group->g0_a1 = 0u;
    if (mode >= 3u) {
        uint32_t widest = 0u;

        for (member = 0; member < group->member_count; member++)
            if (weight->shard_tile_cols[member] > widest)
                widest = weight->shard_tile_cols[member];
        if (g_stash_address == 0u ||
            (uint64_t)widest * kt > (uint64_t)pn_tt_env_u32("PN_TT_G0_STASH_TILES", 0u)) {
            fprintf(stderr, "pn_tt: G0 — no stash big enough (%u x %u tiles)\n", widest, kt);
            goto done;
        }
        group->g0_a1 = 0x80000000u | (g_stash_address >> 4) | (mode == 4u ? 0x40000000u : 0u);
    }
    started = pn_tt_now_seconds();
    if (mode == 1u) {
        for (row = 0; row < mt; row += 2u) {
            pn_tt_activation pin = in, pout = out;
            const uint32_t span = mt - row < 2u ? mt - row : 2u;

            {
                uint32_t r;

                pin.address += row * kt * PN_TT_TILE_BF16;
                for (r = 0; r < pin.replica_count; r++)
                    pin.replica_address[r] += row * kt * PN_TT_TILE_BF16;
            }
            pin.tile_rows = pin.active_rows = span;
            pout.address += row * nt * PN_TT_TILE_BF16;
            pout.replica_address[0] = pout.address;
            pout.tile_rows = pout.active_rows = span;
            if (pn_tt_matmul_group_dispatch(device, weight, &pin, &pout) != FT_TT_OK)
                goto done;
        }
    } else {
        in.tile_rows = in.active_rows = mt;
        out.tile_rows = out.active_rows = mt;
        if (pn_tt_matmul_group_dispatch(device, weight, &in, &out) != FT_TT_OK)
            goto done;
    }
    elapsed = pn_tt_now_seconds() - started;
    group->g0_a1 = 0u;
    if (pn_tt_read_dram(device, out.bank, out.address, host, c_bytes) != 0)
        goto done;
    for (index = 0; index < c_bytes; index++) {
        fnv ^= ((const unsigned char *)host)[index];
        fnv *= 1099511628211ull;
    }
    fprintf(stderr, "pn_tt: G0 mode %u — Mt %u Kt %u Nt %u over %u members: %.3f ms, %.2f "
                    "TFLOP/s, output fnv %016llx\n",
            mode, mt, kt, nt, group->member_count, elapsed * 1e3,
            2.0 * mt * 32.0 * kt * 32.0 * nt * 32.0 / elapsed / 1e12, (unsigned long long)fnv);
    rc = 0;
done:
    group->g0_a1 = 0u;
    free(host);
    return rc;
}

int pn_tt_matmul_group_dispatch(pn_tt_device *device, const pn_tt_weight *weight,
                                const pn_tt_activation *input, const pn_tt_activation *output)
{
    int rc = pn_tt_matmul_group_enqueue(device, weight, input, output);

    if (rc != FT_TT_OK)
        return rc;
    return pn_tt_matmul_group_flush(device);
}

bool pn_tt_matmul_group_can_fall_back(const pn_tt_device *device, const pn_tt_weight *weight)
{
    (void)device;
    /* A sharded weight is physically in several banks; there is no single grid for a
     * one-shot launch to read, so the one-shot escape does not exist for it. */
    return weight->shard_count <= 1u;
}

void pn_tt_matmul_group_report(const pn_tt_device *device)
{
    const pn_tt_matmul_group *group = &device->group;
    uint32_t member;

    if (group->member_count == 1u)
        return;
    fprintf(stderr, "pn_tt: matmul group — %u members on cores", group->member_count);
    for (member = 0; member < group->member_count; member++)
        fprintf(stderr, " %u", group->member[member].core_index);
    fprintf(stderr, "; %u weights sharded over %u banks (%u of them RAGGED — widths differ "
                    "by one), %u left whole (Nt below the member count); weight-door NoC row "
                    "collisions worst %u, activation door rotation %s (row collisions %u); "
                    "completion %s; card-side post %s\n",
            group->sharded_weights, group->member_count, group->ragged_weights,
            group->unsharded_weights, group->worst_weight_row_collisions,
            group->activation_doors ? "ON" : "OFF", group->activation_row_collisions,
            group->group_completion ? "AGGREGATED on member 0" : "per member",
            group->group_completion              ? "to member 0"
            : group->post_probe == PN_TT_GROUP_POST_PROBE_SELF   ? "PROBE self"
            : group->post_probe == PN_TT_GROUP_POST_PROBE_REMOTE ? "PROBE remote"
                                                                 : "disarmed");
}

void pn_tt_matmul_group_op_counts(const pn_tt_device *device, uint64_t *out_split,
                                  uint64_t *out_solo)
{
    if (out_split)
        *out_split = device ? device->group.split_ops : 0u;
    if (out_solo)
        *out_solo = device ? device->group.solo_ops : 0u;
}

/**
 * THE LINK LEDGER: the per-member host term PREDICTED from counted accesses, beside the
 * fitted one.
 *
 * `wall(N) = 0.3885 + 0.00242*N + card(N)` fits every measured member count, but a fit
 * names a SEAM. This names the MECHANISM, from quantities counted at the point of issue and
 * the link's own two rates (DEV_NOTES.md §6):
 *
 *   - the WRITE direction is bandwidth bound, ~53 MB/s through the UC path;
 *   - the READ direction is transaction-count bound, ~3.4 us per access of 16 bytes or 4.
 *
 * The per-member term is then `issue_bytes/N/53e6 + wait_transactions/N * 3.4e-6` per token,
 * and it either reproduces 2.42 ms or one of the two is wrong. **That disagreement is the
 * useful outcome**, which is why both numbers are printed rather than one.
 *
 * `PN_TT_LINK_WRITE_MBPS` and `PN_TT_LINK_READ_US` exist so a session that re-measures the
 * link (a Gen1 retrain moves the first by 2x) can re-quote this without a rebuild.
 */
static void pn_tt_matmul_group_link_report(const pn_tt_device *device)
{
    const pn_tt_matmul_group *group = &device->group;
    const double write_mbps = (double)pn_tt_env_u32("PN_TT_LINK_WRITE_MBPS", 53u);
    const double read_us = (double)pn_tt_env_u32("PN_TT_LINK_READ_NS", 3400u) / 1000.0;
    double issue_seconds, wait_seconds;

    if (group->flushes == 0u || group->member_count == 0u)
        return;
    issue_seconds = (double)group->issue_write_bytes / (write_mbps * 1e6);
    wait_seconds = (double)group->wait_transactions * read_us * 1e-6;
    fprintf(stderr,
            "llama_tt: link ledger — %llu projections, %llu flushes, %u members. "
            "ISSUE %llu writes over %llu issued pairs / %llu B (%.2f B and %.2f writes a "
            "pair) on the %s shape, %llu PCIe accesses (%.2f a pair). "
            "WAIT %llu reads / %llu transactions over %llu awaited pairs (%.2f a "
            "pair).\n",
            (unsigned long long)group->projections, (unsigned long long)group->flushes,
            group->member_count, (unsigned long long)group->issue_writes,
            (unsigned long long)group->issue_pairs,
            (unsigned long long)group->issue_write_bytes,
            group->issue_pairs
                ? (double)group->issue_write_bytes / (double)group->issue_pairs
                : 0.0,
            group->issue_pairs ? (double)group->issue_writes / (double)group->issue_pairs
                               : 0.0,
            pn_tt_issue_shape_string(group->issue_shape),
            (unsigned long long)group->issue_accesses,
            group->issue_pairs ? (double)group->issue_accesses / (double)group->issue_pairs
                               : 0.0,
            (unsigned long long)group->wait_reads,
            (unsigned long long)group->wait_transactions,
            (unsigned long long)group->wait_pairs,
            group->wait_pairs ? (double)group->wait_transactions / (double)group->wait_pairs
                              : 0.0);
    if (group->param_cache) {
        uint64_t hits = 0u, misses = 0u, full = 0u;
        uint32_t used = 0u, member;

        for (member = 0; member < group->member_count; member++) {
            hits += group->member[member].cache_hits;
            misses += group->member[member].cache_misses;
            full += group->member[member].cache_full;
            if (group->member[member].cache_used > used)
                used = group->member[member].cache_used;
        }
        fprintf(stderr,
                "llama_tt: parameter cache — %llu hits, %llu misses, %llu fell back (region "
                "full); %u of %u blocks used on the deepest member. A hit is a 64-byte host "
                "write NOT made.\n",
                (unsigned long long)hits, (unsigned long long)misses,
                (unsigned long long)full, used, FT_TT_QUEUE_PARAM_CACHE_SLOTS);
    }
    fprintf(stderr,
            "llama_tt: link ledger — PREDICTED host cost at %.0f MB/s write and %.1f us a "
            "read: issue %.4f s, wait %.4f s, total %.4f s. MEASURED issue %.4f s, wait "
            "%.4f s. A gap here is the prediction or the counters, not both.\n",
            write_mbps, read_us, issue_seconds, wait_seconds, issue_seconds + wait_seconds,
            group->issue_seconds, group->wait_seconds);
    /* THE RATE THE RUN ITSELF MEASURED, which is the only one that cannot be stale. The
     * bandwidth model above predicted 0.150 s of issue against 0.928 measured at 8 members
     * — a 6x miss — because the write direction is TRANSACTION bound like the read one and
     * not bandwidth bound at all. Printing per-access seconds for both directions makes
     * that visible in one line instead of being derived by hand from two. */
    fprintf(stderr,
            "llama_tt: link ledger — MEASURED per access: issue %.2f us a write, %.2f us an "
            "issued pair; wait %.2f us a transaction, %.2f us an awaited pair.\n",
            group->issue_writes
                ? group->issue_seconds * 1e6 / (double)group->issue_writes : 0.0,
            group->issue_pairs
                ? group->issue_seconds * 1e6 / (double)group->issue_pairs : 0.0,
            group->wait_transactions
                ? group->wait_seconds * 1e6 / (double)group->wait_transactions : 0.0,
            group->wait_pairs
                ? group->wait_seconds * 1e6 / (double)group->wait_pairs : 0.0);
}

void pn_tt_matmul_group_time_report(const pn_tt_device *device)
{
    const pn_tt_matmul_group *group;
    uint64_t ops;
    uint32_t member;

    if (!device)
        return;
    group = &device->group;
    ops = group->split_ops + group->solo_ops;
    if (ops == 0u)
        return;
    fprintf(stderr, "llama_tt: dispatch split — issue %.3f s (%.1f us/op), wait %.3f s "
                    "(%.1f us/op), %llu poll reads, %llu wide group reads (%.2f/op)\n",
            group->issue_seconds, group->issue_seconds * 1e6 / (double)ops,
            group->wait_seconds, group->wait_seconds * 1e6 / (double)ops,
            (unsigned long long)group->poll_reads, (unsigned long long)group->group_reads,
            (double)group->group_reads / (double)ops);
    pn_tt_matmul_group_link_report(device);
    if (group->pull_table_ok)
        fprintf(stderr, "llama_tt: matmul PULL — %llu pulled flushes, %llu member-0 "
                        "descriptors published without their pull (deferred to the chain's "
                        "end; host-issued only), %s sweep\n",
                (unsigned long long)group->pulled_flushes,
                (unsigned long long)group->pulls_deferred,
                group->pull_serial ? "one-at-a-time" : "parallel");
    if (group->swiglu_issues)
        fprintf(stderr, "llama_tt: matmul SWIGLU — %llu fused gate/up/silu_mul projections "
                        "issued\n", (unsigned long long)group->swiglu_issues);
    if (group->resadd_issues)
        fprintf(stderr, "llama_tt: matmul RESADD — %llu projections issued with the residual "
                        "add fused\n", (unsigned long long)group->resadd_issues);
    if (group->member_count == 1u)
        return;
    fprintf(stderr, "llama_tt: per-member wait —");
    for (member = 0; member < group->member_count; member++)
        fprintf(stderr, " core %u %.3f s", group->member[member].core_index,
                group->member_wait_seconds[member]);
    fprintf(stderr, "  (members run concurrently and are awaited in order, so a balanced "
                    "group puts nearly all of it on the first)\n");
}

void pn_tt_matmul_group_read_report(const pn_tt_device *device, double tokens)
{
    const pn_tt_matmul_group *group;
    uint64_t total;
    double seconds;
    uint32_t bank;

    if (!device || tokens <= 0.0)
        return;
    group = &device->group;
    total = group->a_read_bytes + group->b_read_bytes;
    if (total == 0u)
        return;
    /* WAIT, not the whole dispatch phase: it is the only interval in which the card is the
     * thing running. ISSUE overlaps card work by construction, so charging the reads to
     * issue+wait would understate the rate by exactly the overlap. */
    seconds = group->wait_seconds > 0.0 ? group->wait_seconds : 1e-9;
    fprintf(stderr,
            "llama_tt: matmul DRAM reads — A %.3f + B %.3f = %.3f GB/token (A is %.1f%%) over "
            "%.0f A-blocks/token; %.1f GB/s aggregate in %.4f s/token of WAIT, %u member(s); "
            "blocks MBxNB %ux%u of %u DST tiles; requests A %u B / B %u B, %u outstanding "
            "then a barrier\n",
            (double)group->a_read_bytes / 1e9 / tokens,
            (double)group->b_read_bytes / 1e9 / tokens, (double)total / 1e9 / tokens,
            100.0 * (double)group->a_read_bytes / (double)total,
            (double)group->read_blocks / tokens, (double)total / seconds / 1e9,
            group->wait_seconds / tokens, group->member_count, group->row_block,
            group->column_block, group->dst_section_tiles,
            /* The A REQUEST does not grow with MB — a row block is MB separate reads of
             * `depth` contiguous tiles, one per row, so only the count moves. */
            device->resident_depth * PN_TT_TILE_BF16,
            device->resident_depth * device->resident_tile_bytes,
            group->row_block + group->column_block);
    if (pn_tt_env_u32("PN_TT_READ_CENSUS", 0u) != 0u) {
        uint64_t weight_sum = 0u;
        uint32_t row;

        fprintf(stderr,
                "llama_tt: read census — booked A %.3f, wire A %.3f, B %.3f GB/token; of which "
                "multi-row (Mt>1) A %.3f B %.3f GB/token\n",
                (double)group->a_read_bytes / 1e9 / tokens,
                (double)group->wire_a_bytes / 1e9 / tokens,
                (double)group->b_read_bytes / 1e9 / tokens,
                (double)group->multi_row_a_bytes / 1e9 / tokens,
                (double)group->multi_row_b_bytes / 1e9 / tokens);
        for (row = 0; row < group->read_census_rows; row++) {
            const uint64_t weight_bytes = (uint64_t)group->read_census[row].kt *
                                          group->read_census[row].nt *
                                          group->read_census[row].tile_bytes;

            weight_sum += weight_bytes;
            fprintf(stderr,
                    "llama_tt: read census Kt %4u Nt %5u — %.2f calls/token (%llu multi-row), "
                    "weight %.2f MB x calls = %.1f MB; B %.1f MB, wire A %.1f MB (booked "
                    "%.1f) per token\n",
                    group->read_census[row].kt, group->read_census[row].nt,
                    (double)group->read_census[row].calls / tokens,
                    (unsigned long long)group->read_census[row].multi_row_calls,
                    (double)weight_bytes / 1e6,
                    (double)weight_bytes * (double)group->read_census[row].calls / tokens / 1e6,
                    (double)group->read_census[row].b_bytes / tokens / 1e6,
                    (double)group->read_census[row].wire_a_bytes / tokens / 1e6,
                    (double)group->read_census[row].a_bytes / tokens / 1e6);
        }
        fprintf(stderr, "llama_tt: read census — distinct weight shapes sum %.3f GB (one "
                        "of each shape)\n", (double)weight_sum / 1e9);
        fprintf(stderr, "llama_tt: param census — fallback differs from a cached block in "
                        "ONE word:");
        for (row = 0; row < 18u; row++)
            if (group->param_diff_word[row])
                fprintf(stderr, " w%u=%.1f/token", row,
                        (double)group->param_diff_word[row] / tokens);
        fprintf(stderr, "  (w17 = no single-word match)\n");
        for (row = 0; row < group->param_census_rows; row++)
            fprintf(stderr, "llama_tt: param census Kt %4u member-Nt %5u — %llu misses "
                            "(blocks cached), %.2f fallbacks/token\n",
                    group->param_census[row].kt, group->param_census[row].nt,
                    (unsigned long long)group->param_census[row].misses,
                    (double)group->param_census[row].fallbacks / tokens);
    }
    fprintf(stderr, "llama_tt: matmul per-bank reads —");
    for (bank = 0; bank < device->dram_bank_count && bank < FT_TT_MAX_DRAM_BANKS; bank++)
        fprintf(stderr, " b%u %.1f", bank,
                (double)group->bank_read_bytes[bank] / seconds / 1e9);
    fprintf(stderr, " GB/s  (one bank's measured ceiling is 59.2-59.5 GB/s, the card's 510.8)\n");

    /* M1 CARD-BUSY CLOCK, read from each member's writer (control words 12-15,
     * resident_matmul_writer.c): busy = descriptor pickup to completion post, idle = the gaps
     * between. The FRACTION needs no clock rate; ms/token assumes the 1350 MHz AICLK. */
    {
        uint32_t m;
        double busy_max = 0.0, busy_sum = 0.0;

        fprintf(stderr, "llama_tt: matmul card-busy —");
        /* M8: member 0's pull wait (writer word 61), so its busy can be read as work. */
        if (group->member_count > 0u && group->member[0].queue) {
            uint32_t pull = 0u;

            if (device->runtime.noc.read(device->runtime.noc.user, group->member[0].queue->x,
                                         group->member[0].queue->y,
                                         group->member[0].queue->control_address + 61u * 4u,
                                         &pull, sizeof(pull)) == FT_TT_OK)
                fprintf(stderr, " [m0 pull wait %.2fms]", (double)pull * 16.0 / 1.35e9 * 1e3 /
                                                           tokens);
        }
        for (m = 0; m < group->member_count; m++) {
            uint32_t busy = 0u, idle = 0u, count = 0u;

            if (!group->member[m].queue ||
                ft_tt_queue_read_field(&device->runtime.noc, group->member[m].queue,
                                       (ft_tt_queue_control_field)13, &busy) != FT_TT_OK ||
                ft_tt_queue_read_field(&device->runtime.noc, group->member[m].queue,
                                       (ft_tt_queue_control_field)14, &idle) != FT_TT_OK ||
                ft_tt_queue_read_field(&device->runtime.noc, group->member[m].queue,
                                       (ft_tt_queue_control_field)15, &count) != FT_TT_OK)
                continue;
            {
                const double busy_s = (double)busy * 16.0 / 1.35e9;
                const double frac = (busy + idle) ? (double)busy / ((double)busy + idle) : 0.0;

                if (busy_s > busy_max)
                    busy_max = busy_s;
                busy_sum += busy_s;
                fprintf(stderr, " m%u %.1f%% %.2fms %.0fus/d", m, 100.0 * frac,
                        busy_s * 1e3 / tokens, count ? busy_s * 1e6 / count : 0.0);
            }
        }
        fprintf(stderr, " | busiest %.2f ms/token, mean %.2f (since the last arm; 1350 MHz)\n",
                busy_max * 1e3 / tokens,
                group->member_count ? busy_sum * 1e3 / tokens / group->member_count : 0.0);
    }

    /* THE FLUSH CENSUS and its fit, span = fixed + bytes / rate, weighted by flushes. */
    {
        double sw = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0, span_total = 0.0;
        uint32_t row;

        for (row = 0; row < group->flush_census_rows; row++) {
            const double n = (double)group->flush_census[row].flushes;
            const double x = (double)group->flush_census[row].bytes;
            const double y = group->flush_census[row].span / (n > 0.0 ? n : 1.0);

            fprintf(stderr,
                    "llama_tt: matmul flush census — %8.3f MB x %6.2f/token (%.2f proj each): "
                    "span %7.1f us, wait %7.1f us, %6.1f GB/s over span\n",
                    x / 1e6, n / tokens,
                    (double)group->flush_census[row].projections / (n > 0.0 ? n : 1.0),
                    y * 1e6, group->flush_census[row].wait / (n > 0.0 ? n : 1.0) * 1e6,
                    y > 0.0 ? x / y / 1e9 : 0.0);
            if (group->class_clock == 1 && n > 0.0) {
                const double busiest = (double)group->flush_census[row].card_busy_max * 16.0 /
                                       1.35e9 / n;
                const double mean = (double)group->flush_census[row].card_busy_sum * 16.0 /
                                    1.35e9 / n / (double)group->member_count;

                fprintf(stderr,
                        "llama_tt:   CLASS CLOCK — card busy per flush: busiest member %.1f "
                        "us, mean %.1f us; %.1f GB/s over the busiest\n",
                        busiest * 1e6, mean * 1e6, busiest > 0.0 ? x / busiest / 1e9 : 0.0);
                {
                    uint32_t m;

                    fprintf(stderr, "llama_tt:   CLASS CLOCK — m0 pull wait %.1f us; per member "
                                    "us/flush:",
                            (double)group->flush_census[row].card_pull_wait * 16.0 / 1.35e9 /
                                n * 1e6);
                    for (m = 0; m < group->member_count; m++)
                        fprintf(stderr, " %.0f",
                                (double)group->flush_census[row].card_busy_member[m] * 16.0 /
                                    1.35e9 / n * 1e6);
                    fprintf(stderr, "\n");
                }
            }
            {
                const pn_tt_matmul_group *gg = group;
                const double nn = n > 0.0 ? n : 1.0;

                fprintf(stderr,
                        "llama_tt:   split — issue %.1f us, gap %.1f us, await %.1f us = "
                        "busy %.1f us (%.2f members) + done-on-arrival %.1f us (%.2f members, "
                        "%.2f us each), %.1f polls\n",
                        gg->flush_census[row].issue / nn * 1e6,
                        gg->flush_census[row].gap / nn * 1e6,
                        gg->flush_census[row].wait / nn * 1e6,
                        gg->flush_census[row].busy_await / nn * 1e6,
                        (double)gg->flush_census[row].busy_count / nn,
                        gg->flush_census[row].done_await / nn * 1e6,
                        (double)gg->flush_census[row].done_count / nn,
                        gg->flush_census[row].done_count
                            ? gg->flush_census[row].done_await /
                                  (double)gg->flush_census[row].done_count * 1e6
                            : 0.0,
                        (double)gg->flush_census[row].polls / nn);
            }
            sw += n; sx += n * x; sy += n * y; sxx += n * x * x; sxy += n * x * y;
            span_total += group->flush_census[row].span;
        }
        if (group->flush_census_rows >= 2u && sw * sxx - sx * sx > 0.0) {
            const double slope = (sw * sxy - sx * sy) / (sw * sxx - sx * sx);
            const double fixed = (sy - slope * sx) / sw;

            fprintf(stderr,
                    "llama_tt: matmul flush FIT — fixed %.1f us/flush x %.1f flushes/token = "
                    "%.4f s/token, streaming %.1f GB/s; span %.4f s/token; %llu dropped\n",
                    fixed * 1e6, sw / tokens, fixed * sw / tokens,
                    slope > 0.0 ? 1.0 / slope / 1e9 : 0.0, span_total / tokens,
                    (unsigned long long)group->flush_census_dropped);
        }
    }
}


uint32_t pn_tt_matmul_group_incomplete(pn_tt_device *device)
{
    pn_tt_matmul_group *group = device ? &device->group : NULL;
    uint32_t incomplete = 0u;

    if (!group)
        return 0u;
    for (uint32_t member = 0; member < group->member_count; member++) {
        pn_tt_matmul_member *slot = &group->member[member];
        uint32_t completed = 0u;

        if (!slot->queue || !slot->booted)
            continue;
        (void)ft_tt_queue_read_field(&device->runtime.noc, slot->queue,
                                     FT_TT_QUEUE_FIELD_COMPLETED_COUNT, &completed);
        incomplete += completed != slot->queue->issued;
    }
    return incomplete;
}

#endif /* USE_TT */
