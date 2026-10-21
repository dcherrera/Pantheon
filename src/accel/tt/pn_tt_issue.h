/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_issue.h
 * @brief The ISSUE half of a descriptor's host cost — which accessor the parameter block and
 *   the ring slot are written through, and what that costs in PCIe accesses.
 *
 * ---------------------------------------------------------------------------
 * THE OTHER HALF OF THE 58 us, AND IT IS THE SMALLER ONE
 * ---------------------------------------------------------------------------
 * `pn_tt_residency.c` prices the AWAIT: seventeen non-posted reads through the UC / WORD32
 * accessor, taken to six and then to one by `PN_TT_RESIDENCY_AWAIT`. The ISSUE side is the
 * same arithmetic on the same link in the write direction, and it has never been cut:
 *
 *   `write_op_parameters`   64 B -> 16 word writes, POSTED
 *   `stage` (one ring slot) 32 B ->  8 word writes, POSTED
 *   `publish` (head bump)    4 B ->  1 word write,  POSTED
 *                                    -----------------------
 *                                    25 POSTED ACCESSES
 *
 * **Posted stores are not a round trip**, which is why issue was always the cheaper half and
 * why the await was cut first. What the two shapes below change is the ACCESS COUNT, not the
 * number of round trips — there are none.
 *
 * ---------------------------------------------------------------------------
 * THE TWO SHAPES, AND WHY `wide` IS TWO ACCESSORS AND NOT ONE
 * ---------------------------------------------------------------------------
 *   `word`   everything through `runtime->noc`    16 + 8 + 1 = **25**   what shipped
 *   `wide`   the two 16-byte-aligned writes through `runtime->bulk_l1` (WC / BLOCK16) and
 *            the head bump through `runtime->noc`  4 + 2 + 1 = **7**
 *
 * The parameter block and the ring slot are 16 bytes aligned in ADDRESS and in LENGTH, so
 * both fit the bulk L1 accessor. The head bump is four bytes at an arbitrary word and does
 * not — `ft_tt_noc_check_transfer()` refuses it, which is a structural fact and not a
 * convention. `ft_tt_queue_issue_is_wide_addressable()` asks that same function about a
 * queue's placement, so a queue the wide shape cannot reach FALLS BACK rather than failing on
 * its first descriptor.
 *
 * ---------------------------------------------------------------------------
 * THE WC/UC STORE-ORDERING SEAM — THE ONLY THING `wide` PUTS AT RISK
 * ---------------------------------------------------------------------------
 * `wide` splits one descriptor's issue across TWO MAPPINGS of the same BAR: a write-combined
 * window for the payload and an uncached one for the head bump that announces it. "Descriptors
 * before head" is one of the two orderings the whole queue protocol rests on
 * (`ft_tt_queue.h`), and a WC store is allowed to sit in the core's write-combining buffer
 * from the program's point of view. If the head bump overtakes the ring write, the kernel
 * executes **the descriptor that slot held one lap of the ring ago** — well formed, no kernel
 * error, and `completed_count` exactly what the host expected. It is a wrong answer, not a
 * hang, and the ONLY thing that separates it from a correct one is the per-index token.
 *
 * Three things stand between that and the model, and they are named here rather than assumed:
 *
 *  1. **`ft_tt_udrv_noc_transfer()` ends every WRITE slice with `ft_tt_noc_copy_store_fence()`
 *     — an `sfence` — unconditionally, on both widths.** So the WC payload is retired from the
 *     CPU before the UC head bump is issued. This is not new code and not this knob's: the
 *     transport has always done it, because "an unrelated module's later UC access happens to
 *     drain the buffer" is a timing property of one host.
 *  2. **Both windows carry the SAME NoC aim template** — `ft_tt_udrv_noc_path_open()` memsets
 *     it and nothing calls `set_aim_template()`, so neither window selects a different
 *     ordering mode or a static VC. Two PCIe posted writes to one device on one path cannot
 *     pass each other.
 *  3. **The four-agreement check is live on every descriptor**, and the token is what catches
 *     this. `tests/test_queue.c` publishes a head for a descriptor that was never written and
 *     requires the check to refuse and to name the token — with the COUNT passing and the
 *     kernel reporting no error, which is what makes the arm about ORDERING rather than about
 *     arithmetic. `PN_TT_RED=issue_head_first` is the same inversion on the card.
 *
 * The read direction of this seam already ships: `PN_TT_RESIDENCY_AWAIT=line` and the matmul
 * group's fast completion have both polled through `bulk_l1` (WC) while the same descriptors
 * were issued through `noc` (UC) since 2026-08-20.
 *
 * ---------------------------------------------------------------------------
 * MEASURED 2026-08-26: `wide` HITS ITS ACCESS TARGET EXACTLY AND IS 5.7% SLOWER
 * ---------------------------------------------------------------------------
 * 25.00 -> 7.00 accesses an issue, to the digit, with the window byte counters agreeing
 * (the UC window's write bytes fall by 125,354.7 B/token and the WC window's rise by
 * 125,354.7). The matmul ledger goes 9.33 -> 3.08 accesses a pair. **Decode goes
 * 6.51 -> 6.14 tok/s** against a same-session control drift of 0.15%, and the two 256-token
 * soaks are byte-identical, so this is a clean loss and not a correctness problem.
 *
 * **THE ACCESSES WERE NEVER THE COST. THE TLB RE-AIM IS.** The three calls timed apart say
 * so directly: on `word`, sixteen accesses cost 8.69 us and eight cost 0.14, because the
 * FIRST device access of an enqueue re-aims the window and everything after it rides an
 * aperture already pointed at that core. Measured on the card, an aim is **8.92 us, of which
 * 98.0% is its four non-posted readback reads** (2.19 us each) — and the dispatch path takes
 * **4,333 aims a token, 0.0387 s, 25.1% of the decode wall.**
 *
 * `wide` loses because it puts the payload on one window and the head bump on the other, so
 * it pays TWO aims per issue instead of one: 6,483 aims a token, and the whole loss lands in
 * the `head` clock (0.66 -> 7.83 us).
 *
 * **IT WAS A BLOCKED LEVER, NOT A DEAD ONE, AND THE BLOCK IS GONE.** With the aim removed,
 * `word`'s 25 accesses are ~0.8 us an issue and `wide`'s 7 are ~0.2, so `wide` was projected
 * at ~1.4 ms/token: worth having AFTER the aim is removed and worth -17 ms/token before it.
 * Its own confirmation round measured **-1.15 ms/token** against a stated ~1.4 and flipped the
 * default (see @status). It is also the instrument that priced the aim in the first place:
 * two shapes differing by exactly one aim per issue is how 8.9 us was measured three
 * independent ways before the aim had a clock of its own.
 *
 * ONE THING THE ROUND DID NOT SETTLE. The matmul group's own `dispatch split` does not
 * reconcile with the wall: ISSUE falls 0.126 -> 0.075 s over 48 tokens (-1.06 ms/token) and
 * WAIT rises 2.708 -> 2.760 s (+1.09 ms/token), which nets to zero, while the WALL falls
 * 1.15 ms/token. The split's WAIT term is card time the host is partly running ahead of, so
 * it is not a second clock on the same quantity — the wall is the instrument and the split is
 * the evidence the knob armed.
 *
 * @status Green, and `wide` SHIPS since 2026-08-27. It was a 5.7% regression only while every
 *   issue paid a second TLB re-aim; with the window bank absorbing both aims its own
 *   confirmation round measured decode 0.1391 s/token (word, THREE arms, all three to four
 *   decimals — a 0.00% control-vs-control drift) against 0.1381/0.1378: **+0.83%,
 *   -1.15 ms/token**. Prefill 165.19/165.27 -> 166.52 tok/s on the `batched prefill —` line's
 *   own denominator, **+0.80%** against a 0.048% drift, the gain in the `sequential` term.
 *   Every arm `cmp` IDENTICAL at 189 B / 80fc6259; two 256-token soaks byte-identical at
 *   587 B / abca1fa9. The instrument that says the knob armed is the issue clock,
 *   10.0-10.1 -> 5.8-6.1 us/op.
 * @issues None
 * @todo None
 */

#ifndef PN_TT_ISSUE_H
#define PN_TT_ISSUE_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "tt/ft_tt_queue.h"
#include "tt/ft_tt_runtime.h"

/** Which accessor a descriptor's payload is written through. */
typedef enum pn_tt_issue_shape {
    PN_TT_ISSUE_WORD = 0, /**< UC / WORD32 for all three writes — 25 accesses. */
    PN_TT_ISSUE_WIDE      /**< WC / BLOCK16 for the payload, UC for the head — 7. */
} pn_tt_issue_shape;

/** `PN_TT_ISSUE=word|wide`, read ONCE. An unrecognised value is refused to the default and
 *  says so, rather than silently selecting one. */
pn_tt_issue_shape pn_tt_issue_shape_selected(void);

/** "word" or "wide"; never NULL. */
const char *pn_tt_issue_shape_string(pn_tt_issue_shape shape);

/**
 * The shape actually usable for `queue` on `runtime`: `pn_tt_issue_shape_selected()` demoted
 * to `PN_TT_ISSUE_WORD` when the queue's blocks are not 16-byte addressable or the runtime has
 * no bulk L1 accessor. Decide it ONCE at arm time — a per-descriptor `getenv` is 562 of them a
 * token on the resident families alone.
 */
pn_tt_issue_shape pn_tt_issue_shape_for(const ft_tt_runtime *runtime, const ft_tt_queue *queue);

/** The accessor `shape` names. Never NULL for a non-NULL runtime. */
const ft_tt_noc_access *pn_tt_issue_access(const ft_tt_runtime *runtime,
                                           pn_tt_issue_shape shape);

/** Bytes one PCIe access of `shape`'s payload accessor moves: 4 for WORD32, 16 for BLOCK16.
 *  This is the unit the whole link is priced in — a CALL count cannot tell a 4-byte write
 *  from a 64-byte one, and a BYTE count cannot tell sixteen words from four blocks. */
uint32_t pn_tt_issue_unit_bytes(pn_tt_issue_shape shape);

/** Accesses `bytes` of payload cost at `shape`, plus the head bump's single word. */
uint64_t pn_tt_issue_accesses(pn_tt_issue_shape shape, uint32_t payload_bytes);

/**
 * RED PROOF ONLY. Is `PN_TT_RED=issue_head_first` named? The arm publishes the head BEFORE the
 * descriptor is written — the exact inversion the WC/UC seam risks — so the four-agreement
 * check must REFUSE and must name the TOKEN.
 *
 * An ordering claim that cannot be inverted cannot be tested: without this, a clean `wide` run
 * is equally consistent with the check being vacuous.
 *
 * This is only the knob. WHEN it fires is the caller's — `pn_tt_residency.c` waits for the
 * ring to have wrapped once and fires once, because a head published over a ZEROED slot is an
 * unknown opcode and proves something else entirely.
 */
bool pn_tt_issue_red_head_first(void);

#endif /* USE_TT */
#endif /* PN_TT_ISSUE_H */
