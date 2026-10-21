/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdn_core_scaling.h
 * @brief Does the GDN block go faster on more than one Tensix? An INSTRUMENT, armed only by
 *        `PN_TT_GDN_CORES` and never by the shipped path.
 *
 * ---------------------------------------------------------------------------
 * WHY THE QUESTION IS OPEN AND WHY IT NEEDS SILICON
 * ---------------------------------------------------------------------------
 * The GDN op runs on ONE core. Its outer loop is over v-heads, and the heads are independent
 * — each carries its own [S,S] state, reads its own v/decay/beta and writes its own row of y
 * — so the arithmetic parallelises trivially. What is NOT known is whether the CARD does:
 * the descriptor moves the carried state through DRAM, and Phase 1's core-scaling knee at
 * N=4 was measured on the MATMUL path, whose traffic is a broadcast weight strip and nothing
 * like this one. A knee measured on one op is not a property of the machine.
 *
 * So this probe asks the machine directly: run N copies of the SAME descriptor on N cores at
 * once, each against a DIFFERENT SSM layer's carried state so no two touch the same DRAM,
 * and compare the wall time of the set against one core's. Perfect scaling is N descriptors
 * in one descriptor's time. Anything less is a shared resource, and the ratio prices it.
 *
 * **Full-width copies rather than a head split, deliberately.** Splitting 32 heads over N
 * cores would measure the same thing and would also need N sets of per-core operand offsets
 * — new address arithmetic, in a probe, whose bugs would look exactly like a scaling result.
 * N full copies need no new addresses at all: the only word that changes per core is the
 * state operand, and it is a region this device already allocated. If N copies cost one
 * copy's time, then N head-slices certainly do.
 *
 * ---------------------------------------------------------------------------
 * WHAT IT COSTS, WHICH IS NOT NOTHING
 * ---------------------------------------------------------------------------
 * The probe WRITES the carried state of the layers it borrows. It runs once, at the first
 * DELTA descriptor of a run, where the state is still the zero the sequence starts from —
 * but it leaves those layers holding a recurrence step the model did not ask for. **A run
 * with this knob set produces text that is not evidence of anything.** It is a stopwatch.
 *
 * It also holds N-1 extra Tensix for the length of the sweep and then gives them back. A
 * GDN triad that exits on a circular-buffer spin bound wedges its core until a rung-3 reset,
 * so the extra residencies are retired by ASKING, the same way every other one is.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TT_GDN_CORE_SCALING_H
#define PN_TT_GDN_CORE_SCALING_H

#ifdef USE_TT

#include "pn_tt_device.h"

#include "tt/ft_tt_op_gdn.h"

/**
 * Run the sweep, print the table on stderr, and give the extra cores back.
 *
 * Does nothing at all unless `PN_TT_GDN_CORES` is set to 2 or more. Safe to call on every
 * DELTA descriptor — it runs once per device and remembers that it has.
 *
 * @param device    an armed GDN device (`pn_tt_gdn_is_armed`)
 * @param template  a fully built DELTA request; the probe changes only its state operand
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
void pn_tt_gdn_core_scaling_probe(pn_tt_device *device,
                                  const ft_tt_op_gdn_request *template_request);

#endif /* USE_TT */
#endif /* PN_TT_GDN_CORE_SCALING_H */
