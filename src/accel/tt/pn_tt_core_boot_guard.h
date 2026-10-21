/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_core_boot_guard.h
 * @brief The one question asked before ANY of this backend boots firmware onto a Tensix:
 *        **is that core ready, or is something still running on it?**
 *
 * ---------------------------------------------------------------------------
 * THE FAILURE THIS EXISTS FOR — IT CROSSES A PROCESS BOUNDARY
 * ---------------------------------------------------------------------------
 * A refused descriptor makes a resident reader abort and leaves the compute thread spinning
 * out its 400,000,000-iteration bound. That is the family's design. `pn_tt_residency_stop()`
 * waits it out (20 s of grace) and `pn_tt_residency_retired()` reports whether it did, so the
 * process that CAUSED the spin will not re-arm onto it — those guards landed and they fire.
 *
 * **They protect one process. The spinning thread outlives it.** A `serve` restart or a model
 * reload is a NEW process with an empty residency table, and its ordinary first arm boots
 * firmware onto a core that is still running. MEASURED ON SILICON 2026-08-21/22: that is a
 * WEDGE — core 90 came back `instrn_buf_status = 0x66`, twice, and rung 1 does NOT clear it;
 * only a whole-card `ft_tt_core recover --rung 3` does. Nothing in a fresh process's own
 * memory can see this, so the CARD is asked. `ft_tt_core_readiness.h` is that reading, and it
 * is the same verdict `ft_tt_core status` has printed since 2026-08-12.
 *
 * ---------------------------------------------------------------------------
 * IT SITS AT THE BOOT, NOT AT ONE FAMILY'S ARM
 * ---------------------------------------------------------------------------
 * `pn_tt_core_holder()` already refuses a core another residency of THIS process holds, by
 * name. This is the other half — the core the CARD says is busy — and it has to cover every
 * family, so it is called at all three places that boot firmware:
 *
 *   - `pn_tt_prepare_family_on_core()`, which is reduce, eltwise, topk, rope, memory,
 *     attention, gdn, gdnc, the GDN head-split members, the render directions and both
 *     head-norm slots;
 *   - `pn_tt_matmul_group_boot()`, members 1..N;
 *   - `pn_tt_device_open()`'s own core, which is matmul member 0 and the FIRST boot a serve
 *     takes — the one a restart lands on.
 *
 * ---------------------------------------------------------------------------
 * WAIT, THEN REFUSE — AND THE READING DECIDES WHICH
 * ---------------------------------------------------------------------------
 * "Not ready" is two different things. A core spinning out a bound WILL finish, so it is
 * WAITED for, on the same grace `pn_tt_residency_stop()` uses because it is the same spin. A
 * core at `instrn_buf_status = 0x66` never will, and waiting on it is dead time in front of
 * an inevitable refusal. `ft_tt_core_readiness`'s `waiting_can_help` separates them; this
 * file does not guess.
 *
 * Recovery is deliberately NOT attempted. Resetting a core out of a blocked wait is how cores
 * get wedged in the first place (`pn_tt_residency.h` hazard 2), and rung 3 resets the WHOLE
 * CARD — which is an operator's decision, not an arming path's.
 *
 * ---------------------------------------------------------------------------
 * THE REFUSAL IS LOUD, AND THAT IS A REQUIREMENT RATHER THAN A STYLE
 * ---------------------------------------------------------------------------
 * A family that fails to arm can take the WHOLE MODEL to the ~26 s/position CPU forward, and
 * it did so behind a single line — `qwen35 Tenstorrent upload skipped/failed; using CPU
 * forward` — which burned 11 minutes of a run before anyone noticed, because the numbers that
 * came out were plausible and merely very slow. A silent degradation caused BY a reliability
 * fix would be worse than the bug it fixes. So the refusal prints a block, names the core,
 * quotes what the card said, gives the recovery command verbatim, and states the consequence.
 *
 * **`PN_TT_BOOT_GUARD_REFUSAL_TOKEN` is the stable grep handle** for a timing script's abort
 * guard, next to the "using CPU forward" one every script already carries.
 *
 * @status New 2026-08-24. The classifier under it is covered off-hardware by
 *         foundry/tt/tests/test_core_readiness.c.
 * @issues None
 * @todo None
 */

#ifndef PN_TT_CORE_BOOT_GUARD_H
#define PN_TT_CORE_BOOT_GUARD_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "tt/ft_tt_runtime.h"

/** The one string a script needs to grep for. Do not reword it without saying so. */
#define PN_TT_BOOT_GUARD_REFUSAL_TOKEN "REFUSED TO BOOT"

/**
 * May firmware be booted onto `core_index` right now?
 *
 * Reads the core. If it is busy in a way that clears itself, waits for it — up to
 * `PN_TT_RESIDENCY_STOP_GRACE_MS`, the same bound the stop uses for the same spin — and
 * re-reads. Returns false with a LOUD, multi-line explanation on stderr if it is still not
 * ready, or if it never could be.
 *
 * Writes nothing to the card, resets nothing and parks nothing. Reads only.
 *
 * @param runtime     the open card; borrowed
 * @param core_index  Tensix index, as `ft_tt_coord_tensix_by_index` numbers them
 * @param who         what wants the core, for the message, e.g. "gdn" or "matmul member 3"
 * @param core_env    the environment variable that would move this family to another core,
 *                    or NULL when there is not one
 * @return            true when the core may be booted. **A read that itself fails returns
 *                    true**: this guard exists to catch a specific, positively-identified
 *                    busy core, and turning an unreadable status into a refusal would let a
 *                    diagnostic path's failure take the whole model to the CPU forward
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
bool pn_tt_core_boot_is_permitted(ft_tt_runtime *runtime, uint32_t core_index, const char *who,
                                  const char *core_env);

#endif /* USE_TT */
#endif /* PN_TT_CORE_BOOT_GUARD_H */
