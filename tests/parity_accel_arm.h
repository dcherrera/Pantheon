/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file parity_accel_arm.h
 * @brief Pick the accelerator arm a parity harness should score, and refuse the vacuous
 *        CPU-vs-CPU configuration — shared by every harness so the identity control means
 *        the same thing in all of them.
 *
 * @status Working
 * @issues None
 * @todo None
 */
#ifndef PN_TESTS_PARITY_ACCEL_ARM_H
#define PN_TESTS_PARITY_ACCEL_ARM_H

#include "../include/pantheon.h"
#include "../src/core/engine.h"

/**
 * Return WHICHEVER forward the engine selected for `model`, not a hardcoded one.
 *
 * Hardcoding pn_llama_forward_cuda made a harness silently CPU-vs-CPU on any other card — a
 * green run that proves nothing, which is the failure mode this project keeps paying for.
 *
 * PN_PARITY_IDENTITY=1 deliberately makes the arm the CPU oracle itself. That is the one
 * legitimate CPU-vs-CPU configuration: every reported difference must come out exactly zero,
 * and if it does not, no accelerator number from the harness can be believed. It says so on
 * stderr so a stray environment variable can never be mistaken for a passing card run.
 *
 * The rule is an IDENTITY, not a table: the accelerator arm is `model->forward` whenever it
 * differs from the architecture's own CPU forward. A table of known accelerator symbols has
 * to be extended for every family, and the one that was here listed only the llama pair — so
 * on a qwen35moe model (Ornith) it selected nothing and the caller refused, which is the same
 * blindness in a politer form. The identity rule cannot go stale, and it is also the only
 * rule that gets Bonsai right: that architecture's CPU forward is pn_bonsai_forward while its
 * TT forward is pn_qwen35moe_forward_tt, a pair no per-family table would have written down.
 *
 * @param name_out  set to a human name for the arm ("TT", "CUDA", "CPU-IDENTITY-CONTROL")
 * @return          the forward to score, or NULL when there is no accelerator and no
 *                  identity control — in which case the caller must refuse to run
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
pn_forward_fn parity_select_accel_arm(pn_model *model, const char **name_out);

/**
 * The CPU oracle every parity harness scores AGAINST, for `model`'s own architecture.
 *
 * It comes from `pn_get_forward_fn(model->config.arch)` — the engine's own architecture
 * registry — rather than from a second copy of that mapping, so the oracle cannot drift from
 * what the engine calls the CPU path. Harnesses used to hardcode `pn_llama_forward` here,
 * which on a qwen35moe model would have scored a card against a DIFFERENT MODEL's forward
 * reading a qwen35moe weight struct.
 *
 * @param name_out  set to a human name for the oracle, e.g. "CPU/qwen35moe"
 * @return          the CPU forward, or NULL when the architecture has none
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
pn_forward_fn parity_select_cpu_oracle(pn_model *model, const char **name_out);

/**
 * Refuse a model whose CPU oracle is not `expected`.
 *
 * The arm selection above is deliberately architecture-agnostic, but some harnesses' PROBES
 * are not — a residual-divergence probe casts `model->weights` to `pn_llama_qweights *`, which
 * on any other family is a wrong-struct read, and it used to be protected only by the arm
 * table happening to list llama alone. Making selection general moved that guard out from
 * under them, so it is stated here instead of being an accident of a table.
 *
 * @param tool  the harness's name, for the refusal message
 * @return      0 when the oracle matches, -1 when the caller must not run
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int parity_require_cpu_oracle(pn_model *model, pn_forward_fn expected, const char *tool);

/**
 * Detach or reattach whatever accelerator SEAM this model's family installs on shared state,
 * so the CPU oracle is genuinely the CPU.
 *
 * **NOT EVERY ACCELERATOR IS A FORWARD.** The llama family's is: `pn_llama_forward_tt` is a
 * different function and calling `pn_llama_forward` runs on the host. The qwen35/bonsai
 * family's is a HOOK — `qwen35moe_mv()` consults `cfg->mv.fn` on every projection and the
 * upload installs the card's there — so in a card-armed process the architecture's "CPU
 * forward" still runs all 249 projections on the card. A harness that scored the two forwards
 * against each other there would be comparing the card with the card, and against the
 * RESIDENT-ACTIVATION arm it segfaults instead, which is how this was found.
 *
 * Families with no seam are a no-op, so a caller may bracket every oracle call with this
 * unconditionally.
 *
 * @param enable  0 before running the CPU oracle, non-zero before running the accelerator
 * @return        1 if a seam was attached before the call, 0 if not
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int parity_accel_seam_enable(pn_model *model, int enable);

/** @return non-zero when PN_PARITY_IDENTITY selected the identity control. */
int parity_identity_control_active(void);

#endif /* PN_TESTS_PARITY_ACCEL_ARM_H */
