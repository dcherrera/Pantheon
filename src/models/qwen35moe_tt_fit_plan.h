/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_tt_fit_plan.h
 * @brief When the qwen35 family's TT load leaves main's placement: the FIT POLICY trigger.
 *
 * Main's placement is what every shipped TT model was measured on, and any change to where a
 * buffer lands can move speed (a +0.13% TTFT shift on Qwen3.6 from options A/B on a p150). So
 * the fit options (A-D) apply ONLY when main's plan does
 * not fit the card:
 *   - A, `attn_q` uploaded as its Q and gate halves instead of whole: decided HERE, before the
 *     weights go up, because it moves the weight placement;
 *   - B (mirrors-first tall strips) and C (no tall `attn_qg`): taken with A, or on their own at
 *     the tall-strip stage when an exact dry-run of main's order against the allocator's real
 *     free bytes fails (qwen35moe_prefill_tt.c).
 *   - D (one GDN conv weight for both families, pn_tt_gdn_conv_share.h): taken with A, at the
 *     GDN arm. It does not enter the floor below, so the trigger is unchanged.
 *
 * THE TRIGGER for A is a FLOOR of main's DRAM need: the bytes main certainly allocates — the
 * artifact's matmul weights, the second (staged) copy of `attn_q`, the KV region, the GDN
 * family and the GDNC conv-constant copies, the tall strip set and the prefill's mandatory
 * post-load reserve (the GDNC state-split slices), and with a dense MTP head its weights, its
 * KV and its verify's GDN snapshot slots (MTP on). Every term is at most what main
 * allocates. It omits the small arms (decode and attention strips, constants, padding:
 * ~0.21 GiB on Qwen3.8) and per-bank imbalance, which on Qwen3.8 at 16k cost main 0.377 GiB
 * of slack (it failed with that much free). The policy engages when the floor leaves less than
 * PN_QWEN35_TT_FIT_HEADROOM of the card free — OR when THE BANK RULE says the fullest bank
 * would keep less than PN_QWEN35_TT_TIER_HEADROOM (usable spec §3, David's card 20261118-1010):
 * per bank, its free bytes at the decide less main's floor in the fullest bank (the KV at its
 * larger bank role, the rest even, the verify slots and the prefix checkpoint's set included)
 * less a bank's share of the whole-card allowance. A floor that fits the card can still leave
 * every bank ~100 MiB: the 262,144 tier ran main's placement with its fullest bank at 116 MiB
 * before its first verify took 81 more (the 256k tier run, block 1).
 *
 * Measured slack on the p150 (card 31.875 GiB) against this floor: Ornith 32k ~20 GiB,
 * Qwen3.6 36,864 ~2.6 GiB -> main's placement, byte for byte. Qwen3.8 Bfp8 at 16k 0.46 GiB and
 * at 32k -0.54 GiB -> the fit policy; with its tall set off (PN_TT_PREFILL_TILE_ROWS=0, which
 * main loads) 0.95 GiB -> main's.
 *
 * THE BANK BUDGET prices a
 * load the floor does not: per DRAM bank, against a MEASURED anchor. The card's fit is per bank
 * and the fullest bank binds, but no host model reproduces the allocator's per-bank imbalance
 * (emptiest-bank fallbacks, fixed-bank mirrors, the KV sets padded level). So a load is priced
 * as its DIFFERENCE from an anchor load whose fullest bank was measured: every bank had at
 * least that much free, so each bank of the new load has at least
 *     anchor free + its weight bytes freed - its KV growth - the even classes' growth / banks
 * and a one-bank class (the gate-block array) is charged to whichever bank that leaves lowest.
 * Every class the anchor and the load share (GDN, GDNC, tall strips, the prefill reserve,
 * RoPE, the small arms) cancels; only the classes that differ are inputs.
 *
 * No device: arithmetic over a shape, tested on any host (`tests/test_tt_q38_fit.c`; the bank
 * budget in `tests/test_tt_q38_mixw_fit.c`).
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef QWEN35MOE_TT_FIT_PLAN_H
#define QWEN35MOE_TT_FIT_PLAN_H

#include <stdbool.h>
#include <stdint.h>

/** Free bytes below which main's floor engages the fit policy (see the file note). */
#define PN_QWEN35_TT_FIT_HEADROOM (768ull * 1024ull * 1024ull)

/** What the floor needs to know about a model and its load. Feature counts, not tiles. */
typedef struct pn_qwen35_tt_fit_shape {
    uint32_t n_embd, ffn, n_head, n_head_kv, head_dim;
    uint32_t n_attn, n_ssm;
    uint32_t conv_dim, conv_taps, ssm_state, v_heads, d_inner;
    uint32_t ctx;            /**< KV positions reserved (pn_qwen35moe_max_seq) */
    uint32_t kv_k_tile_bytes; /**< K cache's tile: 0 / 2048 bf16, 1088 Bfp8_b, 576 Bfp4_b */
    uint32_t kv_v_tile_bytes; /**< V cache's tile, the same choices (K8V4: 1088 / 576) */
    uint32_t tall_rows;      /**< prefill tall-strip bands, 0 = no tall set */
    uint32_t mirrors;        /**< activation replica count (the decode mirrors) */
    bool outproj_mirrors;    /**< the gated tall strips mirrored too */
    uint32_t gdnc_replicas;  /**< conv-constant copies, copy 0 included */
    bool attn_on_card;
    bool gdn_on_card;
    bool state_split;        /**< the GDNC state-split slices are reserved */
    /** The dense MTP draft head, 0 = none: its artifact bytes plus the draft-vocabulary
     *  LM-head columns it stages (pn_qwen35_tt_mtp_arm). It also takes one attention layer of
     *  KV per position and `mtp_rows` GDN snapshot slots for its verify. */
    uint64_t mtp_bytes;
    uint32_t mtp_rows;
    /** Of `mtp_bytes`, the staged draft-vocabulary LM-head copy; and whether the draft reads
     *  those columns of the trunk's LM head in place instead (lever V, pn_tt_weight_view.h). */
    uint64_t mtp_draft_bytes;
    bool draft_view;
    /** The exact card gates' lookup tables are wanted (qwen35moe_gate_tables.h): priced at
     *  their cap, every SSM layer x V-head slot, spread over the banks. */
    bool gate_tables;
} pn_qwen35_tt_fit_shape;

/** Main's floor, term by term (bytes). */
typedef struct pn_qwen35_tt_fit_floor {
    uint64_t weights, second_q, kv, gdn, gdnc, tall, reserve, mtp, gate_tables, total;
} pn_qwen35_tt_fit_floor;

/** One exact-gate table slot's cap (PN_QWEN35_GATE_TABLE_SLOT_WORDS x 4 B). */
#define PN_QWEN35_TT_FIT_GATE_TABLE_SLOT 40960ull

/** The most banks one layer's decay tables stripe over (ftgc_table.h kFtgtMaxStripes). */
#define PN_QWEN35_TT_FIT_TABLE_STRIPES 8u

/**
 * The exact card gates' tables at their cap: `layers` x `heads` decay slots (rounded up to a
 * whole stripe at the most stripes), one beta copy a bank, and the 256 B table block a layer.
 * The loader packs each layer at its widest table, so this is an upper bound (Qwen3.8:
 * 90.3 MiB priced, ~66 MiB placed).
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint64_t pn_qwen35_tt_fit_gate_table_bytes(uint32_t layers, uint32_t heads);

/**
 * Where the gate tables go, STRIPED: the largest S in {8, 4, 2, 1} (S <= `banks`) for which
 * every layer's heads split over S DISTINCT banks — head h in stripe h % S, ceil(heads / S)
 * slots of `layer_slot[l]` bytes a stripe — and every bank holding a stripe also holds one copy
 * of the beta table (`beta_bytes`), each stripe going to the emptiest bank not yet used by its
 * layer, every bank keeping `keep_free` (the 128 MiB bank rule, plus the MTP verify slots'
 * share while they are still to come). `free_bytes` is updated for the S chosen.
 * @return S with `bank_out[l * PN_QWEN35_TT_FIT_TABLE_STRIPES + s]` filled, or 0 (refused:
 * the host gates stay).
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t pn_qwen35_tt_fit_table_stripes(uint64_t *free_bytes, uint32_t banks,
                                        const uint64_t *layer_slot, uint32_t layers,
                                        uint32_t heads, uint64_t beta_bytes,
                                        uint64_t keep_free, uint32_t *bank_out);

/**
 * Main's DRAM floor for `shape`, with the artifact's matmul bytes and its joint `attn_q` bytes
 * read off the directory (pn_tt_device_artifact_matmul_bytes).
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
void pn_qwen35_tt_fit_floor_compute(const pn_qwen35_tt_fit_shape *shape, uint64_t weight_bytes,
                                    uint64_t joint_q_bytes, pn_qwen35_tt_fit_floor *out);

/** Main's tall strip set in tile columns (the logit row excluded, the joint `attn_qg`
 *  included), as pn_qwen35_tt_tall_requests() builds it without the fit policy. */
uint64_t pn_qwen35_tt_fit_tall_tiles(const pn_qwen35_tt_fit_shape *shape);

/** Does the floor leave less than PN_QWEN35_TT_FIT_HEADROOM of `card_bytes`? */
bool pn_qwen35_tt_fit_policy_wanted(const pn_qwen35_tt_fit_floor *floor, uint64_t card_bytes);

/* ---- The bank budget (see the file note) ---- */

/** Most DRAM banks the bank budget prices (a p150 has 8). */
#define PN_QWEN35_TT_FIT_MAX_BANKS 16u

/** Free bytes every bank must keep after a tier load (usable spec §3 headroom rule). */
#define PN_QWEN35_TT_TIER_HEADROOM (128ull * 1024ull * 1024ull)

/**
 * One GDN snapshot set: every SSM layer's state + conv window. A prefix-checkpoint twin and a
 * compact k1 verify slot are each one set (pn_tt_gdn_checkpoint.h, pn_tt_gdn_verify_slots.h).
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint64_t pn_qwen35_tt_fit_snapshot_bytes(const pn_qwen35_tt_fit_shape *shape);

/**
 * Every SSM layer's recurrent state alone (fp32): one row of a state-only verify.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint64_t pn_qwen35_tt_fit_state_bytes(const pn_qwen35_tt_fit_shape *shape);

/**
 * KV bytes one bank holds over `kv_layers` regions at the shape's context and K/V tiles, as
 * pn_tt_kv_banks_alloc() places them: each layer's K groups go to one 4-bank set and its V
 * groups to the other, one KV head a bank, and a mixed cache alternates which set takes the
 * larger region (pn_tt_kv_layer_v_first), so a bank holds ceil(L/2) of one and floor(L/2) of
 * the other. `larger` picks the bank holding ceil(L/2) of the larger region. Needs the KV
 * heads to divide over half the banks (Qwen3.8: 4 heads, one a bank of 8); 0 otherwise.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint64_t pn_qwen35_tt_fit_kv_bank_bytes(const pn_qwen35_tt_fit_shape *shape, uint32_t kv_layers,
                                        uint32_t banks, bool larger);

/**
 * The GDN GATE family's prefill gate-block arrays (pn_tt_gdngate.c gdngate_alloc_blocks): every
 * SSM layer owns 2 sub-chunk blocks for each 32-row band of a pass, each 64 B aligned, ALL in
 * one bank (the gdnc DELTA constant's). 0 when `rows_a_pass` is 0 (FLAT only).
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint64_t pn_qwen35_tt_fit_gate_array_bytes(uint32_t layers, uint32_t heads, uint32_t rows_a_pass);

/**
 * The GDNC GATE blocks: one per DELTA staging slot, `v_heads * 160` fp32 each, pinned to that
 * slot's DELTA constant bank (pn_tt_gdnc.c gdnc_alloc_regions).
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint64_t pn_qwen35_tt_fit_gdnc_gate_bytes(uint32_t v_heads);

/** What differs between two loads of one model on one card, priced per bank. */
typedef struct pn_qwen35_tt_bank_load {
    pn_qwen35_tt_fit_shape shape; /**< ctx, KV tiles, KV heads (the KV term) */
    uint32_t kv_layers;           /**< KV regions: attention layers + a draft head's */
    uint64_t weight_bank[PN_QWEN35_TT_FIT_MAX_BANKS]; /**< card weight bytes in each bank */
    uint64_t even_bytes;          /**< card-wide classes spread evenly (verify state, ...) */
    uint64_t one_bank_bytes;      /**< a class that sits whole in one unknown bank */
} pn_qwen35_tt_bank_load;

/** A load's bank budget: lower bounds on each bank's free bytes. */
typedef struct pn_qwen35_tt_bank_fit {
    int64_t free_bank[PN_QWEN35_TT_FIT_MAX_BANKS]; /**< the one-bank class where it binds */
    int64_t kv_growth;          /**< the KV growth charged to every bank */
    uint32_t binding_bank;      /**< the lowest bank; the lowest index wins a tie */
    int64_t binding_free;       /**< its free bytes */
    bool fits;                  /**< binding_free >= the headroom asked for */
} pn_qwen35_tt_bank_fit;

/**
 * Price `load` per bank against `anchor`, a load whose fullest bank was measured with
 * `anchor_free` bytes free. KV growth is the worst a bank can see: with the same KV tiles and
 * region count a bank keeps its role (larger role minus larger role), otherwise the larger
 * role of `load` minus the smaller of `anchor`.
 *
 * @return 0, or -1 (bad arguments, or a KV shape the bank rule cannot place)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int pn_qwen35_tt_bank_fit_compute(const pn_qwen35_tt_bank_load *anchor, uint64_t anchor_free,
                                  const pn_qwen35_tt_bank_load *load, uint32_t banks,
                                  uint64_t headroom, pn_qwen35_tt_bank_fit *out);

/* ---- The bank rule (the fit policy's second trigger, see the file note) ---- */

/**
 * Main's floor in its fullest bank: the KV regions (the trunk's and a draft head's) at the bank
 * rule's larger role (pn_qwen35_tt_fit_kv_bank_bytes; spread evenly where it cannot place them),
 * every other floor term — the verify's snapshot slots among them — spread evenly, plus, when
 * `checkpoint`, the prefix checkpoint's one snapshot set (reserved at load under either
 * placement, so not in the floor the whole-card rule reads). 0 for no banks.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint64_t pn_qwen35_tt_fit_bank_floor(const pn_qwen35_tt_fit_shape *shape,
                                     const pn_qwen35_tt_fit_floor *floor, bool checkpoint,
                                     uint32_t banks);

/**
 * The fullest bank's free bytes after main's placement, a lower estimate: the least of
 * `free_bytes[b]` (each bank at the decide) less the bank floor less PN_QWEN35_TT_FIT_HEADROOM /
 * `banks`, a bank's share of what the floor omits (the small arms and the allocator's
 * imbalance: the whole-card rule's allowance). Negative when a bank would run out. `bank_out`
 * (optional) gets that bank, the lowest index on a tie.
 *
 * @return the bytes, or INT64_MIN for no banks or more than PN_QWEN35_TT_FIT_MAX_BANKS
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int64_t pn_qwen35_tt_fit_bank_low(const pn_qwen35_tt_fit_shape *shape,
                                  const pn_qwen35_tt_fit_floor *floor, bool checkpoint,
                                  const uint64_t *free_bytes, uint32_t banks, uint32_t *bank_out);

/**
 * The fit policy's trigger, both rules: the whole-card rule (pn_qwen35_tt_fit_policy_wanted
 * over the banks' sum) or the fullest bank under PN_QWEN35_TT_TIER_HEADROOM. False for no
 * banks.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool pn_qwen35_tt_fit_policy_wanted_banks(const pn_qwen35_tt_fit_shape *shape,
                                          const pn_qwen35_tt_fit_floor *floor, bool checkpoint,
                                          const uint64_t *free_bytes, uint32_t banks);

/* ---- The levers ---- */

/*
 * THE FIT POLICY'S OWN PLACEMENT can still leave a bank under PN_QWEN35_TT_TIER_HEADROOM: the
 * 30,720 default (Bfp8, K8V4, MTP k1, checkpoint) kept 3 MiB in its fullest bank, and 131,072
 * needs ~345 MiB a bank more KV. So the policy also pulls LEVERS that trade prefill speed for
 * DRAM, picked at load from the bank estimate below and only as far as the bank rule needs:
 *   - E, the GDNC conv-constant copies 8 -> 1 (~30 MiB a bank a copy; 8 -> 1 measured +3.0%
 *     serve TTFT on Ornith). Copy 0 stays: the decode conv reads it (option D).
 *   - G, the tall strips' bands 64 -> 32 -> 16 (the tall set halves each step; a pass sweeps
 *     the weights twice as often).
 *   - F, the tall gated strips unmirrored (with G at 32 or below).
 *   - V, the MTP draft reads its vocabulary's columns of the trunk's LM head in place, not a
 *     0.333 GiB staged copy (~42 MiB a bank; its LM head then runs on the 6 source panels the
 *     columns live in, not 16: est. ~1% decode).
 * At each tall step the ladder tries every copy count without V, then with it; main's tall
 * set first, so a load that fits with copies alone keeps its prefill passes and its draft
 * copy. An env knob the operator set (PN_TT_GDNC_CONST_REPLICAS, PN_TT_PREFILL_TILE_ROWS,
 * PN_TT_OUTPROJ_MIRRORS, PN_TT_MTP_DRAFT_VIEW) pins that lever.
 *   - K, last: the profile's long-context KV type (PN_TT_KV_LONG, catalog options.kv_long: k4h,
 *     Bfp4 K rotated + Bfp4 V, pn_tt_kv_rotate.h), a quality trade behind its gate (card
 *     20261119-0008). Only when the whole ladder above fails at the load's own type, and then the
 *     WHOLE ladder again at the long type's tiles, so K comes with the fewest other levers.
 */

/** One choice of the levers. */
typedef struct pn_qwen35_tt_fit_levers {
    uint32_t gdnc_replicas; /**< E: conv-constant copies, copy 0 included */
    uint32_t tall_rows;     /**< G: tall-strip bands, 0 = no tall set */
    bool outproj_mirrors;   /**< F: the tall gated strips mirrored */
    bool draft_view;        /**< V: the MTP draft's LM-head columns read in place */
    bool kv_long;           /**< K: the long-context KV type (the pins' tiles) */
} pn_qwen35_tt_fit_levers;

/** Which levers the operator pinned (the env set them): the ladder keeps the shape's value. */
typedef struct pn_qwen35_tt_fit_pins {
    bool gdnc_replicas, tall_rows, outproj_mirrors, draft_view;
    /** K: the long-context type's K and V tile bytes, 0 = no long type (lever K off). */
    uint32_t kv_long_k_tile, kv_long_v_tile;
} pn_qwen35_tt_fit_pins;

/**
 * A bank's share of what the policy-placement estimate omits: the small arms (decode and
 * attention strips, constants, the late decode mirrors) and the allocator's imbalance.
 * Calibrated on the p150's measured fullest banks, checkpoint in: the 30,720 default at 8 conv
 * copies (2.6 MiB) needs 50 MiB, at 6 copies (63.0 MiB) 50, at 3 (153.1) 49, at 4 + V (165.4)
 * 49, the 262,144 tier (243 MiB) 35. At 60 every estimate sits >= 10 MiB under its measurement
 * (and the gate tables, priced at their cap, are ~3 MiB a bank over what they place).
 */
#define PN_QWEN35_TT_FIT_LEVER_ALLOWANCE (60ull * 1024ull * 1024ull)

/**
 * The allowance with the tall gated strips UNMIRRORED at 32 bands or fewer (lever F with G):
 * the imbalance (the card's mean free bank less its fullest) falls from ~33 MiB with the
 * mirrored set (8, 6, 3 and 4 copies at 30,720) to 9-11 (32 and 16 single bands at 102,400).
 * Calibrated on those loads' fullest banks with the exact gate tables armed (the state the bank
 * rule checks; the tables, priced at their cap, landed ~1 MiB in the fullest bank): 102,400 at
 * 16 single bands (183.8 MiB) needs 13, at 32 (151.0) 10. At 26 every estimate sits >= 10 MiB
 * under its measurement.
 */
#define PN_QWEN35_TT_FIT_LEVER_ALLOWANCE_SINGLE (26ull * 1024ull * 1024ull)

/**
 * The fit policy's placement in its fullest bank: main's floor (`floor`, computed for the
 * shape's levers) without the staged second `attn_q` (option A), the decode conv weight
 * (option D) or the tall `attn_qg` (option C); the MTP verify's COMPACT slots (rows - 1, the
 * set the policy reserves at load); the prefix checkpoint's set when `checkpoint`; the gate
 * tables at their cap. KV at its larger bank role, the rest even. 0 for no banks.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint64_t pn_qwen35_tt_fit_policy_bank_floor(const pn_qwen35_tt_fit_shape *shape,
                                            const pn_qwen35_tt_fit_floor *floor, bool checkpoint,
                                            uint32_t banks);

/**
 * The fullest bank's free bytes under the fit policy, a lower estimate: the least of
 * `free_bytes[b]` less the policy bank floor less PN_QWEN35_TT_FIT_LEVER_ALLOWANCE (its _SINGLE
 * form for an unmirrored gated set at 32 bands or fewer). The floor is
 * computed here from `shape` and the artifact's `weight_bytes` / `joint_q_bytes`.
 *
 * @return the bytes, or INT64_MIN for no banks or more than PN_QWEN35_TT_FIT_MAX_BANKS
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int64_t pn_qwen35_tt_fit_policy_bank_low(const pn_qwen35_tt_fit_shape *shape,
                                         uint64_t weight_bytes, uint64_t joint_q_bytes,
                                         bool checkpoint, const uint64_t *free_bytes,
                                         uint32_t banks);

/**
 * Pick the levers: the first ladder step (see the note above) whose policy estimate keeps
 * PN_QWEN35_TT_TIER_HEADROOM in every bank. `shape` carries the load as configured (main's
 * lever values, or the pinned ones). Nothing fits -> the last step the pins allow, and false.
 * `low_out` (optional) gets that step's estimate.
 *
 * @return true when the picked levers meet the bank rule
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool pn_qwen35_tt_fit_levers_pick(const pn_qwen35_tt_fit_shape *shape, uint64_t weight_bytes,
                                  uint64_t joint_q_bytes, bool checkpoint,
                                  const uint64_t *free_bytes, uint32_t banks,
                                  const pn_qwen35_tt_fit_pins *pins,
                                  pn_qwen35_tt_fit_levers *out, int64_t *low_out);

#endif /* QWEN35MOE_TT_FIT_PLAN_H */
