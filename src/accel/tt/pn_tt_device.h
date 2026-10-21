/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_device.h
 * @brief Pantheon's handle on a Tenstorrent Blackhole card: the weight residency and the
 *        one primitive the model path needs — a row vector times a resident matrix, with
 *        the multiply executing on the card.
 *
 *        This is the model-agnostic half of the TT backend integration. It knows about
 *        FTTW artifacts, DRAM residency, tile layout and matmul dispatch; it knows nothing
 *        about llama, attention or tokens. src/models/llama_forward_tt.c is the caller.
 *
 * ---------------------------------------------------------------------------
 * WHAT RUNS ON THE CARD, AND WHAT IS LEFT ON THE HOST
 * ---------------------------------------------------------------------------
 * Every matmul weight of a converted model lives in the card's DRAM and every projection
 * is a Tensix matmul. The RMS norms, the SwiGLU gate, the residual adds and the LM head's
 * top-N run there too, each on its OWN core, because a resident binary is linked at a fixed
 * slab and one program can be resident on a core at a time (foundry/tt/ft_tt_op_vector.h).
 * They hand activations to each other through card DRAM, which the host never touches —
 * that is the whole mechanism, and it is not a faster transfer but a transfer that does not
 * happen.
 *
 * THREE THINGS ARE STILL HOST-SIDE, and two of them are FORCED rather than chosen:
 *
 *  - **RoPE — the kernel's refusal is FIXED, and fixing it removed no transfer.**
 *    foundry/tt/ft_tt_op_rope.h refuses INTERLEAVED rotary pairing (2i, 2i+1), which is the
 *    style this model family takes, and said the fix was to permute the wq/wk output rows
 *    in the CONVERTER. **The converter does that now** — `--rope-pairing split-half`, with
 *    the choice recorded in the artifact header and read back through
 *    pn_tt_device_rope_is_split_half(). So a split-half artifact's q and k ARE in the form
 *    the kernel accepts. What it does not do on its own is remove the readback: the KV
 *    append and the GQA softmax are still host-side, and they need q, k and v on the host
 *    regardless of pairing. **RoPE, KV-append and attention have to move together or none
 *    of them moves anything.**
 *  - **Embedding.** foundry/tt/ft_tt_op_memory.h requires a ROW-MAJOR UNTILIZED table,
 *    because a row of a tilized table is a set of 32-byte face rows and a 32-byte DRAM read
 *    is not legal. The artifact stores `token_embd.weight` TILED (and block-float), so the
 *    on-card gather cannot read it. This one is CHEAP: it costs one 64-byte-a-tile upload of
 *    the first hidden state per token, in the direction the link is 60x faster.
 *  - **Attention.** The flash-decode op exists and is green; it stays on the host here
 *    because RoPE already forces q/k/v across the link, so moving it would remove no
 *    transfer — only the KV cache would move, and that is a separate piece of work.
 *
 * Which of the movable ops actually move is `PN_TT_ONCARD`, a comma list of
 * `norm,swiglu,residual,topk` (or `all`, the default, or `none`). It exists so the A/B is
 * across ONE binary running the same kernels on the same weights.
 *
 * ---------------------------------------------------------------------------
 * THE DISPATCH PATH — the RESIDENT work queue, with one-shot kept as the fallback
 * ---------------------------------------------------------------------------
 * A projection is enqueued, not launched. Five processors sit resident on the core reading
 * 32-byte descriptors out of its own L1 (foundry/tt/ft_tt_queue.h), so the host's per-op
 * work is a parameter block, a descriptor and a 4-byte head bump with no round trip. The
 * one-shot path it replaces — `ft_tt_op_matmul_run()` — re-uploaded five kernel images per
 * op at 36.2 µs/kB, and that upload was the whole of its fixed cost.
 *
 * **ONE resident program build serves this model, not two, and that is a consequence of a
 * PRECISION decision rather than a dispatch one.** The production ledger used to put
 * `ffn_gate`/`ffn_up` at Bfp4_b, so an artifact carried two matmul tile sizes and a core
 * that can hold ONE resident binary could not serve it. Those two tensors measured +9.50%
 * perplexity and per-op PCC 0.991-0.993 against 0.9999 everywhere else, at no throughput
 * (BFP4 is not faster than BFP8), so they moved to Bfp8_b — and the artifact now holds a
 * single tile size. `rtriad_bfp8` is the whole requirement:
 *   RELOCATE_KERNEL_TEXT=1 READER_CTARGS=0,1,2048,1088,1 COMPUTE_CTARGS=0,1,16,1
 *   WRITER_CTARGS=16,2048,1 CB_TABLE=kernels/cb_tables/matmul_bfp8_b.cbtable
 *
 * The one-shot arm is kept and is reachable with `PN_TT_DISPATCH=oneshot`: it is the same
 * binary running the same kernels on the same weights, which is what makes an A/B across
 * that knob a measurement of DISPATCH rather than of two programs.
 *
 * **IT IS NO LONGER A FALLBACK.** If `rtriad_bfp8` is absent, or was built against other
 * firmware, or the artifact's matmul tile size is not the one it serves, the device REFUSES
 * to open. Falling back cost a round: the program was missing from the host's program root
 * and every timing taken in that window was silently one-shot. MEASURED, with only this
 * knob moved: **0.343 -> 0.372 s a token, 1.085x**, all of it in the dispatch phase
 * (0.156 -> 0.185). Small enough to publish by accident, large enough to be wrong. A slower
 * path that is chosen FOR you is a wrong measurement waiting to be published; one that is
 * asked for is a decision.
 * The same rule now covers the on-card families: a `PN_TT_ONCARD` name whose program
 * directory is missing is a refusal, not an op that quietly stays on the host.
 * The mid-run re-arm after a REFUSED DESCRIPTOR is the one place one-shot is still entered
 * without being asked, because there the alternative is a lost token — and it is counted in
 * `pn_tt_device_dispatch_counts()`.
 *
 * **A refused descriptor is the hazard, and it is handled BEFORE it is sent.** The reader
 * refuses a shape it cannot execute by leaving its loop; `tail` never advances, so the
 * other four processors then wait on work that will never arrive — recoverable only by
 * ASKING them to stop, never by a reset. So the host proves every descriptor executable
 * first, and if one is refused anyway it retires all five, arms a fresh residency and
 * retries once before falling back. Measured: a deliberately malformed descriptor (Nt=0)
 * is refused, the residency is re-armed on the same core, the remaining ops run resident,
 * the token is byte-identical, and the core still probes PASS.
 *
 * ---------------------------------------------------------------------------
 * TWO THINGS CROSS THE LINK, AND BOTH ARE SHAPED FOR IT
 * ---------------------------------------------------------------------------
 * Card->host is ~4.7 MB/s with no device->host DMA, so a token is decided by how many
 * bytes go over PCIe and through which accessor. Two rules hold this path:
 *
 *  1. **Bulk data uses the WC/BLOCK16 window, never the UC/WORD32 register one.** The
 *     register accessor exists for 4-byte registers and go messages, where the width is a
 *     correctness requirement; data inherited it and that alone was 71.1% of a 28 s token.
 *     ft_tt_runtime carries both windows and pn_tt_device_window_traffic() is the evidence.
 *  2. **Only logical row 0 of a tile crosses**, in either direction — a matvec is Mt=1 and
 *     the other 31 rows are the tile quantum's zero padding (ft_tt_tile_row0.h).
 *
 * Together those took the token from 28.0 s to 0.57 s, and the on-card residencies above
 * remove most of what was left: with norm, SwiGLU and the residual adds on the card, the
 * only activations that still cross are q/k/v (the RoPE gap) and the logit row (which the
 * on-card top-N removes in turn).
 *
 * @status Phase 7. Matmul, RMS norm, SwiGLU, residual add and top-N on the card.
 * @issues Moving an op on-card MOVES ITS NUMERICS — the card computes in bf16 with an fp32
 *         destination accumulator where the host used fp32 throughout, and each on-card
 *         stage rounds its output to bf16. That is a real change and is reported, not tuned
 *         away.
 * @todo
 *   - [ ] Permute wq/wk in the converter so RoPE can move too, and q/k/v stop crossing.
 */

#ifndef PN_TT_DEVICE_H
#define PN_TT_DEVICE_H

#ifdef USE_TT

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "pn_tt_column_runs.h"
#include "pn_tt_descriptor_census.h"

/** How many Tensix one projection may be split across. 16 is the cooperative driver's own
 *  bound and not a rectangle — a SPLIT group multicasts nothing.
 *
 *  This was 8, the DRAM bank count, on the argument that a ninth member is a second stream
 *  on a bank rather than a new one. That argument is about BANDWIDTH and it still holds;
 *  what changed is that it was also acting as the ceiling on the experiment. Past 8 the
 *  bank cursor gives members `c` and `c+8` the same bank at DIFFERENT addresses — a
 *  per-bucket cursor, never a constant base, which is the shape that once had two members
 *  silently overwriting each other with every status word clean
 *  (foundry/tt/DEV_NOTES.md §3).
 *
 *  RAISED TO 32 ON 2026-08-19, for the same reason it was raised from 8: the bound was
 *  acting as the ceiling on the experiment. With the hot A strips mirrored across banks the
 *  member count is no longer capped by one bank's ~59 GB/s, and the question "where does
 *  this actually scale to" cannot be asked at a limit of 16. It is a limit on ARRAYS here
 *  and on nothing else — the placement, the shard split and the door chooser are all
 *  written in the member count. What DOES bind past 16 is host ISSUE at 8.9 µs a member per
 *  projection: at 32 members that alone is 184 x 32 x 8.9 µs = 0.052 s/token, which is why
 *  a wide arm is an experiment and not a default. */
#define PN_TT_MATMUL_MAX_CORES 64u

/** Projections that may be issued before one is awaited. The ring holds 8 descriptors per
 *  member and `ft_tt_queue_stage()` does not check occupancy, so this is the bound that
 *  keeps a chain from writing over descriptors the card has not read. The longest chain the
 *  forward actually builds is 3 (q/k/v). */
#define PN_TT_MATVEC_MAX_CHAIN 4u

/** The deepest chain `PN_TT_MATVEC_CHAIN` may ask for. The ring and the parameter-slot table
 *  both hold 8, so 8 is the structural ceiling and not a preference; the DEFAULT stays 4, so
 *  a run that sets nothing is the run every measurement to date was taken on. */
#define PN_TT_MATVEC_CHAIN_CEILING 8u

/**
 * Descriptors this run puts in flight before the host waits — the DISPATCH-phase instrument.
 *
 * `dispatch` is 0.305 s/token over 249 matmuls and the project has read 1.23 ms a matmul as a
 * PCIe round trip. That reading is testable and this is the test: a host round trip is paid
 * once per WAIT, so it divides by the chain depth, and the card's own work does not. Sweeping
 * `PN_TT_MATVEC_CHAIN` over 1, 2, 4, 8 and fitting `C + F/N` separates the two — and what
 * Phase 3 can remove is F, not C.
 *
 * Read from the environment on every call, clamped to 1..PN_TT_MATVEC_CHAIN_CEILING, and 4
 * when unset.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
uint32_t pn_tt_matvec_chain_limit(void);

/** One matmul weight resident in card DRAM, in the orientation the Tensix matmul wants:
 *  K x N = [in_features x out_features], tile grid stored N-major. Both facts come from
 *  the artifact's `layout` field and are checked, never assumed. */
typedef struct pn_tt_weight {
    uint32_t bank;       /**< logical DRAM bank of shard 0 */
    uint32_t address;    /**< byte offset within that bank */
    uint32_t tile_rows;  /**< Kt = in_features / 32 */
    uint32_t tile_cols;  /**< Nt = out_features / 32 of the WHOLE weight */
    uint32_t tile_bytes; /**< 1088 Bfp8_b, 576 Bfp4_b, 2048 bf16 */
    uint32_t in_features;
    uint32_t out_features;
    /**
     * Where the group reads it from, when it was sharded over banks at load.
     *
     * `shard_count` is 1 for a weight that lives whole in one bank — the pre-group layout —
     * in which case only element 0 is meaningful and it repeats `bank`/`address`. Anything
     * else means member c reads `shard_tile_cols[c]` columns from column 0 of its OWN panel:
     * the full-grid column offset is folded into `shard_address`, which is why the reader
     * kernel cannot tell the difference (ft_tt_dram_shard.h).
     *
     * THE WIDTHS ARE PER MEMBER, not one number, because `Nt % members != 0` used to cost a
     * tensor its split entirely. The remainder now goes to the first members one column
     * each, so the widths differ by at most one and `shard_column_first` is the RUNNING SUM
     * rather than `c * width`. MEASURED 2026-08-16: under the old rule N=5 left 3,864 of
     * 3,887 projections on one core and N=7 left ALL of them, which read as a scaling
     * ceiling and was a divisibility one.
     */
    uint32_t shard_count;
    uint32_t shard_tile_cols[PN_TT_MATMUL_MAX_CORES];    /**< this member's own Nt */
    uint32_t shard_column_first[PN_TT_MATMUL_MAX_CORES]; /**< its first column of the whole */
    uint32_t shard_address[PN_TT_MATMUL_MAX_CORES];
    uint16_t shard_x[PN_TT_MATMUL_MAX_CORES]; /**< the DOOR, translated coordinates */
    uint16_t shard_y[PN_TT_MATMUL_MAX_CORES];
    /** The LOGICAL BANK behind each door. The door is a NoC port and three of them lead to
     *  one bank, so the coordinate cannot be reduced back to a bank — it has to be kept.
     *  Only the per-bank read accounting reads this; nothing dispatches from it. */
    uint32_t shard_bank[PN_TT_MATMUL_MAX_CORES];
} pn_tt_weight;

/** Opaque device handle. One per resident model. */
typedef struct pn_tt_device pn_tt_device;

/**
 * Open the card, claim a core, boot its firmware, and load the matmul programs.
 *
 * @param device_index  0 for the first card
 * @param out_device    receives an owned handle
 * @return              0 on success, -1 with the reason on stderr
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_device_open(int device_index, pn_tt_device **out_device);

/**
 * Retire the core by ASKING, release the window, close the card.
 *
 * It never parks: a parked core needs its firmware re-booted, and parking a resident
 * program has broken a core's next boot twice in this project. Safe on NULL.
 */
void pn_tt_device_close(pn_tt_device *device);

/**
 * KEEPALIVE (PM 5): NOP every resident family idle >= `min_idle_seconds` (and not yet past the
 * idle-retire limit), so neither the card's idle exit nor the host's retire fires and the next
 * use pays no re-arm. Does nothing while a stream, replay or recording is open or gdn rows are
 * deferred; re-learns the dispatch model after sending. `PN_TT_KEEPALIVE=0` off.
 * @return NOPs sent (0 when nothing was due), or -1
 */
int pn_tt_device_keepalive(pn_tt_device *device, double min_idle_seconds);

/**
 * Upload every matmul weight of an FTTW artifact into card DRAM, round-robin across the
 * live DRAM banks.
 *
 * Only tensors the artifact marks MATMUL_NMAJOR are uploaded. Norms are 1-D fp32 and the
 * engine already holds them; the embedding table is tiled block-float and no on-card op
 * can read it (see the file note), so neither is moved.
 *
 * @param path  the .fttw file
 * @return      0, or -1 with the reason on stderr
 */
int pn_tt_device_load_artifact(pn_tt_device *device, const char *path);

/** Look up a resident weight by its artifact name, e.g. "blk.3.ffn_down.weight".
 *  @return 0 and fills `out_weight`, or -1 if the name is not resident. */
int pn_tt_device_find_weight(const pn_tt_device *device, const char *name,
                             pn_tt_weight *out_weight);

/**
 * Stage a SECOND, column-permuted copy of an already-loaded weight into card DRAM.
 *
 * This is how a JOINT projection is split: Ornith's `blk.N.attn_q.weight` emits Q and the
 * attention output gate interleaved per head, and every op downstream — the QK-RMSNorm's row
 * blocks, flash decode's contiguous group run — needs one of them compact. Permuting the
 * WEIGHT costs one upload at load and NOTHING per token; the alternatives are a per-token
 * gather op or a stride word in two kernels. `pn_tt_column_runs.h` carries that comparison.
 *
 * The source stays resident: it is one entry in a bank rotation every other tensor was placed
 * against. The cost is therefore a second copy — 71.3 MiB per attention layer for Ornith.
 *
 * @param source_name  the artifact tensor to read, which must be MATMUL_NMAJOR
 * @param runs         destination-ordered runs of SOURCE column tiles
 * @param out_weight   fully written, including the derived `tile_cols` / `out_features`
 * @return             0, or -1 with the reason on stderr
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
/**
 * STEP 4 S2 — a SECOND artifact (the MTP draft head): open it, place one of its matmul weights
 * the way the device's own are placed, stage a column subset of one (as
 * pn_tt_device_stage_weight_columns), or read one of its 1-D fp32 tensors (`n` floats, NULL if
 * the name or the size is wrong). The mapping lives until the device closes.
 */
int pn_tt_device_extra_open(pn_tt_device *device, const char *path);
int pn_tt_device_place_extra_weight(pn_tt_device *device, const char *name,
                                    pn_tt_weight *out_weight);
int pn_tt_device_stage_extra_columns(pn_tt_device *device, const char *source_name,
                                     const pn_tt_column_run *runs, uint32_t run_count,
                                     pn_tt_weight *out_weight);
const float *pn_tt_device_extra_vector(const pn_tt_device *device, const char *name, uint32_t n);

int pn_tt_device_stage_weight_columns(pn_tt_device *device, const char *source_name,
                                      const pn_tt_column_run *runs, uint32_t run_count,
                                      pn_tt_weight *out_weight);

/** How many matmul weights are resident, and how many bytes they occupy. */
uint32_t pn_tt_device_weight_count(const pn_tt_device *device);
uint64_t pn_tt_device_weight_bytes(const pn_tt_device *device);

/**
 * Card DRAM handed out and still free, summed over the live banks.
 *
 * The weight figure alone does not answer a capacity question: the KV regions, the GDN
 * carried state, the activation mirrors, the RoPE table and the constants all come out of
 * the same arenas and none of them are weights. Every "how many sessions fit" claim so far
 * has been ARITHMETIC over a bank size and a file size; this is the measurement it should
 * have been checked against.
 *
 * @param used_out   optional: bytes allocated and not yet freed
 * @param total_out  optional: bytes the live banks could ever hand out
 *
 * Thread-safe: no    Reentrant: yes    Signal-safe: no
 */
void pn_tt_device_dram_usage(const pn_tt_device *device, uint64_t *used_out,
                             uint64_t *total_out);

/** One stderr line naming `tag`, used and total in GiB. Cheap enough to call on every arm,
 *  and the only place a capacity claim can be READ OFF a run rather than derived. */
void pn_tt_device_report_dram(const pn_tt_device *device, const char *tag);

/**
 * Non-zero when the loaded artifact stores `attn_q`/`attn_k` with SPLIT-HALF rotary
 * pairing — the converter's `--rope-pairing split-half`, which runs llama.cpp's HF->GGUF
 * row permutation backwards.
 *
 * **The forward MUST read this rather than infer the pairing from the architecture.** The
 * pairing is a property of the FILE, not of the model: the same GGUF converts either way,
 * the two artifacts are the same size and the same shape and hold the same tensors, and
 * rotating split-half rows with the interleaved form (or the reverse) is wrong with no
 * symptom except the answer.
 *
 * @return 1 when the flag is set, 0 when it is not or no artifact is loaded
 *
 * Thread-safe: yes (read-only)    Reentrant: yes    Signal-safe: yes
 */
int pn_tt_device_rope_is_split_half(const pn_tt_device *device);

/**
 * Whether the artifact's rotary PAIR INDEX is reflected on top of split-half.
 *
 * `--rope-pairing split-half-reversed` is an EXPERIMENT the converter offers so the
 * block-float exponent grouping over `attn_q`/`attn_k` can be drawn a THIRD way while the
 * attention score stays exact. The obligation it puts on a consumer is one sentence: **slot
 * `p` holds the model's rotary pair `half - 1 - p`, so apply frequency `half - 1 - p`
 * there.**
 *
 * Deliberately a SECOND predicate rather than a third value of the first one. The storage
 * really is split-half-shaped, so every caller that cares only about the TILE argument keeps
 * the answer it already had, and only the two callers that care about FREQUENCY ORDER — the
 * host rotary, and the on-card arm which must refuse because its table is built in slot
 * order — grow a branch.
 *
 * @return 1 when the flag is set, 0 when it is not or no artifact is loaded
 *
 * Thread-safe: yes (read-only)    Reentrant: yes    Signal-safe: yes
 */
int pn_tt_device_rope_pairs_reversed(const pn_tt_device *device);

/**
 * y[0..N) = x[0..K) * W[K x N], with the multiply on the card.
 *
 * The row is bf16-rounded on the way in and on the way out, because that is the card's
 * activation format. A caller comparing against an fp32 host reference should expect that
 * difference and not a bit-exact answer.
 *
 * @param x  `weight->in_features` floats
 * @param y  `weight->out_features` floats, written
 * @return   0, or -1 with the reason on stderr
 *
 * Thread-safe: no (one core, one staging buffer)   Reentrant: no   Signal-safe: no
 */
int pn_tt_matvec(pn_tt_device *device, const pn_tt_weight *weight, const float *x, float *y);

/* ================================================================
 * The ON-CARD forward: DRAM activation strips and the ops over them
 * ================================================================ */

/**
 * A vector living in card DRAM, in the shape the Tensix ops read: a strip of consecutive
 * 32x32 bf16 tiles whose LOGICAL ROW 0 carries the value and whose other 31 rows are the
 * tile quantum's zero padding.
 *
 * Those 31 rows must be zero and stay zero, because every op feeds all 32 to the FPU.
 * pn_tt_activation_alloc() zeroes the whole strip once, and every on-card op preserves it
 * for free: a matmul of a zero row is zero, silu(0)*0 is zero, 0+0 is zero, and an RMS norm
 * of a zero row is 0 * gamma * rsqrt(eps).
 */
/** Mirrors one strip may have, primary included. The card has eight DRAM banks and a ninth
 *  copy would be a second copy in a bank that already has one. */
#define PN_TT_ACTIVATION_MAX_REPLICAS 8u

typedef struct pn_tt_activation {
    uint32_t bank;       /**< logical DRAM bank */
    uint32_t address;    /**< byte offset within that bank */
    uint32_t tile_count; /**< features / 32 */

    /* ------------------------------------------------------------------
     * MIRRORS, so N matmul members do not all read ONE bank
     * ------------------------------------------------------------------
     * The card's 510.8 GB/s is EIGHT banks of ~59. A strip is allocated in one of them, and
     * every member of a column-split projection reads the WHOLE strip — so a 16-member
     * matmul puts sixteen readers on one bank. Measured on Ornith during decode: bank 2 at
     * 43.1 GB/s, 73% of its own ceiling, with the other seven under 5 and the card at 16%
     * of peak. That is why matmul card work is flat from four members to sixteen and why
     * the shipped setting is four cores on a 140-Tensix part.
     *
     * `replica[0]` IS `{bank, address}` and is always valid, so every consumer that does
     * not know about mirrors — the host uploads, the GDN op, attention, the elementwise
     * reader, `pn_tt_topk` — is byte-for-byte unchanged. Only the matmul's A side indexes
     * past it, with member `m` reading `replica[m % replica_count]`.
     *
     * `replica_count` 1 is the unreplicated strip and every branch on it is then a no-op.
     * WHAT KEEPS THE MIRRORS CURRENT IS THE PRODUCER, not this struct: the shared resident
     * vector writer fans a tile out to the destinations its replica table names, and a
     * strip whose producer is not that writer must be replicated by whoever writes it or
     * not replicated at all. A mirror nobody refreshes is stale, and a matmul member
     * reading it returns a plausible wrong answer with no error anywhere.
     */
    uint32_t replica_count;
    uint32_t replica_bank[PN_TT_ACTIVATION_MAX_REPLICAS];
    uint32_t replica_address[PN_TT_ACTIVATION_MAX_REPLICAS];

    /* ------------------------------------------------------------------
     * THE SECOND TILE-ROW: `Mt > 1`, and why it is three fields rather than one
     * ------------------------------------------------------------------
     * A strip is `tile_rows` bands of `tile_count` tiles, TILE-ROW-MAJOR — band `m` starts at
     * `m * tile_count * PN_TT_TILE_BF16`. That is the layout both matmul kernels already
     * assume: the reader's A offset is `(mt + row) * tile_inner + kt`
     * (resident_matmul_reader.c) and the writer's destination is `(mt + row) * pitch + nt +
     * column` with the pitch in descriptor word a2 (resident_matmul_writer.c). Nothing on the
     * device had to change for this; only the host ever wrote `Mt = 1`.
     *
     * `tile_rows` is the ALLOCATION. `active_rows` is how many bands the next op covers, and
     * it is a separate number because most ops are NOT tile-row-general: an RMS norm is
     * `row_blocks` (it is), an elementwise op is a wider tile count (it is), and a per-head
     * norm, a RoPE, a KV append, an attention chunk and both GDN mixers have already SPENT
     * their row axis on heads or on positions and can only be run ONCE PER BAND. Those are
     * driven by `row_offset` instead: `pn_tt_activation_seek()` slides `address` and every
     * mirror to band `m` and leaves `active_rows` at one, so an op written for a 32-row strip
     * runs on band `m` UNMODIFIED and cannot tell.
     *
     * **THE DEFAULT IS `tile_rows = active_rows = 1, row_offset = 0`, which is the strip this
     * struct described before these fields existed, byte for byte.** Decode never leaves it,
     * so a decode `cmp` is the evidence that the whole mechanism is inert where it does not
     * pay.
     */
    uint32_t tile_rows;   /**< tile-rows allocated; 1 is the pre-Mt strip */
    uint32_t active_rows; /**< tile-rows the next op spans; 1..tile_rows */
    uint32_t row_offset;  /**< which tile-row `address` currently points at */
} pn_tt_activation;

/** The constants one RMS norm needs, resident in card DRAM: the reduce scaler tile, the eps
 *  tile, then gamma expanded to full tiles. Built once per norm weight, at load. */
typedef struct pn_tt_norm_constants {
    uint32_t bank;
    uint32_t address;
    uint32_t tile_cols; /**< width / 32 */
} pn_tt_norm_constants;

/** Which host-side op a run has moved onto the card. `PN_TT_ONCARD` selects them. */
typedef enum pn_tt_oncard_op {
    PN_TT_ONCARD_NORM = 0,  /**< the RMS norms, on the reduce residency */
    PN_TT_ONCARD_SWIGLU,    /**< silu(gate) * up, on the elementwise residency */
    PN_TT_ONCARD_RESIDUAL,  /**< the two residual adds, on the same residency */
    PN_TT_ONCARD_TOPK,      /**< exact top-N over the logit row */
    /**
     * RoPE + KV append + flash-decode attention, on THREE more resident cores, with the KV
     * cache moved into card DRAM.
     *
     * **OPT-IN ONLY — `PN_TT_ONCARD=all` does NOT include it, and that asymmetry is
     * deliberate.** The other four move an op whose arithmetic the card reproduces in a
     * different precision; this one moves WHERE THE KV CACHE LIVES and changes the order the
     * attention sum accumulates in. It is a different forward, not a faster one, so it has to
     * be asked for by name: `PN_TT_ONCARD=norm,swiglu,residual,topk,attention`.
     */
    PN_TT_ONCARD_ATTENTION,
    /**
     * The whole Gated-Delta-Net block — `conv1d + SiLU` and the delta rule — on ONE more
     * resident core, with the conv window and the `[S,S]` recurrent state living in card DRAM
     * across tokens.
     *
     * **OPT-IN ONLY, and for a different reason from attention's.** Attention is opt-in
     * because it is a different forward; this one is opt-in because it has NEVER EXECUTED ON
     * SILICON and because, until the two per-head norms move too, it is measurably SLOWER
     * than the host recurrence — the L2 norm in the middle of the block forces `conv_out`
     * and `y` back across the link. `pn_tt_gdn.h` carries that arithmetic. Ask for it by
     * name: `PN_TT_ONCARD=norm,swiglu,residual,topk,gdn`.
     */
    PN_TT_ONCARD_GDN,
    PN_TT_ONCARD_COUNT
} pn_tt_oncard_op;

/**
 * Bring up the on-card op residencies this device can serve, one core each.
 *
 * Deliberately never fatal, and each family is independent: a card with no elementwise
 * program still runs the norms on the card and the SwiGLU on the host. What actually armed
 * is `pn_tt_device_oncard()`, and the caller must ask rather than assume.
 *
 * The shapes are the CALLER's because they are the model's, not the artifact's: a reduce
 * residency's circular buffers are placed once for one width, and an elementwise residency
 * runs ONE block depth for every descriptor it will ever see, so the depth has to divide
 * every tile count in `eltwise_tile_counts` and twice the depth has to divide them too (a
 * descriptor that leaves the CB write pointer off the block grid walks the NEXT one up L1 —
 * foundry/tt/ft_tt_op_sample.h states the invariant and the core it cost).
 *
 * @param norm_tile_cols        width / 32 of every vector the norms reduce over
 * @param eltwise_tile_counts   every tile count an elementwise descriptor will carry
 * @param eltwise_count         how many
 * @param vocab_tiles           the logit row's tile count, or 0 for no sampling residency
 * @return                      0 (even when nothing armed), or -1 on a bad argument
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_device_arm_oncard(pn_tt_device *device, uint32_t norm_tile_cols,
                            const uint32_t *eltwise_tile_counts, uint32_t eltwise_count,
                            uint32_t vocab_tiles);

/** Did `op` actually move onto the card for this run? The caller runs the host version
 *  when this is false, so a missing program directory is a slower forward and never a
 *  wrong one. */
bool pn_tt_device_oncard(const pn_tt_device *device, pn_tt_oncard_op op);

/** Was `op` ASKED for, whether or not it armed? The difference matters exactly once: a
 *  family that was asked for and did not arm is a REFUSAL, and a family that was never asked
 *  for is a configuration. */
bool pn_tt_device_oncard_requested(const pn_tt_device *device, pn_tt_oncard_op op);

/**
 * Declare what `PN_TT_ONCARD` UNSET — or `all` — selects for `op` on THIS model, and re-read
 * the selection so `pn_tt_device_oncard_requested()` is answerable immediately.
 *
 * **A DEFAULT IS A PROPERTY OF A MODEL FAMILY, NOT OF THE CARD.** On-card attention is a
 * different forward on each family and the evidence for it is measured per family: on the
 * qwen35/bonsai hybrid it passes a calibrated chat-parity gate and is faster at every length,
 * and on MiniCPM5-1B through `llama_forward_tt.c` it is recorded as FAILING that same gate.
 * A single constant in `pn_tt_read_oncard_selection()` would have shipped the second as a
 * side effect of shipping the first — and silently, because the llama backend arms attention
 * LAZILY, after `pn_tt_device_arm_oncard()` has already run.
 *
 * An explicit `PN_TT_ONCARD` list still wins: this changes the DEFAULT and never overrides an
 * operator who named something. Call it before `pn_tt_device_arm_oncard()`.
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
void pn_tt_device_oncard_default(pn_tt_device *device, pn_tt_oncard_op op, bool on);

/**
 * Was `op` named in `PN_TT_ONCARD` by the OPERATOR, as opposed to being on by default?
 *
 * **A FAILED ARM MEANS TWO DIFFERENT THINGS AND `..._requested()` CANNOT TELL THEM APART.**
 * A NAMED family that will not arm must refuse loudly — the operator asked for that forward,
 * and quietly running the host block instead would be a different model reported as the same
 * one. A family that is merely the DEFAULT must degrade loudly to the host block, because a
 * refusal there propagates to `goto fail` in pn_qwen35moe_tt_upload and drops the ENTIRE
 * model to the CPU forward: a card that cannot arm one op would serve Ornith at ~0.05 tok/s
 * rather than at its host-attention speed.
 *
 * The distinction is inert while attention and GDN are opt-in, and becomes load-bearing the
 * moment either is defaulted on.
 */
bool pn_tt_device_oncard_named(const pn_tt_device *device, pn_tt_oncard_op op);

/**
 * Allocate a `tile_count`-tile bf16 strip in card DRAM and ZERO it.
 *
 * The zeroing is the precondition every later row-0 write depends on and cannot check.
 *
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_activation_alloc(pn_tt_device *device, uint32_t tile_count,
                           pn_tt_activation *out_activation);

/**
 * The same, MIRRORED into `replicas` DISTINCT DRAM banks.
 *
 * `replica[0]` is exactly what `pn_tt_activation_alloc()` would have returned — same bank
 * cursor, same zeroing — so a strip allocated this way is a superset of one allocated the
 * ordinary way and no consumer has to know.
 *
 * WHAT THIS DOES NOT DO IS KEEP THEM IN STEP. Allocating the mirrors is half the mechanism;
 * the other half is the producer writing all of them, which for `normed` and `ffn_swiglu` is
 * the resident vector writer's replica table (`pn_tt_device_publish_replica_table`). A strip
 * mirrored here and produced by something that does not fan out is a strip with seven stale
 * copies, and the matmul member that reads one gets a plausible wrong answer.
 *
 * `replicas` 0 or 1 is exactly `pn_tt_activation_alloc()`, so a caller may pass the knob
 * through without branching. More replicas than banks is a refusal, not a wrap: two mirrors
 * in one bank is the failure the whole thing exists to avoid, wearing the name of the fix.
 *
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_activation_alloc_replicated(pn_tt_device *device, uint32_t tile_count,
                                      uint32_t replicas, pn_tt_activation *out_activation);
/** As `pn_tt_activation_alloc` / `_replicated`, at an explicit height — for a strip SET used
 *  only on its own (PHASE G option 3's prefill-only tall strips). */
int pn_tt_activation_alloc_rows(pn_tt_device *device, uint32_t tile_count, uint32_t tile_rows,
                                pn_tt_activation *out_activation);
int pn_tt_activation_alloc_replicated_rows(pn_tt_device *device, uint32_t tile_count,
                                           uint32_t replicas, uint32_t tile_rows,
                                           pn_tt_activation *out_activation);
/** PHASE G option 3: record the prefill-only set's height (0 = none). */
void pn_tt_device_set_prefill_tile_rows(pn_tt_device *device, uint32_t rows);

/** Write `count` floats into logical row 0 of a strip, bf16-rounded. Rows 1-31 are left
 *  alone, which is correct exactly because the strip was allocated zeroed. */
int pn_tt_activation_upload(pn_tt_device *device, const pn_tt_activation *activation,
                            const float *values, uint32_t count);

/** Read logical row 0 of a strip back as `count` floats. This is the transfer the on-card
 *  forward exists to avoid, so every call site is a place where an op did NOT move. */
int pn_tt_activation_download(pn_tt_device *device, const pn_tt_activation *activation,
                              float *out_values, uint32_t count);

/**
 * Read only SOME of logical row 0: every 16-float chunk i (row 0 columns 16*i .. 16*i+15,
 * i.e. one of a tile's two row-0 pieces) with need[i] != 0 is written into out_values at
 * its columns; the rest of out_values is left alone.  For a caller that needs a handful
 * of entries of a wide row (the grammar mask's allowed tokens of a logits row): the cost
 * is one 64 B read per needed chunk instead of two per tile of the whole row.
 *
 * @param need   count/16 flags
 * @param count  a multiple of 32, no wider than the strip
 * @return       0, or -1
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_activation_download_chunks(pn_tt_device *device, const pn_tt_activation *activation,
                                     const unsigned char *need, float *out_values,
                                     uint32_t count);

/**
 * Read logical tile row `row` of a strip back as `count` floats — the DIAGNOSTIC reader.
 *
 * `pn_tt_activation_download()` reads only the two 64-byte chunks logical row 0 lives in, so
 * it cannot see rows 1-31 at all. This one pulls WHOLE tiles. Rows 1-31 are the rows the strip
 * invariant claims are zero and which flash decode is no longer KNOWN to leave zero
 * (pn_tt_oncard.c, invariant 1) — measuring that claim is what this exists for. It moves 32x
 * the bytes of the row-0 reader, so it is instrumentation and never the forward.
 *
 * @param row    0..31; a row outside the tile is a refusal
 * @param count  a multiple of 32, no wider than the strip
 * @return       0, or -1
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_activation_download_tile_row(pn_tt_device *device, const pn_tt_activation *activation,
                                       uint32_t row, float *out_values, uint32_t count);

/**
 * The BATCHED pair: move logical rows 0..rows-1 of a strip, not only row 0.
 *
 * ---------------------------------------------------------------------------
 * THE 31 ROWS WERE NEVER FREE FOR US, ONLY FREE FOR THE CARD
 * ---------------------------------------------------------------------------
 * A tile is 32x32 and a matvec is Mt=1, so the FPU already multiplies 32 rows against every
 * weight tile it reads and 31 of them are the zero padding `pn_tt_activation_alloc` staged.
 * The card's cost — the reader floor, the weight tiles, the descriptor — is IDENTICAL
 * whether one of those rows carries a token or all 32 do. So a prefill that puts 32 prompt
 * positions into one tile pays the card once for 32 tokens: not an optimisation of the
 * matmul, the same matmul with its waste removed. Everything after it is unchanged —
 * `pn_tt_rms_norm` already submits `row_blocks = 1`, i.e. one 32-row block, and the
 * elementwise ops are per-tile — so the only thing standing between the card and a batch was
 * that nothing could put more than one row across the link.
 *
 * **These move WHOLE tiles and that is not a regression.** The row-0 reader's 16x saving is
 * a saving on ONE row; whole tiles are 16x the bytes for 32x the rows, so the batched
 * transfer is 2x cheaper PER TOKEN than the row-0 one, in both directions. `values` is
 * [rows x count] row-major fp32.
 *
 * The upload re-establishes the strip invariant on every call: `ft_tt_tilize_strip` writes
 * rows `rows..31` as ZERO, so a batched write leaves the strip in exactly the state a later
 * Mt=1 matvec requires and the two paths may be interleaved freely.
 *
 * @param rows   1..32; 0 or >32 is a refusal (a tile has 32 rows and nothing here wraps)
 * @param count  a multiple of 32, no wider than the strip
 * @return       0, or -1
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_activation_upload_rows(pn_tt_device *device, const pn_tt_activation *activation,
                                 const float *values, uint32_t count, uint32_t rows);

/** @copydoc pn_tt_activation_upload_rows */
int pn_tt_activation_download_rows(pn_tt_device *device, const pn_tt_activation *activation,
                                   float *out_values, uint32_t count, uint32_t rows);

/**
 * Re-stage a whole strip as ZERO — the strip invariant, restored on demand.
 *
 * `pn_tt_activation_alloc` zeroes once and every Mt=1 op preserves it for free, so nothing
 * needed this. A batched pass DELIBERATELY fills rows 1-31 with prompt positions, and the
 * card then writes 32 live rows into every output strip. Nothing in the decode path reads
 * those rows — `pn_tt_topk` names row 0 explicitly and every other op is per-row — but
 * leaving a batch's residue behind would make the invariant a claim about who reads what
 * instead of a property of the strip. This is what returns it to a property.
 *
 * @return 0, or -1
 */
int pn_tt_activation_zero(pn_tt_device *device, const pn_tt_activation *activation);

/** `pn_tt_activation_zero` over the first `bands` tile-rows only (0 = all of them). */
int pn_tt_activation_zero_bands(pn_tt_device *device, const pn_tt_activation *activation,
                                uint32_t bands);

/** `pn_tt_activation_zero_bands` with every byte `value` (0xFF = bf16 NaN: a residue poison). */
int pn_tt_activation_fill_bands(pn_tt_device *device, const pn_tt_activation *activation,
                                uint32_t bands, unsigned char value);

/** How many 32-row bands every activation strip is allocated with. See the `tile_rows` block
 *  above. 1 is the shipped strip, and the whole mechanism collapses to the pre-Mt one there. */
#define PN_TT_STRIP_TILE_ROWS_ENV "PN_TT_STRIP_TILE_ROWS"

/**
 * How many tile-rows `pn_tt_activation_alloc` gives every strip — `PN_TT_STRIP_TILE_ROWS`.
 *
 * Read ONCE, before any strip is allocated, because a strip whose height disagrees with its
 * neighbours' is a matmul reading one band and writing another. 1 is the shipped value and
 * the pre-Mt allocation byte for byte.
 */
uint32_t pn_tt_device_strip_tile_rows(pn_tt_device *device);

/**
 * The strip height THIS MODEL wants when `PN_TT_STRIP_TILE_ROWS` is unset. Must be called
 * before the first strip is allocated, and only by a model whose batched prefill runs both
 * chunk paths — any other model at more than one band declines its batched prefill.
 * @return 0, or -1 if the height was already resolved (too late to prefer anything)
 */
int pn_tt_device_prefer_strip_tile_rows(pn_tt_device *device, uint32_t rows);

/**
 * Declare that every strip holds ONE live row, row 0 (a single-token forward), so ops may
 * compute that row only (PHASE E). Rows 1-31 are then whatever the cheap path leaves — the
 * zero-padding invariant holds only because their inputs are zero. Clear it before any
 * multi-row op. `PN_TT_ELTWISE_ROW0`: 0 off, 1 (default) honour the promise, 2 RED: apply
 * row 0 to EVERY descriptor, multi-row prefill included, which must change its output.
 */
void pn_tt_device_set_single_row(pn_tt_device *device, bool single_row);
/** S2c: the verify's live-row promise (pn_tt_device_internal.h `verify_rows`; 0 = none). */
void pn_tt_device_set_verify_rows(pn_tt_device *device, uint32_t rows);
/** S2c: the logits row pn_tt_topk() selects over (0 = the decode row). */
void pn_tt_device_set_topk_row(pn_tt_device *device, uint32_t row);

/** M5: per-processor CB-wait time of the main residencies (`pn_tt: wait clock` lines). */
void pn_tt_device_wait_report(pn_tt_device *device, double tokens);

/**
 * The matmul K-block depth this model wants when `PN_TT_PREFETCH_TILES` is unset. Call before
 * the artifact loads (the residency arms there). Per model, so another model's default does
 * not move as a side effect. Qwen3.5 hybrid: 4 (M1, +0.6-1.0% decode, +0.6% prefill).
 */
void pn_tt_device_prefer_prefetch_tiles(pn_tt_device *device, uint32_t tiles);

/** M4: this model wants the pipelined matmul reader with 3-chunk CBs (env overrides
 *  `PN_TT_MATMUL_PIPE`, `PN_TT_MATMUL_CB_DEEP`). Call before the artifact loads. */
void pn_tt_device_prefer_matmul_pipe(pn_tt_device *device, bool pipe);

/**
 * Point a strip at tile-row `row`, so an op written for a 32-row strip runs on band `row`.
 *
 * THE ONLY MECHANISM THAT REACHES THE SECOND BAND FOR THE OPS THAT CANNOT BE WIDENED, and
 * they are the majority: the per-head norms spend `row_blocks` on heads
 * (`pn_tt_head_norm.c`), RoPE's table is one 32-position block, the KV append's `source_row0`
 * is capped at 32, the attention chunk REFUSES a non-zero `query_row` in chunk mode
 * (`ft_tt_op_attention.c`), and GDN PHASE CONV refuses `first_row` outright
 * (`ft_tt_op_gdnc.c`). None of them needs a new field: every one of them addresses DRAM
 * through the operand's `address`, so sliding the address by a whole band is the same
 * descriptor over different bytes.
 *
 * Slides every MIRROR by the same amount, because a matmul member reads `replica[m %
 * replica_count]` and a mirror left behind is the previous band's answer with no error
 * anywhere.
 *
 * @param row  0..tile_rows-1
 * @return     0, or -1 (out of range, or no strip)
 */
int pn_tt_activation_seek(pn_tt_activation *activation, uint32_t row);

/** How many tile-rows the NEXT op over this strip spans. Only the ops that are genuinely
 *  tile-row-general read it — the RMS norm as `row_blocks`, the elementwise ops as a wider
 *  tile count, the matmul as `Mt`. 1 is the shipped span. */
int pn_tt_activation_span(pn_tt_activation *activation, uint32_t rows);

/**
 * y = x * W with BOTH operands already on the card — the projection of the on-card forward.
 *
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_matvec_resident(pn_tt_device *device, const pn_tt_weight *weight,
                          const pn_tt_activation *input, const pn_tt_activation *output);

/**
 * y = x * W, ISSUED and not awaited — the same projection, without the round trip.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS
 * ---------------------------------------------------------------------------
 * The resident queue was built so the host enqueues and bumps a counter, and the forward
 * then waited on every one of its 169 projections anyway. Two things cost: the host asks
 * every member whether it is done (linear in the member count), and the card sits idle
 * while the host issues the next descriptor. Within a layer **q/k/v are independent** and
 * **gate/up are independent**, so their descriptors can sit in the ring together — the
 * card runs them back to back at its own 0.39-0.45 us op-to-op, and the host waits FOUR
 * times a layer instead of seven.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE CALLER OWES
 * ---------------------------------------------------------------------------
 * Chained projections must be INDEPENDENT: same input, disjoint outputs. They run on
 * different cores concurrently, so a second projection reading the first's output would
 * race even though one core's own queue is ordered.
 *
 * Nothing else has to be remembered. Every other device operation — a download, an upload,
 * a norm, a SwiGLU, an attention, a top-N, even a waiting matvec — flushes first, so a
 * missed `pn_tt_matvec_flush()` cannot produce a wrong answer, only a later wait. A rule
 * enforced at ONE seam beats a rule every call site has to observe.
 *
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_matvec_enqueue(pn_tt_device *device, const pn_tt_weight *weight,
                         const pn_tt_activation *input, const pn_tt_activation *output);

/** Wait for every projection `pn_tt_matvec_enqueue()` has issued. A no-op when none are
 *  pending. @return 0, or -1 with the reason on stderr */
int pn_tt_matvec_flush(pn_tt_device *device);

/**
 * Build one norm weight's constants and upload them. Once per norm, at load — a per-token
 * upload of gamma would put back a transfer this file exists to remove.
 *
 * @param gamma  `width` floats
 * @param eps    the model's rms_norm_eps
 * @return       0, or -1 with the reason on stderr
 */
int pn_tt_norm_constants_upload(pn_tt_device *device, const float *gamma, uint32_t width,
                                float eps, pn_tt_norm_constants *out_constants);

/** out = in * gamma * rsqrt(mean(in^2) + eps), on the card. */
int pn_tt_rms_norm(pn_tt_device *device, const pn_tt_norm_constants *constants,
                   const pn_tt_activation *input, const pn_tt_activation *output);

/** out = a + b, on the card. */
int pn_tt_add(pn_tt_device *device, const pn_tt_activation *a, const pn_tt_activation *b,
              const pn_tt_activation *output);

/** out = silu(a) * b, on the card — the SwiGLU gate, as ONE descriptor rather than two,
 *  because as two it costs a DRAM round trip of the intermediate. */
int pn_tt_silu_mul(pn_tt_device *device, const pn_tt_activation *a, const pn_tt_activation *b,
                   const pn_tt_activation *output);

/**
 * out = sigmoid(gate) * b, on the card — Ornith's ATTENTION output gate.
 *
 * **NOT `pn_tt_silu_mul`**: `silu(x) = x*sigmoid(x)`, so the SwiGLU member is off by a factor
 * of the gate itself. There is no fused `sigmoid(a)*b` in the elementwise table, so this is
 * TWO descriptors through `scratch` — which is why `scratch` is a parameter rather than
 * borrowed: a strip this function chose for itself could alias one of the three the caller
 * passed, and an elementwise op writes its output before the next one reads its input.
 *
 * @param scratch  a strip of at least `gate`'s tile count, distinct from `gate` and `b`
 * @return         0, or -1 with the reason on stderr
 */
int pn_tt_sigmoid_mul(pn_tt_device *device, const pn_tt_activation *gate,
                      const pn_tt_activation *b, const pn_tt_activation *scratch,
                      const pn_tt_activation *output);

/**
 * Exact top-N over a logit row that is already on the card, and NOTHING crosses the link
 * but the pairs.
 *
 * The comparator is the deployed K80 one — highest value wins, LOWEST index breaks a tie —
 * so an index this returns is comparable to a host serial argmax over the same row rather
 * than merely close to it. That is the gate this op is accepted on: EXACT INDEX AGREEMENT,
 * not a correlation.
 *
 * @param out_indices  at least `count` entries
 * @param out_values   at least `count` entries, fp32
 * @param out_count    receives how many pairs the card produced
 * @return             0, or -1 with the reason on stderr
 */
int pn_tt_topk(pn_tt_device *device, const pn_tt_activation *logits, uint32_t count,
               uint32_t *out_indices, float *out_values, uint32_t *out_count);

/* ================================================================
 * ON-CARD ATTENTION: RoPE, the KV append, and flash decode
 * ================================================================
 *
 * These three move together or none of them moves a byte — q, k and v cross the link for
 * whichever of the three is left on the host, and that readback is 0.052 s of a 0.128 s
 * token (~240 KB against a measured 4.7 MB/s card->host).
 *
 * ---------------------------------------------------------------------------
 * WHERE THE KV CACHE LIVES, AND WHAT IT COSTS SESSION EXPORT
 * ---------------------------------------------------------------------------
 * On-card attention wants the KV cache in card DRAM, bf16, in the layout
 * foundry/tt/ft_tt_op_memory.h PART 1 fixes (F1-F5). Session export was built on the
 * opposite fact: it is free BECAUSE the cache is host fp32, so exporting reads a host
 * pointer and zero bytes cross PCIe (49,152 B/token of blob, 0 B of link).
 *
 * **THE CACHE MOVES AND EXPORT PAYS AT EXPORT TIME, NOT PER TOKEN. The arithmetic is not
 * close.** A host mirror maintained by downloading k and v every layer costs
 * 2 x 8 tiles x 128 B = 2,048 B a layer = 49,152 B a token of ROW-0 reads (each 64 B chunk
 * carries 32 B of payload — the read alignment, ft_tt_tile_row0.h), i.e. 10.5 ms EVERY
 * token. Downloading the whole card cache once at export reads WHOLE tiles with no
 * alignment waste: 24,576 B a position, 5.2 ms a position-equivalent, ~10.7 s for a
 * 2048-token session. **So the mirror moves exactly 2x the bytes of the export it makes
 * free, and it charges every session including the ones nobody ever exports.** A
 * 2048-token session would pay 21 s of decode to save 10.7 s of export.
 *
 * Restore is the cheap direction and stays cheap: host->card is 287 MB/s, ~61x the reverse,
 * so a 2048-token restore uploads 50.3 MB in ~0.18 s.
 *
 * The blob therefore becomes bf16 where it was fp32 — which is not a loss of fidelity but a
 * statement of where the truth is: once attention runs on the card, the card's bf16 cache
 * IS the model state, and an fp32 blob would be storing a precision the forward never had.
 *
 * ---------------------------------------------------------------------------
 * ONE REGION SET, AND INTERLEAVED SESSIONS ARE A REFUSAL
 * ---------------------------------------------------------------------------
 * The card holds ONE K/V region set per model, allocated at arm time. Stale contents are
 * harmless — every position a sequence reads is a position it wrote — so a NEW sequence may
 * simply rebind. Two sequences ALTERNATING on one region set would silently overwrite each
 * other, so a host cache key that comes back after being displaced is REFUSED by name.
 */

/**
 * Arm the RoPE, KV-append and flash-decode residencies (three more cores) and allocate the
 * KV regions of every layer.
 *
 * Never fatal on its own: a card missing one of the three programs leaves attention on the
 * host, exactly like the other families — but ONLY when attention was not asked for. When
 * `PN_TT_ONCARD` names it, a missing program is a refusal (pn_tt_device_arm_oncard).
 *
 * @param q_heads             Q heads (16 for MiniCPM5-1B)
 * @param kv_heads            KV heads (2); must divide q_heads
 * @param head_dim            multiple of 32
 * @param rope_dim            rotated features, 0 for the whole head
 * @param rope_theta          `rope.freq_base`, read from the checkpoint
 * @param capacity_positions  the session's max_seq, rounded up to 32 internally
 * @param layer_count         layers to allocate regions for
 * @return                    0 (even when nothing armed), -1 on a bad argument or a refusal
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_device_arm_attention(pn_tt_device *device, uint32_t q_heads, uint32_t kv_heads,
                               uint32_t head_dim, uint32_t rope_dim, double rope_theta,
                               uint32_t capacity_positions, uint32_t layer_count);

/**
 * Bind the card's ONE KV region set to a host sequence, identified by any stable pointer.
 *
 * A key that was displaced and comes back is REFUSED: that is two sequences alternating on
 * one region set, which would be a wrong answer with no error anywhere.
 *
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_kv_bind(pn_tt_device *device, const void *sequence_key);

/** NeoX RoPE at `position`, from `input` into `output`, for `heads` heads.
 *
 *  Out of place on purpose: the reader runs ahead of the writer, so writing over the input
 *  would be an unstated dependency on that lead — the same reason the residual stream keeps
 *  two strips. */
int pn_tt_rope(pn_tt_device *device, const pn_tt_activation *input,
               const pn_tt_activation *output, uint32_t heads, uint32_t position);

/** Place one token's K (`values` false) or V (`values` true) row into layer `layer`'s cache
 *  region at absolute `position`. */
/** STEP 4 S2c: rotate rows 0..rows-1 of a strip at positions first_position.. (any
 *  alignment; the decode row's angles), and append ONE row of a strip at `position`. */
int pn_tt_rope_rows(pn_tt_device *device, const pn_tt_activation *input,
                    const pn_tt_activation *output, uint32_t heads, uint32_t first_position,
                    uint32_t rows);
/** S2f: flash decode for strip rows 0..rows-1 at positions first_position.. in ONE descriptor
 *  a KV group (ROWS mode); every row must end in the same position tile. */
int pn_tt_attention_rows(pn_tt_device *device, uint32_t layer, const pn_tt_activation *query,
                         const pn_tt_activation *output, uint32_t first_position, uint32_t rows);
int pn_tt_kv_append_row(pn_tt_device *device, uint32_t layer, bool values,
                        const pn_tt_activation *source, uint32_t source_row, uint32_t position);
int pn_tt_kv_append(pn_tt_device *device, uint32_t layer, bool values,
                    const pn_tt_activation *source, uint32_t position);

/** Flash-decode attention over `kv_positions` cached positions, every KV group of one
 *  layer, `query` in and `output` out in the model's ordinary activation layout. */
int pn_tt_attention(pn_tt_device *device, uint32_t layer, const pn_tt_activation *query,
                    const pn_tt_activation *output, uint32_t kv_positions);

/* ================================================================
 * The CHUNK forms — the same three ops over C consecutive positions
 * ================================================================
 *
 * A CHUNK IS THE SAME STRIP A SINGLE POSITION IS, with rows 0..C-1 carrying positions
 * instead of padding — the geometry the two chunkwise GDN mixers settled and the reason
 * every activation's DRAM layout is unchanged. These three calls are what the attention
 * layer needs to stop being per-position; the head norms, the sigmoid gate, the residual
 * adds and all nine projections already batch, because a row reduce reduces ALONG the row
 * and a matmul's output row is its input row.
 *
 * `first_position` MUST be a multiple of 32 for all three: RoPE's row-phase contract wants
 * it (`ft_tt_op_rope.h`), the KV append's tile is then whole, and attention's causal mask is
 * one tile only when the chunk IS the last position tile. `rows` is 1..32.
 *
 * They are not a mode: decode keeps calling the per-position forms above, which are
 * untouched, and the two kinds of descriptor interleave on ONE residency over ONE KV cache.
 */

/** NeoX RoPE for positions `first_position .. first_position + rows - 1`, carried in rows
 *  0..rows-1 of `input`. Rebuilds the whole 32-row rotary block rather than its row 0. */
int pn_tt_rope_chunk(pn_tt_device *device, const pn_tt_activation *input,
                     const pn_tt_activation *output, uint32_t heads,
                     uint32_t first_position, uint32_t rows);

/** Place `rows` positions of K or V into layer `layer`'s cache in ONE descriptor. At 32
 *  rows and an aligned first position this is a whole-tile copy per head, not a row
 *  scatter — the memory kernel already had it (`resident_place_writer.c`). */
int pn_tt_kv_append_chunk(pn_tt_device *device, uint32_t layer, bool values,
                          const pn_tt_activation *source, uint32_t first_position,
                          uint32_t rows);

/**
 * Attention for a whole chunk: ONE descriptor per Q HEAD, each serving all `rows`
 * positions, against the cache up to `first_position + rows` with causality inside the
 * chunk.
 *
 * `q_heads` descriptors replace `kv_heads * rows` — 16 against 128 at Ornith's shape, and
 * MEASURED on silicon 2026-08-21 to cost 0.093 ms against a decode descriptor's 0.101 at
 * kv=1024 (0.605 against 0.615 at 8192), because the score matmul already computed all 32
 * rows of its tile and threw 28 of them away. A position therefore costs 8.7x less, and the
 * KV cache is walked 8x fewer times. The chunk's answer is BIT-IDENTICAL to the decode
 * descriptors it replaces, row for row, max-abs 0.000e+00 over the four chunks of a
 * 32+32+32+7 gate.
 */
int pn_tt_attention_chunk(pn_tt_device *device, uint32_t layer, const pn_tt_activation *query,
                          const pn_tt_activation *output, uint32_t first_position,
                          uint32_t rows);

/**
 * Read one layer's K or V back as fp32 in Pantheon's own [head][position][dim] order — the
 * session-export path, and the ONLY place the card cache crosses the link.
 *
 * @param out  `kv_heads * position_count * head_dim` floats
 * @return     0, or -1 with the reason on stderr
 */
int pn_tt_kv_download(pn_tt_device *device, uint32_t layer, bool values,
                      uint32_t position_count, float *out);

/** The reverse, for restore. host->card is ~61x faster, so this is the cheap direction. */
int pn_tt_kv_upload(pn_tt_device *device, uint32_t layer, bool values,
                    uint32_t position_count, const float *in);

/** Matmuls issued, and the wall seconds spent inside pn_tt_matvec(). See
 *  pn_tt_device_phase_seconds() for the three-way split of that total. */
uint64_t pn_tt_device_matvec_count(const pn_tt_device *device);
double pn_tt_device_matvec_seconds(const pn_tt_device *device);

/**
 * The three phases of every matvec, separated because they are three different machines:
 * `upload` is a host tilize plus a ~287 MB/s write, `dispatch` is ft_tt_op_matmul_run()
 * (five kernel images re-uploaded, CB config, GO, poll DONE) and the card's own compute,
 * and `readback` is the ~2-5 MB/s return of the result plus an untilize.
 *
 * Splitting these is what turns "a token takes N seconds" into an attribution. Any of them
 * may be NULL.
 */
void pn_tt_device_phase_seconds(const pn_tt_device *device, double *out_upload,
                                double *out_dispatch, double *out_readback);

/** The FOURTH phase, kept out of the three above because it is a fourth machine: the
 *  enqueue-and-wait for the reduce, elementwise and sampling residencies. Work that used to
 *  be host arithmetic between two transfers, and is now neither.
 *
 *  `out_restarts` counts the times one of those three residencies had to be ASKED to retire
 *  and re-armed. It is reported beside the matmul's own restart count and not folded into
 *  it, because they answer different questions: a matmul restart means a descriptor the
 *  reader would not execute, and an on-card restart is almost always a kernel that had
 *  retired on idle between two requests. Any pointer may be NULL. */
void pn_tt_device_oncard_stats(const pn_tt_device *device, double *out_seconds,
                               uint64_t *out_ops, uint64_t *out_restarts);

/**
 * That fourth phase BROKEN OUT PER RESIDENCY, which is what a re-pricing needs and the sum
 * above cannot give.
 *
 * `pn_tt_device_oncard_stats()` adds seven families together, so "247 µs a descriptor" is an
 * average over a reduce, a SwiGLU, a residual add, a top-N and a GDN block — ops whose work
 * differs by three orders of magnitude. Every residency already counts its own descriptors
 * and its own seconds inside `pn_tt_residency_submit()`; this walks them so the report can
 * print the per-family cost instead of one number that is true of nothing.
 *
 * @param index  0 upward, in arm order; `false` means past the end
 * @return       true and the four figures, or false. Any output pointer may be NULL.
 */
bool pn_tt_device_residency_stat(const pn_tt_device *device, unsigned index,
                                 const char **out_name, uint64_t *out_ops, double *out_seconds,
                                 uint32_t *out_core);

/** Most clocks `pn_tt_device_card_clocks()` can return. */
#define PN_TT_CARD_CLOCKS_MAX 64u

/**
 * EVERY CARD-SIDE CLOCK THIS DEVICE KEEPS, by name, in an order that is stable for the life of
 * the device — so a caller can snapshot it around a bracket and attribute the bracket's wall
 * by subtraction, entry by entry. (PHASE P G0: `sequential` in the batched prefill.)
 *
 * The entries are every residency INCLUDING split members (a member repeats member 0's name;
 * sum by name), then the gdnc split's own clocks, which are not residencies: `gdnc-phase0`,
 * `gdnc-conv`, `gdnc-delta` (clean descriptors, by `FT_TT_OP_GDNC_PHASE_*`), `gdnc-recovered` (any phase, re-armed) and
 * `gdnc-stage` (the host build + upload of the decay/gate blocks). No two entries can contain
 * the same second: the gdnc clocks bracket `gdnc_submit()` and the stage brackets its caller's
 * staging, which runs before that.
 *
 * @return the number of entries written, at most `max`
 */
unsigned pn_tt_device_card_clocks(const pn_tt_device *device, const char **out_names,
                                  double *out_seconds, unsigned max);

/** HOST work inside a forward that no card clock contains, booked by name so the G0
 *  attribution can split its "unclocked" remainder. Appear in pn_tt_device_card_clocks(). */
typedef enum {
    PN_TT_HOST_CLOCK_GDN_MATH = 0,  /**< beta sigmoid + decay gate + expf, on the host */
    PN_TT_HOST_CLOCK_GDN_CONSTANTS, /**< gdn_strip_rms_constants() */
    PN_TT_HOST_CLOCK_ROPE_TABLE,    /**< the rotary table built and uploaded for a chunk */
    PN_TT_HOST_CLOCK_COUNT
} pn_tt_host_clock;

void pn_tt_device_host_clock_add(pn_tt_device *device, pn_tt_host_clock clock, double seconds);

/**
 * THE AWAIT PRICED IN PCIe ACCESSES, over every residency INCLUDING the split members —
 * **SPIN SEPARATED FROM SETTLE, because pooling them hides the thing being measured.**
 *
 * `poll` is every read inside the wait loop. It is CARD-BOUND: it is how the host waits for
 * work that has not finished, and no completion shape removes it. `settle` is the block read
 * issued after the wait returned — **16 on `control`, 5 on `window`, 0 on `line`** — and it
 * is the only term the shapes move, so it is the only one a ladder may be scored on.
 *
 * **THE SPLIT EXISTS BECAUSE THE POOLED FORM WAS MEASURED AND WAS UNREADABLE.** Two card
 * builds both on the six-read `window` reported **49.0 and 54.5** reads an await, because a
 * residency waiting on real card work spins ~44 times. An acceptance criterion of
 * "17.0 / 6.0 / 1.0" against that number is either unreadable or read as a failed arm — the
 * shape of a harness that measures nothing.
 *
 * @param out_fast  how many residencies with at least one await took the ONE-transaction
 *                  path — so an arm cannot report a saving the knob never armed. It is a
 *                  count and not a flag because a queue the BLOCK16 accessor cannot reach
 *                  falls back on its own, and a partial arm must be visible as a number.
 *
 * Any pointer may be NULL.
 */
void pn_tt_device_await_transactions(const pn_tt_device *device, uint64_t *out_awaits,
                                     uint64_t *out_poll, uint64_t *out_settle,
                                     unsigned *out_fast);

/**
 * THE ISSUE PRICED THE SAME WAY, and TIMED PER CALL so the fixed cost can be solved rather
 * than assumed.
 *
 * One enqueue is three device writes — a 64 B parameter block, a 32 B ring slot, a 4 B head
 * bump — which is **25 posted accesses** at WORD32 and **7** on the wide shape. Neither a byte
 * counter nor a call counter can see that difference (same bytes, same three calls), so the
 * accesses are counted at the unit of the accessor that was actually used.
 *
 * **The three clocks are what make the count mean something.** The recorded matmul figures put
 * a 16-access write at 2.08 µs and a 9-access pair at 4.84 — which no `fixed + per_access`
 * model fits, so one of the two must be dominated by a per-CALL cost. Three calls of 16, 8 and
 * 1 accesses timed apart over-determine that split, and the answer decides whether a lever
 * that removes ACCESSES can pay at all.
 *
 * @param out_wide  how many residencies with at least one issue took the wide shape — so an
 *                  arm cannot report a saving the knob never armed.
 *
 * Any pointer may be NULL.
 */
void pn_tt_device_issue_transactions(const pn_tt_device *device, uint64_t *out_issues,
                                     uint64_t *out_accesses, double *out_param_seconds,
                                     double *out_ring_seconds, double *out_head_seconds,
                                     unsigned *out_wide);

/**
 * ONE MEMBER of the elementwise core split — its core, its descriptors and its own AWAIT.
 *
 * The residency stat above reports the elementwise family as ONE line, which is what it was
 * until `PN_TT_ELTWISE_CORES` made it a set; a split reported as one line cannot say whether
 * a member count stopped paying because the cores stopped scaling or because ONE member is
 * being starved. The awaits run in member order after every enqueue, so a balanced set puts
 * nearly all the time on member 0 and a straggler is the member with time of its own.
 *
 * @param index  0 is `eltwise` itself; `false` means past the live members
 * @return       true and the three figures, or false. Any output pointer may be NULL.
 */
bool pn_tt_device_eltwise_member_stat(const pn_tt_device *device, unsigned index,
                                      uint32_t *out_core, uint64_t *out_ops,
                                      double *out_seconds);

/**
 * The SAME on-card descriptors, keyed by (family, function, TILE COUNT) — borrowed, never
 * NULL for a live device.
 *
 * `pn_tt_device_residency_stat()` above replaced "247 us each" with a line per family; this
 * replaces the line per family with a line per SHAPE, because the elementwise residency runs
 * six functions over three widths and reports one mean of them all. It is also the only
 * instrument that can price a family's per-descriptor FIXED cost from an ordinary run —
 * see `pn_tt_descriptor_census_fit()`. Read `dropped` and `refused` beside every figure.
 */
const pn_tt_descriptor_census *pn_tt_device_census(const pn_tt_device *device);

/** `dispatch` split into the host's MMIO ISSUE and the completion WAIT, with the descriptor
 *  count each wait covered. The spec reads 1.23 ms a matmul as a PCIe round trip; these two
 *  counters are what says how much of it is one. Any pointer may be NULL. */
void pn_tt_device_dispatch_split(const pn_tt_device *device, double *out_issue_seconds,
                                 uint64_t *out_issues, double *out_wait_seconds,
                                 uint64_t *out_waits, uint64_t *out_wait_descriptors);

/** The GDN family's two PHASES apart. One residency runs both, so its `ops` and `seconds`
 *  cannot separate the conv1d from the delta rule — and those are the two terms a Phase 3
 *  re-pricing has to tell apart. `phase` is `FT_TT_OP_GDN_PHASE_CONV` or `..._DELTA`.
 *  CLEAN DESCRIPTORS ONLY — see `pn_tt_device_gdn_phase_recovery()`. */
void pn_tt_device_gdn_phase_stats(const pn_tt_device *device, unsigned phase,
                                  double *out_seconds, uint64_t *out_ops);

/** The descriptors of that phase that REFUSED and were replayed after a re-arm, and the
 *  brackets they cost. Deliberately not in the figure above: a re-arm is a residency teardown
 *  and relaunch, and folding one into `seconds / ops` prices it as the op's arithmetic — which
 *  is how PHASE CONV once read 4,502 us a descriptor against 738.2. Anything quoting the clean
 *  figure prints `out_rearms` beside it. */
void pn_tt_device_gdn_phase_recovery(const pn_tt_device *device, unsigned phase,
                                     double *out_seconds, uint64_t *out_ops,
                                     uint64_t *out_rearms);

/** Bytes read back from the card and written to it, over this device's life. The read
 *  figure is the one that matters: the link is ~60x slower in that direction. */
uint64_t pn_tt_device_bytes_read(const pn_tt_device *device);
uint64_t pn_tt_device_bytes_written(const pn_tt_device *device);

/**
 * How many matvecs went through the RESIDENT queue, how many fell back to one-shot, and
 * how many times a residency had to be retired mid-run.
 *
 * EVIDENCE, like the window counters below and for the same reason: both dispatch paths
 * produce identical numbers from identical kernels, so a run that silently fell back to
 * one-shot is indistinguishable from a resident one by its output. Only this counter says
 * which machine ran. A resident claim with `oneshot` non-zero is a claim about a path that
 * did not carry the work. Any output pointer may be NULL.
 */
void pn_tt_device_dispatch_counts(const pn_tt_device *device, uint64_t *out_resident,
                                  uint64_t *out_oneshot, uint64_t *out_restarts);

/**
 * Bytes each of the two TLB windows has actually carried, straight out of the NoC paths'
 * own counters — the register window (UC, one 32-bit access per word) and the bulk DRAM
 * window (WC, 16 B units).
 *
 * This exists as EVIDENCE, not telemetry. The failure mode of splitting the two accessors
 * is a register access silently taking the bulk path, which would not surface until
 * something hung; and the reverse — data quietly still on the register path — is the
 * 27 s/token bug this split exists to fix, and it looks identical from the outside. These
 * four counters distinguish all four cases. Any output pointer may be NULL.
 *
 * TRANSPORT-AGNOSTIC as of 2026-08-17. It used to read the tt-kmd binding directly, so on
 * the userspace-driver transport it reported that ZERO bytes had moved through either
 * window on a run that had just served a token on the card — an instrument that reads
 * clean on the path it was never taught about is worse than none at all.
 */
void pn_tt_device_window_traffic(const pn_tt_device *device, uint64_t *out_register_read,
                                 uint64_t *out_register_written, uint64_t *out_bulk_read,
                                 uint64_t *out_bulk_written);

/**
 * How many times each window was RE-AIMED, and how many aims failed verification.
 *
 * Aiming is in the hot path — the NoC layer re-points a window whenever the target moves —
 * and it is the one thing the two transports do differently at a per-operation rate: the
 * tt-kmd path issues an ioctl, and the userspace driver writes the registers and then READS
 * ALL FOUR BACK over an uncached PCIe link. So the aim count is what converts a per-aim cost
 * difference into a per-token one, and it is the first number to look at when the two
 * transports disagree on wall clock.
 *
 * `out_failures` is udrv-only and MUST be zero: a non-zero value means an aperture was
 * refused because its registers did not read back as intended, not that a wrong address was
 * used. Any output pointer may be NULL.
 */
void pn_tt_device_window_aims(const pn_tt_device *device, uint64_t *out_register_aims,
                              uint64_t *out_bulk_aims, uint64_t *out_failures);

/**
 * WHAT THOSE AIMS COST, and the READBACK's share of it. Zero on tt-kmd, where the aim is an
 * ioctl and there is no readback to separate.
 *
 * The aim count on its own says how often a window moved; it cannot say what that is worth,
 * and the only per-aim figure this program ever had (1.23 us) is the tt-kmd ioctl's, i.e. a
 * different mechanism. On the udrv path an aim is four POSTED register stores plus four
 * NON-POSTED readbacks, and on the dispatch path it happens ONCE PER DESCRIPTOR — so which
 * half dominates decides whether the fix is fewer readbacks or more windows.
 */
void pn_tt_device_window_aim_seconds(const pn_tt_device *device, double *out_seconds,
                                     double *out_verify_seconds);

/**
 * THE WINDOW BANK — how many windows each kind claimed, and whether the set is absorbing the
 * endpoint rotation.
 *
 * The aim count alone cannot distinguish "the bank is working" from "the bank was never
 * armed": both read as a number. `hits` and `misses` are the mechanism measured directly, and
 * the two window counts are what the knob ACHIEVED rather than what it asked for — a knob that
 * reads as armed and is not is this program's most expensive class of instrument defect.
 *
 * REPORTED PER BANK BECAUSE THE TWO STREAMS ARE DIFFERENT MECHANISMS. The WC bank's weight
 * upload walks thousands of consecutive slices inside one 2 MiB window, so a pooled hit rate
 * reads ~93% however badly the dispatch path is thrashing its 26 endpoints.
 *
 * `misaims` is FT_TT_UDRV_RED=misaim's own count, so a red arm that never fired reads as a
 * zero instead of as a silence, and `audit_failures` must be zero on every arm but that one.
 * All zero on tt-kmd, which has one window per kind and no bank. Any pointer may be NULL.
 */
void pn_tt_device_window_bank(const pn_tt_device *device, uint32_t *out_register_windows,
                              uint32_t *out_bulk_windows, uint64_t *out_register_hits,
                              uint64_t *out_register_misses, uint64_t *out_bulk_hits,
                              uint64_t *out_bulk_misses, uint64_t *out_misaims,
                              uint64_t *out_audit_failures);

#endif /* USE_TT */
/** MEASURE (`PN_TT_ARC_TELEMETRY=1`): log one line of ARC telemetry tagged `label`. */
void pn_tt_device_telemetry_log(pn_tt_device *device, const char *label);

/** STEP 3 GATE (`PN_TT_IDLE_CLOCK=1`): stage 0 snapshots every writer's idle-for-work
 *  clock; stage 1 prints card-busy per family and the host-attributable card idle. */
void pn_tt_device_idle_clock(pn_tt_device *device, int stage, double tokens);

#endif /* PN_TT_DEVICE_H */
