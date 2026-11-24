/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_admission.c
 * @brief Implementation of pre-flight admission control (MemoryAware.md §6).
 *        The cost of an N-token prefill is estimated from a measured
 *        per-prompt-length LUT (the transient activation peak — the O(N^2) term
 *        that OOM-rebooted the box) interpolated between anchor points, plus a
 *        linear per-token KV term. Refine the LUT with real K80 measurements
 *        (§11 Q1); the shape is intentionally conservative until then.
 *
 * @status Phase 2 — admission gates, path-aware (GPU-resident requests admit
 *         the full context up to max_window; the LUT gates the CPU-fallback
 *         path only) + the model-load host-spike budget (PN_LOAD_SPIKE_BYTES).
 * @issues LUT anchors model the CPU-forward path (kept conservative on
 *         purpose); GPU-path peaks measured on a K80 host are ~0 — the gpu_resident
 *         branch of pn_admit_request encodes that finding.
 * @todo None
 */

#include "pn_admission.h"

#include <stdlib.h>

#include "pn_memwatch.h"

/* Measured activation-peak LUT (MemoryAware.md §11 Q1). Each row is
 * {prompt_tokens, peak_host_activation_bytes} — the transient, NON-KV host
 * memory a full-attention prefill of that length peaks at. The curve is
 * super-linear (the attention scores buffer is O(n_head * seq)); we interpolate
 * linearly between anchors and extrapolate from the last segment's slope beyond
 * the top anchor. Seeded conservatively from the spec's failure surface (32k
 * full-attention prefill exhausts an 8 GB host); replace with K80 numbers
 * captured at these exact lengths to tighten. */
typedef struct {
    int      tokens;
    uint64_t peak_bytes;
} admit_lut_row;

/* MEASURED (K80) 2026-07-09 (K80, dies 0+1, Q3_K_M 2-die split @ layer 20,
 * QWEN35_KV_Q4=1, QWEN35_MAX_SEQ=32768, serve prefill loop): on the GPU
 * forward path the transient HOST activation peak is ~0 at every measured
 * length — the O(N^2) scores live on-die, and serve RSS never moved:
 *     n=981   -> RSS delta 0 MiB   (avail dip  1 MiB)
 *     n=1958  -> RSS delta 0 MiB   (avail dip 100 MiB, page-cache noise)
 *     n=3933  -> RSS delta 246 MiB (avail dip 273 MiB, page-cache re-fault)
 *     n=8083  -> RSS delta 0 MiB   (avail dip 36 MiB)
 *     n=20360 -> no avail dip over a 200 s capped-prefill sample
 * The anchors below are therefore NOT tightened to the GPU-path numbers: they
 * model the CPU-forward / host-fallback path — the path that actually
 * OOM-rebooted the box and the one a GPU-refused load falls back to. Keeping
 * them conservative keeps the reboot impossible on that path. GPU-resident
 * requests no longer pay this LUT: pn_admit_request's gpu_resident branch
 * admits the full context up to max_window (DECIDED 2026-07-09 — path-aware
 * admission; the only length-triggered host fallback is n > QWEN35_MAX_SEQ). */
static const admit_lut_row PN_ADMIT_LUT[] = {
    {  1024,   64ULL * 1024 * 1024 },   /* ~64 MiB  */
    {  2048,  160ULL * 1024 * 1024 },   /* ~160 MiB */
    {  4096,  440ULL * 1024 * 1024 },   /* ~440 MiB */
    {  8192, 1200ULL * 1024 * 1024 },   /* ~1.2 GiB */
    { 16384, 3400ULL * 1024 * 1024 },   /* ~3.4 GiB — already tips an 8 GB box */
    { 32768, 9800ULL * 1024 * 1024 },   /* ~9.6 GiB — the OOM-reboot point     */
};
static const int PN_ADMIT_LUT_N =
    (int)(sizeof(PN_ADMIT_LUT) / sizeof(PN_ADMIT_LUT[0]));

/* Interpolate the activation peak for an arbitrary prompt length from the LUT.
 * Below the first anchor: scale down linearly from it. Between anchors: linear
 * interpolation. Above the top anchor: extrapolate along the last segment's
 * slope (never clamp — the whole point is to keep escalating past 32k). */
static uint64_t lut_activation_peak(int n_prompt)
{
    if (n_prompt <= 0)
        return 0;

    const admit_lut_row *lo = &PN_ADMIT_LUT[0];
    if (n_prompt <= lo->tokens)
        return lo->peak_bytes * (uint64_t)n_prompt / (uint64_t)lo->tokens;

    for (int i = 1; i < PN_ADMIT_LUT_N; i++) {
        const admit_lut_row *a = &PN_ADMIT_LUT[i - 1];
        const admit_lut_row *b = &PN_ADMIT_LUT[i];
        if (n_prompt <= b->tokens) {
            uint64_t span = (uint64_t)(b->tokens - a->tokens);
            uint64_t off  = (uint64_t)(n_prompt - a->tokens);
            uint64_t rise = b->peak_bytes - a->peak_bytes;
            return a->peak_bytes + rise * off / span;
        }
    }

    /* Extrapolate beyond the top anchor along the last segment's slope. */
    const admit_lut_row *a = &PN_ADMIT_LUT[PN_ADMIT_LUT_N - 2];
    const admit_lut_row *b = &PN_ADMIT_LUT[PN_ADMIT_LUT_N - 1];
    uint64_t span = (uint64_t)(b->tokens - a->tokens);
    uint64_t rise = b->peak_bytes - a->peak_bytes;
    uint64_t off  = (uint64_t)(n_prompt - b->tokens);
    return b->peak_bytes + rise * off / span;
}

uint64_t pn_admit_kv_per_token(uint64_t kv_per_token, bool kv_q4)
{
    if (kv_per_token > 0)
        return kv_per_token;
    return kv_q4 ? PN_ADMIT_KV_BYTES_PER_TOKEN_Q4
                 : PN_ADMIT_KV_BYTES_PER_TOKEN_FP32;
}

uint64_t pn_admit_die_kv_bytes(uint64_t kv_per_token, bool kv_q4, int max_ctx,
                               int n_layers_here, int n_layer)
{
    if (max_ctx <= 0)
        return 0;
    uint64_t per_tok = pn_admit_kv_per_token(kv_per_token, kv_q4);
    uint64_t whole = per_tok * (uint64_t)max_ctx;
    if (n_layer <= 0 || n_layers_here <= 0)
        return whole;                 /* layout unknown — charge all of it */
    if (n_layers_here >= n_layer)
        return whole;                 /* this die holds them all */
    return whole * (uint64_t)n_layers_here / (uint64_t)n_layer;
}

uint64_t pn_admit_host_cost(int n_prompt, bool kv_q4, uint64_t kv_per_token)
{
    if (n_prompt <= 0)
        return 0;
    uint64_t per_tok = pn_admit_kv_per_token(kv_per_token, kv_q4);
    uint64_t kv = per_tok * (uint64_t)n_prompt;
    return kv + lut_activation_peak(n_prompt);
}

uint64_t pn_admit_die_cost(int n_prompt, bool kv_q4, uint64_t kv_per_token)
{
    if (n_prompt <= 0)
        return 0;
    /* On-die KV mirror grows the same per token; the flash-attention scratch is
     * bounded (tiled), so the VRAM growth is dominated by the KV term. Use the
     * same per-token KV term plus a modest activation-scratch slice. */
    uint64_t per_tok = pn_admit_kv_per_token(kv_per_token, kv_q4);
    uint64_t kv = per_tok * (uint64_t)n_prompt;
    return kv + lut_activation_peak(n_prompt) / 4;   /* die scratch is smaller */
}

const char *pn_admit_verdict_name(pn_admit_verdict v)
{
    switch (v) {
    case PN_ADMIT_OK:     return "ADMIT";
    case PN_ADMIT_CAP:    return "CAP";
    case PN_ADMIT_REJECT: return "REJECT";
    }
    return "?";
}

/* Largest prompt length whose host cost fits under `budget`, by walking down
 * from n_prompt in coarse steps then refining. Returns >=1 if any window fits,
 * 0 if even a minimal window won't. */
static int host_fit_window(int n_prompt, bool kv_q4, uint64_t kv_per_token,
                           uint64_t budget)
{
    if (pn_admit_host_cost(n_prompt, kv_q4, kv_per_token) <= budget)
        return n_prompt;

    /* Binary search the largest fitting length in [1, n_prompt). */
    int lo = 1, hi = n_prompt, best = 0;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (pn_admit_host_cost(mid, kv_q4, kv_per_token) <= budget) {
            best = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return best;
}

void pn_admit_request(int n_prompt, bool kv_q4, uint64_t kv_per_token,
                      bool gpu_resident, int die_id, int max_window,
                      const pn_mem_state *state, pn_admit_result *out)
{
    out->verdict = PN_ADMIT_OK;
    out->fit_tokens = n_prompt;
    out->reason = "admitted";
    out->host_need_bytes = pn_admit_host_cost(n_prompt, kv_q4, kv_per_token);
    out->host_avail_bytes = state ? state->host_avail_bytes : 0;

    if (n_prompt <= 0 || !state)
        return;

    /* --- Host stall guard: if the box is already thrashing, do not pile on.
     * Sustained, meaningful pressure only: avg10 decays over ~a minute, so any
     * transient stall (e.g. page-cache churn right after a ~90 s model load)
     * leaves it slightly > 0 long after the box recovered — rejecting on > 0
     * 503s every healthy request for that whole decay window. Threshold via
     * PN_PSI_FULL_REJECT (default PN_PSI_FULL_REJECT_DEFAULT); sub-threshold
     * pressure falls through to the byte math below, which still refuses any
     * request that would not fit. --- */
    if (state->host_psi_full_avg10 >= pn_psi_full_reject_pct()) {
        out->verdict = PN_ADMIT_REJECT;
        out->reason = "host under sustained memory stall (PSI full) — retry shortly";
        return;
    }

    /* --- Host headroom (§6.2, §6.3): MemAvailable minus the reserve. --- */
    uint64_t reserve = pn_host_reserve_bytes(state->host_total_bytes);
    uint64_t host_budget = state->host_avail_bytes > reserve
                               ? state->host_avail_bytes - reserve : 0;

    /* --- GPU-resident path (DECIDED 2026-07-09, measured on a K80 host): per-request
     * prefill host cost is ~0 — the O(N^2) scores live on-die in bounded flash
     * tiles, and host + die KV are preallocated at max_seq by the load
     * admission, so neither the host LUT nor the per-die growth math applies.
     * Admit the FULL context; CAP only past max_window (the QWEN35_MAX_SEQ
     * mirror — the one length that triggers the host-softmax fallback). Real
     * host pressure still refuses: the PSI stall guard above, and MemAvailable
     * already inside the reserve here. --- */
    if (gpu_resident) {
        out->host_need_bytes = 0;
        if (host_budget == 0) {
            out->verdict = PN_ADMIT_REJECT;
            out->reason = "host inside its memory reserve — retry shortly";
            return;
        }
        if (max_window > 0 && n_prompt > max_window) {
            out->verdict = PN_ADMIT_CAP;
            out->fit_tokens = max_window;
            out->reason = "prompt exceeds the max sequence window";
        }
        return;                                 /* ADMIT full context */
    }

    /* --- Die headroom (§6.2): the model's die free VRAM, if reporting. --- */
    uint64_t die_need = pn_admit_die_cost(n_prompt, kv_q4, kv_per_token);
    uint64_t die_free = 0;
    bool have_die = (die_id >= 0 && die_id < state->n_dies &&
                     state->per_die_budget[die_id] > 0);
    if (have_die)
        die_free = state->per_die_free[die_id];

    bool host_fits = out->host_need_bytes <= host_budget;
    bool die_fits = !have_die || die_need <= die_free;

    if (host_fits && die_fits) {
        /* Memory holds it, but the configured window still binds: a CPU-resident
         * model is capped at its --ctx-cap like a GPU one (H2d: a 5,000-token
         * prompt at cap 4096 was answered on the Mac). */
        if (max_window > 0 && n_prompt > max_window) {
            out->verdict = PN_ADMIT_CAP;
            out->fit_tokens = max_window;
            out->reason = "prompt exceeds the max sequence window";
        }
        return;                                 /* ADMIT */
    }

    /* --- Doesn't fit at N. Can a smaller window fit host + die? --- */
    int host_win = host_fits ? n_prompt
                             : host_fit_window(n_prompt, kv_q4, kv_per_token,
                                               host_budget);
    int die_win = n_prompt;
    if (have_die && !die_fits) {
        /* Die cost is ~linear in the KV term; solve for the fitting length. */
        uint64_t per_tok = pn_admit_kv_per_token(kv_per_token, kv_q4);
        die_win = per_tok ? (int)(die_free / per_tok) : 0;
    }
    int fit = host_win < die_win ? host_win : die_win;
    if (max_window > 0 && fit > max_window)
        fit = max_window;

    if (fit >= 1) {
        out->verdict = PN_ADMIT_CAP;
        out->fit_tokens = fit;
        out->reason = "prompt capped to the fitting context window";
    } else {
        out->verdict = PN_ADMIT_REJECT;
        out->reason = "server memory-constrained — retry shortly";
    }
}

bool pn_admit_model_load(uint64_t weights_bytes, int max_ctx, bool kv_q4,
                         uint64_t kv_per_token, uint64_t die_free_bytes,
                         uint64_t *need_out)
{
    uint64_t per_tok = pn_admit_kv_per_token(kv_per_token, kv_q4);
    uint64_t kv = (max_ctx > 0) ? per_tok * (uint64_t)max_ctx : 0;

    /* Activation reserve: the transient peak at the max context we might serve,
     * so a later long prefill on this die still has room. */
    uint64_t activation_reserve = lut_activation_peak(max_ctx > 0 ? max_ctx : 0);

    uint64_t die_reserve = pn_die_reserve_bytes();
    uint64_t need = weights_bytes + kv + activation_reserve;
    if (need_out)
        *need_out = need;

    /* Hold the per-die reserve back from the die's free VRAM. */
    uint64_t usable = die_free_bytes > die_reserve ? die_free_bytes - die_reserve : 0;
    return need <= usable;
}

uint64_t pn_admit_die_scratch_bytes(int max_ctx)
{
    uint64_t scratch = PN_ADMIT_DIE_SCRATCH_BASE_BYTES;
    if (max_ctx > 0)
        scratch += PN_ADMIT_DIE_SCRATCH_BYTES_PER_TOKEN * (uint64_t)max_ctx;
    /* Never charge more than the flat constant this replaces: past the window
     * it was calibrated at we have no measurement, and the slope is sized for
     * the default session count. */
    if (scratch > PN_ADMIT_DIE_SCRATCH_MAX_BYTES)
        scratch = PN_ADMIT_DIE_SCRATCH_MAX_BYTES;
    return scratch;
}

/* The projection both GPU load gates share: this die's weight slice + this
 * die's KV at the capped window + the per-die GPU scratch. Deliberately NO host
 * activation LUT — that models the CPU forward, and charging it to an
 * accelerator load is what refused the deployed config at --ctx-cap 131072. Says nothing about
 * the per-die reserve: that is the caller's question, and mixing the two up is
 * what charged it twice. */
static uint64_t gpu_die_load_need(uint64_t weights_bytes, uint64_t kv_bytes,
                                  int max_ctx)
{
    return weights_bytes + kv_bytes + pn_admit_die_scratch_bytes(max_ctx);
}

bool pn_admit_model_load_die(uint64_t weights_bytes, uint64_t kv_bytes,
                             int max_ctx, uint64_t die_free_bytes,
                             uint64_t *need_out)
{
    uint64_t need = gpu_die_load_need(weights_bytes, kv_bytes, max_ctx);
    if (need_out)
        *need_out = need;

    uint64_t die_reserve = pn_die_reserve_bytes();
    uint64_t usable = die_free_bytes > die_reserve ? die_free_bytes - die_reserve : 0;
    return need <= usable;
}

bool pn_admit_model_load_die_budgeted(uint64_t weights_bytes, uint64_t kv_bytes,
                                      int max_ctx,
                                      uint64_t die_budget_free_bytes,
                                      uint64_t *need_out)
{
    uint64_t need = gpu_die_load_need(weights_bytes, kv_bytes, max_ctx);
    if (need_out)
        *need_out = need;
    /* No reserve subtraction: the caller's figure is already net of it. */
    return need <= die_budget_free_bytes;
}

uint64_t pn_load_spike_bytes(void)
{
    const char *env = getenv("PN_LOAD_SPIKE_BYTES");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        /* Garbage or zero falls back to the default — a zero budget would
         * disarm the one gate that models the real 8 GB-box squeeze. */
        if (end != env && v > 0)
            return (uint64_t)v;
    }
    return PN_ADMIT_LOAD_SPIKE_BYTES_DEFAULT;
}

bool pn_admit_load_spike(uint64_t host_avail_bytes, uint64_t host_total_bytes,
                         uint64_t *spike_out)
{
    uint64_t spike = pn_load_spike_bytes();
    if (spike_out)
        *spike_out = spike;

    uint64_t reserve = pn_host_reserve_bytes(host_total_bytes);
    uint64_t budget = host_avail_bytes > reserve ? host_avail_bytes - reserve : 0;
    return spike <= budget;
}
