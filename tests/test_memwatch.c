/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_memwatch.c — Phase 1 MemoryAware unit tests (host signals + memwatch
 * publish/read + level derivation + reserve policy). Self-contained: no model,
 * no GPU, no external test harness — links only the mem module + libm, so it
 * builds and runs on the Mac (the host-signal path is fully exercised there).
 *
 * Covers:
 *   - pn_host_signals_sample fills a plausible total/avail on this host
 *   - pn_host_reserve_bytes honors the max(1GB,15%) policy + env override
 *   - pn_die_reserve_bytes default + env override
 *   - seqlock read returns a self-consistent snapshot
 *   - the die feed republishes and drives per-die level escalation
 *   - level names are total
 *   - path-aware request admission (GPU-resident admits full context up to
 *     max_seq, CAPs past it; CPU path keeps the conservative LUT)
 *   - the model-load host-spike budget (PN_LOAD_SPIKE_BYTES) refuses a
 *     squeezed load
 */

#include "../src/core/mem/pn_memwatch.h"
#include "../src/core/mem/pn_host_signals.h"
#include "../src/core/mem/pn_admission.h"
#include "../src/core/mem/pn_ladder.h"
#include "../src/serve/signal_handlers.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* signal_handlers.c calls into the serve registry on shutdown. Linking the real
 * one would drag in batch_sched + the registry client and break this target's
 * "no model, no GPU, builds on the Mac" contract, so the CPU-only test provides
 * the one symbol itself. There is nothing registered in this process to
 * deregister. (This target did not link at all before — Phase 10.) */
void serve_registry_deregister_all(void);
void serve_registry_deregister_all(void) { }

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

static void test_host_sample(void)
{
    printf("host_sample:\n");
    pn_host_sample s;
    pn_host_signals_sample(&s);
    /* Every platform we build on has physical RAM; total must be non-zero and
     * available must not exceed total. */
    CHECK(s.total_bytes > 0, "total RAM should be > 0");
    CHECK(s.avail_bytes <= s.total_bytes, "avail should not exceed total");
    printf("    total=%llu MiB avail=%llu MiB psi_some=%.2f\n",
           (unsigned long long)(s.total_bytes >> 20),
           (unsigned long long)(s.avail_bytes >> 20), s.psi_some_avg10);
}

static void test_reserve_policy(void)
{
    printf("reserve_policy:\n");
    unsetenv("PN_HOST_RESERVE_BYTES");
    unsetenv("PN_DIE_RESERVE_BYTES");

    /* 8 GiB host: 15% = ~1.2 GiB, above the 1 GiB floor -> fraction wins. */
    uint64_t eight_gib = 8ULL << 30;
    uint64_t r8 = pn_host_reserve_bytes(eight_gib);
    CHECK(r8 == (uint64_t)(0.15 * (double)eight_gib), "8GiB reserve = 15%");
    CHECK(r8 > (1ULL << 30), "8GiB reserve above the 1GiB floor");

    /* 2 GiB host: 15% = ~300 MiB, below the floor -> floor wins. */
    uint64_t two_gib = 2ULL << 30;
    uint64_t r2 = pn_host_reserve_bytes(two_gib);
    CHECK(r2 == (1ULL << 30), "small host clamps to the 1GiB floor");

    /* Env override takes precedence regardless of total. */
    setenv("PN_HOST_RESERVE_BYTES", "500000000", 1);
    CHECK(pn_host_reserve_bytes(eight_gib) == 500000000ULL, "host env override");
    unsetenv("PN_HOST_RESERVE_BYTES");

    CHECK(pn_die_reserve_bytes() == PN_DIE_RESERVE_BYTES, "die reserve default");
    setenv("PN_DIE_RESERVE_BYTES", "2147483648", 1);
    CHECK(pn_die_reserve_bytes() == 2147483648ULL, "die env override");
    unsetenv("PN_DIE_RESERVE_BYTES");

    /* PSI-full reject threshold: default, env override, garbage/zero fall back
     * (a zero threshold would re-arm the transient-blip hypersensitivity). */
    unsetenv("PN_PSI_FULL_REJECT");
    CHECK(pn_psi_full_reject_pct() == PN_PSI_FULL_REJECT_DEFAULT,
          "psi reject default");
    setenv("PN_PSI_FULL_REJECT", "25.5", 1);
    CHECK(pn_psi_full_reject_pct() == 25.5, "psi reject env override");
    setenv("PN_PSI_FULL_REJECT", "0", 1);
    CHECK(pn_psi_full_reject_pct() == PN_PSI_FULL_REJECT_DEFAULT,
          "psi reject zero falls back to default");
    setenv("PN_PSI_FULL_REJECT", "garbage", 1);
    CHECK(pn_psi_full_reject_pct() == PN_PSI_FULL_REJECT_DEFAULT,
          "psi reject garbage falls back to default");
    unsetenv("PN_PSI_FULL_REJECT");
}

static void test_publish_read(void)
{
    printf("publish_read:\n");
    /* A synchronous poll publishes host truth without starting the thread. */
    pn_memwatch_poll_now();

    pn_mem_state a, b;
    pn_memwatch_read(&a);
    CHECK(a.host_total_bytes > 0, "read sees published host total");

    /* seq is monotonic across a second poll. */
    pn_memwatch_poll_now();
    pn_memwatch_read(&b);
    CHECK(b.seq > a.seq, "seq advances on republish");
}

static void test_die_feed_and_level(void)
{
    printf("die_feed_and_level:\n");
    pn_memwatch_poll_now();

    uint64_t budget = 11ULL << 30;   /* ~11 GiB usable on a 12 GiB die */

    /* Plenty free -> die contributes OK. */
    pn_memwatch_set_die(0, 10ULL << 30, budget, 1ULL << 30);
    pn_mem_state s;
    pn_memwatch_read(&s);
    CHECK(s.n_dies >= 1, "die 0 registered");
    CHECK(s.per_die_budget[0] == budget, "die 0 budget stored");
    CHECK(s.per_die_used[0] == (1ULL << 30), "die 0 used stored");

    /* Nearly exhausted die -> level escalates to at least TIGHT. */
    pn_memwatch_set_die(0, budget / 32, budget, budget);
    pn_memwatch_read(&s);
    CHECK(s.level >= PN_MEM_LEVEL_TIGHT, "near-empty die escalates level");

    /* Fully exhausted die -> CRITICAL. */
    pn_memwatch_set_die(0, 0, budget, budget);
    pn_memwatch_read(&s);
    CHECK(s.level == PN_MEM_LEVEL_CRITICAL, "empty die is CRITICAL");

    /* Out-of-range die id is a no-op (no crash, no n_dies growth past 16). */
    pn_memwatch_set_die(999, 0, budget, 0);
    pn_memwatch_read(&s);
    CHECK(s.n_dies <= PN_MEM_MAX_DIES, "out-of-range die id ignored");
}

/* --- Phase 10: PSI noise floor + escalation dwell -------------------------- */

/* A state with ample byte headroom and no die reporting, so the ONLY thing that
 * can move the level is PSI. 8 GiB total, 6 GiB available — miles above a
 * max(1 GiB, 15%) = ~1.2 GiB reserve. Mirrors a healthy K80 host. */
static pn_mem_state healthy_state(double psi_some, double psi_full)
{
    pn_mem_state s = {0};
    s.host_total_bytes = 8ULL << 30;
    s.host_avail_bytes = 6ULL << 30;
    s.host_psi_some_avg10 = psi_some;
    s.host_psi_full_avg10 = psi_full;
    return s;
}

static void test_psi_noise_floor(void)
{
    printf("psi_noise_floor:\n");
    unsetenv("PN_PSI_FULL_REJECT");
    unsetenv("PN_PSI_FULL_TIGHT");
    unsetenv("PN_PSI_FULL_WATCH");
    unsetenv("PN_PSI_SOME_TIGHT");
    unsetenv("PN_PSI_SOME_WATCH");
    unsetenv("PN_HOST_RESERVE_BYTES");

    /* Hard signals alone say OK on a healthy box, at any PSI. */
    pn_mem_state s = healthy_state(0.2, 0.2);
    CHECK(pn_mem_hard_level(&s) == PN_MEM_LEVEL_OK,
          "6 GiB avail of 8 GiB is OK on bytes");

    /* THE BUG: the exact K80-host reading that tripped TIGHT on a healthy serve
     * (2026-07-27, psi some 0.2 full 0.2 at 5578 MiB avail). 0.2% full-stall
     * avg10 = 20 ms out of 10 s — noise, not pressure. */
    CHECK(pn_mem_psi_level(&s) == PN_MEM_LEVEL_OK,
          "psi(some 0.2 full 0.2) is below the noise floor");

    /* And the recovery reading is obviously OK too. */
    s = healthy_state(0.0, 0.0);
    CHECK(pn_mem_psi_level(&s) == PN_MEM_LEVEL_OK, "psi(0,0) is OK");

    /* Proportionate escalation above the floor. */
    s = healthy_state(0.0, 1.5);
    CHECK(pn_mem_psi_level(&s) == PN_MEM_LEVEL_WATCH, "full 1.5% -> WATCH");
    s = healthy_state(0.0, 6.0);
    CHECK(pn_mem_psi_level(&s) == PN_MEM_LEVEL_TIGHT, "full 6% -> TIGHT");
    s = healthy_state(0.0, 20.0);
    CHECK(pn_mem_psi_level(&s) == PN_MEM_LEVEL_CRITICAL, "full 20% -> CRITICAL");

    /* "some" pressure is normal on any busy box; it needs a floor too. */
    s = healthy_state(0.2, 0.0);
    CHECK(pn_mem_psi_level(&s) == PN_MEM_LEVEL_OK, "some 0.2% is noise");
    s = healthy_state(6.0, 0.0);
    CHECK(pn_mem_psi_level(&s) == PN_MEM_LEVEL_WATCH, "some 6% -> WATCH");
    s = healthy_state(12.0, 0.0);
    CHECK(pn_mem_psi_level(&s) == PN_MEM_LEVEL_TIGHT, "some 12% -> TIGHT");

    /* Thresholds are env-tunable, and garbage/zero falls back to the default. */
    setenv("PN_PSI_FULL_TIGHT", "0.1", 1);
    s = healthy_state(0.0, 0.2);
    CHECK(pn_mem_psi_level(&s) == PN_MEM_LEVEL_TIGHT,
          "PN_PSI_FULL_TIGHT override honored");
    setenv("PN_PSI_FULL_TIGHT", "0", 1);
    CHECK(pn_psi_full_tight_pct() == PN_PSI_FULL_TIGHT_DEFAULT,
          "zero threshold falls back to the default");
    unsetenv("PN_PSI_FULL_TIGHT");

    /* HARD signals must still escalate regardless of PSI: real byte exhaustion
     * is not noise. This is the "genuine pressure still protects" case. */
    s = healthy_state(0.0, 0.0);
    s.host_avail_bytes = 512ULL << 20;      /* inside a ~1.2 GiB reserve */
    CHECK(pn_mem_hard_level(&s) == PN_MEM_LEVEL_CRITICAL,
          "avail inside the reserve is CRITICAL on bytes alone");
    s.host_avail_bytes = (uint64_t)(1.35 * (1ULL << 30));
    CHECK(pn_mem_hard_level(&s) >= PN_MEM_LEVEL_TIGHT,
          "avail just above the reserve is TIGHT on bytes alone");

    /* A near-empty die escalates on hard evidence too, PSI silent. */
    s = healthy_state(0.0, 0.0);
    s.n_dies = 1;
    s.per_die_budget[0] = 11ULL << 30;
    s.per_die_free[0]   = (11ULL << 30) / 32;
    CHECK(pn_mem_hard_level(&s) >= PN_MEM_LEVEL_TIGHT,
          "near-empty die escalates hard level with PSI at zero");
}

static void test_psi_needs_corroboration(void)
{
    printf("psi_needs_corroboration:\n");

    /* Measured on a K80 host during a 3-model load: full-PSI 6.0% (over the TIGHT
     * threshold) with host avail 6338 MiB of 7845 — 81% of the reserve headroom
     * still free. Page-cache reclaim from streaming 17 GB of GGUF, not
     * exhaustion. PSI alone must not push past WATCH here. */
    CHECK(pn_mem_combine_level(PN_MEM_LEVEL_OK, PN_MEM_LEVEL_TIGHT)
              == PN_MEM_LEVEL_WATCH,
          "uncorroborated PSI-TIGHT is capped at WATCH");

    /* WATCH on its own is allowed through — rung 4a only, near-zero cost. */
    CHECK(pn_mem_combine_level(PN_MEM_LEVEL_OK, PN_MEM_LEVEL_WATCH)
              == PN_MEM_LEVEL_WATCH, "PSI-WATCH passes through");

    /* Once the bytes corroborate, PSI escalates in full — that is what a
     * leading indicator is FOR. */
    CHECK(pn_mem_combine_level(PN_MEM_LEVEL_WATCH, PN_MEM_LEVEL_TIGHT)
              == PN_MEM_LEVEL_TIGHT,
          "PSI-TIGHT escalates once bytes show a squeeze");

    /* Whole-box thrash is never capped: at that point MemAvailable's opinion
     * is irrelevant. */
    CHECK(pn_mem_combine_level(PN_MEM_LEVEL_OK, PN_MEM_LEVEL_CRITICAL)
              == PN_MEM_LEVEL_CRITICAL,
          "PSI-CRITICAL is never capped by healthy bytes");

    /* Hard evidence always wins outright, PSI silent. */
    CHECK(pn_mem_combine_level(PN_MEM_LEVEL_CRITICAL, PN_MEM_LEVEL_OK)
              == PN_MEM_LEVEL_CRITICAL, "hard CRITICAL published with PSI at 0");
    CHECK(pn_mem_combine_level(PN_MEM_LEVEL_TIGHT, PN_MEM_LEVEL_OK)
              == PN_MEM_LEVEL_TIGHT, "hard TIGHT published with PSI at 0");
    CHECK(pn_mem_combine_level(PN_MEM_LEVEL_OK, PN_MEM_LEVEL_OK)
              == PN_MEM_LEVEL_OK, "quiet box is OK");
}

static void test_psi_dwell(void)
{
    printf("psi_dwell:\n");
    unsetenv("PN_PSI_TRIP_POLLS");
    int need = pn_psi_trip_polls();
    CHECK(need >= 2, "default dwell is more than one sample");

    pn_psi_dwell d = { 0, PN_MEM_LEVEL_OK };

    /* A single elevated sample must NOT engage a rung. */
    pn_psi_dwell_update(&d, PN_MEM_LEVEL_TIGHT);
    CHECK(d.effective == PN_MEM_LEVEL_OK, "one TIGHT sample does not trip");

    /* Nor does an on/off flap — the counter resets on every quiet sample. */
    for (int i = 0; i < 20; i++) {
        pn_psi_dwell_update(&d, PN_MEM_LEVEL_TIGHT);
        pn_psi_dwell_update(&d, PN_MEM_LEVEL_OK);
        CHECK(d.effective == PN_MEM_LEVEL_OK, "flapping PSI never trips");
    }

    /* Sustained pressure DOES engage, after the dwell. */
    for (int i = 0; i < need; i++)
        pn_psi_dwell_update(&d, PN_MEM_LEVEL_TIGHT);
    CHECK(d.effective == PN_MEM_LEVEL_TIGHT, "sustained TIGHT engages");

    /* Relax is immediate — this is what lets a clamp release. */
    pn_psi_dwell_update(&d, PN_MEM_LEVEL_OK);
    CHECK(d.effective == PN_MEM_LEVEL_OK, "relax is immediate (no latch)");
    CHECK(d.trips == 0, "relax clears the counter");

    /* CRITICAL bypasses the dwell entirely — never delay real protection. */
    pn_psi_dwell d2 = { 0, PN_MEM_LEVEL_OK };
    pn_psi_dwell_update(&d2, PN_MEM_LEVEL_CRITICAL);
    CHECK(d2.effective == PN_MEM_LEVEL_CRITICAL,
          "CRITICAL engages on the first sample");

    /* NULL is a no-op, not a crash. */
    pn_psi_dwell_update(NULL, PN_MEM_LEVEL_CRITICAL);
}

static void test_level_names(void)
{
    printf("level_names:\n");
    CHECK(strcmp(pn_mem_level_name(PN_MEM_LEVEL_OK), "OK") == 0, "OK name");
    CHECK(strcmp(pn_mem_level_name(PN_MEM_LEVEL_CRITICAL), "CRITICAL") == 0,
          "CRITICAL name");
}

static void test_thread_lifecycle(void)
{
    printf("thread_lifecycle:\n");
    CHECK(!pn_memwatch_running(), "not running before start");
    CHECK(pn_memwatch_start() == 0, "start succeeds");
    CHECK(pn_memwatch_running(), "running after start");
    CHECK(pn_memwatch_start() == 0, "second start is an idempotent no-op");
    pn_memwatch_stop();
    CHECK(!pn_memwatch_running(), "stopped after stop");
    pn_memwatch_stop();   /* double stop is safe */
}

/* --- Phase 2: admission cost model + gates --- */

static void test_admit_cost_monotonic(void)
{
    printf("admit_cost_monotonic:\n");
    /* Host cost must rise strictly with prompt length across the LUT range and
     * beyond it (the O(N^2)-ish activation peak is the whole point). */
    uint64_t prev = 0;
    int lens[] = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
    for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); i++) {
        uint64_t c = pn_admit_host_cost(lens[i], true, 0);
        CHECK(c > prev, "host cost strictly increasing");
        prev = c;
    }
    /* 32k must project into multi-GiB territory — that is what OOM-rebooted an
     * 8 GB box, so admission must see it as huge. */
    CHECK(pn_admit_host_cost(32768, true, 0) > (4ULL << 30),
          "32k host cost is multi-GiB");
    CHECK(pn_admit_host_cost(0, true, 0) == 0, "zero-length costs nothing");
}

/* KV Phase 1: admission must price against the model's REAL per-token KV, not
 * the generic 5898-byte constant. Ornith's true geometry (verified on a K80 host):
 * 8 full-attention layers x 4 KV heads x 2(K+V) x 144-byte Q4 row = 9216
 * bytes/token. A dense fp32 model with the same head geometry is far larger.
 * Verify the geometry value drives the cost, and that 0 falls back to the
 * generic constant (never zero-priced). */
static void test_admit_kv_geometry(void)
{
    printf("admit_kv_geometry:\n");
    const uint64_t ORNITH_Q4_PER_TOK = 8ULL * 4 * 2 * 144;  /* 9216 */
    CHECK(ORNITH_Q4_PER_TOK == 9216, "ornith Q4 KV/token is 9216 bytes");

    /* The resolver returns the geometry value when given, else the constant. */
    CHECK(pn_admit_kv_per_token(ORNITH_Q4_PER_TOK, true) == ORNITH_Q4_PER_TOK,
          "geometry value wins over the generic constant");
    CHECK(pn_admit_kv_per_token(0, true) == PN_ADMIT_KV_BYTES_PER_TOKEN_Q4,
          "0 falls back to the generic Q4 constant (never zero-priced)");
    CHECK(pn_admit_kv_per_token(0, false) == PN_ADMIT_KV_BYTES_PER_TOKEN_FP32,
          "0 falls back to the generic fp32 constant");

    /* At a 40K cap, ornith's on-die KV growth is ~368 MiB — comfortably within
     * the die's ~4.8 GiB free budget (the design's single-tenant headroom). */
    uint64_t kv_40k = ORNITH_Q4_PER_TOK * 40000ULL;
    CHECK(kv_40k < (512ULL << 20), "40K ornith KV under 512 MiB");
    CHECK(kv_40k > (256ULL << 20), "40K ornith KV over 256 MiB (real, not tiny)");

    /* The die-cost must scale with the geometry per-token: the ornith value
     * yields a materially different (larger) die cost than the generic constant
     * at the same length — the mispricing the fix removes. */
    uint64_t die_geom = pn_admit_die_cost(10000, true, ORNITH_Q4_PER_TOK);
    uint64_t die_const = pn_admit_die_cost(10000, true, 0);
    CHECK(die_geom > die_const, "geometry prices ornith above the generic const");
}

static void test_admit_request_verdicts(void)
{
    printf("admit_request_verdicts:\n");
    unsetenv("PN_HOST_RESERVE_BYTES");

    pn_mem_state st = {0};
    st.host_total_bytes = 8ULL << 30;   /* an 8 GiB host */
    st.n_dies = 0;                      /* host-only check */

    pn_admit_result r;

    /* Ample free RAM, small prompt -> ADMIT. */
    st.host_avail_bytes = 6ULL << 30;
    pn_admit_request(1024, true, 0, false, -1, 0, &st, &r);
    CHECK(r.verdict == PN_ADMIT_OK, "small prompt with room -> ADMIT");

    /* The catastrophe case: 32k full-attention prefill on ~4 GiB free must NOT
     * be admitted as-is — it is refused or capped, never run whole. */
    st.host_avail_bytes = 4ULL << 30;
    pn_admit_request(32768, true, 0, false, -1, 0, &st, &r);
    CHECK(r.verdict != PN_ADMIT_OK, "32k prefill on a squeezed host is not ADMIT");
    printf("    32k verdict=%s fit=%d need=%llu MiB\n",
           pn_admit_verdict_name(r.verdict), r.fit_tokens,
           (unsigned long long)(r.host_need_bytes >> 20));

    /* A capped verdict must propose a strictly smaller, positive window. */
    if (r.verdict == PN_ADMIT_CAP)
        CHECK(r.fit_tokens > 0 && r.fit_tokens < 32768, "CAP window is smaller");

    /* PSI-full below the reject threshold is a transient blip (avg10 decays
     * over ~a minute after e.g. a model load) — healthy requests must still be
     * ADMITTED on the byte math, not blanket-503'd. */
    unsetenv("PN_PSI_FULL_REJECT");
    st.host_avail_bytes = 7ULL << 30;
    st.host_psi_full_avg10 = 3.5;
    pn_admit_request(1024, true, 0, false, -1, 0, &st, &r);
    CHECK(r.verdict == PN_ADMIT_OK, "transient PSI-full blip still admits");

    /* Sustained PSI-full at/above the threshold -> REJECT regardless of the
     * byte math: the box is genuinely thrashing. */
    st.host_psi_full_avg10 = PN_PSI_FULL_REJECT_DEFAULT + 5.0;
    pn_admit_request(1024, true, 0, false, -1, 0, &st, &r);
    CHECK(r.verdict == PN_ADMIT_REJECT, "sustained PSI-full rejects new work");

    /* The threshold is configurable: lower it under the blip and the same blip
     * now rejects. */
    setenv("PN_PSI_FULL_REJECT", "2.0", 1);
    st.host_psi_full_avg10 = 3.5;
    pn_admit_request(1024, true, 0, false, -1, 0, &st, &r);
    CHECK(r.verdict == PN_ADMIT_REJECT, "lowered threshold rejects the blip");
    unsetenv("PN_PSI_FULL_REJECT");
    st.host_psi_full_avg10 = 0.0;

    /* Die headroom: a nearly-full die caps/refuses even when host is fine. */
    st.host_avail_bytes = 7ULL << 30;
    st.n_dies = 1;
    st.per_die_budget[0] = 11ULL << 30;
    st.per_die_free[0]   = 64ULL << 20;   /* 64 MiB free — tiny */
    pn_admit_request(16384, true, 0, false, 0, 0, &st, &r);
    CHECK(r.verdict != PN_ADMIT_OK, "near-full die blocks a big prefill");
}

/* Path-aware admission (DECIDED 2026-07-09): a GPU-resident model has ~0
 * per-request host cost, so admission admits the FULL context up to max_window
 * and CAPs only past it — while real host pressure (sustained PSI, avail
 * inside the reserve) still refuses, and the CPU path keeps the LUT. */
static void test_admit_request_gpu_path(void)
{
    printf("admit_request_gpu_path:\n");
    unsetenv("PN_HOST_RESERVE_BYTES");
    unsetenv("PN_PSI_FULL_REJECT");

    pn_mem_state st = {0};
    st.host_total_bytes = 8ULL << 30;   /* an 8 GiB host */
    st.host_avail_bytes = 4ULL << 30;   /* squeezes the CPU LUT, fine for GPU */
    st.n_dies = 0;
    pn_admit_result r;

    /* The headline: a 32k prompt on a GPU-resident model ADMITs whole — the
     * same state the CPU path caps (test_admit_request_verdicts above). */
    pn_admit_request(32768, true, 0, true, -1, 32768, &st, &r);
    CHECK(r.verdict == PN_ADMIT_OK, "gpu 32k admits the full context");
    CHECK(r.fit_tokens == 32768, "gpu fit is the whole prompt");
    CHECK(r.host_need_bytes == 0, "gpu per-request host need is ~0");

    /* Past max_seq the flash->host-softmax fallback would fire: CAP there. */
    pn_admit_request(40000, true, 0, true, -1, 32768, &st, &r);
    CHECK(r.verdict == PN_ADMIT_CAP, "gpu caps past max_seq");
    CHECK(r.fit_tokens == 32768, "gpu cap lands exactly on max_seq");

    /* Sustained PSI stall still refuses regardless of path. */
    st.host_psi_full_avg10 = PN_PSI_FULL_REJECT_DEFAULT + 5.0;
    pn_admit_request(1024, true, 0, true, -1, 32768, &st, &r);
    CHECK(r.verdict == PN_ADMIT_REJECT, "gpu path rejects a sustained stall");
    st.host_psi_full_avg10 = 0.0;

    /* MemAvailable already inside the host reserve is genuine pressure. */
    st.host_avail_bytes = 512ULL << 20;
    pn_admit_request(1024, true, 0, true, -1, 32768, &st, &r);
    CHECK(r.verdict == PN_ADMIT_REJECT, "gpu path rejects inside the reserve");
}

/* Model-load host-spike budget: the ~2.4 GB GPU-upload transient (the real
 * 8 GB-box squeeze) must be absorbable on top of the host reserve. */
static void test_admit_load_spike(void)
{
    printf("admit_load_spike:\n");
    unsetenv("PN_LOAD_SPIKE_BYTES");
    unsetenv("PN_HOST_RESERVE_BYTES");
    uint64_t spike = 0;
    uint64_t total = 8ULL << 30;

    /* 6 GiB available minus the ~1.2 GiB reserve absorbs the ~2.4 GB spike. */
    CHECK(pn_admit_load_spike(6ULL << 30, total, &spike),
          "roomy host absorbs the load spike");
    CHECK(spike == PN_ADMIT_LOAD_SPIKE_BYTES_DEFAULT, "default spike budget");

    /* 3 GiB available leaves only ~1.8 GiB over the reserve — refuse. */
    CHECK(!pn_admit_load_spike(3ULL << 30, total, &spike),
          "squeezed host refuses the load spike");

    /* Env override: a 1 GiB budget fits where the default did not. */
    setenv("PN_LOAD_SPIKE_BYTES", "1073741824", 1);
    CHECK(pn_load_spike_bytes() == (1ULL << 30), "spike env override");
    CHECK(pn_admit_load_spike(3ULL << 30, total, &spike),
          "overridden spike fits the squeezed host");

    /* Garbage/zero overrides fall back to the default (never disarm). */
    setenv("PN_LOAD_SPIKE_BYTES", "0", 1);
    CHECK(pn_load_spike_bytes() == PN_ADMIT_LOAD_SPIKE_BYTES_DEFAULT,
          "zero spike falls back to default");
    setenv("PN_LOAD_SPIKE_BYTES", "garbage", 1);
    CHECK(pn_load_spike_bytes() == PN_ADMIT_LOAD_SPIKE_BYTES_DEFAULT,
          "garbage spike falls back to default");
    unsetenv("PN_LOAD_SPIKE_BYTES");
}

static void test_admit_model_load(void)
{
    printf("admit_model_load:\n");
    unsetenv("PN_DIE_RESERVE_BYTES");
    uint64_t need = 0;

    /* 16 GB model must NOT fit a 12 GB die. */
    bool fit = pn_admit_model_load(16ULL << 30, 4096, true, 0, 12ULL << 30, &need);
    CHECK(!fit, "16GB model refused on a 12GB die");
    CHECK(need > (16ULL << 30), "need includes weights + KV + activation");

    /* A 6 GB slice fits a 12 GB die with room for KV + reserve. */
    fit = pn_admit_model_load(6ULL << 30, 4096, true, 0, 12ULL << 30, &need);
    CHECK(fit, "6GB weights fit a 12GB die");
}

static void test_admit_model_load_die(void)
{
    printf("admit_model_load_die:\n");
    unsetenv("PN_DIE_RESERVE_BYTES");
    uint64_t need = 0;

    /* The working 2-die split (Q3_K_M @ layer 20, 32k q4 KV): ~8 GiB weights
     * + ~96 MiB KV on one ~11.2 GiB-free die must ADMIT — the gate must never
     * refuse the configuration that is known to run. */
    uint64_t die_free = 11200ULL << 20;
    bool fit = pn_admit_model_load_die(8ULL << 30, 96ULL << 20, die_free, &need);
    CHECK(fit, "working split half fits an 11.2GiB-free die");
    CHECK(need > (8ULL << 30), "need includes weights + KV + scratch");

    /* A lopsided split (e.g. 38/40 layers -> ~15 GiB on die A) must REFUSE. */
    fit = pn_admit_model_load_die(15ULL << 30, 91ULL << 20, die_free, &need);
    CHECK(!fit, "oversized split share refused");

    /* A die with stranded VRAM (only 4 GiB free) refuses the normal half. */
    fit = pn_admit_model_load_die(8ULL << 30, 96ULL << 20, 4ULL << 30, &need);
    CHECK(!fit, "stranded die refuses the normal half");
}

/* --- Phase 4: the reactive degradation ladder --- */

/* The ladder maps a pressure level onto concrete rungs. Reduce-only and
 * monotonic: OK mirrors config exactly; each higher level pulls the rungs
 * harder and never relaxes one a lower level engaged. */
static void test_ladder_plan(void)
{
    printf("ladder_plan:\n");
    unsetenv("PN_LADDER_DISABLE");

    const int   CFG_SLOTS  = 16;   /* PN_QWEN35MOE_GPU_MAX_SLOTS */
    const int   CFG_WINDOW = 32768;/* QWEN35_MAX_SEQ */
    const bool  CFG_Q4     = false;/* q4 not the operator default */
    pn_ladder_plan p;

    /* OK: byte-identical to config, nothing degraded — this is what makes
     * wiring the rungs low-risk on an unpressured box. */
    pn_ladder_plan_for(PN_MEM_LEVEL_OK, CFG_SLOTS, CFG_WINDOW, CFG_Q4, &p);
    CHECK(p.max_slots == CFG_SLOTS, "OK keeps full slots");
    CHECK(p.prefill_window == CFG_WINDOW, "OK keeps full window");
    CHECK(!p.force_kv_q4, "OK does not force q4");
    CHECK(!p.kv_evict, "OK does not evict");
    CHECK(!p.degraded, "OK is not degraded");

    /* WATCH: force q4 (cheap VRAM win) but no throughput/context hit yet. */
    pn_ladder_plan_for(PN_MEM_LEVEL_WATCH, CFG_SLOTS, CFG_WINDOW, CFG_Q4, &p);
    CHECK(p.force_kv_q4, "WATCH forces q4");
    CHECK(p.max_slots == CFG_SLOTS, "WATCH keeps full slots");
    CHECK(p.prefill_window == CFG_WINDOW, "WATCH keeps full window");
    CHECK(!p.kv_evict, "WATCH does not evict yet");
    CHECK(p.degraded, "WATCH is degraded (q4 forced)");

    /* TIGHT: shrink concurrency, evict, clamp the window. */
    pn_ladder_plan_for(PN_MEM_LEVEL_TIGHT, CFG_SLOTS, CFG_WINDOW, CFG_Q4, &p);
    CHECK(p.max_slots == CFG_SLOTS / 2, "TIGHT halves slots");
    CHECK(p.force_kv_q4, "TIGHT forces q4");
    CHECK(p.kv_evict, "TIGHT evicts");
    CHECK(p.prefill_window == CFG_WINDOW / 2, "TIGHT halves the window");
    CHECK(p.degraded, "TIGHT is degraded");

    /* CRITICAL: single slot, quartered window, q4 + evict. */
    pn_ladder_plan_for(PN_MEM_LEVEL_CRITICAL, CFG_SLOTS, CFG_WINDOW, CFG_Q4, &p);
    CHECK(p.max_slots == 1, "CRITICAL floors slots to 1");
    CHECK(p.prefill_window == CFG_WINDOW / 4, "CRITICAL quarters the window");
    CHECK(p.force_kv_q4 && p.kv_evict, "CRITICAL forces q4 + evict");
    CHECK(p.degraded, "CRITICAL is degraded");

    /* Monotonicity across the whole ladder: slots never rise, window never
     * rises, and once q4/evict engage they stay engaged. */
    int  prev_slots = CFG_SLOTS + 1, prev_win = CFG_WINDOW + 1;
    bool seen_q4 = false, seen_evict = false;
    pn_mem_level levels[] = { PN_MEM_LEVEL_OK, PN_MEM_LEVEL_WATCH,
                              PN_MEM_LEVEL_TIGHT, PN_MEM_LEVEL_CRITICAL };
    for (size_t i = 0; i < sizeof(levels) / sizeof(levels[0]); i++) {
        pn_ladder_plan_for(levels[i], CFG_SLOTS, CFG_WINDOW, CFG_Q4, &p);
        CHECK(p.max_slots <= prev_slots, "slots monotonically non-increasing");
        CHECK(p.prefill_window <= prev_win, "window monotonically non-increasing");
        CHECK(p.max_slots >= 1, "slots never drop below 1");
        if (seen_q4)    CHECK(p.force_kv_q4, "q4 stays engaged once on");
        if (seen_evict) CHECK(p.kv_evict, "evict stays engaged once on");
        seen_q4 = p.force_kv_q4; seen_evict = p.kv_evict;
        prev_slots = p.max_slots; prev_win = p.prefill_window;
    }
}

static void test_ladder_edges(void)
{
    printf("ladder_edges:\n");
    pn_ladder_plan p;

    /* An operator who already runs q4: force_kv_q4 stays true, and q4 alone no
     * longer counts as degradation at WATCH (it was already the config). */
    pn_ladder_plan_for(PN_MEM_LEVEL_WATCH, 16, 32768, true, &p);
    CHECK(p.force_kv_q4, "already-q4 config stays q4 under WATCH");
    CHECK(!p.degraded, "WATCH with q4-config is not degraded");

    /* No known base window (0): the rung-3 clamp is a no-op, stays 0. */
    pn_ladder_plan_for(PN_MEM_LEVEL_CRITICAL, 16, 0, false, &p);
    CHECK(p.prefill_window == 0, "no base window -> no clamp");

    /* A tiny slot config still floors at 1 rather than 0 under TIGHT. */
    pn_ladder_plan_for(PN_MEM_LEVEL_TIGHT, 1, 32768, false, &p);
    CHECK(p.max_slots == 1, "1-slot config floors at 1 under TIGHT");

    /* Disabled: mirror config at every level (the bare-Mac escape hatch). */
    setenv("PN_LADDER_DISABLE", "1", 1);
    CHECK(pn_ladder_disabled(), "PN_LADDER_DISABLE detected");
    pn_ladder_plan_for(PN_MEM_LEVEL_CRITICAL, 16, 32768, false, &p);
    CHECK(p.max_slots == 16, "disabled keeps full slots even at CRITICAL");
    CHECK(p.prefill_window == 32768, "disabled keeps full window");
    CHECK(!p.force_kv_q4 && !p.kv_evict && !p.degraded, "disabled = no rungs");
    unsetenv("PN_LADDER_DISABLE");

    /* from_state wrapper: derives the level from a snapshot; NULL == OK. */
    pn_mem_state st = {0};
    st.level = PN_MEM_LEVEL_TIGHT;
    pn_ladder_plan_from_state(&st, 16, 32768, false, &p);
    CHECK(p.max_slots == 8 && p.kv_evict, "from_state reads the snapshot level");
    pn_ladder_plan_from_state(NULL, 16, 32768, false, &p);
    CHECK(p.max_slots == 16 && !p.degraded, "NULL state treated as OK");
}

/* --- Phase 3: crash/shutdown signal handling --- */

static void test_signal_shutdown(void)
{
    printf("signal_shutdown:\n");
    /* Fork a child that installs the handlers and blocks; send SIGTERM and
     * confirm it terminates via the signal (the handler drains — no live CUDA
     * ctx in this CPU-only test — and re-raises SIG_DFL). The parent must not be
     * affected. This exercises install + the controlled-shutdown path end to
     * end without needing a GPU. */
    pid_t pid = fork();
    if (pid == 0) {
        pn_install_signal_handlers();
        for (;;)
            pause();               /* wait for the signal */
        _exit(0);                  /* unreached */
    }
    CHECK(pid > 0, "fork for signal test");
    if (pid <= 0)
        return;

    usleep(100000);                /* let the child install handlers */
    kill(pid, SIGTERM);

    int status = 0;
    waitpid(pid, &status, 0);
    CHECK(WIFSIGNALED(status), "child terminated by signal");
    if (WIFSIGNALED(status))
        CHECK(WTERMSIG(status) == SIGTERM, "child terminated by SIGTERM");
}

int main(void)
{
    printf("MemoryAware Phase 1-4 Tests\n");
    printf("===========================\n\n");

    test_host_sample();
    test_reserve_policy();
    test_publish_read();
    test_die_feed_and_level();
    test_psi_noise_floor();
    test_psi_needs_corroboration();
    test_psi_dwell();
    test_level_names();
    test_thread_lifecycle();
    test_admit_cost_monotonic();
    test_admit_kv_geometry();
    test_admit_request_verdicts();
    test_admit_request_gpu_path();
    test_admit_load_spike();
    test_admit_model_load();
    test_admit_model_load_die();
    test_ladder_plan();
    test_ladder_edges();
    test_signal_shutdown();

    printf("\n%s\n", g_failed == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_failed > 0 ? 1 : 0;
}
