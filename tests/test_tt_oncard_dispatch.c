/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_oncard_dispatch.c
 * @brief S2 on-card completion through the REAL dispatch recorder and replay builder
 *   (src/accel/tt/pn_tt_dispatch.c), no card: a fake card answers a synthetic decode token
 *   (a 4-core matmul whose members 1..3 go out as one multicast group, a 2-member eltwise with
 *   its rotating parameter block, a reduce, host data writes between them), the recorder
 *   learns it from forwards 4 and 5, and the program the builder uploads to the puller is
 *   captured and parsed.
 *
 *   Each arm is its own process (the knobs and the recorder are process state):
 *     off, off_nohoist   PN_TT_ONCARD_COMPLETION=0: the program must be BYTE-IDENTICAL to
 *                        the base code's (the golden digests below, from
 *                        tests/tt_oncard_golden.sh: this same file built against Pantheon
 *                        bdccd110, the S1 tip on main (with split-K) this work starts from);
 *     on, on_nohoist     =1 (with and without the hoist): sidecar entries in the program, one
 *                        realized AWAIT per original AWAIT or fewer, the builder's own plan
 *                        check passed (it refuses a failing plan and the order stands);
 *     red                =1 + PN_TT_RED=precondition_early: every entry one count short.
 *   `--arm NAME [--dump]` runs one arm and prints its census (and every entry); `--digest`
 *   prints the off arms' digests.
 * Needs USE_TT=1, NOT a card.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_TT

#include "../src/accel/tt/pn_tt_device_internal.h"
#include "../src/accel/tt/pn_tt_dispatch.h"
#include "tt/ft_tt_op_l1_layout.h"

/* The base code's knob-off programs (tests/tt_oncard_golden.sh). */
#define GOLDEN_OFF         0x1358e5072a5dbcceull
#define GOLDEN_OFF_NOHOIST 0xe192370aad35dc52ull
#define GOLDEN_HELD_OFF    0x3a4298df249d1c19ull

#define CORES 7u
#define QUEUE_WORDS (0x10000u / 4u)
#define PULLER_X 20u
#define PULLER_Y 20u
#define PROGRAM_BASE 0x26000u
#define PROGRAM_MAX (0x160000u - 0x26000u)

static uint32_t g_l1[CORES][QUEUE_WORDS];
static uint16_t g_x[CORES], g_y[CORES];
static uint32_t g_capacity[CORES];
static uint32_t g_nonce[CORES];
static uint32_t *g_program;
static size_t g_program_words;
/* HELD shape (arms "held_*"): every reduce publishes one more descriptor before its await,
 * and the fake card finishes that descriptor only at the core's NEXT completion read, so the
 * recorded await names a count below the queue's published one — the shape the card's
 * plain-decode replay showed on queue 0 once PN_TT_MATMUL_PULL_DEFER went to 0. */
static bool g_held;
static uint32_t g_held_seen[CORES];

static int fake_core(uint16_t x, uint16_t y)
{
    uint32_t c;

    for (c = 0; c < CORES; c++)
        if (g_x[c] == x && g_y[c] == y)
            return (int)c;
    return -1;
}

static int fake_write(void *user, uint16_t x, uint16_t y, uint64_t address, const void *src,
                      size_t bytes)
{
    const int c = fake_core(x, y);
    const uint64_t offset = address - FT_TT_OP_QUEUE_BASE;

    (void)user;
    if (x == PULLER_X && y == PULLER_Y && address == PROGRAM_BASE && bytes <= PROGRAM_MAX) {
        free(g_program);
        g_program = malloc(bytes);
        if (g_program)
            memcpy(g_program, src, bytes);
        g_program_words = g_program ? bytes / 4u : 0u;
    }
    if (c < 0 || address < FT_TT_OP_QUEUE_BASE || offset + bytes > 0x10000u)
        return FT_TT_OK; /* DRAM and the rest of L1 are not modelled */
    memcpy((uint8_t *)g_l1[c] + offset, src, bytes);
    return FT_TT_OK;
}

/** A completion read runs every published descriptor, as the writers would. */
static int fake_read(void *user, uint16_t x, uint16_t y, uint64_t address, void *dst,
                     size_t bytes)
{
    const int c = fake_core(x, y);
    const uint64_t offset = address - FT_TT_OP_QUEUE_BASE;

    (void)user;
    memset(dst, 0, bytes);
    if (c < 0 || address < FT_TT_OP_QUEUE_BASE || offset + bytes > 0x10000u)
        return FT_TT_OK;
    if (offset >= 0x10u && offset < 0x40u) {
        uint32_t *control = g_l1[c];

        while (control[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] < control[FT_TT_QUEUE_FIELD_HEAD]) {
            const uint32_t index = control[FT_TT_QUEUE_FIELD_COMPLETED_COUNT];
            const uint32_t *d = &control[(0x100u + (index & (control[1] - 1u)) * 32u) / 4u];

            if (d[5] == 1u && g_held_seen[c] != index + 1u) {
                g_held_seen[c] = index + 1u; /* held: finishes at the next read */
                break;
            }

            control[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] = d[1];
            control[FT_TT_QUEUE_FIELD_STATE] = FT_TT_QUEUE_STATE_IDLE;
            control[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] = index + 1u;
        }
    }
    memcpy(dst, (const uint8_t *)g_l1[c] + offset, bytes);
    return FT_TT_OK;
}

static pn_tt_device *fake_device(void)
{
    pn_tt_device *d = calloc(1, sizeof(*d));
    uint32_t c;

    if (!d)
        return NULL;
    d->runtime.noc.write = fake_write;
    d->runtime.noc.read = fake_read;
    d->runtime.bulk_l1.write = fake_write;
    d->runtime.bulk_l1.read = fake_read;
    d->runtime.bulk_dram.write = fake_write;
    d->runtime.bulk_dram.read = fake_read;
    /* The puller is "already armed": the builder uploads its program to the fake card. */
    d->puller.running = true;
    d->puller.core_x = PULLER_X;
    d->puller.core_y = PULLER_Y;
    for (c = 0; c < CORES; c++) {
        ft_tt_queue queue;

        /* cores 0..3: a matmul-like group, ring 8, members 1..3 sharing one nonce in a row
         * (td_find_groups multicasts them); 4, 5: eltwise; 6: reduce */
        g_x[c] = (uint16_t)(2u + c);
        g_y[c] = 3u;
        g_capacity[c] = c < 4u ? 8u : 64u;
        g_nonce[c] = c >= 1u && c <= 3u ? 0x51A7E000u : 0x1000u + 0x111u * c;
        if (ft_tt_queue_init(&d->runtime.noc, g_x[c], g_y[c], FT_TT_OP_QUEUE_BASE,
                             g_capacity[c], g_nonce[c], &queue) != FT_TT_OK)
            return NULL;
    }
    return d;
}

static uint32_t token_of(uint32_t nonce, uint32_t index)
{
    uint32_t token = nonce ^ ((index + 1u) * 0x9E3779B9u);

    if (token == 0u || token == (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON)
        token = nonce ^ 0xA5A5A5A5u;
    return token;
}

/** One descriptor on core c: its parameter block (rotating slot 0) when `rotating`, its ring
 *  slot, its head. */
static void issue_slot(pn_tt_device *d, uint32_t c, bool rotating, uint32_t a2, uint32_t a3)
{
    const uint32_t index = g_l1[c][FT_TT_QUEUE_FIELD_HEAD];
    uint32_t params[16], slot[8], head = index + 1u;
    uint32_t w;

    for (w = 0; w < 16u; w++)
        params[w] = 0x100u * c + w;
    if (rotating)
        (void)d->runtime.noc.write(d->runtime.noc.user, g_x[c], g_y[c],
                                   FT_TT_OP_QUEUE_BASE + 0x9000u, params, sizeof(params));
    memset(slot, 0, sizeof(slot));
    slot[0] = 0x09u;
    slot[1] = token_of(g_nonce[c], index);
    slot[2] = rotating ? FT_TT_OP_QUEUE_BASE + 0x9000u : FT_TT_OP_QUEUE_BASE + 0xA040u;
    slot[4] = a2;
    slot[5] = a3;
    (void)d->runtime.noc.write(d->runtime.noc.user, g_x[c], g_y[c],
                               FT_TT_OP_QUEUE_BASE + 0x100u + (index & (g_capacity[c] - 1u)) * 32u,
                               slot, sizeof(slot));
    (void)d->runtime.noc.write(d->runtime.noc.user, g_x[c], g_y[c], FT_TT_OP_QUEUE_BASE + 8u,
                               &head, sizeof(head));
}

static void issue(pn_tt_device *d, uint32_t c, bool rotating, uint32_t a2)
{
    issue_slot(d, c, rotating, a2, 0u);
}

static void await_core(pn_tt_device *d, uint32_t c)
{
    uint32_t window[4];

    (void)d->runtime.noc.read(d->runtime.noc.user, g_x[c], g_y[c], FT_TT_OP_QUEUE_BASE + 0x10u,
                              window, sizeof(window));
}

static void host_data_write(pn_tt_device *d)
{
    uint32_t words[16] = { 0 };

    (void)d->runtime.bulk_dram.write(d->runtime.bulk_dram.user, 30u, 1u, 0x100000u, words,
                                     sizeof(words));
}

/** One decode token's traffic: rounds of matmul -> eltwise -> reduce, data writes between. */
static void token(pn_tt_device *d)
{
    uint32_t round, c;

    for (round = 0; round < 6u; round++) {
        for (c = 0; c < 4u; c++)
            issue(d, c, false, round);
        await_core(d, 0u);
        issue(d, 4u, true, round);
        issue(d, 5u, true, round);
        await_core(d, 4u);
        issue(d, 6u, true, round);
        if (g_held) {
            /* the next descriptor goes out (held) before the await reads the count */
            uint32_t window[4];

            issue_slot(d, 6u, true, round, 1u);
            (void)d->runtime.noc.read(d->runtime.noc.user, g_x[6], g_y[6],
                                      FT_TT_OP_QUEUE_BASE + 0x10u, window, sizeof(window));
        } else {
            await_core(d, 6u);
        }
        if (round == 2u || round == 5u)
            host_data_write(d);
    }
}

static uint64_t fnv(const uint32_t *w, size_t n)
{
    uint64_t h = 1469598103934665603ull;
    size_t i;

    for (i = 0; i < n * 4u; i++)
        h = (h ^ ((const uint8_t *)w)[i]) * 1099511628211ull;
    return h;
}

/** Entries of each type in the program's segments; false if one does not parse. `dump`:
 *  print each entry (type, queue word, index / count) on its own line. */
static bool census(uint32_t *heads, uint32_t *awaits, uint32_t *preconds, uint32_t *rings,
                   uint32_t *precond_count_sum, bool dump)
{
    uint32_t s;

    *heads = *awaits = *preconds = *rings = *precond_count_sum = 0u;
    if (!g_program || g_program_words < 16u || g_program[0] != 0x524C5044u)
        return false;
    for (s = 0; s < g_program[4]; s++) {
        size_t at = g_program[g_program[5] / 4u + s] / 4u;

        for (;;) {
            uint32_t type, count, patches;

            if (at >= g_program_words)
                return false;
            type = g_program[at] & 0xFFu;
            count = (g_program[at] >> 8) & 0xFFFu;
            patches = g_program[at] >> 20;
            if (dump)
                printf("  seg %u +%zu type %u q 0x%x %u %u\n", s, at, type,
                       type == 1u ? 0u : g_program[at + 1u], g_program[at + 2u],
                       type == 10u ? g_program[at + 5u] : g_program[at + 3u]);
            if (type == 5u)
                break;
            switch (type) {
            case 1u: at += 3u + count + 2u * patches; break;
            case 2u: (*rings)++; at += 4u + 8u * count + 2u * patches; break;
            case 3u: (*heads)++; at += 4u; break;
            case 4u: (*awaits)++; at += 4u; break;
            case 9u: at += 1u; break;
            case 10u: (*preconds)++; *precond_count_sum += g_program[at + 5u]; at += 7u; break;
            default: return false;
            }
        }
    }
    return true;
}

static int run_arm(const char *name, bool dump)
{
    pn_tt_device *d = fake_device();
    uint32_t heads, awaits, preconds, rings, count_sum;
    uint64_t forward;

    if (!d)
        return 2;
    g_held = strncmp(name, "held", 4) == 0;
    for (forward = 1u; forward <= 5u; forward++) {
        pn_tt_dispatch_token_begin(d, forward);
        token(d);
        pn_tt_dispatch_token_end(d, forward);
    }
    if (!census(&heads, &awaits, &preconds, &rings, &count_sum, dump)) {
        printf("ARM %s no program\n", name);
        return 1;
    }
    printf("ARM %s digest 0x%016llx words %zu heads %u rings %u awaits %u preconds %u countsum %u\n",
           name, (unsigned long long)fnv(g_program, g_program_words), g_program_words, heads,
           rings, awaits, preconds, count_sum);
    return 0;
}

typedef struct {
    unsigned long long digest;
    unsigned long words;
    unsigned heads, rings, awaits, preconds, countsum;
    bool ok;
} arm_result;

static arm_result child(const char *self, const char *env, const char *name)
{
    char command[1024], line[512];
    arm_result r;
    FILE *p;

    memset(&r, 0, sizeof(r));
    snprintf(command, sizeof(command),
             "env -u PN_TT_ONCARD_COMPLETION -u PN_TT_RED -u PN_TT_DISPATCH_HOIST %s %s --arm %s "
             "2>/dev/null", env, self, name);
    p = popen(command, "r");
    if (!p)
        return r;
    while (fgets(line, sizeof(line), p))
        if (sscanf(line, "ARM %*s digest %llx words %lu heads %u rings %u awaits %u preconds %u "
                         "countsum %u", &r.digest, &r.words, &r.heads, &r.rings, &r.awaits,
                   &r.preconds, &r.countsum) == 7)
            r.ok = true;
    r.ok = pclose(p) == 0 && r.ok;
    return r;
}

static int g_checks, g_failures;

static void check(bool ok, const char *what, const arm_result *r)
{
    g_checks++;
    printf("%s %s", ok ? "PASS" : "FAIL", what);
    if (r)
        printf(" (digest 0x%016llx, %lu words, %u heads, %u awaits, %u entries)", r->digest,
               r->words, r->heads, r->awaits, r->preconds);
    printf("\n");
    if (!ok)
        g_failures++;
}

int main(int argc, char **argv)
{
    arm_result off, off_nohoist, on, on_nohoist, red, dflt, held_off, held_on, held_on8;

    if (argc > 2 && strcmp(argv[1], "--arm") == 0)
        return run_arm(argv[2], argc > 3 && strcmp(argv[3], "--dump") == 0);
    off = child(argv[0], "PN_TT_ONCARD_COMPLETION=0", "off");
    off_nohoist = child(argv[0], "PN_TT_ONCARD_COMPLETION=0 PN_TT_DISPATCH_HOIST=0", "off_nohoist");
    held_off = child(argv[0], "PN_TT_ONCARD_COMPLETION=0", "held_off");
    if (argc > 1 && strcmp(argv[1], "--digest") == 0) {
        printf("off 0x%016llx off_nohoist 0x%016llx held_off 0x%016llx\n", off.digest,
               off_nohoist.digest, held_off.digest);
        return off.ok && off_nohoist.ok && held_off.ok ? 0 : 1;
    }
    held_on = child(argv[0], "PN_TT_ONCARD_COMPLETION=1", "held_on");
    held_on8 = child(argv[0], "PN_TT_ONCARD_COMPLETION=1 PN_TT_ONCARD_LOOKAHEAD=8", "held_on8");
    on = child(argv[0], "PN_TT_ONCARD_COMPLETION=1", "on");
    on_nohoist = child(argv[0], "PN_TT_ONCARD_COMPLETION=1 PN_TT_DISPATCH_HOIST=0", "on_nohoist");
    red = child(argv[0], "PN_TT_ONCARD_COMPLETION=1 PN_TT_RED=precondition_early", "red");
    dflt = child(argv[0], "", "default");
    check(dflt.ok && dflt.digest == on.digest, "knob unset: on by default (the same program as =1)", &dflt);
    check(off.ok && off.preconds == 0u && off.digest == GOLDEN_OFF,
          "knob 0: the program is the base code's, byte for byte", &off);
    check(off_nohoist.ok && off_nohoist.preconds == 0u && off_nohoist.digest == GOLDEN_OFF_NOHOIST,
          "knob 0, no hoist: the program is the base code's, byte for byte", &off_nohoist);
    check(on.ok && on.preconds > 0u && on.digest != off.digest && on.heads == off.heads &&
              on.rings == off.rings && on.awaits <= off.awaits,
          "knob 1: sidecar entries; every head and ring slot kept; no AWAIT added", &on);
    check(on_nohoist.ok && on_nohoist.preconds == on.preconds &&
              on_nohoist.digest != off_nohoist.digest,
          "knob 1, no hoist: the same entries on the host's order", &on_nohoist);
    check(red.ok && red.preconds == on.preconds && red.countsum < on.countsum &&
              red.digest != on.digest,
          "RED precondition_early: the same entries, aimed short", &red);
    check(held_off.ok && held_off.preconds == 0u && held_off.digest == GOLDEN_HELD_OFF,
          "held awaits, knob 0: the program is the base code's, byte for byte", &held_off);
    check(held_on.ok && held_on.preconds > 0u && held_on.heads == held_off.heads &&
              held_on.awaits == held_off.awaits,
          "held awaits, knob 1: the S2 plan passes its check (not refused), every await kept",
          &held_on);
    check(held_on8.ok && held_on8.preconds > 0u && held_on8.awaits == held_off.awaits,
          "held awaits, lookahead 8: the same", &held_on8);
    printf("test_tt_oncard_dispatch: %d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

#else
int main(void)
{
    printf("test_tt_oncard_dispatch: needs USE_TT=1\n");
    return 0;
}
#endif
