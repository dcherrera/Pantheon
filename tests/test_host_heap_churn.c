/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_host_heap_churn.c — B5: host anon RAM must stay flat across sessions.
 *
 * Replays the serve's per-request host churn the way the batched serve runs
 * it: every request on its own short-lived thread, several at once, each
 *   - building a tool-grammar token table (one strdup per vocab id plus the
 *     [n_vocab] pointer array, as sg_tg_* in serve_generation.c does),
 *   - capturing a REAL qwen35moe sys-prefix snapshot (30 GDN layers at the
 *     qwen3.6-35b-a3b dims) into one of 4 slots, freeing the slot's previous
 *     snapshot (captured by another thread),
 *   - holding a prompt-sized buffer and a few small records that outlive the
 *     request (session history / bookkeeping, freed rounds later).
 * RssAnon (Linux /proc/self/status) is sampled after every round.
 *
 * Each mode runs in its own child process (allocator settings are global):
 *   baseline  allocator untouched           — must GROW (proves the harness
 *                                             can fail)
 *   fixed     pn_host_heap_init + release   — must stay flat
 * Linux/glibc only; prints SKIP elsewhere.
 */

#include "../src/core/mem/pn_host_heap.h"
#include "../src/models/qwen35moe_internal.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

void *pn_qwen35moe_state_snapshot(const void *state_v, int pos);
void pn_qwen35moe_state_snapshot_free(void *snap_v);
/* Link stub: the snapshot restore path seeds kv content generations. */
unsigned pn_qwen35moe_kv_next_content_gen(void) { return 1u; }

#define N_VOCAB      248320   /* qwen3.6 */
#define N_SSM        30
#define N_SLOTS      4
#define CONC         8        /* max requests in flight; bursts vary 2..8 */
#define ROUNDS       100
#define WARMUP       5
#define KEEP_ROUNDS  3        /* long-lived records survive this many rounds */
#define FLAT_MB      48.0     /* fixed mode: max growth after warm-up */
#define GROW_MB      150.0    /* baseline mode: must grow at least this
                                 * over its first round */

#if defined(__GLIBC__)
static pn_qwen35moe_state g_state;
static void *g_slot_snap[N_SLOTS];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static void *g_keep[ROUNDS][CONC][4];
static int g_fixed;

static double rss_anon_mb(void)
{
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    double mb = -1.0;
    if (!f)
        return mb;
    while (fgets(line, sizeof(line), f))
        if (strncmp(line, "RssAnon:", 8) == 0) {
            mb = atof(line + 8) / 1024.0;
            break;
        }
    fclose(f);
    return mb;
}

typedef struct { int round, i; unsigned seed; } job;

static void *request_thread(void *arg)
{
    job *j = (job *)arg;
    unsigned s = j->seed;

    /* Records that outlive the request (history, sids, bookkeeping). */
    for (int k = 0; k < 4; k++)
        g_keep[j->round][j->i][k] = malloc(64 + (rand_r(&s) % 4096));

    /* Prompt text / tokens / JSON for this turn. */
    size_t pb = 64 * 1024 + (size_t)(rand_r(&s) % (2 * 1024 * 1024));
    char *prompt = malloc(pb);
    memset(prompt, 'p', pb);

    /* Tool-grammar token table. */
    char **pieces = calloc(N_VOCAB, sizeof(char *));
    for (int t = 0; t < N_VOCAB; t++) {
        char buf[24];
        int n = 1 + (int)(rand_r(&s) % 14);
        memset(buf, 'a' + t % 26, (size_t)n);
        buf[n] = '\0';
        pieces[t] = strdup(buf);
    }

    /* Sys-prefix snapshot into a slot; the old one was another thread's. */
    void *snap = pn_qwen35moe_state_snapshot(&g_state, 1);
    pthread_mutex_lock(&g_mu);
    int slot = (j->round * CONC + j->i) % N_SLOTS;
    void *old = g_slot_snap[slot];
    g_slot_snap[slot] = snap;
    pthread_mutex_unlock(&g_mu);
    pn_qwen35moe_state_snapshot_free(old);

    for (int t = 0; t < N_VOCAB; t++)
        free(pieces[t]);
    free(pieces);
    free(prompt);
    if (g_fixed)
        pn_host_heap_release();
    return NULL;
}

static int run_mode(int fixed)
{
    g_fixed = fixed;
    if (fixed && !pn_host_heap_init()) {
        fprintf(stderr, "pn_host_heap_init refused\n");
        return 2;
    }
    g_state.n_ssm = N_SSM;
    g_state.ssm = calloc(N_SSM, sizeof(*g_state.ssm));
    for (int l = 0; l < N_SSM; l++) {
        pn_qwen35moe_ssm_state *st = &g_state.ssm[l];
        st->conv_dim = 8192;
        st->conv_minus_1 = 3;
        st->state_size = 128;
        st->n_v_heads = 32;
        st->conv_state = calloc((size_t)st->conv_minus_1 * st->conv_dim, sizeof(float));
        st->ssm_state = calloc((size_t)st->state_size * st->state_size * st->n_v_heads,
                               sizeof(float));
    }

    double first = 0.0, base = 0.0, last = 0.0;
    unsigned seed = 5;
    int requests = 0;
    for (int r = 0; r < ROUNDS; r++) {
        pthread_t th[CONC];
        job jobs[CONC];
        int n = 2 + (int)((seed >> 16) % (CONC - 1));
        requests += n;
        for (int i = 0; i < n; i++) {
            jobs[i] = (job){ r, i, seed = seed * 1103515245u + 12345u };
            pthread_create(&th[i], NULL, request_thread, &jobs[i]);
        }
        for (int i = 0; i < n; i++)
            pthread_join(th[i], NULL);
        if (r >= KEEP_ROUNDS)
            for (int i = 0; i < CONC; i++)
                for (int k = 0; k < 4; k++) {
                    free(g_keep[r - KEEP_ROUNDS][i][k]);
                    g_keep[r - KEEP_ROUNDS][i][k] = NULL;
                }
        last = rss_anon_mb();
        if (r == 0)
            first = last;
        if (r == WARMUP - 1)
            base = last;
        if (r % 10 == 9 || r == ROUNDS - 1)
            printf("  %-8s round %2d  requests %3d  RssAnon %7.1f MiB\n",
                   fixed ? "fixed" : "baseline", r + 1, requests, last);
    }
    printf("  %-8s growth: %+.1f MiB since round 1, %+.1f MiB since warm-up\n",
           fixed ? "fixed" : "baseline", last - first, last - base);
    if (fixed)
        return last - base <= FLAT_MB ? 0 : 1;
    return last - first >= GROW_MB ? 0 : 1;
}
#endif /* __GLIBC__ */

int main(int argc, char **argv)
{
#if !defined(__GLIBC__)
    (void)argc; (void)argv;
    printf("SKIP test_host_heap_churn: glibc only (the B5 retention is a glibc "
           "arena behaviour)\n");
    return 0;
#else
    if (argc == 2 && strcmp(argv[1], "--mode=baseline") == 0)
        return run_mode(0);
    if (argc == 2 && strcmp(argv[1], "--mode=fixed") == 0)
        return run_mode(1);

    int fails = 0;
    const char *modes[2] = { "--mode=baseline", "--mode=fixed" };
    for (int m = 0; m < 2; m++) {
        fflush(stdout);
        pid_t pid = fork();
        if (pid == 0) {
            /* glibc's default arena cap is 8 x cores: pin the bank's (2 cores)
             * so the baseline does not depend on this machine's core count. */
            setenv("GLIBC_TUNABLES", "glibc.malloc.arena_max=16", 1);
            execl("/proc/self/exe", argv[0], modes[m], (char *)NULL);
            _exit(3);
        }
        int st = 0;
        waitpid(pid, &st, 0);
        int ok = WIFEXITED(st) && WEXITSTATUS(st) == 0;
        printf("%s %s: %s\n", ok ? "PASS" : "FAIL", modes[m] + 7,
               m == 0 ? "unmanaged heap grows (harness can fail)"
                      : "managed heap stays flat");
        fails += !ok;
    }
    return fails ? 1 : 0;
#endif
}
