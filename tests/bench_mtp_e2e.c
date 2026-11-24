/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_mtp_e2e.c
 * @brief End-to-end MTP speculative decode vs plain decode on the K80: real prompts padded
 *        to a context depth, the real Q8_0 MTP head, greedy and sampled, one model load.
 *
 * Per (depth, prompt, temperature) every arm prefills the same tokens from a fresh state
 * (chunks of 128; the spec arms feed the MTP head's backlog), draws the first token from
 * the prefill logits with a seeded RNG, then generates n_gen tokens:
 *   plain  — bound to decode slot 0, one pn_qwen35moe_decode_slots step per token (the
 *            serve's decode path);
 *   ad     — pn_mtp_spec_generate, adaptive depth (spec_depth.h), plain switch allowed;
 *   dN     — pn_mtp_spec_generate at a fixed N drafts per verify.
 * Decode wall time excludes prefill. Every spec arm's stream is compared token for token
 * with the plain arm's ("EXACT" / "DIVERGED at i"); the exit code is 1 on any divergence.
 *
 * Build (CUDA host): make USE_CUDA=1 tests/bench_mtp_e2e
 * Run:   QWEN35_MTP=1 <production env> ./tests/bench_mtp_e2e <model.gguf> <filler.txt> \
 *          <prompts.txt> <depth[,depth...]> [n_gen=256] [n_prompts=2]
 * Env:   MTP_ARMS  ("plain,ad")   arms, in order (plain must come first)
 *        MTP_TEMPS ("0,0.7")      temperatures (top_p 0.95, top_k 20 when > 0)
 *        MTP_SEED  (1234)         sampler seed (same for every arm)
 *        MTP_DMAX  (4)            adaptive arm's largest D
 *        MTP_DUMP_TOKS            append each plain trajectory to this file (lookup studies)
 *        MTP_WIDE  (0)            1: prefill through the wide pipelined path (bit-exact)
 * Arms ending in 'c' (adc, d3c) draw coupled drafts on sampled rows (spec_mtp_couple.h);
 * a further 'l' (adcl) adds prompt-lookup drafts (MTP_LOOKUP mode, default 1).
 * The prompts file holds prompts separated by lines "===PROMPT===", optionally led by a
 * "===SYSTEM===" block. The filler is plain text cut to make each prompt `depth` tokens.
 *
 * @status complete
 * @issues None
 * @todo None
 */
#include "../include/pantheon.h"
#include "../src/core/chat.h"
#include "../src/core/engine.h"
#include "../src/core/spec_mtp.h"
#include "../src/models/qwen35moe.h"
#include "../src/serve/serve_spec_lookup.h"
#include "tensor/tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef USE_CUDA
int main(void) { fprintf(stderr, "built without USE_CUDA\n"); return 1; }
#else

#define E2E_CHUNK      128
#define E2E_MAX_PROMPT 16
#define E2E_MAX_ARMS   8
#define E2E_MAX_GEN    4096
#define E2E_TOPK       40
#define E2E_RECENT     64

typedef struct {
    char name[16];
    int fixed_d;          /* -2 plain, -1 adaptive, >= 0 fixed */
    int couple;           /* name ends in 'c' (before any 'l'): coupled drafts */
    int lookup;           /* name ends in 'l': prompt-lookup drafts, mode MTP_LOOKUP (1) */
    double tok, ms;       /* per (depth, temp) totals */
    long cycles, drafted, accepted, exact_runs, runs;
} e2e_arm;

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static char *slurp(const char *path)
{
    FILE *f = fopen(path, "rb");
    char *buf = NULL;
    long len;

    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = malloc((size_t)len + 1);
    if (buf && fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf);
        buf = NULL;
    }
    if (buf)
        buf[len] = '\0';
    fclose(f);
    return buf;
}

/* Split on "===PROMPT===" lines; an optional leading "===SYSTEM===" block sets *system. */
static int load_prompts(char *buf, char **out, int max_out, const char **system)
{
    int n = 0;
    char *cur = buf;

    if (strncmp(cur, "===SYSTEM===\n", 13) == 0) {
        char *sep = strstr(cur, "\n===PROMPT===\n");
        if (!sep)
            return 0;
        *sep = '\0';
        *system = cur + 13;
        cur = sep + 14;
    }
    while (cur && *cur && n < max_out) {
        char *sep = strstr(cur, "\n===PROMPT===\n");
        if (sep)
            *sep = '\0';
        if (strncmp(cur, "===PROMPT===\n", 13) == 0)
            cur += 13;
        if (*cur)
            out[n++] = cur;
        cur = sep ? sep + 14 : NULL;
    }
    return n;
}

/* The prompt padded with filler text in front to (just under) `depth` tokens. */
static pn_token_seq build_prompt(pn_model *model, const char *system, const char *filler,
                                 const char *prompt, int depth)
{
    size_t flen = strlen(filler), take = (size_t)depth * 3;
    pn_token_seq seq = { 0 };

    for (int it = 0; it < 6; it++) {
        if (take > flen)
            take = flen;
        size_t ulen = take + strlen(prompt) + 64;
        char *user = malloc(ulen);
        if (!user)
            break;
        snprintf(user, ulen, "%s%.*s%s%s", take ? "Reference material:\n\n" : "", (int)take,
                 filler, take ? "\n\n---\n\n" : "", prompt);
        free(seq.tokens);
        seq = pn_chat_format_first((pn_chat_style)model->chat_style, model->tokenizer, system,
                                   user);
        free(user);
        if (!seq.tokens || (seq.n_tokens <= depth && seq.n_tokens > depth - 96) || take == 0)
            break;
        /* ~chars per token of what is there now; move the filler by the token error. */
        double cpt = (double)(take + strlen(prompt)) / seq.n_tokens;
        long delta = (long)((depth - 48 - seq.n_tokens) * cpt);
        take = delta < 0 && (size_t)(-delta) >= take ? 0 : (size_t)((long)take + delta);
        if (take >= flen && delta > 0)
            break;
    }
    return seq;
}

/* Fresh state, prefill in chunks (feeding the MTP backlog when asked), first token drawn. */
static pn_state *prefill(pn_model *model, const pn_token_seq *seq, int feed_mtp,
                         pn_sampling_params *sp, uint32_t *recent, int *n_recent,
                         uint32_t *first, double *ms)
{
    pn_state *st = pn_state_new(model);
    double t0 = now_ms();

    if (!st)
        return NULL;
    /* MTP_WIDE=1: the serve's wide pipelined prefill (bit-exact; it feeds the MTP backlog
     * itself). Refused -> nothing advanced -> the chunk loop below. */
    if (getenv("MTP_WIDE") && atoi(getenv("MTP_WIDE")) != 0) {
        tm_tensor *lg = NULL;
        if (pn_qwen35moe_prefill_range_pipelined_cuda(model, seq->tokens, seq->n_tokens, st,
                                                      E2E_CHUNK, NULL, NULL, &lg) == 0 && lg) {
            *first = pn_sample_next_token(tm_data_f32(lg), model->config.n_vocab, sp, recent,
                                          *n_recent);
            pn_mtp_recent_push(recent, n_recent, E2E_RECENT, *first);
            tm_free(lg);
            *ms = now_ms() - t0;
            return st;
        }
        if (lg)
            tm_free(lg);
    }
    for (int off = 0; off < seq->n_tokens; off += E2E_CHUNK) {
        int n = seq->n_tokens - off < E2E_CHUNK ? seq->n_tokens - off : E2E_CHUNK;
        tm_tensor *out = pn_qwen35moe_forward_batch_cuda(model, seq->tokens + off, n, st, NULL);
        if (!out ||
            (feed_mtp && pn_qwen35moe_mtp_push_rows(model, st, seq->tokens + off, n) != 0)) {
            if (out)
                tm_free(out);
            fprintf(stderr, "prefill failed at %d\n", off);
            pn_state_free(st);
            return NULL;
        }
        if (off + n >= seq->n_tokens) {
            *first = pn_sample_next_token(tm_data_f32(out), model->config.n_vocab, sp, recent,
                                          *n_recent);
            pn_mtp_recent_push(recent, n_recent, E2E_RECENT, *first);
        }
        tm_free(out);
    }
    *ms = now_ms() - t0;
    return st;
}

/* The plain arm: slot 0, one decode-slots step per token. Returns tokens written. */
static int run_plain(pn_model *model, pn_state *st, uint32_t seed, int n_gen, int stop,
                     pn_sampling_params *sp, uint32_t *recent, int *n_recent, uint32_t *out)
{
    int n = 0;

    if (pn_qwen35moe_decode_slot_bind(model, st, 0, 1) != 0)
        return -1;
    while (n < n_gen) {
        float *logits[1];
        pn_state *sts[1] = { st };
        if (pn_qwen35moe_decode_slots(model, &seed, sts, 1, logits) != 0)
            return -1;
        seed = pn_sample_next_token(logits[0], model->config.n_vocab, sp, recent, *n_recent);
        pn_mtp_recent_push(recent, n_recent, E2E_RECENT, seed);
        if ((int)seed == stop)
            break;
        out[n++] = seed;
    }
    return n;
}

static pn_sampling_params sampler(float temp, uint64_t seed)
{
    pn_sampling_params sp = pn_sampling_defaults();
    sp.temperature = temp;
    sp.top_p = 0.95f;
    sp.top_k = 20;
    sp.repeat_penalty = 1.0f;
    sp.repeat_last_n = E2E_RECENT;
    sp.rng_seeded = 1;
    sp.rng_state = seed;
    return sp;
}

static int parse_arms(e2e_arm *arms)
{
    const char *spec = getenv("MTP_ARMS") ? getenv("MTP_ARMS") : "plain,ad";
    int n = 0;

    while (*spec && n < E2E_MAX_ARMS) {
        e2e_arm *a = &arms[n];
        int len = (int)strcspn(spec, ",");
        memset(a, 0, sizeof(*a));
        snprintf(a->name, sizeof(a->name), "%.*s", len, spec);
        int base = len;
        a->lookup = base > 1 && a->name[base - 1] == 'l' && strcmp(a->name, "plain") != 0;
        base -= a->lookup;
        a->couple = base > 1 && a->name[base - 1] == 'c';
        base -= a->couple;
        a->fixed_d = strcmp(a->name, "plain") == 0 ? -2
                   : strncmp(a->name, "ad", 2) == 0 && base == 2 ? -1
                   : a->name[0] == 'd' ? atoi(a->name + 1) : -3;
        if (a->fixed_d == -3 || a->fixed_d > PN_SPEC_DEPTH_MAX) {
            fprintf(stderr, "bad arm %s\n", a->name);
            return -1;
        }
        n++;
        spec += len + (spec[len] == ',');
    }
    return n > 0 && arms[0].fixed_d == -2 ? n : -1;
}

/* FNV-1a over a token stream: the plain arm prints it so two BUILDS can be compared for
 * an identical greedy/sampled stream (spec arms are compared to plain in-process). */
static uint64_t toks_fnv(const uint32_t *t, int n)
{
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++)
        for (int b = 0; b < 4; b++) {
            h ^= (t[i] >> (8 * b)) & 0xFFu;
            h *= 1099511628211ULL;
        }
    return h;
}

/* MTP_DUMP_TOKS=<file>: append the plain arm's trajectory (int32 n_prompt, int32 n_total,
 * n_total ids: the prompt, the first draw, the generated tokens) for offline lookup studies. */
static void dump_toks(const pn_token_seq *seq, uint32_t first, const uint32_t *gen, int n)
{
    const char *path = getenv("MTP_DUMP_TOKS");
    FILE *f = path ? fopen(path, "ab") : NULL;
    int32_t hdr[2] = { seq->n_tokens, seq->n_tokens + 1 + n };

    if (!f)
        return;
    fwrite(hdr, sizeof(hdr), 1, f);
    fwrite(seq->tokens, sizeof(uint32_t), (size_t)seq->n_tokens, f);
    fwrite(&first, sizeof(first), 1, f);
    fwrite(gen, sizeof(uint32_t), (size_t)n, f);
    fclose(f);
}

/* pn_mtp_gen_opts' lookup callbacks over a sg_spec_lookup. */
static int lk_draft(void *c, int cap, uint32_t *d) { return sg_spec_lookup_draft(c, cap, d); }
static void lk_report(void *c, int n, int a) { sg_spec_lookup_report(c, n, a); }
static void lk_push(void *c, uint32_t t) { sg_spec_lookup_push(c, t); }

/* One spec arm on one prompt: prefill, generate, compare with the plain stream. */
static int run_spec(pn_model *model, const pn_token_seq *seq, e2e_arm *a, float temp,
                    uint64_t seed, int n_gen, int stop, const uint32_t *ref, int n_ref,
                    const char *tag)
{
    static uint32_t out[E2E_MAX_GEN];
    uint32_t recent[E2E_RECENT], first;
    int n_recent = 0, n, diff = -1;
    double pre_ms, t0;
    pn_sampling_params sp = sampler(temp, seed);
    pn_mtp_gen_opts opts = { .fixed_d = a->fixed_d, .dmax = getenv("MTP_DMAX") ?
                             atoi(getenv("MTP_DMAX")) : PN_SPEC_DEPTH_MAX,
                             .plain_switch = a->fixed_d < 0, .use_more = a->fixed_d < 0,
                             .topk = E2E_TOPK, .repeat_window = E2E_RECENT,
                             .couple = a->couple };
    sg_spec_lookup lk;
    memset(&lk, 0, sizeof(lk));
    pn_spec_depth ctl;
    pn_mtp_gen_stats gs;
    pn_state *st = prefill(model, seq, 1, &sp, recent, &n_recent, &first, &pre_ms);

    if (!st)
        return -1;
    if (a->lookup) {
        int mode = getenv("MTP_LOOKUP") ? atoi(getenv("MTP_LOOKUP")) : 1;
        char mbuf[8];
        snprintf(mbuf, sizeof(mbuf), "%d", mode);
        setenv("QWEN35_MTP_LOOKUP", mbuf, 1);
        if (sg_spec_lookup_init_cuda(&lk, seq->tokens, seq->n_tokens, n_gen + 1) == 0 && lk.on) {
            sg_spec_lookup_push(&lk, first);
            opts.lookup_draft = lk_draft;
            opts.lookup_report = lk_report;
            opts.lookup_push = lk_push;
            opts.lookup_ctx = &lk;
            opts.lookup_mode = lk.mode;
        }
    }
    pn_spec_depth_init(&ctl, opts.dmax);
    if (pn_qwen35moe_dense_ffn(model))
        pn_spec_depth_dense_priors(&ctl);
    t0 = now_ms();
    n = pn_mtp_spec_generate(model, st, first, n_gen, stop, &sp, recent, &n_recent, &opts,
                             a->fixed_d < 0 ? &ctl : NULL, out, &gs);
    double ms = now_ms() - t0;
    pn_state_free(st);
    lk.cycles = 0;   /* the RUN line reports it */
    sg_spec_lookup_free(&lk);
    if (n < 0)
        return -1;
    for (int i = 0; i < (n < n_ref ? n : n_ref); i++)
        if (out[i] != ref[i]) { diff = i; break; }
    if (diff < 0 && n != n_ref)
        diff = n < n_ref ? n : n_ref;
    a->tok += n; a->ms += ms; a->runs++; a->exact_runs += diff < 0;
    a->cycles += gs.cycles; a->drafted += gs.drafted; a->accepted += gs.accepted;
    printf("RUN %s arm=%s n=%d decode_ms=%.0f tok/s=%.2f %s cycles=%ld acc=%ld/%ld "
           "hist=%ld/%ld/%ld/%ld/%ld switch=%d ms(draft %.0f verify %.0f plain %.0f sample %.0f)"
           " prefill_ms=%.0f lookup=%ld/%ld\n", tag, a->name, n, ms, n / (ms / 1e3),
           diff < 0 ? "EXACT" : "DIVERGED", gs.cycles, gs.accepted, gs.drafted,
           gs.depth_hist[0], gs.depth_hist[1], gs.depth_hist[2], gs.depth_hist[3],
           gs.depth_hist[4], gs.switched_at, gs.ms_draft, gs.ms_verify, gs.ms_plain,
           gs.ms_sample, pre_ms, gs.lookup_accepted, gs.lookup_cycles);
    if (diff >= 0)
        printf("   first_diff=%d plain=%u spec=%u\n", diff, diff < n_ref ? ref[diff] : 0,
               diff < n ? out[diff] : 0);
    fflush(stdout);
    return diff < 0 ? 0 : 1;
}

static int run_depth(pn_model *model, const char **systems, const char *filler, char **prompts,
                     int n_prompts, int depth, int n_gen, e2e_arm *arms, int n_arms,
                     const float *temps, int n_temps, uint64_t seed, int stop)
{
    static uint32_t ref[E2E_MAX_GEN];
    int bad = 0;

    for (int ti = 0; ti < n_temps; ti++) {
        for (int a = 0; a < n_arms; a++)
            arms[a].tok = arms[a].ms = 0, arms[a].cycles = arms[a].drafted = 0,
            arms[a].accepted = arms[a].exact_runs = arms[a].runs = 0;
        for (int p = 0; p < n_prompts; p++) {
            pn_token_seq seq = build_prompt(model, systems[p], filler, prompts[p], depth);
            char tag[96];
            uint32_t recent[E2E_RECENT], first;
            int n_recent = 0;
            double pre_ms;
            pn_sampling_params sp = sampler(temps[ti], seed);
            if (!seq.tokens)
                return -1;
            snprintf(tag, sizeof(tag), "depth=%d ctx=%d prompt=%d temp=%.1f", depth,
                     seq.n_tokens, p, temps[ti]);
            pn_state *st = prefill(model, &seq, 0, &sp, recent, &n_recent, &first, &pre_ms);
            if (!st)
                return -1;
            double t0 = now_ms();
            int n_ref = run_plain(model, st, first, n_gen, stop, &sp, recent, &n_recent, ref);
            double ms = now_ms() - t0;
            pn_state_free(st);
            if (n_ref < 0)
                return -1;
            arms[0].tok += n_ref; arms[0].ms += ms; arms[0].runs++;
            printf("RUN %s arm=plain n=%d decode_ms=%.0f tok/s=%.2f prefill_ms=%.0f "
                   "toks=%016llx\n", tag, n_ref, ms, n_ref / (ms / 1e3), pre_ms,
                   (unsigned long long)toks_fnv(ref, n_ref));
            fflush(stdout);
            dump_toks(&seq, first, ref, n_ref);
            for (int a = 1; a < n_arms; a++) {
                int r = run_spec(model, &seq, &arms[a], temps[ti], seed, n_gen, stop, ref,
                                 n_ref, tag);
                if (r < 0)
                    return -1;
                bad += r;
            }
            free(seq.tokens);
        }
        double plain = arms[0].tok / (arms[0].ms / 1e3);
        printf("SUMMARY depth=%d temp=%.1f plain %.2f tok/s", depth, temps[ti], plain);
        for (int a = 1; a < n_arms; a++)
            printf(" | %s %.2f tok/s (%.3fx, acc %.3f, tau %.2f, exact %ld/%ld)", arms[a].name,
                   arms[a].tok / (arms[a].ms / 1e3), arms[a].tok / (arms[a].ms / 1e3) / plain,
                   arms[a].drafted ? (double)arms[a].accepted / arms[a].drafted : 0.0,
                   arms[a].cycles ? (double)(arms[a].accepted + arms[a].cycles) /
                                        arms[a].cycles : 0.0,
                   arms[a].exact_runs, arms[a].runs);
        printf("\n");
        fflush(stdout);
    }
    return bad;
}

int main(int argc, char **argv)
{
    e2e_arm arms[E2E_MAX_ARMS];
    float temps[4];
    int n_temps = 0, n_arms = parse_arms(arms), bad = 0;
    const char *systems[E2E_MAX_PROMPT];
    char *prompts[E2E_MAX_PROMPT];

    if (argc < 5 || n_arms < 0) {
        fprintf(stderr, "usage: %s <model.gguf> <filler.txt> <prompts.txt[,more.txt]> "
                        "<depth[,..]> [n_gen] [prompts_per_file]  (MTP_ARMS must start with "
                        "plain)\n", argv[0]);
        return 2;
    }
    int n_gen = argc > 5 ? atoi(argv[5]) : 256;
    int per_file = argc > 6 ? atoi(argv[6]) : 1;
    uint64_t seed = getenv("MTP_SEED") ? strtoull(getenv("MTP_SEED"), NULL, 10) : 1234;
    for (const char *p = getenv("MTP_TEMPS") ? getenv("MTP_TEMPS") : "0,0.7"; *p && n_temps < 4;) {
        char *end = NULL;
        temps[n_temps++] = strtof(p, &end);
        if (end == p)
            break;
        p = (*end == ',') ? end + 1 : end;
    }
    char *filler = slurp(argv[2]);
    int n_prompts = 0;
    char paths[1024];
    snprintf(paths, sizeof(paths), "%s", argv[3]);
    for (char *path = strtok(paths, ","); path; path = strtok(NULL, ",")) {
        char *buf = slurp(path), *got[E2E_MAX_PROMPT];
        const char *sys = NULL;
        int n = buf ? load_prompts(buf, got, E2E_MAX_PROMPT, &sys) : 0;
        for (int i = 0; i < n && i < per_file && n_prompts < E2E_MAX_PROMPT; i++) {
            systems[n_prompts] = sys;
            prompts[n_prompts++] = got[i];
        }
    }
    if (!filler || n_prompts == 0 || n_gen < 1 || n_gen > E2E_MAX_GEN) {
        fprintf(stderr, "bad filler/prompts/n_gen\n");
        return 2;
    }
    pn_model *model = pn_load(argv[1]);
    if (!model)
        return 2;
    if (n_arms > 1 && !pn_qwen35moe_mtp_ready(model)) {
        fprintf(stderr, "no MTP head (QWEN35_MTP=1?)\n");
        return 2;
    }
    int stop = ft_eos_id(model->tokenizer);
    printf("mtp e2e: arms=%s n_gen=%d prompts=%d seed=%llu stop=%d\n",
           getenv("MTP_ARMS") ? getenv("MTP_ARMS") : "plain,ad", n_gen, n_prompts,
           (unsigned long long)seed, stop);
    for (const char *p = argv[4]; *p;) {
        char *end = NULL;
        int depth = (int)strtol(p, &end, 10);
        if (end == p)
            break;
        int r = run_depth(model, systems, filler, prompts, n_prompts, depth, n_gen, arms,
                          n_arms, temps, n_temps, seed, stop);
        if (r < 0) {
            printf("RUN depth=%d FAILED\n", depth);
            bad = 1;
        } else {
            bad |= r > 0;
        }
        p = (*end == ',') ? end + 1 : end;
    }
    pn_mtp_stats ms;
    if (pn_qwen35moe_mtp_stats(model, &ms) == 0)
        printf("MTP head: drafts=%ld kv_rows=%ld full_rows=%ld draft_ms=%.0f (%.2f ms/draft)\n",
               ms.drafts, ms.kv_rows, ms.full_rows, ms.draft_ms,
               ms.drafts ? ms.draft_ms / ms.drafts : 0.0);
    if (getenv("QWEN35_MTP_PROF") && ms.drafts)
        printf("MTP head prof (ms/draft, sync'd): kv rows %.2f attention %.2f moe %.2f head %.2f\n",
               ms.prof_ms[0] / ms.drafts, ms.prof_ms[1] / ms.drafts, ms.prof_ms[2] / ms.drafts,
               ms.prof_ms[3] / ms.drafts);
    long vc, vr, vf;
    if (pn_qwen35moe_verify_graph_stats(model, &vc, &vr, &vf) == 0)
        printf("verify graphs: captures=%ld replays=%ld failed=%ld\n", vc, vr, vf);
    printf(bad ? "E2E: DIVERGED or FAILED\n" : "E2E: ALL EXACT\n");
    fflush(stdout);
    if (!getenv("QWEN35_EXIT_CLEAN"))
        _Exit(bad);
    pn_model_free(model);
    return bad;
}
#endif
