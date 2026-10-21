/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* test_spec_batch.c - the M4/M5 losslessness gate for BATCHED speculative decode.
 *
 * Proves the batched speculative decoder (pn_batch_decoder_step_spec) emits a
 * transcript BIT-IDENTICAL to plain batched greedy decode for ANY draft, across
 * batch sizes, draft widths, draft qualities, mixed spec/plain slots, staggered
 * admits, and stop (EOS / max_tokens) handling. Drafts only change how many
 * tokens each verify pass yields, never which tokens are produced, because
 * step_spec emits only the penalized argmax of the model's own verify logits and
 * accepts a draft solely when it already equals that argmax.
 *
 * References (greedy, temperature 0):
 *   A  = sequential pn_llama_forward greedy, mirroring the plain per-slot step
 *        loop (push pending -> forward -> penalized-argmax next pending), giving
 *        both the token stream and its decoded UTF-8 text.
 *   B  = plain pn_batch_decoder (the shipped batched greedy decoder).
 *   C  = pn_batch_decoder_step_spec under drafts {none, oracle=replay of A,
 *        garbage, truncated}.
 * The shipped decoders (B, C) expose only decoded text, so the end-to-end gate
 * is TEXT byte-for-byte: A-text == B-text == C-text for every page and config.
 * A separate TOKEN-for-token check (sequential A vs a batched forward_batch loop
 * captured in-test) confirms the batched kernel is token-lossless per page.
 *
 * Usage: test_spec_batch [model.gguf]
 */

#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/llama.h"
#include "tokenizer/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PROMPT_LEN 8
#define GEN        24
#define MAXB       6

static int g_nv = 0, g_eos = -1, g_imend = -1;

/* --- greedy sampler mirror (sample_token's temperature<=0 path) --- */
static uint32_t pen_argmax(float *logits, int nv, float rp, uint32_t *recent, int nrec) {
    if (rp != 1.0f && recent && nrec > 0) {
        for (int i = 0; i < nrec; i++) {
            uint32_t t = recent[i];
            if (t < (uint32_t)nv) { if (logits[t] > 0.0f) logits[t] /= rp; else logits[t] *= rp; }
        }
    }
    int b = 0;
    for (int i = 1; i < nv; i++) if (logits[i] > logits[b]) b = i;
    return (uint32_t)b;
}
static void ring_push(uint32_t *recent, int *nrec, int maxrec, uint32_t tok) {
    if (*nrec < maxrec) recent[(*nrec)++] = tok;
    else { memmove(recent, recent + 1, (maxrec - 1) * sizeof(uint32_t)); recent[maxrec - 1] = tok; }
}

static pn_sampling_params mkparams(float rp) {
    pn_sampling_params p = pn_sampling_defaults();
    p.temperature = 0.0f; p.repeat_penalty = rp; p.repeat_last_n = 64;
    p.top_k = 0; p.top_p = 1.0f;
    return p;
}

static void make_prompt(uint32_t *p, int n, int idx) {
    uint32_t x = 0x9e3779b9u * (uint32_t)(idx + 7);
    for (int i = 0; i < n; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        p[i] = (x % (uint32_t)(g_nv - 4)) + 2;
    }
}

static int is_stop(uint32_t t) { return ((g_eos >= 0 && (int)t == g_eos) || (int)t == g_imend); }

/* Prefill a fresh state, return (state, seed=argmax of prefill logits). */
static pn_state* prefill_state(pn_model *m, const uint32_t *prompt, int plen, uint32_t *seed) {
    pn_state *s = pn_state_new(m);
    tm_tensor *l = pn_llama_prefill(m, prompt, NULL, plen, s);
    if (!l) { pn_state_free(s); return NULL; }
    *seed = pen_argmax((float *)l->data, g_nv, 1.0f, NULL, 0);
    tm_free(l);
    return s;
}

/* Reference A: sequential greedy via pn_llama_forward, mirroring plain step.
 * Fills tok[0..*ntok-1] (tokens the decoder would output), returns 0/-1. */
static int gen_seq(pn_model *m, const uint32_t *prompt, int plen, float rp,
                   int max_tokens, uint32_t *tok, int *ntok) {
    uint32_t seed; pn_state *s = prefill_state(m, prompt, plen, &seed);
    if (!s) return -1;
    uint32_t recent[64]; int nrec = 0, maxrec = 64;
    uint32_t pending = seed; int no = 0;
    while (!(is_stop(pending) || no >= max_tokens)) {
        tok[no++] = pending;
        ring_push(recent, &nrec, maxrec, pending);
        tm_tensor *l = m->forward(m, pending, s);
        if (!l) { pn_state_free(s); return -1; }
        pending = pen_argmax((float *)l->data, g_nv, rp, recent, nrec);
        tm_free(l);
    }
    *ntok = no;
    pn_state_free(s);
    return 0;
}

/* Batched token reference: decode `np` pages together via pn_llama_forward_batch
 * (argmax + penalty per slot), capturing each page's token stream. This is the
 * plain batched greedy path at the token level (what pn_batch_decoder does
 * internally), used for the token-for-token check vs sequential A. */
typedef struct { uint32_t tok[GEN]; int no; uint32_t recent[64]; int nrec; uint32_t pending; int done; } bslot;
static int gen_batch_tokens(pn_model *m, const uint32_t *const *prompts, int np,
                            float rp, int max_tokens, bslot *slots) {
    pn_state **st = malloc((size_t)np * sizeof(pn_state *));
    for (int p = 0; p < np; p++) {
        uint32_t seed; st[p] = prefill_state(m, prompts[p], PROMPT_LEN, &seed);
        slots[p].no = 0; slots[p].nrec = 0; slots[p].pending = seed; slots[p].done = 0;
    }
    float *logits = malloc((size_t)np * g_nv * sizeof(float));
    uint32_t *toks = malloc((size_t)np * sizeof(uint32_t));
    pn_state **act = malloc((size_t)np * sizeof(pn_state *));
    int *aslot = malloc((size_t)np * sizeof(int));
    int remaining = np;
    while (remaining > 0) {
        int na = 0;
        for (int p = 0; p < np; p++) {
            bslot *b = &slots[p];
            if (b->done) continue;
            if (is_stop(b->pending) || b->no >= max_tokens) { b->done = 1; remaining--; continue; }
            b->tok[b->no++] = b->pending;
            ring_push(b->recent, &b->nrec, 64, b->pending);
            act[na] = st[p]; toks[na] = b->pending; aslot[na] = p; na++;
        }
        if (na == 0) break;
        if (pn_llama_forward_batch(m, toks, NULL, act, na, logits) != 0) {
            for (int p = 0; p < np; p++) pn_state_free(st[p]);
            free(st); free(logits); free(toks); free(act); free(aslot); return -1;
        }
        for (int j = 0; j < na; j++) {
            bslot *b = &slots[aslot[j]];
            b->pending = pen_argmax(logits + (size_t)j * g_nv, g_nv, rp, b->recent, b->nrec);
        }
    }
    for (int p = 0; p < np; p++) pn_state_free(st[p]);
    free(st); free(logits); free(toks); free(act); free(aslot);
    return 0;
}

static char* decode_text(pn_model *m, const uint32_t *tok, int n) {
    int *ids = malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    for (int i = 0; i < n; i++) ids[i] = (int)tok[i];
    char *raw = ft_decode(m->tokenizer, ids, n);
    char *text = raw ? gpt2_bytes_decode(raw) : NULL;
    free(raw); free(ids);
    return text ? text : strdup("");
}

/* --- draft providers --- */
typedef enum { D_NONE, D_ORACLE, D_GARBAGE, D_TRUNC } dkind;
typedef struct { const uint32_t *ref; int ref_len; dkind kind; unsigned rng; } dctx;
static int draft_cb(const uint32_t *prefix, int n, uint32_t *out, int max_k, void *ud) {
    dctx *c = (dctx *)ud; (void)prefix;
    if (c->kind == D_NONE || max_k <= 0) return 0;
    int k = max_k;
    if (c->kind == D_TRUNC) k = (max_k > 3) ? 3 : max_k;   /* short draft */
    for (int j = 0; j < k; j++) {
        if (c->kind == D_GARBAGE) {
            c->rng ^= c->rng << 13; c->rng ^= c->rng >> 17; c->rng ^= c->rng << 5;
            out[j] = c->rng % (uint32_t)g_nv;
        } else {
            int idx = n + j;                       /* oracle: the true next tokens */
            out[j] = (idx < c->ref_len) ? c->ref[idx] : 0;
        }
    }
    return k;
}

/* Run the shipped decoder (plain or spec) over `np` pre-supplied (state,seed)
 * pages admitted up front, collect each page's text by page_id. */
static int run_decoder(pn_model *m, pn_state **states, const uint32_t *seeds,
                       int np, float rp, int max_tokens, int max_batch,
                       int spec, int max_k, pn_draft_fn *dfns, void **duds,
                       char **out_text, long *passes, long *drafted, long *accepted) {
    pn_sampling_params params = mkparams(rp);
    pn_batch_decoder *d = pn_batch_decoder_new(m, max_batch, max_tokens, &params);
    if (!d) return -1;
    if (spec && pn_batch_decoder_spec_config(d, max_k, 0.10, 4) != 0) { pn_batch_decoder_free(d); return -1; }
    for (int p = 0; p < np; p++) {
        int rc = spec ? pn_batch_decoder_admit_spec(d, states[p], p, seeds[p], dfns ? dfns[p] : NULL, duds ? duds[p] : NULL)
                      : pn_batch_decoder_admit(d, states[p], p, seeds[p]);
        if (rc != 0) { pn_batch_decoder_free(d); return -1; }
    }
    pn_page_out outbuf[MAXB];
    int rc = 0;
    while (pn_batch_decoder_active(d) > 0) {
        int nf = spec ? pn_batch_decoder_step_spec(d, outbuf, max_batch)
                      : pn_batch_decoder_step(d, outbuf, max_batch);
        if (nf < 0) { rc = -1; break; }
        for (int i = 0; i < nf; i++) {
            out_text[outbuf[i].page_id] = outbuf[i].text;   /* borrow; caller frees */
            pn_state_free(outbuf[i].state);
        }
    }
    if (spec) pn_batch_decoder_spec_stats(d, passes, drafted, accepted);
    pn_batch_decoder_free(d);
    return rc;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "models/lightonocr-lm-Q4_K_M.gguf";
    pn_model *model = pn_load(path);
    if (!model) { fprintf(stderr, "load failed: %s\n", path); return 1; }
    g_nv = model->config.n_vocab;
    g_eos = ft_eos_id(model->tokenizer);
    g_imend = ft_token_to_id(model->tokenizer, "<|im_end|>");
    printf("model: %s  n_vocab=%d n_layer=%d arch=%d  eos=%d im_end=%d\n",
           path, g_nv, model->config.n_layer, model->config.arch, g_eos, g_imend);

    { /* support gate */
        pn_state *s = pn_state_new(model); uint32_t t = 2; int nt = 1;
        float *lg = malloc((size_t)g_nv * sizeof(float));
        int ok = lg && pn_llama_verify_batch(model, &t, &s, &nt, 1, lg) == 0;
        free(lg); pn_state_free(s);
        if (!ok) { printf("[SKIP] model not on the quantized non-MoE verify path\n");
                   pn_model_free(model); return 0; }
    }

    int fail = 0;
    const int Bs[] = { 1, 2, 6 };
    const float RPs[] = { 1.0f, 1.15f };
    const int Ks[] = { 0, 1, 8 };
    const dkind kinds[] = { D_NONE, D_ORACLE, D_GARBAGE, D_TRUNC };
    const char *kname[] = { "none", "oracle", "garbage", "trunc" };

    uint32_t prompts[MAXB][PROMPT_LEN];

    for (int bi = 0; bi < 3; bi++) {
      int B = Bs[bi];
      const uint32_t *pp[MAXB];
      for (int p = 0; p < B; p++) { make_prompt(prompts[p], PROMPT_LEN, bi * 10 + p); pp[p] = prompts[p]; }

      for (int ri = 0; ri < 2; ri++) {
        float rp = RPs[ri];
        printf("\n==== B=%d  repeat_penalty=%.2f ====\n", B, rp);

        /* Reference A: sequential greedy, per page (tokens + text). */
        uint32_t refTok[MAXB][GEN]; int refN[MAXB]; char *refText[MAXB];
        for (int p = 0; p < B; p++) {
            if (gen_seq(model, pp[p], PROMPT_LEN, rp, GEN, refTok[p], &refN[p]) != 0) {
                fprintf(stderr, "gen_seq failed\n"); fail = 1; goto out;
            }
            refText[p] = decode_text(model, refTok[p], refN[p]);
        }

        /* Token-for-token: batched forward_batch loop must equal sequential A. */
        bslot *bs = malloc((size_t)B * sizeof(bslot));
        if (gen_batch_tokens(model, pp, B, rp, GEN, bs) != 0) { fprintf(stderr, "batch tokens failed\n"); fail = 1; free(bs); goto out; }
        int tokdiff = 0;
        for (int p = 0; p < B; p++) {
            if (bs[p].no != refN[p]) tokdiff++;
            else for (int j = 0; j < refN[p]; j++) if (bs[p].tok[j] != refTok[p][j]) { tokdiff++; break; }
        }
        printf("  [tok] batched forward_batch vs sequential: %s\n", tokdiff ? "DIFFERS" : "token-identical");
        if (tokdiff) fail = 1;
        free(bs);

        /* B: plain batch decoder. Compare text byte-for-byte to A. */
        {
            pn_state *st[MAXB]; uint32_t seeds[MAXB]; char *txt[MAXB] = {0};
            for (int p = 0; p < B; p++) st[p] = prefill_state(model, pp[p], PROMPT_LEN, &seeds[p]);
            if (run_decoder(model, st, seeds, B, rp, GEN, B, 0, 0, NULL, NULL, txt, NULL, NULL, NULL) != 0) { fail = 1; }
            int bad = 0;
            for (int p = 0; p < B; p++) { if (!txt[p] || strcmp(txt[p], refText[p]) != 0) bad++; free(txt[p]); }
            printf("  [B ] plain batch decoder text: %s\n", bad ? "DIFFERS" : "byte-identical");
            if (bad) fail = 1;
        }

        /* C: step_spec across K and draft strategies. */
        for (int ki = 0; ki < 3; ki++) {
            int K = Ks[ki];
            int nkind = (K == 0) ? 1 : 4;   /* K=0: only "none" */
            for (int kd = 0; kd < nkind; kd++) {
                dkind kind = kinds[kd];
                pn_state *st[MAXB]; uint32_t seeds[MAXB]; char *txt[MAXB] = {0};
                pn_draft_fn dfns[MAXB]; void *duds[MAXB]; dctx *ctxs[MAXB];
                for (int p = 0; p < B; p++) {
                    st[p] = prefill_state(model, pp[p], PROMPT_LEN, &seeds[p]);
                    ctxs[p] = malloc(sizeof(dctx));
                    *ctxs[p] = (dctx){ refTok[p], refN[p], kind, 0xBEEF + p * 131 + kd };
                    dfns[p] = (K == 0 || kind == D_NONE) ? NULL : draft_cb;
                    duds[p] = ctxs[p];
                }
                long passes = 0, drafted = 0, accepted = 0;
                if (run_decoder(model, st, seeds, B, rp, GEN, B, 1, K, dfns, duds,
                                txt, &passes, &drafted, &accepted) != 0) fail = 1;
                int bad = 0;
                for (int p = 0; p < B; p++) { if (!txt[p] || strcmp(txt[p], refText[p]) != 0) bad++; free(txt[p]); free(ctxs[p]); }
                double ar = drafted ? (double)accepted / (double)drafted : 0.0;
                printf("  [C ] K=%d %-8s text %s  passes=%ld drafted=%ld accepted=%ld acc=%.2f\n",
                       K, kname[kd], bad ? "DIFFERS" : "byte-identical",
                       passes, drafted, accepted, ar);
                if (bad) fail = 1;
            }
        }

        for (int p = 0; p < B; p++) free(refText[p]);
      }
    }

    /* Mixed spec/plain slots + staggered admit, B=6: pages 0,2,4 speculate
     * (oracle), pages 1,3,5 ride plain; pages 3..5 admitted after two steps. */
    printf("\n==== mixed spec/plain + staggered admit (B=6, rp=1.15) ====\n");
    {
        float rp = 1.15f; int B = 6, max_batch = 6, max_tokens = GEN;
        const uint32_t *pp[6];
        for (int p = 0; p < 6; p++) { make_prompt(prompts[p], PROMPT_LEN, 200 + p); pp[p] = prompts[p]; }
        uint32_t refTok[6][GEN]; int refN[6]; char *refText[6];
        for (int p = 0; p < 6; p++) { gen_seq(model, pp[p], PROMPT_LEN, rp, max_tokens, refTok[p], &refN[p]); refText[p] = decode_text(model, refTok[p], refN[p]); }

        pn_sampling_params params = mkparams(rp);
        pn_batch_decoder *d = pn_batch_decoder_new(model, max_batch, max_tokens, &params);
        pn_batch_decoder_spec_config(d, 8, 0.10, 4);
        dctx *ctxs[6]; char *txt[6] = {0};
        pn_state *st[6]; uint32_t seeds[6];
        for (int p = 0; p < 6; p++) {
            st[p] = prefill_state(model, pp[p], PROMPT_LEN, &seeds[p]);
            ctxs[p] = malloc(sizeof(dctx));
            *ctxs[p] = (dctx){ refTok[p], refN[p], D_ORACLE, 0u };
        }
        /* admit 0..2 up front; 0 & 2 speculate, 1 plain */
        pn_batch_decoder_admit_spec(d, st[0], 0, seeds[0], draft_cb, ctxs[0]);
        pn_batch_decoder_admit_spec(d, st[1], 1, seeds[1], NULL, NULL);           /* plain slot */
        pn_batch_decoder_admit_spec(d, st[2], 2, seeds[2], draft_cb, ctxs[2]);
        pn_page_out ob[6];
        int admitted_more = 0, steps = 0;
        while (pn_batch_decoder_active(d) > 0 || !admitted_more) {
            int nf = pn_batch_decoder_step_spec(d, ob, 6);
            if (nf < 0) { fail = 1; break; }
            for (int i = 0; i < nf; i++) { txt[ob[i].page_id] = ob[i].text; pn_state_free(ob[i].state); }
            steps++;
            if (steps == 2 && !admitted_more) {   /* staggered admit of 3..5 */
                pn_batch_decoder_admit_spec(d, st[3], 3, seeds[3], NULL, NULL);       /* plain */
                pn_batch_decoder_admit_spec(d, st[4], 4, seeds[4], draft_cb, ctxs[4]);
                pn_batch_decoder_admit_spec(d, st[5], 5, seeds[5], draft_cb, ctxs[5]);
                admitted_more = 1;
            }
            if (admitted_more && pn_batch_decoder_active(d) == 0) break;
        }
        int bad = 0;
        for (int p = 0; p < 6; p++) { if (!txt[p] || strcmp(txt[p], refText[p]) != 0) bad++; free(txt[p]); free(ctxs[p]); free(refText[p]); }
        printf("  mixed/staggered text: %s\n", bad ? "DIFFERS" : "byte-identical (all 6 pages)");
        if (bad) fail = 1;
        pn_batch_decoder_free(d);
    }

    /* Forced stop cases: (1) seed == EOS -> immediate finalize, empty output;
     * (2) max_tokens truncation -> both plain and spec cut at the same point. */
    printf("\n==== forced-stop handling ====\n");
    if (g_eos >= 0) {
        float rp = 1.0f;
        uint32_t pr[PROMPT_LEN]; make_prompt(pr, PROMPT_LEN, 999);
        /* (1) EOS seed */
        uint32_t junk; pn_state *sa = prefill_state(model, pr, PROMPT_LEN, &junk);
        pn_state *sb = prefill_state(model, pr, PROMPT_LEN, &junk);
        uint32_t seeds[1]; pn_state *st1[1];
        char *tp = NULL, *ts = NULL;
        seeds[0] = (uint32_t)g_eos; st1[0] = sa;
        run_decoder(model, st1, seeds, 1, rp, GEN, 1, 0, 0, NULL, NULL, &tp, NULL, NULL, NULL);
        st1[0] = sb; dctx c = { NULL, 0, D_NONE, 0 }; pn_draft_fn df[1] = { NULL }; void *du[1] = { &c };
        run_decoder(model, st1, seeds, 1, rp, GEN, 1, 1, 8, df, du, &ts, &(long){0}, &(long){0}, &(long){0});
        int ok = tp && ts && tp[0] == '\0' && ts[0] == '\0';
        printf("  EOS seed: plain=%s spec=%s  %s\n", tp && !tp[0] ? "empty" : "NONEMPTY",
               ts && !ts[0] ? "empty" : "NONEMPTY", ok ? "OK" : "FAIL");
        if (!ok) fail = 1;
        free(tp); free(ts);

        /* (2) max_tokens truncation: plain and spec must both equal A[0..N-1]. */
        int N = 5;
        uint32_t rt[GEN]; int rn; gen_seq(model, pr, PROMPT_LEN, rp, N, rt, &rn);
        char *refT = decode_text(model, rt, rn);
        uint32_t s2; pn_state *sc = prefill_state(model, pr, PROMPT_LEN, &s2);
        pn_state *sd = prefill_state(model, pr, PROMPT_LEN, &s2);
        char *tcp = NULL, *tcs = NULL; uint32_t sd2[1] = { s2 }; pn_state *stc[1] = { sc };
        run_decoder(model, stc, sd2, 1, rp, N, 1, 0, 0, NULL, NULL, &tcp, NULL, NULL, NULL);
        stc[0] = sd; dctx c2 = { rt, rn, D_ORACLE, 0 }; pn_draft_fn df2[1] = { draft_cb }; void *du2[1] = { &c2 };
        run_decoder(model, stc, sd2, 1, rp, N, 1, 1, 8, df2, du2, &tcs, &(long){0}, &(long){0}, &(long){0});
        int ok2 = tcp && tcs && strcmp(tcp, refT) == 0 && strcmp(tcs, refT) == 0;
        printf("  max_tokens=%d truncation: plain=%s spec=%s  %s\n", N,
               (tcp && !strcmp(tcp, refT)) ? "match" : "DIFF",
               (tcs && !strcmp(tcs, refT)) ? "match" : "DIFF", ok2 ? "OK" : "FAIL");
        if (!ok2) fail = 1;
        free(tcp); free(tcs); free(refT);
    }

out:
    printf("\nRESULT: %s\n", fail ? "FAIL"
        : "PASS (batched speculative decode == plain batched greedy, byte for byte)");
    pn_model_free(model);
    return fail ? 1 : 0;
}
