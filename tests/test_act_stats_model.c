/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_act_stats_model.c
 * @brief PN_ACT_STATS end to end on a tiny synthetic dense qwen35 GGUF (3 GDN layers + 1
 *   attention layer, dense SwiGLU FFN, untied LM head: the Qwen3.8 layer mix), through the real
 *   pn_load / forward / pn_model_free. Each run is a child process (the split knobs are parsed
 *   once per process).
 *
 *   - logits with PN_ACT_STATS on equal logits with it off, byte for byte
 *   - every dense matmul tensor the forward runs is in the file, count = tokens, n_in = cols
 *   - blk.0.attn_qkv's sum x^2 = a naive recomputation of rmsnorm(embedding) * attn_norm (the
 *     input after the norm); tensors sharing an input carry identical stats
 *   - PN_ACT_STATS_FULL: X^T X diagonal = the imatrix, symmetric, = the naive outer products
 *   - PN_SPLIT_LAYERS halves: disjoint tensors, the first half has no LM head, and the merge of
 *     the two files equals the unsplit file exactly
 *   - an unknown PN_ACT_STATS_FULL name refuses the load
 *
 * @status None
 * @issues None
 * @todo None
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "tensor/tensor.h"
#include "../include/pantheon.h"
#include "../src/core/act_stats_file.h"
#include "../src/core/engine.h"

#define OUT_DIR "build/act_stats"
#define GGUF_PATH OUT_DIR "/tiny_qwen35.gguf"

/* Tiny Qwen3.8-shaped config: interval 4 -> layers 0..2 GDN, layer 3 attention. */
#define N_LAYER 4
#define N_EMBD 64
#define N_HEAD 2
#define N_HEAD_KV 1
#define HEAD_DIM 16
#define SSM_STATE 16
#define SSM_GROUPS 2
#define SSM_DT_RANK 4
#define SSM_INNER (SSM_STATE * SSM_DT_RANK)
#define SSM_CONV 4
#define CONV_DIM (2 * SSM_STATE * SSM_GROUPS + SSM_INNER)
#define FFN 96
#define VOCAB 32
#define RMS_EPS 1e-6f
#define MAX_TENSORS 64

static const int run_tokens[] = { 1, 5, 9, 2, 30, 7 };
#define N_TOK (int)(sizeof(run_tokens) / sizeof(run_tokens[0]))

static int checks, failures;

static void check(int ok, const char *what) {
    checks++;
    failures += !ok;
    if (!ok)
        printf("  FAIL  %s\n", what);
}

/* ---- the synthetic GGUF ---------------------------------------------------------------- */

typedef struct {
    char     name[64];
    uint64_t d0, d1;   /* d1 = 0: 1-D */
    float   *data;
} tiny_tensor;

static tiny_tensor tiny[MAX_TENSORS];
static int n_tiny;

static float rand_value(unsigned *state) {
    *state = *state * 1664525u + 1013904223u;
    return (float)(*state >> 8) / 16777216.0f - 0.5f;
}

/* Add a tensor filled with base + scale * U(-0.5, 0.5). */
static float *add_tensor(const char *name, uint64_t d0, uint64_t d1, float base, float scale) {
    static unsigned seed = 12345;
    tiny_tensor *t = &tiny[n_tiny++];
    uint64_t n = d0 * (d1 ? d1 : 1);
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->d0 = d0;
    t->d1 = d1;
    t->data = malloc((size_t)n * sizeof(float));
    for (uint64_t i = 0; i < n; i++)
        t->data[i] = base + scale * rand_value(&seed);
    return t->data;
}

static float *find_tiny(const char *name) {
    for (int i = 0; i < n_tiny; i++)
        if (strcmp(tiny[i].name, name) == 0)
            return tiny[i].data;
    return NULL;
}

static void build_tensors(void) {
    char n[64];
    add_tensor("token_embd.weight", N_EMBD, VOCAB, 0.0f, 2.0f);
    add_tensor("output_norm.weight", N_EMBD, 0, 1.0f, 0.2f);
    add_tensor("output.weight", N_EMBD, VOCAB, 0.0f, 0.3f);
    for (int l = 0; l < N_LAYER; l++) {
#define T(suffix, d0, d1, base, scale) \
        (snprintf(n, sizeof(n), "blk.%d." suffix, l), add_tensor(n, d0, d1, base, scale))
        T("attn_norm.weight", N_EMBD, 0, 1.0f, 0.5f);
        T("post_attention_norm.weight", N_EMBD, 0, 1.0f, 0.2f);
        T("ffn_gate.weight", N_EMBD, FFN, 0.0f, 0.3f);
        T("ffn_up.weight", N_EMBD, FFN, 0.0f, 0.3f);
        T("ffn_down.weight", FFN, N_EMBD, 0.0f, 0.2f);
        if (l == N_LAYER - 1) {
            T("attn_q.weight", N_EMBD, 2 * HEAD_DIM * N_HEAD, 0.0f, 0.3f);
            T("attn_k.weight", N_EMBD, HEAD_DIM * N_HEAD_KV, 0.0f, 0.3f);
            T("attn_v.weight", N_EMBD, HEAD_DIM * N_HEAD_KV, 0.0f, 0.3f);
            T("attn_output.weight", HEAD_DIM * N_HEAD, N_EMBD, 0.0f, 0.3f);
            T("attn_q_norm.weight", HEAD_DIM, 0, 1.0f, 0.2f);
            T("attn_k_norm.weight", HEAD_DIM, 0, 1.0f, 0.2f);
        } else {
            T("attn_qkv.weight", N_EMBD, CONV_DIM, 0.0f, 0.3f);
            T("attn_gate.weight", N_EMBD, SSM_INNER, 0.0f, 0.3f);
            T("ssm_alpha.weight", N_EMBD, SSM_DT_RANK, 0.0f, 0.3f);
            T("ssm_beta.weight", N_EMBD, SSM_DT_RANK, 0.0f, 0.3f);
            T("ssm_out.weight", SSM_INNER, N_EMBD, 0.0f, 0.3f);
            T("ssm_conv1d.weight", SSM_CONV, CONV_DIM, 0.0f, 0.5f);
            snprintf(n, sizeof(n), "blk.%d.ssm_a", l);
            add_tensor(n, SSM_DT_RANK, 0, -0.3f, 0.2f);
            T("ssm_dt.bias", SSM_DT_RANK, 0, 0.0f, 0.2f);
            T("ssm_norm.weight", SSM_STATE, 0, 1.0f, 0.2f);
        }
#undef T
    }
}

static void put_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void put_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void put_str(FILE *f, const char *s) {
    put_u64(f, strlen(s));
    fwrite(s, 1, strlen(s), f);
}
static void kv_u32(FILE *f, const char *key, uint32_t v) {
    put_str(f, key);
    put_u32(f, 4);   /* GGUF_TYPE_UINT32 */
    put_u32(f, v);
}
static void kv_f32(FILE *f, const char *key, float v) {
    put_str(f, key);
    put_u32(f, 6);   /* GGUF_TYPE_FLOAT32 */
    fwrite(&v, 4, 1, f);
}
static void kv_str(FILE *f, const char *key, const char *v) {
    put_str(f, key);
    put_u32(f, 8);   /* GGUF_TYPE_STRING */
    put_str(f, v);
}

/* Little-endian host assumed (every Pantheon host is). */
static int write_tiny_gguf(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f)
        return -1;
    fwrite("GGUF", 1, 4, f);
    put_u32(f, 3);
    put_u64(f, (uint64_t)n_tiny);
    put_u64(f, 22);
    kv_str(f, "general.architecture", "qwen35");
    kv_u32(f, "qwen35.block_count", N_LAYER);
    kv_u32(f, "qwen35.full_attention_interval", 4);
    kv_u32(f, "qwen35.embedding_length", N_EMBD);
    kv_u32(f, "qwen35.feed_forward_length", FFN);
    kv_u32(f, "qwen35.attention.head_count", N_HEAD);
    kv_u32(f, "qwen35.attention.head_count_kv", N_HEAD_KV);
    kv_u32(f, "qwen35.attention.key_length", HEAD_DIM);
    kv_u32(f, "qwen35.attention.value_length", HEAD_DIM);
    kv_u32(f, "qwen35.rope.dimension_count", 8);
    kv_f32(f, "qwen35.rope.freq_base", 10000.0f);
    kv_f32(f, "qwen35.attention.layer_norm_rms_epsilon", RMS_EPS);
    kv_u32(f, "qwen35.ssm.state_size", SSM_STATE);
    kv_u32(f, "qwen35.ssm.group_count", SSM_GROUPS);
    kv_u32(f, "qwen35.ssm.time_step_rank", SSM_DT_RANK);
    kv_u32(f, "qwen35.ssm.inner_size", SSM_INNER);
    kv_u32(f, "qwen35.ssm.conv_kernel", SSM_CONV);
    kv_u32(f, "qwen35.context_length", 64);
    kv_str(f, "general.name", "tiny-qwen35-act-stats");
    kv_str(f, "tokenizer.ggml.model", "gpt2");
    kv_u32(f, "general.alignment", 32);
    put_str(f, "tokenizer.ggml.tokens");
    put_u32(f, 9);   /* GGUF_TYPE_ARRAY */
    put_u32(f, 8);   /* of GGUF_TYPE_STRING */
    put_u64(f, VOCAB);
    for (int v = 0; v < VOCAB; v++) {
        char tok[8];
        snprintf(tok, sizeof(tok), "t%d", v);
        put_str(f, tok);
    }
    uint64_t offset = 0;
    for (int i = 0; i < n_tiny; i++) {
        put_str(f, tiny[i].name);
        put_u32(f, tiny[i].d1 ? 2 : 1);
        put_u64(f, tiny[i].d0);
        if (tiny[i].d1)
            put_u64(f, tiny[i].d1);
        put_u32(f, 0);   /* F32 */
        put_u64(f, offset);
        offset += (tiny[i].d0 * (tiny[i].d1 ? tiny[i].d1 : 1) * 4 + 31) & ~31ull;
    }
    long pos = ftell(f);
    for (long p = pos; p % 32; p++)
        fputc(0, f);
    for (int i = 0; i < n_tiny; i++) {
        uint64_t bytes = tiny[i].d0 * (tiny[i].d1 ? tiny[i].d1 : 1) * 4;
        fwrite(tiny[i].data, 1, bytes, f);
        for (uint64_t p = bytes; p % 32; p++)
            fputc(0, f);
    }
    return fclose(f) == 0 ? 0 : -1;
}

/* ---- child runs ------------------------------------------------------------------------ */

/* Child: load, forward run_tokens, write the logits, free (which writes the stats). */
static int child_run(const char *gguf, const char *logits_path) {
    pn_model *model = pn_load(gguf);
    if (!model)
        return 3;
    pn_state *state = pn_state_new(model);
    FILE *f = fopen(logits_path, "wb");
    int rc = state && f ? 0 : 4;
    if (rc)
        fprintf(stderr, "child: %s\n", state ? "cannot open the logits file" : "pn_state_new failed");
    for (int i = 0; i < N_TOK && rc == 0; i++) {
        tm_tensor *out = model->forward(model, (uint32_t)run_tokens[i], state);
        if (!out) {
            rc = 5;
            break;
        }
        fwrite(tm_data_f32(out), sizeof(float), (size_t)out->shape[0], f);
        tm_free(out);
    }
    if (f)
        fclose(f);
    pn_state_free(state);
    pn_model_free(model);
    return rc;
}

/* Run this binary as a child with `env` ("K=V" strings, NULL-terminated). Returns its exit code. */
static int spawn_run(const char *self, const char *tag, const char *const *env) {
    char logits[256], log[256];
    snprintf(logits, sizeof(logits), OUT_DIR "/%s.logits", tag);
    snprintf(log, sizeof(log), OUT_DIR "/%s.log", tag);
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        unsetenv("PN_ACT_STATS");
        unsetenv("PN_ACT_STATS_FULL");
        unsetenv("PN_SPLIT_LAYERS");
        setenv("QWEN35_KV_Q4", "0", 1);   /* as the oracle runs; Q4 KV needs head_dim >= 32 */
        for (int i = 0; env && env[i]; i++)
            putenv((char *)env[i]);
        if (!freopen(log, "w", stderr))
            _exit(9);
        execl(self, self, "--run", GGUF_PATH, logits, (char *)NULL);
        _exit(8);
    }
    int status = 0;
    if (pid <= 0 || waitpid(pid, &status, 0) != pid)
        return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

static int files_equal(const char *a, const char *b) {
    FILE *fa = fopen(a, "rb"), *fb = fopen(b, "rb");
    int same = fa && fb, ca, cb;
    while (same && (ca = fgetc(fa)) != EOF) {
        cb = fgetc(fb);
        same = ca == cb;
    }
    same = same && fgetc(fb) == EOF;
    if (fa)
        fclose(fa);
    if (fb)
        fclose(fb);
    return same;
}

/* ---- checks ---------------------------------------------------------------------------- */

static int entry_equal(const pn_act_stats_entry *a, const pn_act_stats_entry *b, int xtx) {
    uint64_t len = pn_act_stats_payload_len(xtx, a->n_in);
    return a->n_in == b->n_in && a->count == b->count &&
           memcmp(a->payload, b->payload, (size_t)len * sizeof(double)) == 0;
}

/* Every dense matmul tensor of the forward, with its input width. */
static void check_coverage(pn_act_stats_table *t) {
    static const char *const gdn[] = { "attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta" };
    static const char *const attn[] = { "attn_q", "attn_k", "attn_v" };
    char name[64];
    int expected = 0, ok = 1;
#define EXPECT(width) do { \
        pn_act_stats_entry *e = pn_act_stats_table_find(t, name); \
        expected++; \
        if (!e || e->count != (uint64_t)N_TOK || e->n_in != (uint32_t)(width)) { \
            ok = 0; \
            printf("        coverage: %s %s\n", name, e ? "count/n_in wrong" : "missing"); \
        } \
    } while (0)
    for (int l = 0; l < N_LAYER; l++) {
        int is_attn = l == N_LAYER - 1;
        for (int k = 0; k < (is_attn ? 3 : 4); k++) {
            snprintf(name, sizeof(name), "blk.%d.%s.weight", l, is_attn ? attn[k] : gdn[k]);
            EXPECT(N_EMBD);
        }
        snprintf(name, sizeof(name), "blk.%d.%s.weight", l, is_attn ? "attn_output" : "ssm_out");
        EXPECT(is_attn ? HEAD_DIM * N_HEAD : SSM_INNER);
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
        EXPECT(N_EMBD);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
        EXPECT(N_EMBD);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
        EXPECT(FFN);
    }
    snprintf(name, sizeof(name), "output.weight");
    EXPECT(N_EMBD);
#undef EXPECT
    check(ok, "every dense matmul tensor present, count = tokens, n_in = cols");
    check(t->n_entries == (uint32_t)expected, "no other tensors recorded");
    check(t->n_tokens == (uint64_t)N_TOK, "header n_tokens = tokens");
}

/* The layer-0 input after attn_norm, recomputed naively in double, one row per token. */
static void naive_layer0_inputs(double x[N_TOK][N_EMBD]) {
    const float *embd = find_tiny("token_embd.weight");
    const float *norm = find_tiny("blk.0.attn_norm.weight");
    for (int t = 0; t < N_TOK; t++) {
        const float *e = embd + (size_t)run_tokens[t] * N_EMBD;
        double ss = 0.0;
        for (int j = 0; j < N_EMBD; j++)
            ss += (double)e[j] * e[j];
        double r = 1.0 / sqrt(ss / N_EMBD + RMS_EPS);
        for (int j = 0; j < N_EMBD; j++)
            x[t][j] = e[j] * r * norm[j];
    }
}

static int close_rel(double a, double b, double tol) {
    return fabs(a - b) <= tol * (fabs(b) + 1e-3);
}

static void check_semantics(pn_act_stats_table *t, pn_act_stats_table *xtx) {
    double x[N_TOK][N_EMBD];
    naive_layer0_inputs(x);
    pn_act_stats_entry *qkv = pn_act_stats_table_find(t, "blk.0.attn_qkv.weight");
    if (!qkv) {
        check(0, "blk.0.attn_qkv present");
        return;
    }
    int bad = 0;
    for (int j = 0; j < N_EMBD; j++) {
        double ref = 0.0;
        for (int k = 0; k < N_TOK; k++)
            ref += x[k][j] * x[k][j];
        bad += !close_rel(qkv->payload[j], ref, 1e-5);
    }
    check(bad == 0, "blk.0.attn_qkv sum x^2 = naive sum of (rmsnorm(embedding) * attn_norm)^2");
    static const char *const same_input[] = { "blk.0.attn_gate.weight", "blk.0.ssm_alpha.weight",
                                              "blk.0.ssm_beta.weight" };
    for (int k = 0; k < 3; k++) {
        pn_act_stats_entry *e = pn_act_stats_table_find(t, same_input[k]);
        check(e && entry_equal(e, qkv, 0), "GDN projections sharing attn_qkv's input match it");
    }
    pn_act_stats_entry *gate = pn_act_stats_table_find(t, "blk.2.ffn_gate.weight");
    pn_act_stats_entry *up = pn_act_stats_table_find(t, "blk.2.ffn_up.weight");
    check(gate && up && entry_equal(gate, up, 0), "ffn_gate and ffn_up share stats");

    pn_act_stats_entry *xq = pn_act_stats_table_find(xtx, "blk.0.attn_qkv.weight");
    pn_act_stats_entry *xo = pn_act_stats_table_find(xtx, "blk.3.attn_output.weight");
    check(xtx->n_entries == 2 && xq && xo && xq->count == (uint64_t)N_TOK &&
          xo->count == (uint64_t)N_TOK, "X^T X: both FULL tensors, count = tokens");
    if (!xq)
        return;
    int diag = 0, sym = 0, outer = 0;
    for (int i = 0; i < N_EMBD; i++) {
        diag += xq->payload[(size_t)i * N_EMBD + i] != qkv->payload[i];
        for (int j = 0; j < N_EMBD; j++) {
            double ref = 0.0;
            for (int k = 0; k < N_TOK; k++)
                ref += x[k][i] * x[k][j];
            sym += xq->payload[(size_t)i * N_EMBD + j] != xq->payload[(size_t)j * N_EMBD + i];
            /* Error scale for a sum of products of float-rounded inputs: the Cauchy-Schwarz
             * bound sqrt(D_ii D_jj), not |ref| (an off-diagonal sum can cancel to ~0). */
            double scale = sqrt(qkv->payload[i] * qkv->payload[j]);
            outer += fabs(xq->payload[(size_t)i * N_EMBD + j] - ref) > 1e-5 * scale;
        }
    }
    check(diag == 0, "X^T X diagonal = imatrix sum x^2 exactly");
    check(sym == 0, "X^T X symmetric");
    check(outer == 0, "X^T X = naive sum of outer products");
}

/* Split halves: disjoint, the head only in the second, merge = the whole run. */
static void check_split(pn_act_stats_table *full) {
    pn_act_stats_table h1, h2;
    check(pn_act_stats_table_read(&h1, OUT_DIR "/h1.imatrix", 0) == 0 &&
          pn_act_stats_table_read(&h2, OUT_DIR "/h2.imatrix", 0) == 0, "both halves wrote a file");
    check(pn_act_stats_table_find(&h1, "output.weight") == NULL,
          "first half records no LM head (its input is mid-stack)");
    int overlap = 0;
    for (uint32_t i = 0; i < h1.n_entries; i++)
        overlap += pn_act_stats_table_find(&h2, h1.entries[i].name) != NULL;
    check(overlap == 0, "halves hold disjoint tensors");
    check(pn_act_stats_table_merge(&h1, &h2) == 0, "halves merge");
    int same = h1.n_entries == full->n_entries;
    for (uint32_t i = 0; i < full->n_entries && same; i++) {
        pn_act_stats_entry *e = pn_act_stats_table_find(&h1, full->entries[i].name);
        same = e && entry_equal(e, &full->entries[i], 0);
    }
    check(same, "merged halves = unsplit run, every tensor byte for byte");
    pn_act_stats_table_free(&h1);
    pn_act_stats_table_free(&h2);
}

static void test_model_runs(const char *self) {
    build_tensors();
    check(write_tiny_gguf(GGUF_PATH) == 0, "synthetic GGUF written");

    const char *const on_env[] = { "PN_ACT_STATS=" OUT_DIR "/full.imatrix",
        "PN_ACT_STATS_FULL=blk.0.attn_qkv.weight,blk.3.attn_output.weight", NULL };
    const char *const h1_env[] = { "PN_ACT_STATS=" OUT_DIR "/h1.imatrix", "PN_SPLIT_LAYERS=0:2",
        "PN_SPLIT_RESID_OUT=" OUT_DIR "/resid.f32", NULL };
    const char *const h2_env[] = { "PN_ACT_STATS=" OUT_DIR "/h2.imatrix", "PN_SPLIT_LAYERS=2:4",
        "PN_SPLIT_RESID_IN=" OUT_DIR "/resid.f32", NULL };
    const char *const bad_env[] = { "PN_ACT_STATS=" OUT_DIR "/bad.imatrix",
        "PN_ACT_STATS_FULL=blk.0.no_such.weight", NULL };

    check(spawn_run(self, "off", NULL) == 0, "knob-off run");
    check(spawn_run(self, "on", on_env) == 0, "knob-on run");
    check(files_equal(OUT_DIR "/off.logits", OUT_DIR "/on.logits"), "logits byte-identical, knob on vs off");
    check(spawn_run(self, "h1", h1_env) == 0 && spawn_run(self, "h2", h2_env) == 0, "split halves run");
    check(files_equal(OUT_DIR "/off.logits", OUT_DIR "/h2.logits"), "split second half = unsplit logits");
    check(spawn_run(self, "bad", bad_env) == 3, "unknown PN_ACT_STATS_FULL name refuses the load");

    pn_act_stats_table full, xtx;
    if (pn_act_stats_table_read(&full, OUT_DIR "/full.imatrix", 0) != 0 ||
        pn_act_stats_table_read(&xtx, OUT_DIR "/full.imatrix.xtx", 1) != 0) {
        check(0, "knob-on run wrote the imatrix and X^T X files");
        return;
    }
    check_coverage(&full);
    check_semantics(&full, &xtx);
    check_split(&full);
    pn_act_stats_table_free(&full);
    pn_act_stats_table_free(&xtx);
}

int main(int argc, char **argv) {
    if (argc == 4 && strcmp(argv[1], "--run") == 0)
        return child_run(argv[2], argv[3]);
    mkdir("build", 0755);
    mkdir(OUT_DIR, 0755);
    test_model_runs(argv[0]);
    printf("test_act_stats_model: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
