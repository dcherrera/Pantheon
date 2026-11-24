/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_ingot.c — Ingot Phase 1b: catalog files[].ingot downloads, decoded while they
 * arrive, against the local HTTP fixture (tests/local_fixture_httpd.h; it cuts the first
 * GET of every file after 1 MiB). Fixtures are synthetic GGUF v3 files (Q8_0 tensors whose
 * blocks are built like ggml's: d = fp16(amax / 127) of BF16-valued weights, codes =
 * round(w / d); plus an F32 norm), encoded with foundry's ingot_encode_file.
 *   - a 200+ MB model installs from its .ingot alone (the whole file is not served),
 *     byte-identical, in a child whose peak RSS stays under a fixed bound (so the decode
 *     cannot hold the file);
 *   - a cut transfer restarts curl at the bytes read and the same decode finishes;
 *   - a corrupt .ingot fails and leaves no part file (no fallback);
 *   - a delta .ingot decodes against its installed base; with the base missing or wrong the
 *     box downloads the whole file, and fails cleanly when that is not there either;
 *   - a part file left by an earlier run resumes the whole file when that moves fewer bytes
 *     than the .ingot, else the .ingot is fetched again.
 */

#include "local_test_support.h"
#include "local_fixture_httpd.h"

#include <fcntl.h>
#include <math.h>
#include <sys/resource.h>

#include "ingot/container/ingot_base.h"
#include "ingot/container/ingot_writer.h"

#include "../src/local/curl_runner.h"
#include "../src/local/downloader.h"
#include "../src/local/ingot_fetch.h"
#include "../src/local/model_store.h"
#include "../src/local/sha256.h"

#define DIR    "build/test_local/ingot"
#define REV    "0123456789abcdef0123456789abcdef01234567"
#define WWW    DIR "/www/teamide/test-fixtures/resolve/" REV
#define STORE  DIR "/store"
#define CUT    (1024 * 1024)
#define RSS_BOUND ((uint64_t)64 << 20)   /* the decode child's peak: far below the 200 MB file */

#define GGML_F32  0
#define GGML_Q8_0 8
#define Q8_BLOCK  34

typedef struct {
    const char *name;
    uint32_t    type;
    uint64_t    ne0, ne1;   /* ne1 = 1: a vector */
    uint64_t    offset;     /* out: file offset of the data */
} fx_tensor;

typedef struct {
    char     path[512];
    char     sha[65];
    uint64_t size;
} fx_file;

/* ---- synthetic GGUF ---------------------------------------------------------------- */

typedef struct {
    FILE         *f;
    pn_sha256_ctx sha;
    uint64_t      pos;
} fx_out;

static void out_bytes(fx_out *o, const void *p, size_t n)
{
    if (fwrite(p, 1, n, o->f) != n)
        CHECK(0, "fixture write");
    pn_sha256_update(&o->sha, p, n);
    o->pos += n;
}

static void out_u32(fx_out *o, uint32_t v) { out_bytes(o, &v, 4); }
static void out_u64(fx_out *o, uint64_t v) { out_bytes(o, &v, 8); }

static void out_str(fx_out *o, const char *s)
{
    out_u64(o, strlen(s));
    out_bytes(o, s, strlen(s));
}

static void out_pad(fx_out *o)
{
    static const uint8_t zero[32];
    out_bytes(o, zero, (size_t)((32 - o->pos % 32) % 32));
}

static uint64_t tensor_bytes(const fx_tensor *t)
{
    return t->type == GGML_Q8_0 ? t->ne0 / 32 * Q8_BLOCK * t->ne1 : t->ne0 * 4 * t->ne1;
}

static uint32_t rng_next(uint32_t *s)
{
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s;
}

/* Unit-variance noise (Irwin-Hall of 4), rounded to a BF16 value. */
static float bf16_noise(uint32_t *s, float sigma)
{
    float u = 0;
    for (int i = 0; i < 4; i++)
        u += (float)(rng_next(s) >> 8) * (1.0f / 16777216.0f);
    float w = (u - 2.0f) * 1.7320508f * sigma;
    uint32_t x;
    memcpy(&x, &w, 4);
    x = (x + 0x7fffu + ((x >> 16) & 1u)) & 0xffff0000u;
    memcpy(&w, &x, 4);
    return w;
}

static uint16_t fp16_bits(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u, man = x & 0x7fffffu;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    if (exp >= 31)
        return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10)
            return (uint16_t)sign;
        uint32_t m = man | 0x800000u, shift = (uint32_t)(14 - exp);
        uint32_t h = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        return (uint16_t)(sign | (h + (rem > half || (rem == half && (h & 1)))));
    }
    uint32_t h = ((uint32_t)exp << 10) | (man >> 13), rem = man & 0x1fffu;
    return (uint16_t)(sign | (h + (rem > 0x1000u || (rem == 0x1000u && (h & 1)))));
}

/* One Q8_0 row, quantized the way ggml's quantize_row_q8_0_ref does. */
static void q8_row(uint8_t *row, uint64_t ne0, uint32_t *s, float sigma)
{
    float w[32];
    for (uint64_t b = 0; b < ne0 / 32; b++) {
        uint8_t *blk = row + b * Q8_BLOCK;
        float amax = 0;
        for (int j = 0; j < 32; j++) {
            w[j] = bf16_noise(s, sigma);
            amax = fabsf(w[j]) > amax ? fabsf(w[j]) : amax;
        }
        float d = amax / 127.0f, id = d != 0 ? 1.0f / d : 0.0f;
        uint16_t h = fp16_bits(d);
        memcpy(blk, &h, 2);
        for (int j = 0; j < 32; j++)
            blk[2 + j] = (uint8_t)(int8_t)roundf(w[j] * id);
    }
}

static void tensor_data(fx_out *o, const fx_tensor *t, uint32_t seed)
{
    uint32_t s = seed * 2654435761u + 1u;
    uint64_t row_bytes = tensor_bytes(t) / t->ne1;
    uint8_t *row = malloc((size_t)row_bytes);

    for (uint64_t r = 0; row && r < t->ne1; r++) {
        if (t->type == GGML_Q8_0) {
            q8_row(row, t->ne0, &s, 0.02f * (1.0f + (float)(r % 7) / 7.0f));
        } else {
            for (uint64_t i = 0; i < t->ne0; i++) {
                float v = 1.0f + bf16_noise(&s, 0.05f);
                memcpy(row + i * 4, &v, 4);
            }
        }
        out_bytes(o, row, (size_t)row_bytes);
    }
    free(row);
}

/* Write a GGUF v3 file (one metadata key, tensors in order, 32-byte aligned). */
static void write_gguf(const char *path, fx_tensor *t, int n, uint32_t seed, fx_file *out)
{
    fx_out o;
    uint64_t rel = 0;

    memset(&o, 0, sizeof(o));
    o.f = fopen(path, "wb");

    CHECK(o.f != NULL, path);
    if (!o.f)
        return;
    pn_sha256_init(&o.sha);
    out_bytes(&o, "GGUF", 4);
    out_u32(&o, 3);
    out_u64(&o, (uint64_t)n);
    out_u64(&o, 1);
    out_str(&o, "general.architecture");
    out_u32(&o, 8);
    out_str(&o, "llama");
    for (int i = 0; i < n; i++) {
        out_str(&o, t[i].name);
        out_u32(&o, t[i].ne1 > 1 ? 2 : 1);
        out_u64(&o, t[i].ne0);
        if (t[i].ne1 > 1)
            out_u64(&o, t[i].ne1);
        out_u32(&o, t[i].type);
        out_u64(&o, rel);
        rel += (tensor_bytes(&t[i]) + 31) / 32 * 32;
    }
    out_pad(&o);
    for (int i = 0; i < n; i++) {
        t[i].offset = o.pos;
        tensor_data(&o, &t[i], seed + (uint32_t)i);
        out_pad(&o);
    }
    fclose(o.f);
    uint8_t d[32];
    pn_sha256_final(&o.sha, d);
    pn_hex_encode(d, 32, out->sha);
    snprintf(out->path, sizeof(out->path), "%s", path);
    out->size = o.pos;
}

/* ---- helpers ------------------------------------------------------------------------ */

static void file_identity(const char *path, fx_file *out)
{
    uint8_t d[32];
    snprintf(out->path, sizeof(out->path), "%s", path);
    out->size = pn_model_store_file_size(path);
    out->sha[0] = '\0';
    if (pn_sha256_file(path, d) == 0)
        pn_hex_encode(d, 32, out->sha);
}

/* Run fn in a child; its exit status, and its rusage (it and the children it reaped). */
static int run_child(int (*fn)(void *), void *arg, struct rusage *ru)
{
    int status = 0;
    struct rusage scratch;

    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid == 0)
        _exit(fn(arg));
    if (pid < 0 || wait4(pid, &status, 0, ru ? ru : &scratch) != pid)
        return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

static uint64_t maxrss_bytes(const struct rusage *ru)
{
#ifdef __APPLE__
    return (uint64_t)ru->ru_maxrss;            /* bytes */
#else
    return (uint64_t)ru->ru_maxrss * 1024u;    /* KiB */
#endif
}

typedef struct {
    const char *src, *dst, *base;
} encode_job;

static int encode_child(void *arg)
{
    const encode_job *j = arg;
    ingot_encode_options opt = { .threads = 2 };
    ingot_base_set *base = NULL;

    if (j->base) {
        if (ingot_base_set_open(&j->base, 1, 2, &base) != 0)
            return 2;
        opt.base = base;
        opt.policy = INGOT_POLICY_DELTA;
    }
    int rc = ingot_encode_file(j->src, j->dst, &opt, NULL);
    ingot_base_set_close(base);
    return rc == 0 ? 0 : 1;
}

static void encode(const char *src, const char *dst, const char *base)
{
    encode_job j = { src, dst, base };
    CHECK(run_child(encode_child, &j, NULL) == 0, dst);
}

/* A one-file model of the catalog: `file` with an optional .ingot. */
static pn_cat_model *add_model(pn_catalog_manifest *cat, const char *id, const char *file,
                               const fx_file *whole, const char *ingot, uint64_t ingot_size,
                               const char *base_sha)
{
    pn_cat_model *m = &cat->models[cat->n_models++];
    pn_cat_file *f = &m->files[0];

    snprintf(m->id, sizeof(m->id), "%s", id);
    m->n_files = 1;
    snprintf(f->role, sizeof(f->role), "weights");
    snprintf(f->hf_repo, sizeof(f->hf_repo), "teamide/test-fixtures");
    snprintf(f->revision, sizeof(f->revision), REV);
    snprintf(f->filename, sizeof(f->filename), "%s", file);
    snprintf(f->sha256, sizeof(f->sha256), "%s", whole->sha);
    f->size = whole->size;
    if (ingot) {
        f->ingot.present = true;
        snprintf(f->ingot.hf_repo, sizeof(f->ingot.hf_repo), "%s", f->hf_repo);
        snprintf(f->ingot.revision, sizeof(f->ingot.revision), REV);
        snprintf(f->ingot.filename, sizeof(f->ingot.filename), "%s", ingot);
        f->ingot.size = ingot_size;
        snprintf(f->ingot.base_sha256, sizeof(f->ingot.base_sha256), "%s",
                 base_sha ? base_sha : "");
    }
    return m;
}

static int copy_file(const char *from, const char *to)
{
    char cmd[1200];
    snprintf(cmd, sizeof(cmd), "cp '%s' '%s'", from, to);
    return system(cmd);
}

/* Change one Q8_0 code byte at `at` (+-1, stays in range). */
static void nudge_code(const char *path, uint64_t at)
{
    int fd = open(path, O_RDWR);
    int8_t q = 0;
    if (fd < 0 || pread(fd, &q, 1, (off_t)at) != 1) {
        CHECK(0, "nudge");
    } else {
        q = (int8_t)(q > 0 ? q - 1 : q + 1);
        CHECK(pwrite(fd, &q, 1, (off_t)at) == 1, "nudge write");
    }
    if (fd >= 0)
        close(fd);
}

static char g_base_url[64];
static pn_catalog_manifest *g_cat;

static pn_download_cfg cfg_with(const pn_catalog_manifest *cat)
{
    pn_download_cfg c = { STORE, g_base_url, PN_CURL_PROTOCOLS_ANY, cat };
    return c;
}

static int installed_as(const pn_cat_model *m, const fx_file *want)
{
    char path[1024];
    fx_file got;
    if (pn_model_store_path(STORE, m, &m->files[0], path, sizeof(path)) != 0)
        return 0;
    file_identity(path, &got);
    return got.size == want->size && strcmp(got.sha, want->sha) == 0;
}

static int part_gone(const pn_cat_model *m)
{
    char path[1024], part[1100];
    pn_model_store_path(STORE, m, &m->files[0], path, sizeof(path));
    snprintf(part, sizeof(part), "%s.part", path);
    return access(part, F_OK) != 0 && access(path, F_OK) != 0;
}

/* The first n bytes of `from` as `to` (an interrupted download's part file). */
static void write_prefix(const char *from, const char *to, uint64_t n)
{
    FILE *in = fopen(from, "rb"), *out = fopen(to, "wb");
    char buf[65536];
    while (in && out && n > 0) {
        size_t want = n < sizeof(buf) ? (size_t)n : sizeof(buf);
        size_t got = fread(buf, 1, want, in);
        if (got == 0 || fwrite(buf, 1, got, out) != got)
            break;
        n -= got;
    }
    CHECK(in && out && n == 0, to);
    if (in)
        fclose(in);
    if (out)
        fclose(out);
}

static void part_path_of(const pn_cat_model *m, char *part, size_t cap)
{
    char path[1024];
    pn_model_store_path(STORE, m, &m->files[0], path, sizeof(path));
    snprintf(part, cap, "%s.part", path);
}

/* A part file left by an earlier run (a restarted box): resume the whole file when that
 * moves fewer bytes than the .ingot, else fetch the .ingot again. */
static void test_part_choice(const fx_file *mid, const fx_file *mid_ing, const fx_file *target,
                             const fx_file *target_ing, const char *base_sha)
{
    char err[256], part[1100];
    pn_download_cfg c = cfg_with(g_cat);

    printf("part_choice:\n");
    /* 1 MB of 4.4 left 3.4 MB to fetch, under the 4.2 MB .ingot (gone.ingot: not served). */
    copy_file(DIR "/mid.gguf", WWW "/mid-whole.gguf");
    const pn_cat_model *a = add_model(g_cat, "part-a", "mid-whole.gguf", mid, "gone.ingot",
                                      mid_ing->size, NULL);
    part_path_of(a, part, sizeof(part));
    write_prefix(DIR "/mid.gguf", part, (uint64_t)CUT);
    int rc = pn_download_file(&c, a, &a->files[0], NULL, NULL, err, sizeof(err));
    if (rc != 0 && strstr(err, "curl 18"))   /* the fixture cuts the first GET */
        rc = pn_download_file(&c, a, &a->files[0], NULL, NULL, err, sizeof(err));
    CHECK(rc == 0 && installed_as(a, mid), "a long part resumes the whole file");
    /* 800 KB of 852 left 52 KB, over the 2.4 KB delta: the .ingot (no.gguf: not served). */
    const pn_cat_model *b = add_model(g_cat, "part-b", "no.gguf", target, "target.gguf.ingot",
                                      target_ing->size, base_sha);
    part_path_of(b, part, sizeof(part));
    write_prefix(WWW "/target.gguf", part, 800 * 1024);
    CHECK(pn_download_file(&c, b, &b->files[0], NULL, NULL, err, sizeof(err)) == 0 &&
          installed_as(b, target), "a part with more left than the .ingot: the .ingot instead");
}

/* ---- tests -------------------------------------------------------------------------- */

static const pn_cat_model *g_big;

static int download_big_child(void *arg)
{
    char err[256];
    pn_download_cfg c = cfg_with(g_cat);
    (void)arg;
    if (pn_download_file(&c, g_big, &g_big->files[0], NULL, NULL, err, sizeof(err)) != 0) {
        fprintf(stderr, "  child: %s\n", err);
        return 1;
    }
    return 0;
}

/* 200+ MB from its .ingot alone (big.gguf itself is not on the server), in bounded memory. */
static void test_big_bounded(void)
{
    fx_tensor t[7] = {
        { "blk.0.ffn_up.weight", GGML_Q8_0, 4096, 8192, 0 },
        { "blk.0.ffn_down.weight", GGML_Q8_0, 4096, 8192, 0 },
        { "blk.1.ffn_up.weight", GGML_Q8_0, 4096, 8192, 0 },
        { "blk.1.ffn_down.weight", GGML_Q8_0, 4096, 8192, 0 },
        { "blk.2.ffn_up.weight", GGML_Q8_0, 4096, 8192, 0 },
        { "blk.2.ffn_down.weight", GGML_Q8_0, 4096, 8192, 0 },
        { "output_norm.weight", GGML_F32, 4096, 1, 0 },
    };
    fx_file whole, ing;
    struct rusage ru;

    printf("big_bounded:\n");
    write_gguf(DIR "/big.gguf", t, 7, 11, &whole);
    encode(DIR "/big.gguf", WWW "/big.gguf.ingot", NULL);
    file_identity(WWW "/big.gguf.ingot", &ing);
    printf("  big.gguf %llu bytes, .ingot %llu bytes\n", (unsigned long long)whole.size,
           (unsigned long long)ing.size);
    CHECK(whole.size >= 200000000ull && ing.size > CUT, "a 200+ MB model, cut once in transit");
    g_big = add_model(g_cat, "big-model", "big.gguf", &whole, "big.gguf.ingot", ing.size, NULL);
    int rc = run_child(download_big_child, NULL, &ru);
    uint64_t rss = maxrss_bytes(&ru);
    printf("  decode child peak RSS %.1f MiB for a %.1f MiB file (bound %.0f MiB)\n",
           (double)rss / 1048576.0, (double)whole.size / 1048576.0,
           (double)RSS_BOUND / 1048576.0);
    CHECK(rc == 0, "installed from the .ingot (the whole file is not served)");
    CHECK(installed_as(g_big, &whole), "byte-identical to the original (sha256, size)");
    CHECK(rss > 0 && rss < RSS_BOUND, "peak RSS under the bound: the file is never held");
    unlink(DIR "/big.gguf");
}

/* A cut transfer: curl restarts at the bytes read, the same decode finishes. */
static void test_resume_after_cut(const fx_file *whole, const fx_file *ing)
{
    char url[512], err[256];
    pn_ingot_fetch_stats st = { 0 };
    fx_file got;

    printf("resume_after_cut:\n");
    snprintf(url, sizeof(url), "%s/teamide/test-fixtures/resolve/" REV "/mid.gguf.ingot",
             g_base_url);
    pn_ingot_fetch_req req = { url, PN_CURL_PROTOCOLS_ANY, ing->size, NULL,
                               DIR "/mid.part", whole->sha, whole->size };
    CHECK(pn_ingot_fetch(&req, &st, err, sizeof(err)) == 0, err);
    CHECK(st.restarts == 1 && st.ingot_bytes == ing->size,
          "cut once at 1 MiB, restarted once, every .ingot byte read");
    file_identity(DIR "/mid.part", &got);
    CHECK(got.size == whole->size && strcmp(got.sha, whole->sha) == 0, "decoded file identical");
    req.sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
    CHECK(pn_ingot_fetch(&req, &st, err, sizeof(err)) == -1 && strstr(err, "sha256 mismatch") &&
          access(DIR "/mid.part", F_OK) != 0, "a catalog sha256 the file does not have: deleted");
    req.sha256 = whole->sha;
    req.ingot_size = ing->size + 1;
    CHECK(pn_ingot_fetch(&req, &st, err, sizeof(err)) == -1 && strstr(err, "catalog says") &&
          access(DIR "/mid.part", F_OK) != 0, "the server's .ingot shorter than the catalog's");
    snprintf(url, sizeof(url), "%s/teamide/test-fixtures/resolve/" REV "/gone.ingot", g_base_url);
    req.ingot_size = ing->size;
    CHECK(pn_ingot_fetch(&req, &st, err, sizeof(err)) == -1 && strstr(err, "curl 22") &&
          st.restarts == 0 && access(DIR "/mid.part", F_OK) != 0,
          "a missing .ingot (HTTP 404) fails at once, no retries");
}

/* A flipped byte: the decode fails, no part, no fallback to the whole file. */
static void test_corrupt(const fx_file *whole, const fx_file *ing)
{
    char err[256];
    pn_download_cfg c = cfg_with(g_cat);

    printf("corrupt:\n");
    copy_file(WWW "/mid.gguf.ingot", WWW "/bad.gguf.ingot");
    int fd = open(WWW "/bad.gguf.ingot", O_RDWR);
    uint8_t b = 0;
    off_t at = (off_t)(ing->size / 2);
    if (fd >= 0 && pread(fd, &b, 1, at) == 1) {
        b ^= 0x10;
        CHECK(pwrite(fd, &b, 1, at) == 1, "flip");
    }
    if (fd >= 0)
        close(fd);
    const pn_cat_model *m = add_model(g_cat, "bad-model", "mid.gguf", whole, "bad.gguf.ingot",
                                      ing->size, NULL);
    int rc = pn_download_file(&c, m, &m->files[0], NULL, NULL, err, sizeof(err));
    printf("  %s\n", err);
    CHECK(rc == -1 && strstr(err, "bad.gguf.ingot") && !strstr(err, "curl 22"),
          "a corrupt .ingot is refused (no whole-file fallback)");
    CHECK(part_gone(m), "no part file, nothing at the final name");
}

/* Delta: base installed / missing / wrong. Leaves the right base installed; the target and
 * its .ingot in *target / *ing. */
static void test_delta(fx_file *target_out, fx_file *ing_out, char *base_sha)
{
    fx_tensor t[3] = {
        { "blk.0.attn_q.weight", GGML_Q8_0, 4096, 96, 0 },
        { "blk.0.attn_k.weight", GGML_Q8_0, 4096, 96, 0 },
        { "output_norm.weight", GGML_F32, 4096, 1, 0 },
    };
    fx_file base, target, ing;
    char err[256], base_path[1024];
    pn_catalog_manifest *cat = g_cat;

    printf("delta:\n");
    write_gguf(DIR "/base.gguf", t, 3, 77, &base);
    copy_file(DIR "/base.gguf", WWW "/target.gguf");
    for (int k = 0; k < 6; k++)
        nudge_code(WWW "/target.gguf", t[k % 2].offset + (uint64_t)k * 997 * Q8_BLOCK + 7);
    file_identity(WWW "/target.gguf", &target);
    encode(WWW "/target.gguf", WWW "/target.gguf.ingot", DIR "/base.gguf");
    file_identity(WWW "/target.gguf.ingot", &ing);
    printf("  target %llu bytes, delta .ingot %llu bytes\n", (unsigned long long)target.size,
           (unsigned long long)ing.size);
    CHECK(strcmp(base.sha, target.sha) != 0 && ing.size > 0 && ing.size < target.size / 4,
          "a small delta against the base");
    const pn_cat_model *bm = add_model(cat, "base-model", "base.gguf", &base, NULL, 0, NULL);
    pn_model_store_path(STORE, bm, &bm->files[0], base_path, sizeof(base_path));

    /* Base installed: decodes from the delta alone (no.gguf is not on the server). */
    copy_file(DIR "/base.gguf", base_path);
    const pn_cat_model *d1 = add_model(cat, "delta-a", "no.gguf", &target, "target.gguf.ingot",
                                       ing.size, base.sha);
    pn_download_cfg c = cfg_with(cat);
    CHECK(pn_download_file(&c, d1, &d1->files[0], NULL, NULL, err, sizeof(err)) == 0, err);
    CHECK(installed_as(d1, &target), "delta decoded against the installed base: identical");

    /* No catalog to find it by, or the base not installed: the whole file instead. */
    const pn_cat_model *d2 = add_model(cat, "delta-b", "target.gguf", &target,
                                       "target.gguf.ingot", ing.size, base.sha);
    pn_download_cfg no_cat = cfg_with(NULL);
    CHECK(pn_download_file(&no_cat, d2, &d2->files[0], NULL, NULL, err, sizeof(err)) == 0 &&
          installed_as(d2, &target), "no catalog: downloaded the whole file");
    unlink(base_path);
    const pn_cat_model *d3 = add_model(cat, "delta-c", "target.gguf", &target,
                                       "target.gguf.ingot", ing.size, base.sha);
    CHECK(pn_download_file(&c, d3, &d3->files[0], NULL, NULL, err, sizeof(err)) == 0 &&
          installed_as(d3, &target), "base not installed: downloaded the whole file");

    /* A wrong base under the base's store name: rejected, then the whole file. */
    copy_file(DIR "/base.gguf", base_path);
    nudge_code(base_path, t[0].offset + 5 * Q8_BLOCK + 3);
    pn_ingot_fetch_stats st = { 0 };
    char url[512];
    snprintf(url, sizeof(url), "%s/teamide/test-fixtures/resolve/" REV "/target.gguf.ingot",
             g_base_url);
    pn_ingot_fetch_req req = { url, PN_CURL_PROTOCOLS_ANY, ing.size, base_path,
                               DIR "/delta.part", target.sha, target.size };
    CHECK(pn_ingot_fetch(&req, &st, err, sizeof(err)) == PN_INGOT_EBASE &&
          access(DIR "/delta.part", F_OK) != 0, "a wrong base is rejected; part deleted");
    const pn_cat_model *d4 = add_model(cat, "delta-d", "target.gguf", &target,
                                       "target.gguf.ingot", ing.size, base.sha);
    CHECK(pn_download_file(&c, d4, &d4->files[0], NULL, NULL, err, sizeof(err)) == 0 &&
          installed_as(d4, &target), "wrong base: downloaded the whole file");
    const pn_cat_model *d5 = add_model(cat, "delta-e", "no.gguf", &target,
                                       "target.gguf.ingot", ing.size, base.sha);
    CHECK(pn_download_file(&c, d5, &d5->files[0], NULL, NULL, err, sizeof(err)) == -1 &&
          part_gone(d5), "wrong base and no whole file: fails cleanly");
    printf("  %s\n", err);
    copy_file(DIR "/base.gguf", base_path);
    *target_out = target;
    *ing_out = ing;
    snprintf(base_sha, 65, "%s", base.sha);
}

int main(void)
{
    int port = lt_pick_port(5);
    fx_tensor mid[2] = {
        { "blk.0.ffn_up.weight", GGML_Q8_0, 4096, 1024, 0 },
        { "output_norm.weight", GGML_F32, 4096, 1, 0 },
    };
    fx_file mid_whole, mid_ing, target, target_ing;
    char base_sha[65];

    lt_clean_dir(DIR);
    lt_make_dir(STORE);
    lt_make_dir(WWW);
    if (!pn_curl_path()) {
        printf("  [SKIP] curl not installed\n");
        return lt_report("test_local_ingot");
    }
    g_cat = calloc(1, sizeof(*g_cat));
    snprintf(g_base_url, sizeof(g_base_url), "http://127.0.0.1:%d", port);
    write_gguf(DIR "/mid.gguf", mid, 2, 5, &mid_whole);
    encode(DIR "/mid.gguf", WWW "/mid.gguf.ingot", NULL);
    file_identity(WWW "/mid.gguf.ingot", &mid_ing);
    fflush(stdout);
    pid_t httpd = fork();
    if (httpd == 0)
        _exit(fx_httpd_run(port, DIR "/www", CUT, 0));
    lt_wait_port(port, 5000);
    test_big_bounded();
    test_resume_after_cut(&mid_whole, &mid_ing);
    test_corrupt(&mid_whole, &mid_ing);
    test_delta(&target, &target_ing, base_sha);
    test_part_choice(&mid_whole, &mid_ing, &target, &target_ing, base_sha);
    kill(httpd, SIGKILL);
    waitpid(httpd, NULL, 0);
    lt_clean_dir(DIR);
    free(g_cat);
    return lt_report("test_local_ingot");
}
