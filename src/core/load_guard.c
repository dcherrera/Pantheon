/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   load_guard.c
 * @brief  Implementation of the load-placement guard (see load_guard.h).
 *
 * @status Complete
 * @issues None
 * @todo   None
 */

#include "load_guard.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "../accel/backend_identity.h"
#include "engine.h"
#include "../formats/gguf.h"
#include "../formats/gguf_layer_bytes.h"
#include "mem/pn_host_admission.h"

#define LOAD_GUARD_NOTE_LEN 200

static _Thread_local char g_accel_note[LOAD_GUARD_NOTE_LEN];
static _Thread_local char g_load_refusal[PN_LOAD_REFUSAL_LEN];
static _Thread_local uint64_t g_load_refusals = 0;
static const char *g_fake_accel_reason = NULL;   /* test hook; set before loads */

void pn_accel_refusal_note(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_accel_note, sizeof(g_accel_note), fmt, ap);
    va_end(ap);
}

void pn_accel_refusal_clear(void)
{
    g_accel_note[0] = '\0';
}

const char *pn_accel_refusal_last(void)
{
    return g_accel_note;
}

const char *pn_load_refusal_last(void)
{
    return g_load_refusal;
}

uint64_t pn_load_thread_refusals(void)
{
    return g_load_refusals;
}

bool pn_load_guard_accel_build(void)
{
    return pn_backend_supports_die_placement() || g_fake_accel_reason != NULL;
}

bool pn_load_guard_cpu_planned(bool cpu_only)
{
    return cpu_only || !pn_load_guard_accel_build();
}

/* The file name without its directories, for messages. */
static const char *base_name(const char *path)
{
    const char *slash = path ? strrchr(path, '/') : NULL;
    return slash ? slash + 1 : (path ? path : "model");
}

/* Make `msg` this thread's load refusal and log it. */
static void set_load_refusal(const char *msg)
{
    snprintf(g_load_refusal, sizeof(g_load_refusal), "%s", msg);
    g_load_refusals++;
    fprintf(stderr, "pn_load: REFUSED - %s\n", msg);
}

uint64_t pn_load_guard_cpu_working_set(const char *path)
{
    struct stat st;
    if (!path || stat(path, &st) != 0 || !S_ISREG(st.st_mode))
        return 0;
    uint64_t bytes = (uint64_t)st.st_size;
    /* Header + tensor table only; the data is mmap'd and never touched here. */
    pn_gguf_file *gf = pn_is_gguf(path) ? pn_gguf_open(path) : NULL;
    if (gf) {
        uint64_t ws = pn_gguf_cpu_forward_bytes(gf);
        if (ws > 0 && ws < bytes)
            bytes = ws;
        pn_gguf_close(gf);
    }
    return bytes;
}

int pn_load_guard_cpu_host_fit(const char *path)
{
    uint64_t need = pn_load_guard_cpu_working_set(path);
    if (need == 0)
        return 0;
    char what[160], why[PN_HOST_ADMIT_REASON_LEN], msg[PN_LOAD_REFUSAL_LEN];
    snprintf(what, sizeof(what), "%s on the CPU (weights read every token)",
             base_name(path));
    if (pn_host_admit_check(need, what, "use a smaller quant or model, or free host memory",
                            why, sizeof(why)) == 0)
        return 0;
    snprintf(msg, sizeof(msg), "host RAM: %s", why);
    set_load_refusal(msg);
    return -1;
}

void pn_load_guard_refuse_accel(const char *path, const int *dies, int n_dies)
{
    char where[64] = "the accelerator";
    if (dies && n_dies > 0) {
        int off = snprintf(where, sizeof(where), "die%s ", n_dies > 1 ? "s" : "");
        for (int i = 0; i < n_dies && off < (int)sizeof(where); i++)
            off += snprintf(where + off, sizeof(where) - (size_t)off, "%s%d", i ? "," : "",
                            dies[i]);
    }
    char why[LOAD_GUARD_NOTE_LEN + 96];
    if (g_accel_note[0])
        snprintf(why, sizeof(why), "%s", g_accel_note);
    else
        snprintf(why, sizeof(why), "%s did not take it (upload failed, no device visible, or "
                 "no accelerator forward for this architecture)", where);
    char msg[PN_LOAD_REFUSAL_LEN];
    snprintf(msg, sizeof(msg), "%s not loaded: %s. No CPU fallback: try a smaller context, "
             "fewer users or a smaller quant (--cpu runs it on the host CPU)",
             base_name(path), why);
    set_load_refusal(msg);
}

void pn_load_guard_test_fake_accel(const char *reason)
{
    g_fake_accel_reason = reason;
}

const char *pn_load_guard_test_fake_reason(void)
{
    return g_fake_accel_reason;
}
