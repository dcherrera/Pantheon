/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file dump_reqlog.c
 * @brief The hidden-dump request log (`PN_TT_DUMP_REQLOG`): record counter + one JSON line per
 *        request. See dump_reqlog.h.
 *
 * status: complete.
 * issues: none.
 * todo:   none.
 */
#include "dump_reqlog.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

static pthread_mutex_t g_reqlog_lock = PTHREAD_MUTEX_INITIALIZER;
static FILE *g_reqlog_out;
static int g_reqlog_env_read;
static uint64_t g_reqlog_records;   /* records the dump has written */
static uint64_t g_reqlog_first;     /* first record of the request in progress */
static uint64_t g_reqlog_index;     /* index of the next request line */

int pn_dump_reqlog_open(const char *path)
{
    FILE *f = path && *path ? fopen(path, "w") : NULL;

    pthread_mutex_lock(&g_reqlog_lock);
    if (g_reqlog_out)
        fclose(g_reqlog_out);
    g_reqlog_out = f;
    g_reqlog_env_read = 1;
    g_reqlog_records = g_reqlog_first = g_reqlog_index = 0;
    pthread_mutex_unlock(&g_reqlog_lock);
    if (!f) {
        fprintf(stderr, "dump_reqlog: %s could not be opened\n", path ? path : "(null)");
        return -1;
    }
    return 0;
}

void pn_dump_reqlog_close(void)
{
    pthread_mutex_lock(&g_reqlog_lock);
    if (g_reqlog_out)
        fclose(g_reqlog_out);
    g_reqlog_out = NULL;
    pthread_mutex_unlock(&g_reqlog_lock);
}

bool pn_dump_reqlog_enabled(void)
{
    bool on;

    pthread_mutex_lock(&g_reqlog_lock);
    if (!g_reqlog_env_read) {
        const char *path = getenv("PN_TT_DUMP_REQLOG");

        g_reqlog_env_read = 1;
        if (path && *path) {
            g_reqlog_out = fopen(path, "w");
            if (!g_reqlog_out)
                fprintf(stderr, "dump_reqlog: PN_TT_DUMP_REQLOG %s could not be opened\n", path);
        }
    }
    on = g_reqlog_out != NULL;
    pthread_mutex_unlock(&g_reqlog_lock);
    return on;
}

void pn_dump_reqlog_count_records(uint64_t n)
{
    pthread_mutex_lock(&g_reqlog_lock);
    g_reqlog_records += n;
    pthread_mutex_unlock(&g_reqlog_lock);
}

/** `s` as a JSON string body: printable ASCII only, quote and backslash escaped, anything else
 *  dropped (the values written here are fixed tokens and hex). */
static void reqlog_put_json_string(FILE *f, const char *s)
{
    fputc('"', f);
    for (; s && *s; s++) {
        if (*s == '"' || *s == '\\')
            fputc('\\', f);
        if (*s >= 0x20 && *s < 0x7f)
            fputc(*s, f);
    }
    fputc('"', f);
}

int pn_dump_reqlog_write_request(const pn_dump_reqlog_request *r)
{
    int rc = -1;

    if (!r)
        return -1;
    pthread_mutex_lock(&g_reqlog_lock);
    if (g_reqlog_out) {
        FILE *f = g_reqlog_out;

        fprintf(f, "{\"req\":%llu,\"first_record\":%llu,\"records\":%llu,\"prompt_tokens\":%d,"
                   "\"completion_tokens\":%d,\"max_tokens\":%d,\"temperature\":%.9g,\"top_k\":%d,"
                   "\"top_p\":%.9g,\"seed\":",
                (unsigned long long)g_reqlog_index, (unsigned long long)g_reqlog_first,
                (unsigned long long)(g_reqlog_records - g_reqlog_first), r->prompt_tokens,
                r->completion_tokens, r->max_tokens, (double)r->temperature, r->top_k,
                (double)r->top_p);
        if (r->seed_set)
            fprintf(f, "%llu", (unsigned long long)r->seed);
        else
            fputs("null", f);
        fputs(",\"finish\":", f);
        reqlog_put_json_string(f, r->finish ? r->finish : "");
        fputs(",\"prompt_sha256\":", f);
        reqlog_put_json_string(f, r->prompt_sha256_hex ? r->prompt_sha256_hex : "");
        if (r->has_spec)
            fprintf(f, ",\"spec_cycles\":%llu,\"spec_drafts\":%llu,\"spec_accepted\":%llu",
                    (unsigned long long)r->spec_cycles, (unsigned long long)r->spec_drafts,
                    (unsigned long long)r->spec_accepted);
        fputs("}\n", f);
        rc = fflush(f) == 0 && !ferror(f) ? 0 : -1;
        if (rc != 0)
            fprintf(stderr, "dump_reqlog: write failed at request %llu\n",
                    (unsigned long long)g_reqlog_index);
        g_reqlog_index++;
        g_reqlog_first = g_reqlog_records;
    }
    pthread_mutex_unlock(&g_reqlog_lock);
    return rc;
}
