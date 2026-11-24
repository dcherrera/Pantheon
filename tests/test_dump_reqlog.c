/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_dump_reqlog.c
 * @brief Unit test of the hidden-dump request log (src/core/dump_reqlog.c +
 *        src/serve/serve_dump_reqlog.c): off by default, record ranges per request, the line's
 *        exact bytes, the prompt-token hash.
 *
 * @status complete
 * @issues None
 * @todo None
 */
#include "../src/core/dump_reqlog.h"
#include "../src/serve/serve_dump_reqlog.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int g_fail;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            printf("FAIL %s (%s:%d)\n", msg, __FILE__, __LINE__);           \
            g_fail++;                                                       \
        }                                                                   \
    } while (0)

#define LOG_PATH "build/test_dump_reqlog.jsonl"

static char *read_whole(const char *path)
{
    FILE *f = fopen(path, "rb");
    char *buf = calloc(1, 8192);
    size_t n = f && buf ? fread(buf, 1, 8191, f) : 0;

    (void)n;
    if (f) fclose(f);
    return buf;
}

int main(void)
{
    static const char *expect =
        "{\"req\":0,\"first_record\":0,\"records\":5,\"prompt_tokens\":4,\"completion_tokens\":2,"
        "\"max_tokens\":400,\"temperature\":0,\"top_k\":0,\"top_p\":1,\"seed\":null,"
        "\"finish\":\"length\",\"prompt_sha256\":"
        "\"5833cf3440a0b08839b87a2116b6a62892e4f9f2f2cb3d0c8ca06c0c3593e04a\"}\n"
        "{\"req\":1,\"first_record\":5,\"records\":7,\"prompt_tokens\":3,\"completion_tokens\":5,"
        "\"max_tokens\":256,\"temperature\":0.600000024,\"top_k\":20,\"top_p\":0.949999988,"
        "\"seed\":1235,\"finish\":\"stop\",\"prompt_sha256\":\"ab\",\"spec_cycles\":4,"
        "\"spec_drafts\":9,\"spec_accepted\":6}\n"
        "{\"req\":2,\"first_record\":12,\"records\":0,\"prompt_tokens\":1,\"completion_tokens\":1,"
        "\"max_tokens\":8,\"temperature\":0,\"top_k\":0,\"top_p\":1,\"seed\":0,"
        "\"finish\":\"q\\\"\",\"prompt_sha256\":\"\"}\n";
    const uint32_t prompt[4] = { 1, 2, 3, 248044 };
    pn_dump_reqlog_request r;
    char *got;

    /* OFF by default: no env, no file, every call inert. */
    unsetenv("PN_TT_DUMP_REQLOG");
    CHECK(!pn_dump_reqlog_enabled(), "off without PN_TT_DUMP_REQLOG");
    memset(&r, 0, sizeof(r));
    CHECK(pn_dump_reqlog_write_request(&r) == -1, "no line without a log");

    mkdir("build", 0755);
    CHECK(pn_dump_reqlog_open(LOG_PATH) == 0, "open");
    CHECK(pn_dump_reqlog_enabled(), "on after open");
    pn_dump_reqlog_count_records(5);
    r = (pn_dump_reqlog_request){ .prompt_tokens = 4, .completion_tokens = 2, .max_tokens = 400,
                                  .top_p = 1.0f, .finish = "length" };
    CHECK(sg_dump_reqlog_request(&r, prompt, 4) == 0, "line 0");
    pn_dump_reqlog_count_records(3);
    pn_dump_reqlog_count_records(4);
    r = (pn_dump_reqlog_request){ .prompt_tokens = 3, .completion_tokens = 5, .max_tokens = 256,
                                  .top_k = 20, .temperature = 0.6f, .top_p = 0.95f,
                                  .seed_set = true, .seed = 1235, .finish = "stop",
                                  .prompt_sha256_hex = "ab", .has_spec = true, .spec_cycles = 4,
                                  .spec_drafts = 9, .spec_accepted = 6 };
    CHECK(pn_dump_reqlog_write_request(&r) == 0, "line 1");
    r = (pn_dump_reqlog_request){ .prompt_tokens = 1, .completion_tokens = 1, .max_tokens = 8,
                                  .top_p = 1.0f, .seed_set = true, .finish = "q\"\x01" };
    CHECK(pn_dump_reqlog_write_request(&r) == 0, "line 2 (no records, escaped finish)");
    pn_dump_reqlog_close();
    CHECK(!pn_dump_reqlog_enabled(), "off after close");

    got = read_whole(LOG_PATH);
    CHECK(got && strcmp(got, expect) == 0, "log bytes");
    if (got && strcmp(got, expect) != 0)
        printf("got:\n%s\nwant:\n%s\n", got, expect);
    free(got);

    /* Re-open truncates and resets the counters. */
    CHECK(pn_dump_reqlog_open(LOG_PATH) == 0, "re-open");
    pn_dump_reqlog_count_records(2);
    r = (pn_dump_reqlog_request){ .finish = "length", .top_p = 1.0f };
    CHECK(pn_dump_reqlog_write_request(&r) == 0, "line after re-open");
    pn_dump_reqlog_close();
    got = read_whole(LOG_PATH);
    CHECK(got && strncmp(got, "{\"req\":0,\"first_record\":0,\"records\":2,", 38) == 0,
          "counters reset on open");
    free(got);
    CHECK(pn_dump_reqlog_open("build/no/such/dir/x.jsonl") == -1, "unopenable path refused");
    CHECK(!pn_dump_reqlog_enabled(), "off after a refused open");

    printf("test_dump_reqlog: %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
