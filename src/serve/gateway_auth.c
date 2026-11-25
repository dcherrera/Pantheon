/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gateway_auth.c
 * @brief Implementation of gateway_auth.h.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include "gateway_auth.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>

#define KEY_FILE_READ_MAX 4096

static bool is_space(char c)
{
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

int pn_gateway_key_parse(pn_gateway_key *out, const char *text, size_t n,
                         char *err, size_t errlen)
{
    size_t start = 0, end = n;

    memset(out, 0, sizeof(*out));
    while (start < end && is_space(text[start]))
        start++;
    while (end > start && is_space(text[end - 1]))
        end--;
    if (end == start) {
        snprintf(err, errlen, "the key file is empty");
        return -1;
    }
    if (end - start > PN_GATEWAY_KEY_MAX) {
        snprintf(err, errlen, "the key is longer than %d bytes", PN_GATEWAY_KEY_MAX);
        return -1;
    }
    for (size_t i = start; i < end; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c <= 0x20 || c == 0x7f) {
            snprintf(err, errlen, "the key holds whitespace or a control byte");
            return -1;
        }
    }
    memcpy(out->key, text + start, end - start);
    out->len = end - start;
    return 0;
}

int pn_gateway_key_load(pn_gateway_key *out, const char *path, char *err,
                        size_t errlen)
{
    char buf[KEY_FILE_READ_MAX];
    struct stat st;
    FILE *f;
    size_t n;
    int rc;

    memset(out, 0, sizeof(*out));
    f = fopen(path, "rb");
    if (!f) {
        snprintf(err, errlen, "cannot open the key file: %s", strerror(errno));
        return -1;
    }
    if (fstat(fileno(f), &st) == 0 && (st.st_mode & (S_IRWXG | S_IRWXO)))
        fprintf(stderr, "serve: warning: the gateway key file is readable by group "
                        "or others; chmod 600 it\n");
    n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    if (n == sizeof(buf)) {
        memset(buf, 0, sizeof(buf));
        snprintf(err, errlen, "the key file is larger than %d bytes", KEY_FILE_READ_MAX - 1);
        return -1;
    }
    rc = pn_gateway_key_parse(out, buf, n, err, errlen);
    memset(buf, 0, sizeof(buf));
    return rc;
}

bool pn_gateway_key_matches(const pn_gateway_key *key, const char *authorization)
{
    static const char scheme[] = "Bearer ";
    const char *presented;
    size_t plen;
    unsigned char diff = 0;

    if (!key || key->len == 0 || !authorization)
        return false;
    if (strncasecmp(authorization, scheme, sizeof(scheme) - 1) != 0)
        return false;
    presented = authorization + sizeof(scheme) - 1;
    plen = strlen(presented);
    while (plen > 0 && is_space(presented[plen - 1]))
        plen--;
    /* Walk the whole loaded key whatever was presented: the time depends on
     * the key's length, never on how many leading bytes matched. */
    for (size_t i = 0; i < key->len; i++) {
        unsigned char p = i < plen ? (unsigned char)presented[i] : 0;
        diff |= (unsigned char)(p ^ (unsigned char)key->key[i]);
    }
    return diff == 0 && plen == key->len;
}

void pn_gateway_key_clear(pn_gateway_key *key)
{
    if (!key)
        return;
    volatile char *p = key->key;
    for (size_t i = 0; i < sizeof(key->key); i++)
        p[i] = 0;
    key->len = 0;
}
