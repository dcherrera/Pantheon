/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_core_list.c
 * @brief Implementation of the "96-110,0-15" core-list parser (pn_tt_core_list.h).
 *
 * @status Added 2026-11-19.
 * @issues None
 * @todo   None
 */
#include "pn_tt_core_list.h"

#include <ctype.h>
#include <stdlib.h>

/** One decimal index at `*at`; advances `*at`. @return 0, or -1 when there are no digits. */
static int core_list_read_index(const char **at, unsigned long *value)
{
    char *end;

    if (!isdigit((unsigned char)**at))
        return -1;
    *value = strtoul(*at, &end, 10);
    *at = end;
    return 0;
}

/** Whether `core` is already among the first `count` of `cores`. */
static int core_list_holds(const uint32_t *cores, uint32_t count, uint32_t core)
{
    for (uint32_t index = 0; index < count; index++)
        if (cores[index] == core)
            return 1;
    return 0;
}

int pn_tt_core_list_parse(const char *text, uint32_t *cores, uint32_t capacity,
                          uint32_t max_index)
{
    const char *at = text;
    uint32_t count = 0u;

    if (!text || !*text)
        return 0;
    for (;;) {
        unsigned long first, last;

        if (core_list_read_index(&at, &first) != 0)
            return -1;
        last = first;
        if (*at == '-') {
            at++;
            if (core_list_read_index(&at, &last) != 0 || last < first)
                return -1;
        }
        if (last > max_index)
            return -1;
        for (unsigned long core = first; core <= last; core++) {
            if (count == capacity || core_list_holds(cores, count, (uint32_t)core))
                return -1;
            cores[count++] = (uint32_t)core;
        }
        if (*at == '\0')
            return (int)count;
        if (*at != ',')
            return -1;
        at++;
    }
}
