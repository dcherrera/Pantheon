/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_core_list.h
 * @brief A Tensix core list spelled as text, "96-110,0-15": comma-separated indices or
 *   inclusive ranges, parsed into an array. Used by `PN_TT_MATMUL_CORE_LIST` (decode frontier
 *   F1.1: matmul members 1..N-1 on cores that are not one arithmetic run).
 *
 * @status Added 2026-11-19.
 * @issues None
 * @todo   None
 */
#ifndef PN_TT_CORE_LIST_H
#define PN_TT_CORE_LIST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Parse `text` into `cores`, in the order written. Every entry is a decimal index or `a-b`
 * with a <= b; entries are separated by single commas; no index may repeat and none may exceed
 * `max_index`.
 *
 * @param text       the list (NULL or "" is an empty list)
 * @param cores      output, `capacity` entries
 * @param capacity   how many indices `cores` holds
 * @param max_index  largest legal core index
 * @return           the number of indices, 0 for an empty list, or -1 when the text is
 *                   malformed, a range is reversed, an index repeats or is out of range, or
 *                   the list is longer than `capacity`
 */
int pn_tt_core_list_parse(const char *text, uint32_t *cores, uint32_t capacity,
                          uint32_t max_index);

#ifdef __cplusplus
}
#endif

#endif /* PN_TT_CORE_LIST_H */
