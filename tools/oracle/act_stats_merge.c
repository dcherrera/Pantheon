/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   act_stats_merge.c
 * @brief  Sum PN_ACT_STATS files ("mixw imatrix v1", src/core/act_stats_file.h): the halves of
 *         a PN_SPLIT_LAYERS run and separate runs over different tokens merge into one file.
 *
 *   act_stats_merge OUT IN1 [IN2 ...]          imatrix files
 *   act_stats_merge --xtx OUT IN1 [IN2 ...]    their .xtx (X^T X) companions
 *
 * Same tensor name: n_in must match, counts and payloads add; other names are copied. Exit 0
 * on success, 1 on any refusal (stderr says which input and tensor).
 *
 * @status None
 * @issues None
 * @todo   None
 */
#include <stdio.h>
#include <string.h>

#include "../../src/core/act_stats_file.h"

int main(int argc, char **argv) {
    int xtx = argc > 1 && strcmp(argv[1], "--xtx") == 0;
    int first = 1 + xtx;
    if (argc - first < 2) {
        fprintf(stderr, "usage: %s [--xtx] OUT IN1 [IN2 ...]\n", argv[0]);
        return 1;
    }
    pn_act_stats_table merged = { .xtx = xtx };
    int rc = 0;
    for (int i = first + 1; i < argc && rc == 0; i++) {
        pn_act_stats_table in;
        if (pn_act_stats_table_read(&in, argv[i], xtx) != 0) {
            rc = 1;
            break;
        }
        if (pn_act_stats_table_merge(&merged, &in) != 0) {
            fprintf(stderr, "act_stats_merge: %s does not merge\n", argv[i]);
            rc = 1;
        }
        pn_act_stats_table_free(&in);
    }
    if (rc == 0 && pn_act_stats_table_write(&merged, argv[first]) != 0)
        rc = 1;
    if (rc == 0)
        printf("act_stats_merge: %s: %u tensors from %d files\n", argv[first], merged.n_entries,
               argc - first - 1);
    pn_act_stats_table_free(&merged);
    return rc;
}
