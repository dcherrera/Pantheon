/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_precondition_dump.c
 * @brief `PN_TT_ONCARD_DUMP`: an S2 planning problem (queues, multicast groups, the items in the
 *        order the planner sees them) and its plan, written as text on the card and read back
 *        off-card (tests/test_tt_precondition --replay FILE re-plans and re-checks it).
 *
 * Format, one record a line:
 *   PNPRE 1 | config LOOKAHEAD RED | queues N | q CAPACITY STEP LEADER HONOURS (N lines)
 *   groups N | g COUNT Q0 Q1 Q2 Q3 (N lines)
 *   items N  | i KIND GROUP QUEUE INDEX SLOTS ADDRESS STEP PULL_SEQUENCE GROUP_SEQUENCE
 *   actions N | a KIND GROUP QUEUE PRODUCER ITEM INDEX INDEX_STEP COUNT COUNT_STEP | end
 *
 * @status S2 card fix 3.
 * @issues None
 * @todo None
 */
#include "pn_tt_precondition.h"

#include <stdlib.h>
#include <string.h>

int pn_tt_precondition_dump_write(FILE *f, const pn_tt_pre_dump *d)
{
    uint32_t q, g;
    size_t i;

    if (!f || !d)
        return -1;
    fprintf(f, "PNPRE 1\nconfig %u %d\nqueues %u\n", d->config.lookahead, d->config.red_early ? 1 : 0,
            d->queue_count);
    for (q = 0; q < d->queue_count; q++)
        fprintf(f, "q %u %u %d %d\n", d->capacity[q], d->step[q], d->leader[q],
                d->honours[q] ? 1 : 0);
    fprintf(f, "groups %u\n", d->group_count);
    for (g = 0; g < d->group_count; g++)
        fprintf(f, "g %u %u %u %u %u\n", d->groups[g].count, d->groups[g].queues[0],
                d->groups[g].queues[1], d->groups[g].queues[2], d->groups[g].queues[3]);
    fprintf(f, "items %zu\n", d->n_items);
    for (i = 0; i < d->n_items; i++) {
        const pn_tt_pre_item *it = &d->items[i];

        fprintf(f, "i %u %d %u %u %u %u %u %u %u\n", it->kind, it->group, it->queue, it->index,
                it->slots, it->address, it->step, it->pull_sequence, it->group_sequence);
    }
    fprintf(f, "actions %zu\n", d->n_actions);
    for (i = 0; i < d->n_actions; i++) {
        const pn_tt_pre_action *a = &d->actions[i];

        fprintf(f, "a %u %d %u %u %u %u %u %u %u\n", a->kind, a->group, a->queue, a->producer,
                a->item, a->index, a->index_step, a->count, a->count_step);
    }
    fprintf(f, "end\n");
    return ferror(f) ? -1 : 0;
}

void pn_tt_precondition_dump_free(pn_tt_pre_dump *d)
{
    if (!d)
        return;
    free(d->capacity);
    free(d->step);
    free(d->leader);
    free(d->honours);
    free(d->groups);
    free(d->items);
    free(d->actions);
    memset(d, 0, sizeof(*d));
}

/** The header line `word N`; false when it is not one. */
static bool dump_count(FILE *f, const char *word, size_t *n)
{
    char tag[16];
    unsigned long value = 0;

    if (fscanf(f, "%15s %lu", tag, &value) != 2 || strcmp(tag, word) != 0)
        return false;
    *n = (size_t)value;
    return true;
}

static bool dump_read_tables(FILE *f, pn_tt_pre_dump *d)
{
    size_t n = 0, i;
    int red = 0;

    if (fscanf(f, " config %u %d", &d->config.lookahead, &red) != 2 || !dump_count(f, "queues", &n))
        return false;
    d->config.red_early = red != 0;
    d->queue_count = (uint32_t)n;
    d->capacity = calloc(n + 1u, sizeof(*d->capacity));
    d->step = calloc(n + 1u, sizeof(*d->step));
    d->leader = calloc(n + 1u, sizeof(*d->leader));
    d->honours = calloc(n + 1u, sizeof(*d->honours));
    if (!d->capacity || !d->step || !d->leader || !d->honours)
        return false;
    for (i = 0; i < n; i++) {
        int honours = 0;

        if (fscanf(f, " q %u %u %d %d", &d->capacity[i], &d->step[i], &d->leader[i], &honours) != 4)
            return false;
        d->honours[i] = honours != 0;
    }
    if (!dump_count(f, "groups", &n))
        return false;
    d->group_count = (uint32_t)n;
    d->groups = calloc(n + 1u, sizeof(*d->groups));
    if (!d->groups)
        return false;
    for (i = 0; i < n; i++) {
        unsigned q[4];

        if (fscanf(f, " g %u %u %u %u %u", &d->groups[i].count, &q[0], &q[1], &q[2], &q[3]) != 5)
            return false;
        d->groups[i].queues[0] = (uint16_t)q[0];
        d->groups[i].queues[1] = (uint16_t)q[1];
        d->groups[i].queues[2] = (uint16_t)q[2];
        d->groups[i].queues[3] = (uint16_t)q[3];
    }
    return true;
}

static bool dump_read_lists(FILE *f, pn_tt_pre_dump *d)
{
    size_t n = 0, i;

    if (!dump_count(f, "items", &n))
        return false;
    d->items = calloc(n + 1u, sizeof(*d->items));
    if (!d->items)
        return false;
    d->n_items = n;
    for (i = 0; i < n; i++) {
        pn_tt_pre_item *it = &d->items[i];
        unsigned kind, queue;
        int group;

        if (fscanf(f, " i %u %d %u %u %u %u %u %u %u", &kind, &group, &queue, &it->index, &it->slots,
                   &it->address, &it->step, &it->pull_sequence, &it->group_sequence) != 9)
            return false;
        it->kind = (uint8_t)kind;
        it->group = (int16_t)group;
        it->queue = (uint16_t)queue;
    }
    if (!dump_count(f, "actions", &n))
        return false;
    d->actions = calloc(n + 1u, sizeof(*d->actions));
    if (!d->actions)
        return false;
    d->n_actions = n;
    for (i = 0; i < n; i++) {
        pn_tt_pre_action *a = &d->actions[i];
        unsigned kind, queue, producer;
        int group;

        if (fscanf(f, " a %u %d %u %u %u %u %u %u %u", &kind, &group, &queue, &producer, &a->item,
                   &a->index, &a->index_step, &a->count, &a->count_step) != 9)
            return false;
        a->kind = (uint8_t)kind;
        a->group = (int16_t)group;
        a->queue = (uint16_t)queue;
        a->producer = (uint16_t)producer;
    }
    return true;
}

int pn_tt_precondition_dump_read(FILE *f, pn_tt_pre_dump *d)
{
    char tag[16];
    unsigned version = 0;

    if (!f || !d)
        return -1;
    memset(d, 0, sizeof(*d));
    if (fscanf(f, "%15s %u", tag, &version) != 2 || strcmp(tag, "PNPRE") != 0 || version != 1u ||
        !dump_read_tables(f, d) || !dump_read_lists(f, d)) {
        pn_tt_precondition_dump_free(d);
        return -1;
    }
    return 0;
}
