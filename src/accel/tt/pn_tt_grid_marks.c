/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   pn_tt_grid_marks.c
 * @brief  Grid step marks — see pn_tt_grid_marks.h.
 *
 * @status None
 * @issues None
 * @todo   None
 */
#include "pn_tt_grid_marks.h"

#include <string.h>

void pn_tt_grid_marks_reset(pn_tt_grid_marks *m, uint32_t cores)
{
    memset(m, 0, sizeof(*m));
    m->cores = cores < PN_TT_GRID_MARK_CORES ? cores : PN_TT_GRID_MARK_CORES;
}

void pn_tt_grid_marks_issued(pn_tt_grid_marks *m, uint32_t depth, bool detached)
{
    m->ring_total += depth;
    if (detached)
        m->detached_total++;
}

uint32_t pn_tt_grid_marks_push(pn_tt_grid_marks *m, const uint32_t *counts)
{
    pn_tt_grid_mark *mark = &m->mark[m->next % PN_TT_GRID_MARKS];

    if (m->next - m->oldest >= PN_TT_GRID_MARKS)
        m->oldest = m->next - PN_TT_GRID_MARKS + 1u; /* evict: a wait for it waits for all */
    mark->ring_total = m->ring_total;
    mark->detached_total = m->detached_total;
    memset(mark->count, 0, sizeof(mark->count));
    memcpy(mark->count, counts, m->cores * sizeof(counts[0]));
    return m->next++;
}

int pn_tt_grid_marks_take(pn_tt_grid_marks *m, uint32_t id, uint32_t *need)
{
    const pn_tt_grid_mark *mark, *from;
    int n = 0;

    memset(need, 0, m->cores * sizeof(need[0]));
    if (id == PN_TT_GRID_MARK_NONE || (id < m->done_through && id < m->next))
        return 0; /* nothing to wait for, or done already */
    if (id < m->oldest || id >= m->next)
        return -1; /* evicted, or never pushed */
    mark = &m->mark[id % PN_TT_GRID_MARKS];
    from = mark;
#ifdef PN_TT_GRID_MARKS_RED
    if (id > m->oldest)
        from = &m->mark[(id - 1u) % PN_TT_GRID_MARKS];
#endif
    for (uint32_t c = 0; c < m->cores; c++)
        if (from->count[c] > m->awaited[c]) {
            need[c] = from->count[c];
            m->awaited[c] = from->count[c];
            n++;
        }
    if (mark->ring_total > m->ring_done)
        m->ring_done = mark->ring_total;
    if (mark->detached_total > m->detached_done)
        m->detached_done = mark->detached_total;
    m->oldest = id + 1u;
    m->done_through = id + 1u;
    return n;
}

void pn_tt_grid_marks_all(pn_tt_grid_marks *m, const uint32_t *counts)
{
    for (uint32_t c = 0; c < m->cores; c++)
        if (counts[c] > m->awaited[c])
            m->awaited[c] = counts[c];
    m->ring_done = m->ring_total;
    m->detached_done = m->detached_total;
    m->oldest = m->next;
    m->done_through = m->next;
}

uint32_t pn_tt_grid_marks_ring(const pn_tt_grid_marks *m)
{
    return m->ring_total - m->ring_done;
}

uint32_t pn_tt_grid_marks_detached(const pn_tt_grid_marks *m)
{
    return m->detached_total - m->detached_done;
}

uint32_t pn_tt_grid_marks_room(const pn_tt_grid_marks *m, uint32_t depth, uint32_t chained,
                               uint32_t limit)
{
    if (pn_tt_grid_marks_ring(m) + depth <= limit &&
        chained + pn_tt_grid_marks_detached(m) < limit)
        return PN_TT_GRID_MARK_NONE;
    for (uint32_t id = m->oldest; id != m->next; id++) {
        const pn_tt_grid_mark *mark = &m->mark[id % PN_TT_GRID_MARKS];

        if (m->ring_total - mark->ring_total + depth <= limit &&
            chained + (m->detached_total - mark->detached_total) < limit)
            return id;
    }
    return PN_TT_GRID_MARK_FULL;
}
