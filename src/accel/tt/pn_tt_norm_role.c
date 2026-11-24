/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_norm_role.c
 * @brief The persistent norm role on the card: arm, step, teardown (pn_tt_norm_role.h).
 *
 * @status P0. Off by default.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_norm_role.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pn_tt_core_list.h"
#include "pn_tt_device_internal.h"
#include "pn_tt_group_pull.h"
#include "pn_tt_replica_map.h"
#include "pn_tt_role_table.h"

#include "tt/ft_tt_queue.h"

_Static_assert(PN_TT_NORM_ROLE_EXCHANGE_FACES == (unsigned)kNrExchangeFaces &&
                   PN_TT_NORM_ROLE_EXCHANGE_STAGGER == (unsigned)kNrExchangeStagger,
               "exchange mode bits mirror the wire contract");

/** Wall-clock ticks (16-cycle units at 1.35 GHz) to milliseconds. */
#define NORM_ROLE_TICK_MS (16.0 / 1.35e9 * 1e3)

/** The role's state, one per device (device->norm_role). */
struct pn_tt_norm_role {
    pn_tt_residency member[PN_TT_NORM_ROLE_MAX_MEMBERS];
    pn_tt_group_pull pull;
    ft_tt_op_norm_role_schedule schedule;
    ft_tt_launch_cb_entry placed[FT_TT_OP_NORM_ROLE_CB_COUNT];
    uint32_t cores[PN_TT_NORM_ROLE_MAX_MEMBERS];
    uint32_t members;
    bool armed;
    bool broken;
    uint32_t arms;
    double arm_seconds;
    uint64_t steps;
};

static void role_table_teardown(void *user);
static int role_table_arm(void *user, void *context);
static void norm_role_down(pn_tt_device *device);

static struct pn_tt_norm_role *role_of(pn_tt_device *device)
{
    if (!device)
        return NULL;
    if (!device->norm_role)
        device->norm_role = calloc(1u, sizeof(*device->norm_role));
    return device->norm_role;
}

bool pn_tt_norm_role_armed(const pn_tt_device *device)
{
    return device && device->norm_role && device->norm_role->armed && !device->norm_role->broken;
}

const pn_tt_residency *pn_tt_norm_role_member(const pn_tt_device *device, uint32_t m)
{
    if (!device || !device->norm_role || m >= device->norm_role->members)
        return NULL;
    return &device->norm_role->member[m];
}

pn_tt_residency *pn_tt_norm_role_member_mut(pn_tt_device *device, uint32_t m)
{
    if (!device || !device->norm_role || m >= device->norm_role->members)
        return NULL;
    return &device->norm_role->member[m];
}

void pn_tt_norm_role_fill_step(const pn_tt_device *device, ft_tt_op_norm_role_step *step,
                               uint32_t kind, const pn_tt_activation *input,
                               const pn_tt_norm_constants *constants,
                               const pn_tt_activation *output)
{
    const ft_tt_op_operand in = pn_tt_operand(device, input);
    const ft_tt_op_operand out = pn_tt_operand(device, output);

    step->kind = kind;
    step->input.x = in.x;
    step->input.y = in.y;
    step->input.base = in.address;
    step->constants.x = device->dram[constants->bank].x;
    step->constants.y = device->dram[constants->bank].y;
    step->constants.base = constants->address;
    step->output.x = out.x;
    step->output.y = out.y;
    step->output.base = out.address;
}

static void release_members(struct pn_tt_norm_role *role)
{
    for (uint32_t m = 0; m < PN_TT_NORM_ROLE_MAX_MEMBERS; m++)
        pn_tt_residency_release(&role->member[m]);
}

/** Boot and place every member; fills the schedule's member coordinates. */
static bool boot_members(pn_tt_device *device, struct pn_tt_norm_role *role)
{
    const uint32_t slice = role->schedule.tile_cols / role->members;

    for (uint32_t m = 0; m < role->members; m++) {
        pn_tt_residency *member = &role->member[m];

        if (!pn_tt_prepare_family_on_core(device, member, FT_TT_OP_NORM_ROLE_PROGRAM,
                                          m == 0u ? "norm-role" : "norm-role-member",
                                          PN_TT_PERSIST_NORM_CORES_ENV, role->cores[m],
                                          PN_TT_ONCARD_TRIAD_MASK))
            return false;
        if (ft_tt_op_norm_role_place_cbs(role->schedule.tile_cols, slice, role->placed,
                                         member->cb_buffer, sizeof(member->cb_buffer),
                                         &member->cb_mask) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: norm role: the circular buffers do not fit (T %u, S %u)\n",
                    role->schedule.tile_cols, slice);
            return false;
        }
        role->schedule.member_x[m] = member->core_x;
        role->schedule.member_y[m] = member->core_y;
    }
    return true;
}

/** Start every member and write its schedule table into its table page. */
static bool start_members(pn_tt_device *device, struct pn_tt_norm_role *role)
{
    static uint32_t table[FT_TT_OP_NORM_ROLE_TABLE_WORDS];
    const uint32_t address = ft_tt_op_norm_role_cb_address(role->placed, kNrCbTable);
    const ft_tt_noc_access *noc =
        device->runtime.bulk_l1.write ? &device->runtime.bulk_l1 : &device->runtime.noc;
    uint32_t used = 0u;

    for (uint32_t m = 0; m < role->members; m++) {
        pn_tt_residency *member = &role->member[m];

        if (ft_tt_op_norm_role_table(&role->schedule, m, table, &used) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: norm role: member %u's schedule table is refused\n", m);
            return false;
        }
        if (!pn_tt_start_family(member))
            return false;
        if (address == 0u || noc->write(noc->user, member->core_x, member->core_y, address,
                                        table, used * sizeof(uint32_t)) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: norm role: member %u's table write failed\n", m);
            return false;
        }
    }
    return true;
}

int pn_tt_norm_role_arm(pn_tt_device *device, const ft_tt_op_norm_role_schedule *schedule)
{
    struct pn_tt_norm_role *role = role_of(device);
    const char *list = getenv(PN_TT_PERSIST_NORM_CORES_ENV);
    const char *exchange_text = getenv(PN_TT_PERSIST_NORM_EXCHANGE_ENV);
    const int exchange = pn_tt_norm_role_exchange_from(exchange_text);
    double started;
    int count;

    if (!role || !schedule || role->broken)
        return -1;
    if (role->armed && role->schedule.steps == schedule->steps &&
        memcmp(role->schedule.step, schedule->step, sizeof(schedule->step)) == 0)
        return 0;
    if (role->armed)
        norm_role_down(device);
    started = pn_tt_now_seconds();
    count = pn_tt_core_list_parse(list && *list ? list : PN_TT_PERSIST_NORM_CORES_DEFAULT,
                                  role->cores, PN_TT_NORM_ROLE_MAX_MEMBERS, 139u);
    role->schedule = *schedule;
    if (count <= 0 || schedule->tile_cols % (uint32_t)count != 0u) {
        fprintf(stderr, "pn_tt: norm role: core list '%s' does not split T=%u\n",
                list ? list : PN_TT_PERSIST_NORM_CORES_DEFAULT, schedule->tile_cols);
        role->broken = true;
        return -1;
    }
    role->members = (uint32_t)count;
    role->schedule.members = role->members;
    if (exchange < 0) {
        fprintf(stderr, "pn_tt: norm role: %s='%s' is not whole, faces, stagger or "
                        "faces,stagger — the role stays off\n",
                PN_TT_PERSIST_NORM_EXCHANGE_ENV, exchange_text);
        role->broken = true;
        return -1;
    }
    role->schedule.exchange = (uint32_t)exchange;
    if (pn_tt_red("norm_role_gamma_next")) {
        fprintf(stderr, "pn_tt: RED PROOF norm_role_gamma_next — every step norms with the "
                        "NEXT step's gamma. THE ANSWER IS GARBAGE; it must differ.\n");
        for (uint32_t k = 0; k < role->schedule.steps; k++)
            role->schedule.step[k].constants = schedule->step[(k + 1u) % schedule->steps].constants;
    }
    if (!boot_members(device, role) || !start_members(device, role) ||
        pn_tt_replica_map_publish_role(device) != 0) {
        fprintf(stderr, "pn_tt: norm role REFUSED — the norms stay on the reduce / eltwise "
                        "residencies for this process\n");
        release_members(role);
        role->broken = true;
        return -1;
    }
    if (pn_tt_role_table_claim("norm-role", role->cores, role->members, PN_TT_ROLE_MODE_DECODE,
                               role_table_teardown, role_table_arm, device) != 0)
        fprintf(stderr, "pn_tt: norm role: its cores overlap another role in decode — not in "
                        "the role table (mode switches will not move it)\n");
    role->armed = true;
    role->arms++;
    role->arm_seconds = pn_tt_now_seconds() - started;
    fprintf(stderr, "pn_tt: norm role up on %u cores (%s), %u steps a token, armed in %.2f ms "
                    "(arm %u), exchange %s%s\n",
            role->members, list && *list ? list : PN_TT_PERSIST_NORM_CORES_DEFAULT,
            role->schedule.steps, role->arm_seconds * 1e3, role->arms,
            role->schedule.exchange & PN_TT_NORM_ROLE_EXCHANGE_FACES ? "faces" : "whole",
            role->schedule.exchange & PN_TT_NORM_ROLE_EXCHANGE_STAGGER ? " + stagger" : "");
    return 0;
}

/** At a token's first step: a member the idle retire stopped re-arms the whole role. */
static int rearm_if_retired(pn_tt_device *device, struct pn_tt_norm_role *role)
{
    ft_tt_op_norm_role_schedule *schedule;
    int rc;

    for (uint32_t m = 0; m < role->members; m++)
        if (!pn_tt_residency_enqueue_ready(&role->member[m]))
            break;
        else if (m + 1u == role->members)
            return 0;
    schedule = malloc(sizeof(*schedule));
    if (!schedule)
        return -1;
    *schedule = role->schedule;
    norm_role_down(device);
    rc = pn_tt_norm_role_arm(device, schedule);
    free(schedule);
    return rc;
}

/** `words` words of CB `cb` (its base) on `member`, on one line. */
static void norm_role_dump_words(pn_tt_device *device, const pn_tt_residency *member,
                                 const char *what, unsigned cb, uint32_t words)
{
    const struct pn_tt_norm_role *role = device->norm_role;
    const uint32_t address = ft_tt_op_norm_role_cb_address(role->placed, cb);
    uint32_t buffer[kNrHeadWords > kNrFlagsBytes / 4u ? kNrHeadWords : kNrFlagsBytes / 4u];
    char line[16u * 10u + 1u];
    size_t used = 0u;

    if (address == 0u || words > sizeof(buffer) / sizeof(buffer[0]) ||
        device->runtime.noc.read(device->runtime.noc.user, member->core_x, member->core_y,
                                 address, buffer, words * sizeof(uint32_t)) != FT_TT_OK)
        return;
    for (uint32_t w = 0; w < words && used + 10u < sizeof(line); w++)
        used += (size_t)snprintf(line + used, sizeof(line) - used, " %08x", buffer[w]);
    fprintf(stderr, "pn_tt:   %s @%#x:%s\n", what, address, line);
}

/** On a failed step, before teardown: every member's control words, its refusal reason (word
 *  kNrWordReason) and the five processors' status slots — "did not complete: error" alone does
 *  not say which thread refused or why. */
static void norm_role_report_failure(pn_tt_device *device, const struct pn_tt_norm_role *role)
{
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    uint32_t role_words[3];
    uint32_t status[FT_TT_QUEUE_STATUS_WORDS];

    for (uint32_t m = 0; m < role->members; m++) {
        const pn_tt_residency *member = &role->member[m];

        if (!member->running ||
            ft_tt_queue_read_control(&device->runtime.noc, &member->queue, fields) != FT_TT_OK)
            continue;
        if (device->runtime.noc.read(device->runtime.noc.user, member->queue.x, member->queue.y,
                                     member->queue.control_address + kNrWordWatchdog * 4u,
                                     role_words, sizeof(role_words)) != FT_TT_OK)
            memset(role_words, 0, sizeof(role_words));
        fprintf(stderr, "pn_tt: norm role FAIL member %u (%u,%u): state %s error %s reason %u "
                        "head %u tail %u completed %u issued %u last %s watchdog %u steps %u\n",
                m, member->core_x, member->core_y,
                ft_tt_queue_state_string(fields[FT_TT_QUEUE_FIELD_STATE]),
                ft_tt_queue_kernel_error_string(fields[FT_TT_QUEUE_FIELD_ERROR]),
                fields[kNrWordReason], fields[FT_TT_QUEUE_FIELD_HEAD],
                fields[FT_TT_QUEUE_FIELD_TAIL], fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT],
                member->queue.issued,
                ft_tt_queue_opcode_string(fields[FT_TT_QUEUE_FIELD_LAST_OPCODE]),
                role_words[0], role_words[1]);
        for (uint32_t proc = 0; proc < 5u; proc++)
            if (ft_tt_queue_read_status(&device->runtime.noc, &member->queue, proc, status) ==
                FT_TT_OK)
                fprintf(stderr, "pn_tt:   proc %u: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                        proc, status[0], status[1], status[2], status[3], status[4], status[5],
                        status[6], status[7]);
        /* The table header as L1 holds it, and the exchange flags page (one word a member,
         * then the writers' stage words 8 + member): a bad table and a stalled exchange read
         * differently here. */
        norm_role_dump_words(device, member, "table", kNrCbTable, kNrHeadWords);
        norm_role_dump_words(device, member, "flags", kNrCbFlags, kNrFlagsBytes / 4u);
    }
}

int pn_tt_norm_role_step(pn_tt_device *device, uint32_t step)
{
    struct pn_tt_norm_role *role = role_of(device);
    uint32_t words[PN_TT_NORM_ROLE_MAX_MEMBERS][FT_TT_QUEUE_PARAM_WORDS];
    pn_tt_residency *set[PN_TT_NORM_ROLE_MAX_MEMBERS];
    bool bad[PN_TT_NORM_ROLE_MAX_MEMBERS];
    bool failed = false;
    uint32_t m;

    if (!pn_tt_norm_role_armed(device) || pn_tt_matvec_flush(device) != 0)
        return -1;
    if (step == 0u && rearm_if_retired(device, role) != 0)
        return -1;
    if (ft_tt_op_norm_role_parameters(&role->schedule, step, words[0]) != FT_TT_OK)
        return -1;
    for (m = 0; m < role->members; m++) {
        memcpy(words[m], words[0], sizeof(words[0]));
        set[m] = &role->member[m];
        bad[m] = false;
    }
    if (pn_tt_group_pull_decide(&role->pull, pn_tt_group_pull_decode(device), set,
                                role->members, role->members, "norm-role")) {
        (void)pn_tt_group_pull_enqueue(&role->pull, true, set, role->members,
                                       FT_TT_QUEUE_OP_REDUCE, words[0], bad);
        (void)pn_tt_group_pull_await(&role->pull, true, set, role->members, bad, NULL);
    } else {
        for (m = 0; m < role->members; m++)
            bad[m] = pn_tt_residency_enqueue(set[m], FT_TT_QUEUE_OP_REDUCE, words[m], 1u) != 0;
        for (m = 0; m < role->members; m++)
            if (!bad[m] && pn_tt_residency_await(set[m], PN_TT_OP_TIMEOUT_MS) != 0)
                bad[m] = true;
    }
    for (m = 0; m < role->members; m++)
        failed = failed || bad[m];
    if (failed) {
        fprintf(stderr, "pn_tt: norm role step %u FAILED — the role is torn down and stays "
                        "off; this token is refused\n", step);
        norm_role_report_failure(device, role);
        pn_tt_norm_role_teardown(device);
        role->broken = true;
        return -1;
    }
    role->steps++;
    return 0;
}

/** Member `m`'s arrival clocks (flags page words kNrFlagArrival + k), then every flag seen,
 *  the SqAll reserve wait, the first output page and the output acked, us a step. */
static void norm_role_report_arrivals(pn_tt_device *device, const struct pn_tt_norm_role *role,
                                      const pn_tt_residency *member, uint32_t m, double per)
{
    const uint32_t address = ft_tt_op_norm_role_cb_address(role->placed, kNrCbFlags);
    uint32_t page[kNrFlagsBytes / 4u - kNrFlagArrival];
    const uint32_t *arrival = page;
    char line[PN_TT_NORM_ROLE_MAX_MEMBERS * 10u + 1u];
    size_t used = 0u;

    if (address == 0u || role->members > PN_TT_NORM_ROLE_MAX_MEMBERS ||
        device->runtime.noc.read(device->runtime.noc.user, member->core_x, member->core_y,
                                 address + (uint32_t)kNrFlagArrival * 4u, page,
                                 sizeof(page)) != FT_TT_OK)
        return;
    for (uint32_t k = 0; k < role->members && used + 10u < sizeof(line); k++)
        used += (size_t)snprintf(line + used, sizeof(line) - used, " %.2f",
                                 (double)arrival[k] * per);
    fprintf(stderr, "pn_tt: norm role member %u arrivals, us a step from precondition, by "
                    "sender:%s; all seen %.2f, SqAll wait %.2f, output first %.2f, output acked "
                    "%.2f\n",
            m, line, (double)page[kNrFlagAllSeen - kNrFlagArrival] * per,
            (double)page[kNrFlagSqAllWait - kNrFlagArrival] * per,
            (double)page[kNrFlagOutFirst - kNrFlagArrival] * per,
            (double)page[kNrFlagOutDone - kNrFlagArrival] * per);
}

/** Per member, each phase clock as microseconds a step (all from that member's precondition
 *  stamp): input read, own flag sent, every flag in, published. The gap between a member's flag
 *  sent and its flags in is its wait on the other members. Then when each sender's flag first
 *  showed (flags page arrival clocks, same origin): arrival of k at m minus k's own flag sent is
 *  k's precondition skew against m, plus the flag's flight. */
static void norm_role_report_phases(pn_tt_device *device, const struct pn_tt_norm_role *role)
{
    for (uint32_t m = 0; m < role->members; m++) {
        const pn_tt_residency *member = &role->member[m];
        uint32_t phase[4]; /* words 12-15: reason, input read, flag sent, flags in */
        uint32_t tail[3];  /* words 61-63: watchdog, steps, busy */
        double per;

        if (!member->running ||
            device->runtime.noc.read(device->runtime.noc.user, member->queue.x, member->queue.y,
                                     member->queue.control_address + kNrWordReason * 4u, phase,
                                     sizeof(phase)) != FT_TT_OK ||
            device->runtime.noc.read(device->runtime.noc.user, member->queue.x, member->queue.y,
                                     member->queue.control_address + kNrWordWatchdog * 4u, tail,
                                     sizeof(tail)) != FT_TT_OK ||
            tail[1] == 0u)
            continue;
        per = NORM_ROLE_TICK_MS * 1e3 / (double)tail[1];
        fprintf(stderr, "pn_tt: norm role member %u (%u,%u): us a step from precondition — "
                        "input %.2f, flag sent %.2f, flags in %.2f, published %.2f; watchdog %u\n",
                m, member->core_x, member->core_y,
                (double)phase[kNrWordInputRead - kNrWordReason] * per,
                (double)phase[kNrWordFlagSent - kNrWordReason] * per,
                (double)phase[kNrWordFlagsIn - kNrWordReason] * per, (double)tail[2] * per,
                tail[0]);
        norm_role_report_arrivals(device, role, member, m, per);
    }
}

/** Stop and release every member, keeping the schedule (a mode switch re-arms from it). */
static void norm_role_down(pn_tt_device *device)
{
    struct pn_tt_norm_role *role = device ? device->norm_role : NULL;
    uint32_t words[3] = { 0u, 0u, 0u };
    double started;

    if (!role || !role->armed)
        return;
    if (role->member[0].running &&
        device->runtime.noc.read(device->runtime.noc.user, role->member[0].queue.x,
                                 role->member[0].queue.y,
                                 role->member[0].queue.control_address + kNrWordWatchdog * 4u,
                                 words, sizeof(words)) == FT_TT_OK && words[1] != 0u)
        fprintf(stderr, "pn_tt: norm role: %u steps since arm, busy %.3f ms (%.2f us a step, "
                        "precondition met -> published), watchdog %u\n",
                words[1], (double)words[2] * NORM_ROLE_TICK_MS,
                (double)words[2] * NORM_ROLE_TICK_MS * 1e3 / (double)words[1], words[0]);
    norm_role_report_phases(device, role);
    started = pn_tt_now_seconds();
    release_members(role);
    role->armed = false;
    fprintf(stderr, "pn_tt: norm role down: armed in %.2f ms, torn down in %.2f ms, %llu steps "
                    "issued\n",
            role->arm_seconds * 1e3, (pn_tt_now_seconds() - started) * 1e3,
            (unsigned long long)role->steps);
}

void pn_tt_norm_role_teardown(pn_tt_device *device)
{
    norm_role_down(device);
    pn_tt_role_table_release("norm-role");
}

int pn_tt_norm_role_resume(pn_tt_device *device)
{
    struct pn_tt_norm_role *role = device ? device->norm_role : NULL;
    ft_tt_op_norm_role_schedule *schedule;
    int rc;

    if (!role || role->broken || role->armed || role->schedule.steps == 0u)
        return -1;
    schedule = malloc(sizeof(*schedule));
    if (!schedule)
        return -1;
    *schedule = role->schedule;
    rc = pn_tt_norm_role_arm(device, schedule);
    free(schedule);
    return rc;
}

/* The role table's callbacks (pn_tt_role_table.h): a mode switch tears the role down and
 * re-arms it from the saved schedule. */
static void role_table_teardown(void *user)
{
    norm_role_down((pn_tt_device *)user);
}

static int role_table_arm(void *user, void *context)
{
    (void)context;
    return pn_tt_norm_role_resume((pn_tt_device *)user);
}

#endif /* USE_TT */
