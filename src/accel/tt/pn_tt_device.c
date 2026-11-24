/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_device.c
 * @brief Implementation of the Tenstorrent device handle (see pn_tt_device.h).
 *
 * @status Phase 7. The card, the artifact residency and the matmul live here; the reduce,
 *   elementwise and sampling residencies that keep the activation ON the card between
 *   matmuls live in pn_tt_oncard.c. 0.568 -> 0.229 s/token with the four on-card ops.
 *   Tensix programs are resolved BY NAME out of the binary's own blob (Phase 12): there is
 *   no program path in this file, and PN_TT_PROGRAM_DIR is an override, not a default.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_attention_grid.h"
#include "pn_tt_device_internal.h"
#include "pn_tt_attention_splitk.h"
#include "pn_tt_gdngate.h"
#include "pn_tt_reader.h"
#include "pn_tt_core_boot_guard.h"
#include "pn_tt_dispatch.h"
#include "pn_tt_eltwise_group.h"
#include "pn_tt_gdnc.h"
#include "pn_tt_gate.h"
#include "pn_tt_kv_append_split.h"
#include "pn_tt_render.h"
#include "pn_tt_moe.h"
#include "pn_tt_norm_role.h"
#include "pn_tt_replica_map.h"
#include "pn_tt_strip_placement.h"
#include "pn_tt_weight_view.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tt/convert/ft_float_convert.h"
#include "tt/ft_tt_coord.h"
#include "tt/ft_tt_device.h"
#include "tt/ft_tt_kmd.h"
#include "tt/ft_tt_op_dispatch.h"
#include "tt/ft_tt_op_l1_layout.h"
#include "tt/ft_tt_op_matmul.h"
#include "tt/ft_tt_program_resolve.h"
#include "tt/ft_tt_tile_row0.h"

/** The one-shot matmul triads, by NAME. A name is the thing a shipping binary can hold; the
 *  directory they were built in is a property of one machine and is gone from this file. */
#define PN_TT_MATMUL_BFP8_PROGRAM "matmul_bfp8"
#define PN_TT_MATMUL_BFP4_PROGRAM "matmul_bfp4"

/* PN_TT_RESIDENT_PROGRAM moved to pn_tt_device_internal.h: the matmul group's DST interlock
 * reads the ARMED program's recorded build defines, so the program's NAME is no longer
 * private to this file. */

/** `resident` (default when the program is there) or `oneshot`. Both arms are the SAME
 *  binary running the same kernels on the same weights, so an A/B across this knob measures
 *  DISPATCH and nothing else. */
#define PN_TT_DISPATCH_ENV "PN_TT_DISPATCH"

/**
 * RED PROOF ONLY. `PN_TT_RED_BAD_DESCRIPTOR=N` malforms the Nth resident descriptor's
 * parameter block, AFTER the host check that would have refused it, so the kernel's own
 * refusal runs on real silicon.
 *
 * It is a knob and not a test-only build because the thing being proven is a property of
 * the SHIPPING path — that a refused descriptor is recoverable rather than a lost core —
 * and a separate binary would prove it about a different program. It defaults off, it is
 * read once, and it fires once.
 */
#define PN_TT_RED_BAD_DESCRIPTOR_ENV "PN_TT_RED_BAD_DESCRIPTOR"

/** Where to append every projection's result row. See the struct field for why bytes. */
#define PN_TT_MATVEC_DUMP_ENV "PN_TT_MATVEC_DUMP"

/** Which enabled Tensix to drive. 71 is in the set the bring-up log records as verified
 *  healthy after the recovery round; it is a default, not a constant — a serve with a
 *  placement should pass its own. */
#define PN_TT_CORE_DEFAULT 71u
#define PN_TT_CORE_ENV     "PN_TT_CORE"

/** Every processor of the one-shot matmul triad. */
#define PN_TT_TRIAD_MASK                                                                  \
    ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1) | (1u << FT_TT_PROC_MATH0) |         \
     (1u << FT_TT_PROC_MATH1) | (1u << FT_TT_PROC_MATH2))

/** K-tiles the reader fetches per NoC barrier. Shrunk to a divisor of Kt by the driver, so
 *  this is a ceiling and not a promise; 8 divides every Kt this model family uses (48, 64,
 *  144) and keeps the circular buffers far inside the arena.
 *
 *  `PN_TT_PREFETCH_TILES` overrides it, and it is a knob rather than a constant because it
 *  is the ONE number that moves the cooperative matmul's cost: the multicast rendezvous is
 *  paid ONCE PER BLOCK, so doubling the depth halves how many of them a projection pays.
 *  The driver still shrinks whatever is asked for to a divisor of the artifact's Kt gcd,
 *  and the circular buffers are `2 * depth` pages of each input — at depth 16 that is
 *  ~100 KiB of a 1.32 MB arena, so the ceiling is nowhere near. */
#define PN_TT_PREFETCH_TILES_ENV     "PN_TT_PREFETCH_TILES"
#define PN_TT_PREFETCH_TILES_DEFAULT 8u

/* ================================================================
 * Small helpers
 * ================================================================ */

double pn_tt_now_seconds(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/** Parse `name` from the environment: the value, or `fallback` when unset or not a number. */
static uint32_t pn_tt_env_u32_parse(const char *name, uint32_t fallback)
{
    const char *value = getenv(name);
    char *end = NULL;
    unsigned long parsed;

    if (!value || !*value)
        return fallback;
    parsed = strtoul(value, &end, 10);
    if (!end || *end != '\0')
        return fallback;
    return (uint32_t)parsed;
}

/**
 * CACHED BY NAME POINTER (qwen36 plan S2): descriptor builders call this per descriptor, and
 * getenv() is a scan of the whole environment — 13 ms of host CPU a Qwen3.6 token in the
 * replay profile. No PN_TT_* variable changes after start (nothing in Pantheon sets one), so
 * the first answer for a (name, fallback) pair is kept. Called from the forward's thread.
 */
#define PN_TT_ENV_CACHE 128u
uint32_t pn_tt_env_u32(const char *name, uint32_t fallback)
{
    static struct {
        const char *name;
        uint32_t fallback, value;
    } cache[PN_TT_ENV_CACHE];
    static uint32_t used;

    for (uint32_t i = 0; i < used; i++)
        if (cache[i].name == name && cache[i].fallback == fallback)
            return cache[i].value;
    if (used == PN_TT_ENV_CACHE)
        return pn_tt_env_u32_parse(name, fallback);
    cache[used].name = name;
    cache[used].fallback = fallback;
    cache[used].value = pn_tt_env_u32_parse(name, fallback);
    return cache[used++].value;
}

/** The program directory for a weight tile size, or NULL if we hold no program for it. */
static const ft_tt_kernel_program *pn_tt_program_for(const pn_tt_device *device,
                                                     uint32_t tile_bytes)
{
    if (tile_bytes == FT_TT_OP_TILE_BYTES_BFP8_B)
        return &device->program_bfp8.program;
    if (tile_bytes == FT_TT_OP_TILE_BYTES_BFP4_B && device->have_bfp4)
        return &device->program_bfp4.program;
    return NULL;
}

const char *pn_tt_program_override_root(void)
{
    const char *root = getenv(PN_TT_PROGRAM_ROOT_ENV);

    return (root && *root) ? root : NULL;
}

const char *pn_tt_program_where(const char *root)
{
    return root ? root : "this binary's embedded program blob";
}

/**
 * Refuse two programs whose firmware differs.
 *
 * A kernel's text is linked at its own firmware's `__fw_export_text_end`, so booting one
 * program's firmware and then loading another program's kernels is not a configuration,
 * it is a wedged core — and a wedged core here costs a card-level reset. Comparing the five
 * ELFs byte for byte turns that into a refusal.
 *
 * By NAME, not by directory: on the embedded arm the two firmware sets are usually the same
 * deduplicated object, and comparing bytes rather than that pointer is deliberate — the
 * shared pointer is the generator's claim and the bytes are the fact.
 */
static int pn_tt_firmware_matches(const char *name_a, const char *name_b, const char *root)
{
    ft_tt_runtime_images a, b;
    unsigned processor;
    int same = 1;

    if (ft_tt_program_open_firmware(name_a, root, &a) != FT_TT_OK)
        return 0;
    if (ft_tt_program_open_firmware(name_b, root, &b) != FT_TT_OK) {
        ft_tt_runtime_images_release(&a);
        return 0;
    }
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        if (a.elf_size[processor] != b.elf_size[processor] ||
            memcmp(a.elf_data[processor], b.elf_data[processor], a.elf_size[processor]) != 0) {
            same = 0;
            break;
        }
    }
    ft_tt_runtime_images_release(&a);
    ft_tt_runtime_images_release(&b);
    return same;
}

/* ================================================================
 * Open / close
 * ================================================================ */

/** Resolve the core, boot its firmware, and load the matmul programs. */
static int pn_tt_bring_up_core(pn_tt_device *device, const char *root)
{
    const char *dispatch = getenv(PN_TT_DISPATCH_ENV);
    ft_tt_noc_coord core;
    int rc;

    device->red_bad_descriptor = pn_tt_env_u32(PN_TT_RED_BAD_DESCRIPTOR_ENV, 0u);
    device->red_split = pn_tt_matmul_group_red_mode();
    /* How many DRAM banks the hot activation strips are mirrored into. Clamped rather than
     * refused, because the bank count is a property of the part and a run that asked for
     * more should get every bank rather than a failure to open the card. */
    device->activation_replicas = pn_tt_env_u32(PN_TT_ACTIVATION_REPLICAS_ENV,
                                                PN_TT_ACTIVATION_REPLICAS_DEFAULT);
    if (device->activation_replicas == 0u)
        device->activation_replicas = 1u;
    device->core_index = pn_tt_env_u32(PN_TT_CORE_ENV, PN_TT_CORE_DEFAULT);
    rc = ft_tt_coord_tensix_by_index(&device->runtime.grid, device->core_index,
                                     FT_TT_COORD_TRANSLATED, &core);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: Tensix index %u is not on this part\n", device->core_index);
        return -1;
    }
    device->core_x = (uint16_t)core.x;
    device->core_y = (uint16_t)core.y;

    if (ft_tt_program_open_firmware(PN_TT_MATMUL_BFP8_PROGRAM, root, &device->firmware) !=
        FT_TT_OK) {
        fprintf(stderr, "pn_tt: no %s firmware in %s\n", PN_TT_MATMUL_BFP8_PROGRAM,
                pn_tt_program_where(root));
        return -1;
    }
    /* Say WHERE the programs came from, once, before anything runs on them. The whole point
     * of the blob is that a serve no longer depends on a directory existing, and a run that
     * cannot state which arm it took is a run whose measurements name no artifact. */
    fprintf(stderr, "pn_tt: Tensix programs — source %s (%s)\n",
            ft_tt_program_source_string(device->firmware.source), pn_tt_program_where(root));

    if (ft_tt_program_open_kernels(PN_TT_MATMUL_BFP8_PROGRAM, root, PN_TT_TRIAD_MASK,
                                   &device->program_bfp8) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: no Bfp8_b matmul program in %s\n", pn_tt_program_where(root));
        return -1;
    }
    /* Bfp4_b is optional: a model whose FFN is not block-float-4 never asks for it, and
     * refusing the whole card because one program is missing would be wrong. */
    device->have_bfp4 = pn_tt_firmware_matches(PN_TT_MATMUL_BFP8_PROGRAM,
                                               PN_TT_MATMUL_BFP4_PROGRAM, root) &&
                        ft_tt_program_open_kernels(PN_TT_MATMUL_BFP4_PROGRAM, root,
                                                   PN_TT_TRIAD_MASK,
                                                   &device->program_bfp4) == FT_TT_OK;
    if (!device->have_bfp4)
        fprintf(stderr, "pn_tt: no usable Bfp4_b matmul program in %s "
                        "(missing, or built against different firmware)\n",
                pn_tt_program_where(root));

    /* The resident triad is NOT optional, and the previous "serve one-shot rather than
     * refuse" is retracted (2026-08-15). The program went missing from the host's program
     * root for a day and nothing said so: MEASURED on this card, same binary, same weights,
     * same on-card families, only this knob moved — resident 0.343 s a token against
     * one-shot 0.372, i.e. **+0.029 s, 1.085x**, entirely in the dispatch phase (0.156 ->
     * 0.185). That is small enough to publish by accident and large enough to be wrong, and
     * it is exactly the size of regression a silent fallback is good at hiding. A
     * configuration this backend cannot honour is a refusal; the slow path stays reachable,
     * but it has to be ASKED for.
     *
     * The firmware comparison is not a formality either: a kernel's text is linked at ITS
     * OWN firmware's `__fw_export_text_end`, so booting one program's firmware and then
     * loading another's kernels is a wedged core, and a wedged core here costs a card-level
     * reset. That is why a MISMATCH is refused on the same terms as an absence. */
    device->resident_requested = !(dispatch && strcmp(dispatch, "oneshot") == 0);
    device->have_resident = false;
    if (device->resident_requested) {
        if (!pn_tt_firmware_matches(PN_TT_MATMUL_BFP8_PROGRAM, PN_TT_RESIDENT_PROGRAM, root) ||
            ft_tt_program_open_kernels(PN_TT_RESIDENT_PROGRAM, root, PN_TT_TRIAD_MASK,
                                       &device->program_resident) != FT_TT_OK) {
            fprintf(stderr,
                    "pn_tt: REFUSED — no usable %s program in %s (missing, or "
                    "built against different firmware than %s).\n"
                    "pn_tt: one-shot dispatch re-uploads five kernel images per op and costs "
                    "1.085x the seconds per token (measured 0.343 -> 0.372 on MiniCPM5-1B), "
                    "while reporting a token that looks healthy — so falling back to it "
                    "silently is the defect this refusal replaces.\n"
                    "pn_tt: build it (foundry/tt/tests/Makefile.phase6 carries the recipe) and "
                    "regenerate the embedded blob (cd foundry/tt/tools && make regen-programs "
                    "— that cwd is required), or ask for the slow path deliberately with "
                    "%s=oneshot.\n",
                    PN_TT_RESIDENT_PROGRAM, pn_tt_program_where(root),
                    PN_TT_MATMUL_BFP8_PROGRAM, PN_TT_DISPATCH_ENV);
            return -1;
        }
        device->have_resident = true;
    }

    /* THE FIRST BOOT A SERVE TAKES, AND THEREFORE THE ONE A RESTART LANDS ON. This core is
     * matmul member 0; if the previous process left a compute thread spinning here, this is
     * where an ordinary restart wedges the card. The card is asked before anything is
     * written — `pn_tt_core_boot_guard.h`. */
    if (!pn_tt_core_boot_is_permitted(&device->runtime, device->core_index, "matmul member 0",
                                      PN_TT_CORE_ENV))
        return -1;

    rc = ft_tt_runtime_boot_core(&device->runtime.noc, device->core_x, device->core_y,
                                 &device->firmware);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: core %u (%u,%u) firmware boot failed: %s\n", device->core_index,
                core.x, core.y, ft_tt_error_string(rc));
        return -1;
    }
    fprintf(stderr, "pn_tt: core %u at translated (%u,%u), firmware up\n", device->core_index,
            core.x, core.y);
    return 0;
}

/* The devices this process has open, so the hardware report can read each card's DRAM
 * allocators. A device is listed only between a successful open and the start of its close,
 * and its allocator is read under the lock, so a read never sees a device being freed. */
#define PN_TT_OPEN_MAX 8
static pthread_mutex_t g_open_lock = PTHREAD_MUTEX_INITIALIZER;
static struct {
    const pn_tt_device *device;
    int index;
} g_open[PN_TT_OPEN_MAX];

static void pn_tt_open_note(const pn_tt_device *device, int device_index)
{
    pthread_mutex_lock(&g_open_lock);
    for (int i = 0; i < PN_TT_OPEN_MAX; i++) {
        if (!g_open[i].device) {
            g_open[i].device = device;
            g_open[i].index = device_index;
            break;
        }
    }
    pthread_mutex_unlock(&g_open_lock);
}

static void pn_tt_open_forget(const pn_tt_device *device)
{
    pthread_mutex_lock(&g_open_lock);
    for (int i = 0; i < PN_TT_OPEN_MAX; i++)
        if (g_open[i].device == device)
            g_open[i].device = NULL;
    pthread_mutex_unlock(&g_open_lock);
}

int pn_tt_device_dram_usage_on(int device_index, uint64_t *used_out, uint64_t *total_out)
{
    uint64_t used = 0u, total = 0u;
    int found = 0;

    pthread_mutex_lock(&g_open_lock);
    for (int i = 0; i < PN_TT_OPEN_MAX; i++) {
        uint64_t u = 0u, t = 0u;

        if (!g_open[i].device || g_open[i].index != device_index)
            continue;
        pn_tt_device_dram_usage(g_open[i].device, &u, &t);
        used += u;
        total = t > total ? t : total;
        found = 1;
    }
    pthread_mutex_unlock(&g_open_lock);
    if (!found)
        return -1;
    if (used_out)
        *used_out = used;
    if (total_out)
        *total_out = total;
    return 0;
}

int pn_tt_device_open(int device_index, pn_tt_device **out_device)
{
    /* NULL means the embedded blob, and NULL is the normal case. Passed through rather than
     * defaulted to a path: the default IS the binary's own programs. */
    const char *root = pn_tt_program_override_root();
    pn_tt_device *device = NULL;
    uint32_t bank;
    int rc;

    if (!out_device)
        return -1;
    *out_device = NULL;

    device = calloc(1, sizeof(*device));
    if (!device)
        return -1;
    /* 2, which is where the artifact load has always started its bank rotation. It is a
     * device field now so a post-load restage continues the same walk. */
    device->weight_bank_cursor = 2u;
    /* EVERY on-card family is on by default EXCEPT attention, and the exception is per
     * FAMILY rather than global — `pn_tt_device_oncard_default()` is how a backend that has
     * the evidence turns it on for its own model. The qwen35/bonsai hybrid does; the llama
     * backend does not, because on-card attention there fails MiniCPM5-1B's chat-parity gate
     * and a shared default would have shipped that as a side effect. */
    for (uint32_t op = 0; op < PN_TT_ONCARD_COUNT; op++)
        device->oncard_default[op] = (op != (uint32_t)PN_TT_ONCARD_ATTENTION);

    rc = ft_tt_runtime_open(device_index, &device->runtime);
    if (rc != FT_TT_OK) {
        free(device);
        return -1;
    }
    /* Which transport served this open, stated at the boundary. `path=gpu` and a
     * non-zero dispatch count prove the CARD ran the forward; this line is what
     * says WHICH DRIVER it ran through, and the two questions are independent. */
    {
        char address[32];

        if (ft_tt_runtime_pci_address(&device->runtime, address, sizeof(address)) != FT_TT_OK)
            (void)snprintf(address, sizeof(address), "(address unavailable)");
        fprintf(stderr, "pn_tt: device %d open over the %s transport, PCI %s\n", device_index,
                ft_tt_runtime_driver_string(device->runtime.driver), address);
    }
    if (pn_tt_bring_up_core(device, root) != 0)
        goto fail;

    device->dram_bank_count = ft_tt_coord_dram_bank_count(&device->runtime.grid);
    if (device->dram_bank_count == 0 || device->dram_bank_count > FT_TT_MAX_DRAM_BANKS)
        goto fail;
    for (bank = 0; bank < device->dram_bank_count; bank++) {
        if (ft_tt_coord_dram_bank(&device->runtime.grid, bank, FT_TT_COORD_TRANSLATED,
                                  &device->dram[bank]) != FT_TT_OK)
            goto fail;
    }
    if (ft_tt_allocator_init(&device->allocator, device->runtime.ctx, &device->runtime.grid, 0) !=
        FT_TT_OK)
        goto fail;
    /* CLAMPED HERE and not where the env is read, because the bank count is a property of
     * the part. More mirrors than banks would have to put two of them in one bank, which is
     * the exact condition mirroring exists to remove. */
    if (device->activation_replicas > device->dram_bank_count) {
        fprintf(stderr, "pn_tt: %s=%u is more than this card's %u banks — using %u\n",
                PN_TT_ACTIVATION_REPLICAS_ENV, device->activation_replicas,
                device->dram_bank_count, device->dram_bank_count);
        device->activation_replicas = device->dram_bank_count;
    }
    if (device->activation_replicas > PN_TT_ACTIVATION_MAX_REPLICAS)
        device->activation_replicas = PN_TT_ACTIVATION_MAX_REPLICAS;
    if (device->activation_replicas > 1u)
        fprintf(stderr, "pn_tt: the hot activation strips are mirrored into %u DRAM banks — "
                        "matmul member m reads mirror m %% %u\n",
                device->activation_replicas, device->activation_replicas);
    /* AFTER the banks, because the group's activation door table is one assignment per
     * bank, and after the core, because member 0 IS that core. */
    if (pn_tt_matmul_group_configure(device) != 0)
        goto fail;

    *out_device = device;
    pn_tt_open_note(device, device_index);
    return 0;

fail:
    pn_tt_device_close(device);
    return -1;
}

/* ================================================================
 * The resident triad
 *
 * The one-shot path re-uploads five kernel images per op — 36.2 µs/kB, ~0.22 ms before
 * anything is computed. A resident triad is the same three kernels built to NOT return:
 * five processors sit on the core reading 32-byte descriptors out of its own L1, and the
 * host's per-op work becomes a parameter block, a descriptor and a 4-byte head bump with
 * no round trip. ft_tt_queue.h is the protocol and states the consequence for op authors:
 * DO NOT FUSE OPS TO SAVE DISPATCH — there is no dispatch to save.
 *
 * ALL OF IT — including the one-member case — now lives in pn_tt_matmul_group.c, because a
 * projection may be split across several cores and a residency per core is what that costs.
 * A group of one is this file's old path with the same core, the same queue and the same
 * sixteen descriptor words, which is what makes PN_TT_MATMUL_CORES=1 a control rather than
 * a second implementation.
 * ================================================================ */

int pn_tt_device_keepalive(pn_tt_device *device, double min_idle_seconds)
{
    static int enabled = -1;
    int sent;

    if (enabled < 0) {
        const char *value = getenv("PN_TT_KEEPALIVE");

        enabled = (value && *value) ? (strcmp(value, "0") != 0) : 1;
    }
    if (!enabled || !device || !pn_tt_dispatch_quiet() || device->gdn_pending_members != 0u)
        return 0;
    sent = pn_tt_residency_keepalive_all(min_idle_seconds);
    {
        const int group = pn_tt_matmul_group_keepalive(device, min_idle_seconds);

        sent = (sent < 0 || group < 0) ? -1 : sent + group;
    }
    if (sent != 0)
        pn_tt_dispatch_sequence_reset();
    if (sent > 0)
        device->keepalive_nops += (uint64_t)sent;
    return sent;
}

void pn_tt_device_close(pn_tt_device *device)
{
    if (!device)
        return;
    pn_tt_open_forget(device);
    /* The reader thread holds its own window bank: joined before anything is torn down. */
    pn_tt_reader_close(device);
    /* The residency is retired by ASKING; the one-shot path has nothing to retire, because
     * it leaves the core in its firmware's GO wait, which is where a healthy idle core sits.
     * Neither is PARKED, deliberately — parking costs the core its next boot even after a
     * textbook-clean shutdown (foundry/tt/DEV_NOTES.md §1). */
    /* Say what the dump CONTAINS before anything can read it. A comparison of two empty
     * files is green, and "the two arms agreed" would then be a statement about nothing. */
    if (device->matvec_dump) {
        fprintf(stderr, "pn_tt: matvec dump — %llu result rows, %llu bytes\n",
                (unsigned long long)device->matvec_dump_rows,
                (unsigned long long)device->matvec_dump_bytes);
        fclose(device->matvec_dump);
        device->matvec_dump = NULL;
    }
    /* Clear the death-stop BEFORE retiring, so the driver's write cannot fire on a clean
     * exit. It would be harmless (a store to L1 nobody reads), but an action the driver
     * performs on every close is an action nobody can attribute to a cause — and the whole
     * value of this registration is that when it DOES fire, it means the process died. */
    {
        ft_tt_kmd_device *kmd = ft_tt_ctx_kmd_device(device->runtime.ctx);

        if (kmd)
            (void)ft_tt_kmd_clear_noc_cleanup(kmd);
    }
    pn_tt_matmul_grid_close(device);
    pn_tt_matmul_group_retire(device);
    pn_tt_moe_release(device);
    pn_tt_norm_role_teardown(device);
    free(device->norm_role);
    device->norm_role = NULL;
    pn_tt_residency_release(&device->reduce);
    {
        uint32_t m;

        for (m = 0; m + 1u < PN_TT_REDUCE_MAX_MEMBERS; m++)
            pn_tt_residency_release(&device->reduce_member[m]);
    }
    {
        uint32_t f;

        for (f = 0; f < device->idle_fill_count; f++)
            pn_tt_residency_release(&device->idle_fill[f]);
        device->idle_fill_count = 0u;
    }
    device->reduce_members = 0u;
    /* THE ELEMENTWISE SPLIT'S EXTRA MEMBERS GO FIRST, because the release below is what makes
     * `eltwise_members` fall back to 0 and the set is walked from that count. A leaked triad
     * is a WEDGED Tensix for the next process to ask for that core — the failure the head-norm
     * slots below cost on 2026-08-21. */
    pn_tt_eltwise_group_release(device);
    pn_tt_residency_release(&device->eltwise);
    device->eltwise_members = 0u;
    pn_tt_residency_release(&device->sample);
    {
        uint32_t m;

        for (m = 1u; m < device->sample_members; m++)
            pn_tt_residency_release(&device->sample_member[m - 1u]);
        device->sample_members = 0u;
    }
    pn_tt_residency_release(&device->rope);
    pn_tt_kv_append_members_release(device);
    pn_tt_residency_release(&device->memory);
    pn_tt_residency_release(&device->puller);
    pn_tt_residency_release(&device->attention);
    for (uint32_t member = 1u; member < device->attention_members; member++)
        pn_tt_residency_release(&device->attention_member[member - 1u]);
    device->attention_members = 0u;
    pn_tt_attention_splitk_release(device);
    /* THE PER-PHASE FIGURE PRINTS AT RELEASE, NOT ON A TOKEN MULTIPLE, and that is a fix
     * rather than a duplicate of the report line.
     *
     * `qwen35_tt_report_gdn()` runs under `tokens % PN_TT_REPORT == 0`, so an arm that
     * CORRUPTS the model's output — which every data arm on this op does by construction —
     * shortens the run below the threshold and the instrument that would have scored the arm
     * silently does not print at all. Measured 2026-08-27: three CONV data arms generated 8
     * tokens against the control's 48 at `PN_TT_REPORT=48` and produced no `gdn phase` line,
     * leaving a result file that reads like a completed arm with a number missing. The
     * chunkwise family never had the defect because `pn_tt_gdnc_release()` prints
     * unconditionally; this is the decode family being given the same property. */
    pn_tt_gdn_report_phases(device);
    pn_tt_residency_release(&device->gdn);
    /* The chunkwise family, which also frees its constants. A leaked triad is a WEDGED Tensix
     * for the next process to ask for that core — see the per-head reduce note below. */
    pn_tt_gdnc_release(device);
    pn_tt_gdngate_release(device);
    /* The head split's extra members. `gdn_members` counts member 0, released above, so this
     * gives back one fewer; at the shipped default of 1 the loop does not run. */
    {
        uint32_t member;

        for (member = 1u; member < device->gdn_members; member++)
            pn_tt_residency_release(&device->gdn_member[member - 1u]);
        device->gdn_members = 0u;
    }
    /* THE PER-HEAD REDUCE SLOTS, WHICH THIS FUNCTION USED TO LEAK. Every other family is a
     * NAMED FIELD released above; these are a SET whose size is discovered at arm time, and
     * being a loop rather than a field is how they were missed. MEASURED 2026-08-21 on the
     * first arm that ran attention and the GDN block together: after a clean exit
     * `ft_tt_core probe` reported cores 91 and 103 as `SKIP refused: a kernel is running on
     * this core` while every named residency came back PASS — and one slot leaked the same
     * way on every single-width arm before it, which is why `before` probed 20 PASS and
     * `after` probed 19. A leaked triad is not untidiness: the next process to ask for that
     * core boots a second program's firmware onto a core that already holds one, which is a
     * WEDGED Tensix rather than a configuration. A `serve` reloading a model does exactly
     * that. */
    {
        uint32_t slot;

        for (slot = 0; slot < device->head_norm_count; slot++) {
            pn_tt_residency_release(&device->head_norm[slot].residency);
            for (uint32_t m = 1u; m < device->head_norm[slot].members; m++)
                pn_tt_residency_release(&device->head_norm[slot].member[m - 1u]);
            device->head_norm[slot].members = 0u;
        }
        device->head_norm_count = 0u;
    }
    /* The activation strip bridge's two directions, and their zero operands with them. A
     * residency this function forgets WEDGES THE NEXT PROCESS — the head-norm slots were leaked
     * on every arm since the norms moved and made cores 91/103 probe "a kernel is running"
     * after a clean exit. The merged gate is released for the same reason. */
    pn_tt_render_release(device);
    pn_tt_gate_release(device);
    free(device->kv_region);
    /* C's borrowed grid members, if a prefill stopped while they lived: retired and parked
     * like every other residency here, never left running for the next process. */
    if (device->attention_grid_count != 0u)
        (void)pn_tt_attention_grid_park_all(device, NULL, NULL);
    free(device->attention_grid);
    device->attention_grid = NULL;
    device->kv_region = NULL;
    free(device->gdn_region);
    device->gdn_region = NULL;
    free(device->gdn_checkpoint);
    device->gdn_checkpoint = NULL;
    if (device->artifact_open)
        ft_tt_fttw_close(&device->artifact);
    if (device->extra_open)
        ft_tt_fttw_close(&device->extra_artifact);
    free(device->attn_constants_host);
    device->attn_constants_host = NULL;
    ft_tt_runtime_images_release(&device->program_bfp8);
    ft_tt_runtime_images_release(&device->program_bfp4);
    ft_tt_runtime_images_release(&device->program_resident);
    ft_tt_runtime_images_release(&device->firmware);
    ft_tt_runtime_close(&device->runtime);
    free(device->resident);
    free(device->row_in);
    free(device->row_out);
    free(device->tile_stage);
    free(device->rows_stage);
    free(device->gdn_stage);
    free(device->gdn_window_shadow);
    free(device->gdn_qkv_history);
    free(device);
}

/* ================================================================
 * Weight residency
 * ================================================================ */

static uint32_t pn_tt_gcd(uint32_t a, uint32_t b)
{
    while (b != 0u) {
        uint32_t r = a % b;

        a = b;
        b = r;
    }
    return a;
}

/**
 * Widest shape any matmul in the artifact will ask for, plus the two facts a RESIDENCY
 * needs decided before it is launched rather than per op.
 *
 * A resident binary is bound to one weight tile size and its circular buffers are placed
 * once, so both of those are properties OF THE ARTIFACT: `nmajor_tile_bytes` is the single
 * matmul tile size, or 0 if the artifact mixes two (which needs two residencies and one
 * core cannot hold two — ft_tt_op_l1_layout.h's divergence 2). `kt_gcd` is what the one
 * prefetch depth has to divide, because a depth that does not divide a later descriptor's
 * Kt collapses to 1 on the card and walks the CB write pointer off the block grid.
 */
/**
 * The MoE block's own weights, which no dense family reads: a routed-expert stack (Bfp6_b,
 * per-expert placement) and an fp32 router (`ffn_gate_inp`, spec §11 D4: 4096 B tiles, its
 * own program in qwen36 plan Phase 5.3). Neither decides the dense triad's tile size.
 */
static bool pn_tt_entry_is_moe_block(const ft_tt_artifact_entry *entry)
{
    if (ft_tt_fttw_expert_count(entry) > 1u)
        return true;
    return entry->format == FT_TT_FORMAT_FP32 && strstr(entry->name, "ffn_gate_inp") != NULL &&
           strstr(entry->name, "ffn_gate_inp_shexp") == NULL;
}

static void pn_tt_scan_extents(pn_tt_device *device)
{
    uint32_t index;
    uint32_t kt_gcd = 0u;
    uint32_t tile_bytes = 0u;
    bool mixed = false;

    for (index = 0; index < device->artifact.header.tensor_count; index++) {
        const ft_tt_artifact_entry *entry = &device->artifact.directory[index];
        uint32_t entry_tile_bytes;

        /* Routed-expert stacks and the fp32 router are the MoE block's, never the dense
         * triad's: their tile sizes (832 B, 4096 B) do not decide the dense program. */
        if (entry->layout != FT_TT_LAYOUT_MATMUL_NMAJOR || pn_tt_entry_is_moe_block(entry))
            continue;
        if (entry->tile_rows > device->max_tile_rows)
            device->max_tile_rows = entry->tile_rows;
        if (entry->tile_cols > device->max_tile_cols)
            device->max_tile_cols = entry->tile_cols;
        kt_gcd = pn_tt_gcd(kt_gcd, entry->tile_rows);
        entry_tile_bytes = ft_tt_fttw_tile_bytes(entry->format);
        if (tile_bytes == 0u)
            tile_bytes = entry_tile_bytes;
        else if (tile_bytes != entry_tile_bytes)
            mixed = true;
    }
    device->max_in_features = device->max_tile_rows * PN_TT_TILE_DIM;
    device->max_out_features = device->max_tile_cols * PN_TT_TILE_DIM;
    device->artifact_kt_gcd = kt_gcd;
    device->artifact_tile_bytes = mixed ? 0u : tile_bytes;
}

/**
 * Launch the residency for THIS artifact, or say in one line why it could not be.
 *
 * Itself never fatal, because it serves TWO callers with opposite needs: the load path,
 * where not arming means the whole run is 1.9x slower and must refuse (that check is at the
 * call site, on `resident_running`), and the mid-run re-arm after a refused descriptor,
 * where falling through to one-shot for the rest of the token is the difference between a
 * recoverable refusal and a lost token — and is counted, not hidden.
 */
static void pn_tt_arm_death_stop(pn_tt_device *device);

static void pn_tt_arm_residency(pn_tt_device *device)
{
    uint32_t depth;

    if (!device->resident_requested) {
        fprintf(stderr, "pn_tt: dispatch = one-shot (%s)\n", PN_TT_DISPATCH_ENV);
        return;
    }
    if (device->artifact_tile_bytes != FT_TT_OP_TILE_BYTES_BFP8_B) {
        /* One binary, one slab, one weight tile size. An artifact that mixes Bfp8_b and
         * Bfp4_b matmul weights needs two resident programs on one core, which needs XIP or
         * a second slab set and is not built. */
        fprintf(stderr, "pn_tt: artifact matmul tile size is %u B, the resident program "
                        "serves %u B — one-shot dispatch\n",
                device->artifact_tile_bytes, FT_TT_OP_TILE_BYTES_BFP8_B);
        return;
    }
    depth = ft_tt_op_block_depth(device->artifact_kt_gcd,
                                 pn_tt_env_u32(PN_TT_PREFETCH_TILES_ENV,
                                               device->prefetch_tiles_preferred
                                                   ? device->prefetch_tiles_preferred
                                                   : PN_TT_PREFETCH_TILES_DEFAULT));
    if (pn_tt_matmul_group_arm(device, device->artifact_tile_bytes, depth) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: residency failed to launch — one-shot dispatch\n");
        device->resident_running = false;
        return;
    }
    fprintf(stderr, "pn_tt: resident triad up on %u core(s), weight tile %u B, depth %u "
                    "(Kt gcd %u)\n",
            device->group.member_count, device->resident_tile_bytes, device->resident_depth,
            device->artifact_kt_gcd);
    pn_tt_arm_death_stop(device);
}

/**
 * Hand tt-kmd the one NoC write it will perform for us if this process dies abnormally.
 *
 * ---------------------------------------------------------------------------
 * WHICH CORE, AND WHY IT IS A CHOICE AND NOT AN ACCIDENT
 * ---------------------------------------------------------------------------
 * The driver holds ONE registration per fd, so a forward that runs four matmul members plus
 * the on-card families protects exactly one core and the rest keep only `idle_spin_limit`.
 * This registers **member 0's queue**, which is `device->queue` on `device->core_x/core_y`,
 * for three reasons that are properties of the design rather than preferences:
 *
 *  - It is the residency that exists in EVERY configuration. `PN_TT_MATMUL_CORES=1` is the
 *    pre-group path and member 0 is the whole of it, so registering any other member would
 *    leave the default configuration unprotected.
 *  - It is live for essentially all of decode — 169 projections a token go through it — so a
 *    kill "mid-generation" almost certainly lands while this core is running or waiting.
 *  - It is the core a RESTARTED process brings up FIRST (`pn_tt_bring_up_core()` resolves the
 *    same index before the group does), so it is the core that loses the boot race, and
 *    booting firmware onto a core with a live resident program is the documented wedge.
 *
 * Re-armed on every residency arm, including the mid-run re-arm, because `ft_tt_queue_init()`
 * gives the ring a fresh nonce and the address is derived from the queue rather than assumed.
 *
 * A failure here is reported and not fatal: an older driver without the ioctl answers EINVAL,
 * and a backend that runs without the safety net is strictly what we had yesterday.
 */
static void pn_tt_arm_death_stop(pn_tt_device *device)
{
    ft_tt_kmd_device *kmd = ft_tt_ctx_kmd_device(device->runtime.ctx);
    uint32_t address = ft_tt_queue_stop_request_address(&device->queue);
    int rc;

    if (address == 0u)
        return;
    if (!kmd) {
        /* The userspace-driver transport has no kernel to register the write
         * with, and that is the ONE thing on ft_tt_udrv.h §5's loss list that
         * cannot be replaced — the value of the ioctl is precisely that it
         * survives the death of the process. Said out loud once per arm rather
         * than silently skipped, because "the death-stop is armed" is a claim
         * somebody will otherwise carry across a transport that cannot make it.
         * The residency idle-spin bound is what remains. */
        fprintf(stderr, "pn_tt: NO DEATH-STOP on the %s transport — SET_NOC_CLEANUP is a "
                        "tt-kmd feature and there is no kernel here to hold it. An abnormal "
                        "death leaves core %u resident until its idle spin limit (measured at "
                        "over 40 s).\n",
                ft_tt_runtime_driver_string(device->runtime.driver), device->core_index);
        return;
    }
    rc = ft_tt_kmd_set_noc_cleanup(kmd, 0u, (uint8_t)device->core_x, (uint8_t)device->core_y,
                                   address, FT_TT_QUEUE_STOP_REQUESTED);
    if (rc != 0) {
        fprintf(stderr, "pn_tt: SET_NOC_CLEANUP refused (%s) — an abnormal death will leave "
                        "core %u resident until its idle spin limit\n",
                strerror(-rc), device->core_index);
        return;
    }
    fprintf(stderr, "pn_tt: death-stop armed — if this process dies, tt-kmd writes "
                    "stop_request=1 to core %u (%u,%u) L1 0x%05x. ONE core: members 1..%u and "
                    "the on-card families keep only the idle spin limit.\n",
            device->core_index, device->core_x, device->core_y, address,
            device->group.member_count ? device->group.member_count - 1u : 0u);
}

/** Card and host staging, sized from the extents. Allocated BEFORE the weights so the
 *  bump allocator's LIFO free order matches the close order. */
static int pn_tt_alloc_scratch(pn_tt_device *device)
{
    size_t act_bytes = (size_t)device->max_tile_rows * PN_TT_TILE_BF16;
    size_t out_bytes = (size_t)device->max_tile_cols * PN_TT_TILE_BF16;

    if (act_bytes == 0 || out_bytes == 0)
        return -1;
    /* Two different banks: the reader pulls the activation while the writer pushes the
     * result, and Phase 6.7 measured two endpoints on one NoC row sharing a ceiling. With
     * one core in flight it is not a throughput question, but it costs nothing to keep. */
    if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, 0, act_bytes,
                           &device->activation) != FT_TT_OK)
        return -1;
    if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM,
                           device->dram_bank_count > 1 ? 1u : 0u, out_bytes,
                           &device->result) != FT_TT_OK)
        return -1;

    /* BOTH rows are sized to the widest of the two directions. They stage activation
     * STRIPS as well as matvec operands now, and a strip may be an input in one op and an
     * output in the next — the lm_head's logit row is read into row_out and the attention
     * output is written from row_in, and neither is bounded by the other's extent. */
    device->max_row_features = device->max_in_features > device->max_out_features
                                   ? device->max_in_features
                                   : device->max_out_features;
    device->row_in = calloc(device->max_row_features, sizeof(*device->row_in));
    device->row_out = calloc(device->max_row_features, sizeof(*device->row_out));
    if (!device->row_in || !device->row_out)
        return -1;
    return 0;
}

/**
 * Zero the whole activation strip on the card, ONCE.
 *
 * pn_tt_matvec() writes only logical row 0 of each activation tile from then on, because
 * rows 1-31 are the tile quantum's zero padding and re-sending 2016 zero bytes per tile
 * per token is 32x the traffic for no information. That is only correct if rows 1-31 are
 * ALREADY zero — DRAM out of the bump allocator is not, and the reader feeds all 32 rows
 * to the FPU. So this is the precondition ft_tt_tile_row0_write() states and cannot check.
 */
static int pn_tt_stage_activation_zeros(pn_tt_device *device)
{
    size_t act_bytes = (size_t)device->max_tile_rows * PN_TT_TILE_BF16;
    const ft_tt_noc_access *bulk = &device->runtime.bulk_dram;
    unsigned char *zeros = calloc(1, act_bytes);
    int rc;

    if (!zeros)
        return -1;
    rc = bulk->write(bulk->user, (uint16_t)device->dram[device->activation.logical_bank].x,
                     (uint16_t)device->dram[device->activation.logical_bank].y,
                     device->activation.address, zeros, act_bytes);
    free(zeros);
    if (rc != FT_TT_OK)
        return -1;
    device->bytes_written += act_bytes;
    return 0;
}

/**
 * A routed-expert stack's DRAM, reserved and NOT written: the Qwen3.6 MoE census arm
 * (`PN_TT_Q36_CENSUS=1`) needs every expert byte held so the fit is the real one, and no
 * expert placement exists before the MoE block does.
 * Spread evenly over the banks, as a per-member expert placement will spread them.
 */
static int pn_tt_reserve_expert_stack(pn_tt_device *device, const ft_tt_artifact_entry *entry)
{
    const uint32_t banks = device->dram_bank_count;
    const uint64_t share = (entry->byte_count / banks + 63u) / 64u * 64u;

    for (uint32_t bank = 0; bank < banks; bank++) {
        ft_tt_buffer buffer;

        if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, bank, share, &buffer) != FT_TT_OK) {
            fprintf(stderr, "pn_tt: DRAM full reserving %s (%.1f MiB a bank)\n", entry->name,
                    (double)share / (1024.0 * 1024.0));
            return -1;
        }
    }
    device->resident_bytes += share * banks;
    return 0;
}

/** Resident-table slots this artifact needs: one per tensor, plus the extra parts of every
 *  tensor the split rule uploads as parts. */
static uint32_t pn_tt_resident_slots_needed(const pn_tt_device *device)
{
    uint32_t slots = device->artifact.header.tensor_count;

    if (!device->weight_split_set)
        return slots;
    for (uint32_t index = 0; index < device->artifact.header.tensor_count; index++) {
        const ft_tt_artifact_entry *entry = &device->artifact.directory[index];

        if (entry->layout == FT_TT_LAYOUT_MATMUL_NMAJOR &&
            pn_tt_weight_split_applies(&device->weight_split, entry->name, entry->tile_cols))
            slots += pn_tt_weight_split_part_count(&device->weight_split) - 1u;
    }
    return slots;
}

/**
 * Place one N-major tensor: whole, or as the column parts the split rule names.
 *
 * Where each upload goes is the GROUP's decision, because the number of banks a weight is
 * spread over is the number of cores that will read it (pn_tt_matmul_group.h). Parts take
 * consecutive cursor positions from the tensor's own, and the cursor then moves on by ONE as
 * it would for the whole tensor, so every later tensor keeps the bank run it had before.
 */
static int pn_tt_place_resident_entry(pn_tt_device *device, const ft_tt_artifact_entry *entry,
                                      uint32_t tile_bytes)
{
    pn_tt_weight_upload_part parts[PN_TT_WEIGHT_SPLIT_MAX_PARTS];
    const uint32_t start = device->weight_bank_cursor;
    const uint32_t count = pn_tt_weight_upload_parts(
        device->weight_split_set ? &device->weight_split : NULL, entry->name, entry->tile_cols,
        entry->byte_count, parts);

    if (count == 0u) {
        fprintf(stderr, "pn_tt: %s (%u column tiles, %llu B) cannot be cut into its parts\n",
                entry->name, entry->tile_cols, (unsigned long long)entry->byte_count);
        return -1;
    }
    for (uint32_t i = 0; i < count; i++) {
        pn_tt_resident *slot = &device->resident[device->resident_count];
        const bool whole = parts[i].part < 0;

        snprintf(slot->name, sizeof(slot->name), "%s", entry->name);
        slot->part = parts[i].part;
        slot->weight.tile_rows = entry->tile_rows;
        slot->weight.tile_cols = parts[i].tile_cols;
        slot->weight.tile_bytes = tile_bytes;
        slot->weight.in_features = entry->tile_rows * PN_TT_TILE_DIM;
        slot->weight.out_features = parts[i].tile_cols * PN_TT_TILE_DIM;
        if (pn_tt_matmul_group_place_weight_columns(device, entry, whole ? NULL : parts[i].runs,
                                                    whole ? 0u : parts[i].run_count,
                                                    &device->weight_bank_cursor,
                                                    &slot->weight) != 0)
            return -1;
        device->resident_count++;
        device->resident_bytes += parts[i].bytes;
    }
    device->weight_bank_cursor = (start + 1u) % device->dram_bank_count;
    return 0;
}

int pn_tt_device_open_artifact(pn_tt_device *device, const char *path)
{
    if (!device || !path || device->artifact_open)
        return -1;
    if (ft_tt_fttw_open(path, &device->artifact) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: cannot open artifact %s\n", path);
        return -1;
    }
    device->artifact_open = true;

    pn_tt_scan_extents(device);
    if (pn_tt_alloc_scratch(device) != 0) {
        fprintf(stderr, "pn_tt: scratch allocation failed\n");
        return -1;
    }
    if (pn_tt_stage_activation_zeros(device) != 0) {
        fprintf(stderr, "pn_tt: could not stage the activation strip\n");
        return -1;
    }
    /* After the scratch rows, because the dump writes through row_out. */
    if (getenv(PN_TT_MATVEC_DUMP_ENV)) {
        device->matvec_dump = fopen(getenv(PN_TT_MATVEC_DUMP_ENV), "wb");
        if (!device->matvec_dump) {
            fprintf(stderr, "pn_tt: cannot open %s for the matvec dump\n",
                    getenv(PN_TT_MATVEC_DUMP_ENV));
            return -1;
        }
    }
    return 0;
}

int pn_tt_device_artifact_matmul_bytes(const pn_tt_device *device, const char *suffix,
                                       uint64_t *total, uint64_t *suffix_total)
{
    const size_t suffix_length = suffix ? strlen(suffix) : 0u;

    if (!device || !device->artifact_open || !total || !suffix_total)
        return -1;
    *total = *suffix_total = 0u;
    for (uint32_t index = 0; index < device->artifact.header.tensor_count; index++) {
        const ft_tt_artifact_entry *entry = &device->artifact.directory[index];
        const size_t length = strlen(entry->name);

        if (entry->layout != FT_TT_LAYOUT_MATMUL_NMAJOR)
            continue;
        *total += entry->byte_count;
        if (suffix_length && length >= suffix_length &&
            strcmp(entry->name + length - suffix_length, suffix) == 0)
            *suffix_total += entry->byte_count;
    }
    return 0;
}

int pn_tt_device_load_artifact(pn_tt_device *device, const char *path)
{
    return pn_tt_device_open_artifact(device, path) == 0 ? pn_tt_device_place_artifact(device)
                                                         : -1;
}

int pn_tt_device_place_artifact(pn_tt_device *device)
{
    const char *census = getenv("PN_TT_Q36_CENSUS");
    uint32_t expert_stacks = 0;
    uint32_t fp32_routers = 0;
    uint32_t deferred_moe = 0;
    uint32_t index;
    double started;

    if (!device || !device->artifact_open || device->resident)
        return -1;
    device->resident = calloc(pn_tt_resident_slots_needed(device), sizeof(*device->resident));
    if (!device->resident)
        return -1;

    started = pn_tt_now_seconds();
    for (index = 0; index < device->artifact.header.tensor_count; index++) {
        const ft_tt_artifact_entry *entry = &device->artifact.directory[index];
        uint32_t tile_bytes;

        /* Only matmul weights. A tensor in SOURCE layout is either a 1-D norm the engine
         * already holds as fp32, or the embedding table, which no on-card op can read in
         * this format (pn_tt_device.h). Uploading either would be dead weight in DRAM. */
        if (entry->layout != FT_TT_LAYOUT_MATMUL_NMAJOR)
            continue;
        /* A routed-expert stack or an fp32 router is not a matmul weight of the dense
         * families: it needs the MoE block's placement and programs. Reserved on the census
         * arm; refused otherwise. */
        if (pn_tt_entry_is_moe_block(entry)) {
            /* The MoE block places these itself (pn_tt_moe.c), after the dense load. */
            if (device->moe_defer_weights) {
                deferred_moe++;
                continue;
            }
            if (!census || atoi(census) != 1) {
                fprintf(stderr, "pn_tt: %s is a MoE-block weight (expert stack or fp32 router) and no "
                                "MoE block is built (PN_TT_Q36_CENSUS=1 reserves its DRAM for the "
                                "census)\n", entry->name);
                return -1;
            }
            if (pn_tt_reserve_expert_stack(device, entry) != 0)
                return -1;
            if (ft_tt_fttw_expert_count(entry) > 1u)
                expert_stacks++;
            else
                fp32_routers++;
            continue;
        }
        tile_bytes = ft_tt_fttw_tile_bytes(entry->format);
        if (tile_bytes == 0 || pn_tt_program_for(device, tile_bytes) == NULL) {
            fprintf(stderr, "pn_tt: no matmul program for %s (tile %u bytes)\n", entry->name,
                    tile_bytes);
            return -1;
        }
        if (pn_tt_place_resident_entry(device, entry, tile_bytes) != 0)
            return -1;
    }
    fprintf(stderr, "pn_tt: %u matmul weights resident, %.1f MiB, %.1f s\n",
            device->resident_count, (double)device->resident_bytes / (1024.0 * 1024.0),
            pn_tt_now_seconds() - started);
    if (deferred_moe)
        fprintf(stderr, "pn_tt: %u MoE-block weights (expert stacks, fp32 routers) left to the "
                        "MoE block's own placement\n", deferred_moe);
    if (expert_stacks || fp32_routers)
        fprintf(stderr, "pn_tt: CENSUS — %u routed-expert stacks and %u fp32 routers RESERVED in "
                        "DRAM, not uploaded (counted as weights above)\n", expert_stacks, fp32_routers);
    pn_tt_matmul_group_report(device);
    if (device->resident_count == 0)
        return -1;
    pn_tt_arm_residency(device);
    /* IF YOU DID NOT ASK FOR ONE-SHOT, YOU DO NOT GET ONE-SHOT. Every reason the residency
     * did not arm — a missing program, a mismatched artifact tile size, a launch that failed
     * — costs 1.085x the seconds per token and produces a token indistinguishable from a
     * healthy one. `pn_tt_arm_residency()` has already printed which reason it was. */
    if (device->resident_requested && !device->resident_running) {
        fprintf(stderr, "pn_tt: REFUSED — the resident dispatch path did not come up (reason "
                        "above). Re-run with %s=oneshot if the slow path is what you want.\n",
                PN_TT_DISPATCH_ENV);
        return -1;
    }
    return 0;
}

int pn_tt_device_find_weight(const pn_tt_device *device, const char *name,
                             pn_tt_weight *out_weight)
{
    uint32_t index;

    if (!device || !name || !out_weight)
        return -1;
    for (index = 0; index < device->resident_count; index++) {
        if (device->resident[index].part < 0 && strcmp(device->resident[index].name, name) == 0) {
            *out_weight = device->resident[index].weight;
            return 0;
        }
    }
    return -1;
}

int pn_tt_device_find_weight_part(const pn_tt_device *device, const char *name, uint32_t part,
                                  pn_tt_weight *out_weight)
{
    if (!device || !name || !out_weight)
        return -1;
    for (uint32_t index = 0; index < device->resident_count; index++) {
        const pn_tt_resident *slot = &device->resident[index];

        if (slot->part == (int32_t)part && strcmp(slot->name, name) == 0) {
            *out_weight = slot->weight;
            return 0;
        }
    }
    return -1;
}

int pn_tt_device_split_weights_at_load(pn_tt_device *device, const char *suffix,
                                       uint32_t fields, uint32_t field_tiles,
                                       uint32_t part_tiles)
{
    if (!device || device->resident || device->weight_split_set)
        return -1;
    if (pn_tt_weight_split_rule_init(&device->weight_split, suffix, fields, field_tiles,
                                     part_tiles) != 0) {
        fprintf(stderr, "pn_tt: cannot split *%s into %u-tile parts of %u fields x %u tiles\n",
                suffix ? suffix : "(null)", part_tiles, fields, field_tiles);
        return -1;
    }
    device->weight_split_set = true;
    return 0;
}

int pn_tt_device_stage_weight_columns(pn_tt_device *device, const char *source_name,
                                      const pn_tt_column_run *runs, uint32_t run_count,
                                      pn_tt_weight *out_weight)
{
    const ft_tt_artifact_entry *entry = NULL;
    uint32_t destination_columns;
    uint32_t index;

    if (!device || !source_name || !runs || !out_weight || !device->artifact_open)
        return -1;
    destination_columns = pn_tt_column_runs_total(runs, run_count);
    if (destination_columns == 0u)
        return -1;
    for (index = 0; index < device->artifact.header.tensor_count; index++) {
        if (strcmp(device->artifact.directory[index].name, source_name) == 0) {
            entry = &device->artifact.directory[index];
            break;
        }
    }
    if (!entry || entry->layout != FT_TT_LAYOUT_MATMUL_NMAJOR) {
        fprintf(stderr, "pn_tt: %s is not an N-major matmul weight in this artifact\n",
                source_name);
        return -1;
    }

    memset(out_weight, 0, sizeof(*out_weight));
    out_weight->tile_rows = entry->tile_rows;
    out_weight->tile_cols = destination_columns;
    out_weight->tile_bytes = ft_tt_fttw_tile_bytes(entry->format);
    out_weight->in_features = entry->tile_rows * PN_TT_TILE_DIM;
    out_weight->out_features = destination_columns * PN_TT_TILE_DIM;
    if (out_weight->tile_bytes == 0u || pn_tt_program_for(device, out_weight->tile_bytes) ==
                                            NULL) {
        fprintf(stderr, "pn_tt: no matmul program for %s (tile %u bytes)\n", source_name,
                out_weight->tile_bytes);
        return -1;
    }
    /* A SECOND COPY IN DRAM when the source is resident too: the bump allocator cannot free
     * one tensor out of the middle of the rotation. A tensor that is only ever read as its
     * parts is split AT LOAD instead (pn_tt_device_split_weights_at_load) — the joint
     * `attn_q` was 0.996 GiB of such copies on Qwen3.8-27B. */
    return pn_tt_matmul_group_place_weight_columns(device, entry, runs, run_count,
                                                   &device->weight_bank_cursor, out_weight);
}

int pn_tt_device_view_weight_columns(const pn_tt_device *device, const pn_tt_weight *source,
                                     const pn_tt_column_run *runs, uint32_t run_count,
                                     pn_tt_weight *out_weight)
{
    if (!device || device->group.cooperative_requested || device->group.cooperative)
        return -1;
    return pn_tt_weight_column_view(source, runs, run_count,
                                    device->group.member_count ? device->group.member_count : 1u,
                                    out_weight);
}

/* ================================================================
 * STEP 4 S2: the second artifact (the MTP draft head)
 * ================================================================ */

int pn_tt_device_extra_open(pn_tt_device *device, const char *path)
{
    if (!device || !path || device->extra_open)
        return -1;
    if (ft_tt_fttw_open(path, &device->extra_artifact) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: cannot open the extra artifact %s\n", path);
        return -1;
    }
    device->extra_open = true;
    return 0;
}

/* The placement paths read `device->artifact`; for the extra one they run with the two
 * swapped, and swapped back whatever happened. */
static void pn_tt_extra_swap(pn_tt_device *device)
{
    ft_tt_fttw t = device->artifact;

    device->artifact = device->extra_artifact;
    device->extra_artifact = t;
}

int pn_tt_device_reserve_extra_expert_stacks(pn_tt_device *device)
{
    uint32_t reserved = 0;

    if (!device || !device->extra_open)
        return -1;
    for (uint32_t i = 0; i < device->extra_artifact.header.tensor_count; i++) {
        const ft_tt_artifact_entry *entry = &device->extra_artifact.directory[i];

        if (entry->layout != FT_TT_LAYOUT_MATMUL_NMAJOR || !pn_tt_entry_is_moe_block(entry))
            continue;
        if (pn_tt_reserve_expert_stack(device, entry) != 0)
            return -1;
        reserved++;
    }
    fprintf(stderr, "pn_tt: CENSUS — %u MoE-block weights (expert stacks, fp32 router) of the extra "
                    "artifact RESERVED\n", reserved);
    return (int)reserved;
}

int pn_tt_dram_alloc_any(pn_tt_device *device, size_t bytes, ft_tt_buffer *out)
{
    uint32_t first, tried[FT_TT_MAX_DRAM_BANKS] = { 0 };

    if (!device || !out || device->dram_bank_count == 0u)
        return -1;
    first = device->strip_bank_cursor++ % device->dram_bank_count;
    if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, first, bytes, out) == FT_TT_OK)
        return 0;
    tried[first] = 1u;
    for (uint32_t attempt = 1u; attempt < device->dram_bank_count; attempt++) {
        uint32_t best = device->dram_bank_count;
        size_t best_free = 0u;

        for (uint32_t b = 0; b < device->dram_bank_count; b++) {
            const size_t free_bytes =
                ft_tt_allocator_free_bytes(&device->allocator, FT_TT_MEM_SPACE_DRAM, b);

            if (!tried[b] && free_bytes >= bytes && free_bytes > best_free) {
                best = b;
                best_free = free_bytes;
            }
        }
        if (best == device->dram_bank_count)
            break;
        tried[best] = 1u;
        if (ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, best, bytes, out) ==
            FT_TT_OK) {
            fprintf(stderr, "pn_tt: DRAM bank %u is full for %zu bytes — placed in bank %u\n",
                    first, bytes, best);
            return 0;
        }
    }
    return -1;
}

uint32_t pn_tt_device_dram_free_banks(const pn_tt_device *device, uint64_t *free_bytes,
                                      uint32_t max_banks)
{
    if (!device || !free_bytes || device->dram_bank_count == 0u ||
        device->dram_bank_count > max_banks)
        return 0u;
    for (uint32_t b = 0; b < device->dram_bank_count; b++)
        free_bytes[b] = ft_tt_allocator_free_bytes(&device->allocator, FT_TT_MEM_SPACE_DRAM, b);
    return device->dram_bank_count;
}

uint32_t pn_tt_device_strip_cursor(const pn_tt_device *device)
{
    return device ? device->strip_bank_cursor : 0u;
}

int pn_tt_dram_alloc_emptiest(pn_tt_device *device, size_t bytes, ft_tt_buffer *out)
{
    uint64_t free_bytes[FT_TT_MAX_DRAM_BANKS];
    uint32_t bank;

    if (!device || !out || device->dram_bank_count == 0u ||
        device->dram_bank_count > FT_TT_MAX_DRAM_BANKS)
        return -1;
    for (uint32_t b = 0; b < device->dram_bank_count; b++)
        free_bytes[b] = ft_tt_allocator_free_bytes(&device->allocator, FT_TT_MEM_SPACE_DRAM, b);
    bank = pn_tt_strip_placement_emptiest_bank(free_bytes, device->dram_bank_count, bytes);
    if (bank >= device->dram_bank_count)
        return -1;
    return ft_tt_buffer_alloc(&device->allocator, FT_TT_MEM_SPACE_DRAM, bank, bytes, out) ==
                   FT_TT_OK
               ? 0
               : -1;
}

uint64_t pn_tt_device_extra_tensor_bytes(const pn_tt_device *device, const char *name)
{
    const ft_tt_artifact_entry *entry;

    if (!device || !name || !device->extra_open)
        return 0u;
    entry = ft_tt_fttw_find(&device->extra_artifact, name);
    return entry ? entry->byte_count : 0u;
}

int pn_tt_device_artifact_tensor_tiles(const pn_tt_device *device, const char *name,
                                       uint32_t *tile_rows, uint32_t *tile_cols,
                                       uint32_t *tile_bytes)
{
    const ft_tt_artifact_entry *entry;

    if (!device || !name || !tile_rows || !tile_cols || !tile_bytes || !device->artifact_open)
        return -1;
    entry = ft_tt_fttw_find(&device->artifact, name);
    if (!entry || entry->layout != FT_TT_LAYOUT_MATMUL_NMAJOR)
        return -1;
    *tile_rows = entry->tile_rows;
    *tile_cols = entry->tile_cols;
    *tile_bytes = ft_tt_fttw_tile_bytes(entry->format);
    return 0;
}

int pn_tt_moe_place_extra_stack(pn_tt_device *device, const char *name, pn_tt_moe_stack *out)
{
    int rc;

    if (!device || !device->extra_open)
        return -1;
    pn_tt_extra_swap(device);
    rc = pn_tt_moe_place_stack(device, name, out);
    pn_tt_extra_swap(device);
    return rc;
}

int pn_tt_moe_place_extra_router(pn_tt_device *device, const char *name, const float *shexp_gate,
                                 pn_tt_moe_router *out)
{
    int rc;

    if (!device || !device->extra_open)
        return -1;
    pn_tt_extra_swap(device);
    rc = pn_tt_moe_place_router(device, name, shexp_gate, out);
    pn_tt_extra_swap(device);
    return rc;
}

int pn_tt_device_place_extra_weight(pn_tt_device *device, const char *name,
                                    pn_tt_weight *out_weight)
{
    const ft_tt_artifact_entry *entry;
    uint32_t tile_bytes;
    int rc;

    if (!device || !name || !out_weight || !device->extra_open)
        return -1;
    entry = ft_tt_fttw_find(&device->extra_artifact, name);
    if (!entry || entry->layout != FT_TT_LAYOUT_MATMUL_NMAJOR) {
        fprintf(stderr, "pn_tt: the extra artifact has no N-major %s\n", name);
        return -1;
    }
    tile_bytes = ft_tt_fttw_tile_bytes(entry->format);
    if (tile_bytes == 0u || pn_tt_program_for(device, tile_bytes) == NULL) {
        fprintf(stderr, "pn_tt: no matmul program for %s (tile %u bytes)\n", name, tile_bytes);
        return -1;
    }
    memset(out_weight, 0, sizeof(*out_weight));
    out_weight->tile_rows = entry->tile_rows;
    out_weight->tile_cols = entry->tile_cols;
    out_weight->tile_bytes = tile_bytes;
    out_weight->in_features = entry->tile_rows * PN_TT_TILE_DIM;
    out_weight->out_features = entry->tile_cols * PN_TT_TILE_DIM;
    pn_tt_extra_swap(device);
    rc = pn_tt_matmul_group_place_weight(device, entry, &device->weight_bank_cursor, out_weight);
    pn_tt_extra_swap(device);
    return rc;
}

int pn_tt_device_stage_extra_columns(pn_tt_device *device, const char *source_name,
                                     const pn_tt_column_run *runs, uint32_t run_count,
                                     pn_tt_weight *out_weight)
{
    int rc;

    if (!device || !device->extra_open)
        return -1;
    pn_tt_extra_swap(device);
    rc = pn_tt_device_stage_weight_columns(device, source_name, runs, run_count, out_weight);
    pn_tt_extra_swap(device);
    return rc;
}

const float *pn_tt_device_extra_vector(const pn_tt_device *device, const char *name, uint32_t n)
{
    const ft_tt_artifact_entry *entry;

    if (!device || !name || !device->extra_open)
        return NULL;
    entry = ft_tt_fttw_find(&device->extra_artifact, name);
    if (!entry || entry->format != FT_TT_FORMAT_FP32 || entry->byte_count != (uint64_t)n * 4u ||
        entry->data_offset + entry->byte_count > (uint64_t)device->extra_artifact.bytes) {
        fprintf(stderr, "pn_tt: the extra artifact has no %u-float fp32 %s\n", n, name);
        return NULL;
    }
    return (const float *)(const void *)(device->extra_artifact.base + entry->data_offset);
}

uint32_t pn_tt_device_weight_count(const pn_tt_device *device)
{
    return device ? device->resident_count : 0u;
}

uint64_t pn_tt_device_weight_bytes(const pn_tt_device *device)
{
    return device ? device->resident_bytes : 0u;
}

void pn_tt_device_dram_usage(const pn_tt_device *device, uint64_t *used_out,
                             uint64_t *total_out)
{
    uint64_t used = 0u, total = 0u;

    if (device) {
        for (uint32_t bank = 0; bank < device->dram_bank_count; bank++) {
            uint64_t bank_used =
                ft_tt_allocator_used_bytes(&device->allocator, FT_TT_MEM_SPACE_DRAM, bank);

            used += bank_used;
            total += bank_used +
                ft_tt_allocator_free_bytes(&device->allocator, FT_TT_MEM_SPACE_DRAM, bank);
        }
    }
    if (used_out)
        *used_out = used;
    if (total_out)
        *total_out = total;
}

void pn_tt_device_report_dram(const pn_tt_device *device, const char *tag)
{
    const double gib = 1024.0 * 1024.0 * 1024.0;
    uint64_t used = 0u, total = 0u, fullest_used = 0u, fullest_free = 0u;
    uint32_t fullest = 0u;

    pn_tt_device_dram_usage(device, &used, &total);
    /* The banks fill independently (a bump arena each), so the card is full when its FULLEST
     * bank is, not when the sum is: the qwen36 fit question is read off this column. */
    for (uint32_t bank = 0; device && bank < device->dram_bank_count; bank++) {
        uint64_t bank_used =
            ft_tt_allocator_used_bytes(&device->allocator, FT_TT_MEM_SPACE_DRAM, bank);

        if (bank_used >= fullest_used) {
            fullest_used = bank_used;
            fullest_free = ft_tt_allocator_free_bytes(&device->allocator, FT_TT_MEM_SPACE_DRAM,
                                                      bank);
            fullest = bank;
        }
    }
    fprintf(stderr, "pn_tt: DRAM %s — %.3f GiB used of %.3f GiB over %u bank(s), %.3f GiB "
                    "free; weights %.3f GiB, non-weight %.3f GiB; fullest bank %u %.3f GiB "
                    "used %.3f GiB free\n", tag ? tag : "usage", (double)used / gib,
            (double)total / gib, device ? device->dram_bank_count : 0u,
            (double)(total - used) / gib,
            (double)(device ? device->resident_bytes : 0u) / gib,
            (double)(used - (device ? device->resident_bytes : 0u)) / gib, fullest,
            (double)fullest_used / gib, (double)fullest_free / gib);
}

bool pn_tt_device_dram_report_wanted(void)
{
    const char *env = getenv("PN_TT_DRAM_REPORT");

    return env && atoi(env) != 0;
}

void pn_tt_device_report_dram_stage(const pn_tt_device *device, const char *tag)
{
    if (pn_tt_device_dram_report_wanted())
        pn_tt_device_report_dram(device, tag);
}

int pn_tt_device_rope_is_split_half(const pn_tt_device *device)
{
    if (!device || !device->artifact.base)
        return 0;
    return (device->artifact.header.flags & FT_TT_ARTIFACT_FLAG_ROPE_SPLIT_HALF) != 0u;
}

int pn_tt_device_rope_pairs_reversed(const pn_tt_device *device)
{
    if (!device || !device->artifact.base)
        return 0;
    return (device->artifact.header.flags & FT_TT_ARTIFACT_FLAG_ROPE_PAIRS_REVERSED) != 0u;
}

/* ================================================================
 * The one primitive
 * ================================================================ */

/**
 * Append logical row 0 of one projection's output to the dump file.
 *
 * Row 0 is the whole answer at Mt = 1: the other 31 rows of every output tile are the tile
 * quantum multiplied against the activation's staged zeros, so they carry no information
 * about the split and dumping them would make the file 32x larger without making it
 * stricter. The bytes are read back off the card, not from any host mirror — a comparison
 * against something the host computed would be measuring the host.
 */
static void pn_tt_dump_result_row(pn_tt_device *device, const pn_tt_weight *weight,
                                  const pn_tt_activation *output)
{
    const ft_tt_noc_access *bulk = &device->runtime.bulk_dram;
    size_t bytes = (size_t)weight->out_features * sizeof(*device->row_out);

    if (weight->out_features > device->max_row_features)
        return;
    if (ft_tt_tile_row0_read(bulk, (uint16_t)device->dram[output->bank].x,
                             (uint16_t)device->dram[output->bank].y, output->address,
                             weight->tile_cols, device->row_out) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: matvec dump could not read the result row\n");
        return;
    }
    if (fwrite(device->row_out, 1, bytes, device->matvec_dump) != bytes) {
        fprintf(stderr, "pn_tt: matvec dump write failed\n");
        return;
    }
    device->matvec_dump_bytes += bytes;
    device->matvec_dump_rows++;
}

/**
 * THE DISPATCH, with both operands already on the card.
 *
 * Both arms compute the same thing with the same three kernels on the same weights; what
 * differs is whether the five kernel images cross the link again first. The resident arm is
 * taken whenever the residency is live AND the reader would accept the shape — a descriptor
 * the reader refuses stops it mid-residency, so the check is a precondition here, never a
 * discovery there.
 */
/**
 * THE MATMUL GROUP'S IDLE RETIRE, retire AND re-arm in one step, because this file owns the
 * arm. Every member triad idles out on its own counters ~3 s apart (fastest ~8 s); a
 * projection sent into that window wedges a member. So a group idle past the shared limit
 * with nothing outstanding is ASKED to stop — whole, since every processor still alive is in
 * its idle poll — and re-armed fresh before the projection is written.
 */
static void pn_tt_resident_idle_refresh(pn_tt_device *device)
{
    double idle = 0.0;

    if (!device->resident_running || !pn_tt_matmul_group_idle_expired(device, &idle))
        return;
    fprintf(stderr, "pn_tt: matmul group idle %.1f s (limit %ld ms) — retiring all %u "
                    "member(s) by ASKING and re-arming before a projection can reach a "
                    "half-exited triad\n",
            idle, pn_tt_residency_idle_retire_limit_ms(), device->group.member_count);
    {
        const double started = pn_tt_now_seconds();

        pn_tt_matmul_group_retire(device);
        device->resident_restarts++;
        device->group.idle_retires++;
        pn_tt_arm_residency(device);
        pn_tt_residency_lifecycle_add(pn_tt_now_seconds() - started);
    }
}

int pn_tt_matvec_dispatch(pn_tt_device *device, const pn_tt_weight *weight,
                          const pn_tt_activation *input, const pn_tt_activation *output)
{
    ft_tt_op_matmul_request request;
    const ft_tt_kernel_program *program;
    bool dispatched;
    double phase;
    int rc;

    program = pn_tt_program_for(device, weight->tile_bytes);
    if (!program)
        return -1;
    phase = pn_tt_now_seconds();

    dispatched = false;
    pn_tt_resident_idle_refresh(device);
    if (device->resident_running) {
        dispatched = pn_tt_matmul_group_dispatch(device, weight, input, output) == FT_TT_OK;
        if (!dispatched) {
            fprintf(stderr, "pn_tt: resident matmul Mt=1 Kt=%u Nt=%u refused — retiring the "
                            "residency by ASKING and re-arming\n",
                    weight->tile_rows, weight->tile_cols);
            /* A REFUSED DESCRIPTOR MUST NOT COST THE CORE. The reader leaves its loop and
             * `tail` never advances, so the other four processors are waiting on work that
             * will never arrive — a state only the ASK can leave, never a reset. Retire all
             * five, then arm a FRESH residency: ft_tt_queue_init() re-poisons the control
             * block and issues a new nonce, so the refused descriptor is not still in the
             * ring for the new kernel to re-attempt and refuse again. Every member is
             * retired, not only the one that refused — a partly-armed group would leave one
             * slice of the next projection unwritten. */
            pn_tt_matmul_group_retire(device);
            device->resident_restarts++;
            pn_tt_arm_residency(device);
            /* Retry ONCE on the re-armed residency. Falling through to one-shot rather than
             * failing is the difference between a recoverable refusal and a lost token. */
            if (device->resident_running)
                dispatched = pn_tt_matmul_group_dispatch(device, weight, input, output) ==
                             FT_TT_OK;
            if (!dispatched) {
                fprintf(stderr, "pn_tt: re-armed residency refused it too\n");
                pn_tt_matmul_group_retire(device);
            }
        }
    }
    /* THE ONE-SHOT ESCAPE DOES NOT EXIST FOR A SHARDED WEIGHT, and that is arithmetic
     * rather than a policy: its panels are in several banks, so there is no single grid a
     * one-shot launch could read. A group wide enough to shard has traded the fallback for
     * the bandwidth, and it says so here instead of computing from one bank and returning a
     * plausible wrong answer. */
    if (!dispatched && !pn_tt_matmul_group_can_fall_back(device, weight)) {
        fprintf(stderr, "pn_tt: a weight Kt=%u Nt=%u sharded over %u banks has no one-shot "
                        "grid to fall back to\n",
                weight->tile_rows, weight->tile_cols, weight->shard_count);
        return -1;
    }
    if (!dispatched) {
        memset(&request, 0, sizeof(request));
        request.tile_rows = 1u;
        request.tile_inner = weight->tile_rows;
        request.tile_cols = weight->tile_cols;
        request.prefetch_tiles = pn_tt_env_u32(PN_TT_PREFETCH_TILES_ENV,
                                              device->prefetch_tiles_preferred
                                                  ? device->prefetch_tiles_preferred
                                                  : PN_TT_PREFETCH_TILES_DEFAULT);
        /* A STATEMENT about DRAM, not a request: the artifact's MATMUL_NMAJOR layout IS the
         * N-major tile grid, and a wrong value here is a wrong answer with no error. */
        request.weight_n_major = 1u;
        request.activation.x = (uint16_t)device->dram[input->bank].x;
        request.activation.y = (uint16_t)device->dram[input->bank].y;
        request.activation.address = input->address;
        request.activation.tile_bytes = PN_TT_TILE_BF16;
        request.weight.x = (uint16_t)device->dram[weight->bank].x;
        request.weight.y = (uint16_t)device->dram[weight->bank].y;
        request.weight.address = weight->address;
        request.weight.tile_bytes = weight->tile_bytes;
        request.output.x = (uint16_t)device->dram[output->bank].x;
        request.output.y = (uint16_t)device->dram[output->bank].y;
        request.output.address = output->address;
        request.output.tile_bytes = PN_TT_TILE_BF16;

        rc = ft_tt_op_matmul_run(&device->runtime.noc, device->core_x, device->core_y, program,
                                 &request, PN_TT_OP_TIMEOUT_MS);
        if (rc != FT_TT_OK) {
            fprintf(stderr, "pn_tt: matmul Mt=1 Kt=%u Nt=%u failed: %s\n", weight->tile_rows,
                    weight->tile_cols, ft_tt_error_string(rc));
            return -1;
        }
    }
    device->dispatch_seconds += pn_tt_now_seconds() - phase;
    device->matvec_count++;
    /* OUTSIDE the timed phase: this is an instrument, and a phase split that included it
     * could not be compared with a run that did not have it armed. */
    if (device->matvec_dump)
        pn_tt_dump_result_row(device, weight, output);
    return 0;
}

/** The strips a projection is asked to write into have to hold it. Shared by the waiting
 *  and the issuing forms, because a wrong shape is a card refusal either way. */
static bool pn_tt_matvec_strips_fit(const pn_tt_weight *weight, const pn_tt_activation *input,
                                    const pn_tt_activation *output)
{
    if (input->tile_count < weight->tile_rows || output->tile_count < weight->tile_cols) {
        fprintf(stderr, "pn_tt: matvec strip too small (Kt=%u into %u tiles, Nt=%u into %u)\n",
                weight->tile_rows, input->tile_count, weight->tile_cols, output->tile_count);
        return false;
    }
    /* THE OTHER AXIS, AND IT HAS NO CARD-SIDE CHECK AT ALL. At `Mt > 1` the writer places
     * output tile `(m, n)` at `C_BASE + (m*a2 + n) * 2048` — DRAM, no bounds — so an output
     * strip spanning fewer BANDS than the input is a write into whatever the allocator put
     * after it, with every kernel retiring clean. Refused here, where both strips are in
     * scope, rather than left to the descriptor writer's clamp: the clamp keeps the answer
     * safe and SILENTLY narrows the pass, which is a lever that measures itself away. */
    if (input->active_rows > 1u && output->active_rows < input->active_rows) {
        fprintf(stderr, "pn_tt: matvec spans %u tile-row bands of A into an output that spans "
                        "%u — the writer would place a row past the end of the destination "
                        "strip\n", input->active_rows, output->active_rows);
        return false;
    }
    return true;
}

int pn_tt_matvec_resident(pn_tt_device *device, const pn_tt_weight *weight,
                          const pn_tt_activation *input, const pn_tt_activation *output)
{
    double started;
    int rc;

    if (!device || !weight || !input || !output)
        return -1;
    if (!pn_tt_matvec_strips_fit(weight, input, output))
        return -1;
    /* A WAITING projection is one of the operations that flushes first — see
     * pn_tt_matvec_enqueue()'s contract. Chained work may write the very strip this one
     * reads, and the members run concurrently. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    started = pn_tt_now_seconds();
    rc = pn_tt_matvec_dispatch(device, weight, input, output);
    device->matvec_seconds += pn_tt_now_seconds() - started;
    return rc;
}

uint32_t pn_tt_matvec_chain_limit(void)
{
    const char *value = getenv("PN_TT_MATVEC_CHAIN");
    long parsed = (value && *value) ? strtol(value, NULL, 10) : 0;

    if (parsed <= 0)
        return PN_TT_MATVEC_MAX_CHAIN;
    if (parsed > (long)PN_TT_MATVEC_CHAIN_CEILING)
        return PN_TT_MATVEC_CHAIN_CEILING;
    return (uint32_t)parsed;
}

int pn_tt_matvec_enqueue(pn_tt_device *device, const pn_tt_weight *weight,
                         const pn_tt_activation *input, const pn_tt_activation *output)
{
    double started;
    int rc;

    if (!device || !weight || !input || !output)
        return -1;
    if (!pn_tt_matvec_strips_fit(weight, input, output))
        return -1;
    /* Without a residency there is nothing to chain onto: the one-shot escape launches five
     * kernel images and waits, so the waiting form IS the only form it has. */
    if (device->pending_count == 0u)
        pn_tt_resident_idle_refresh(device);
    if (!device->resident_running)
        return pn_tt_matvec_resident(device, weight, input, output);
    if (device->pending_count >= pn_tt_matvec_chain_limit() &&
        pn_tt_matvec_flush(device) != 0)
        return -1;

    started = pn_tt_now_seconds();
    rc = pn_tt_matmul_group_enqueue(device, weight, input, output);
    /* ONE clock read for both accumulators. Two would charge the two phases slightly
     * different amounts of the same work, and these are the numbers the sweep is read from. */
    started = pn_tt_now_seconds() - started;
    device->matvec_seconds += started;
    device->dispatch_seconds += started;
    /* THE HOST'S OWN HALF OF THE DISPATCH PHASE, kept apart from the wait. What runs above
     * is parameter words and a ring entry per member over MMIO and nothing else — no poll,
     * no completion read, no card work. So this counter IS "the host in the dispatch loop",
     * to the microsecond, and `dispatch - issue - flush_wait` cannot hide anywhere. */
    device->dispatch_issue_seconds += started;
    device->dispatch_issues++;
    if (rc != FT_TT_OK) {
        /* Refused BEFORE it was issued, so nothing is outstanding for it and the waiting
         * form — which owns the retire, the re-arm and the one-shot escape — can simply be
         * asked for the same projection. */
        return pn_tt_matvec_resident(device, weight, input, output);
    }
    device->pending[device->pending_count].weight = weight;
    device->pending[device->pending_count].input = *input;
    device->pending[device->pending_count].output = *output;
    device->pending[device->pending_count].swiglu_up = NULL;
    device->pending[device->pending_count].resadd = false;
    device->pending_count++;
    device->matvec_count++;
    return 0;
}

int pn_tt_matvec_enqueue_swiglu(pn_tt_device *device, const pn_tt_weight *gate,
                                const pn_tt_weight *up, const pn_tt_activation *input,
                                const pn_tt_activation *output)
{
    double started;
    int rc;

    if (!device || !gate || !up || !input || !output)
        return -1;
    if (!pn_tt_fuse_swiglu_mode() || !pn_tt_matvec_strips_fit(gate, input, output) ||
        !pn_tt_matvec_strips_fit(up, input, output))
        return 1;
    if (device->pending_count == 0u)
        pn_tt_resident_idle_refresh(device);
    if (!device->resident_running)
        return 1;
    if (device->pending_count >= pn_tt_matvec_chain_limit() && pn_tt_matvec_flush(device) != 0)
        return -1;
    started = pn_tt_now_seconds();
    rc = pn_tt_matmul_group_enqueue_swiglu(device, gate, up, input, output);
    started = pn_tt_now_seconds() - started;
    device->matvec_seconds += started;
    device->dispatch_seconds += started;
    device->dispatch_issue_seconds += started;
    device->dispatch_issues++;
    /* REFUSED BEFORE ANYTHING WAS WRITTEN: the caller runs gate, up and silu_mul unfused. */
    if (rc == FT_TT_EINVAL)
        return 1;
    if (rc != FT_TT_OK)
        return -1;
    device->pending[device->pending_count].weight = gate;
    device->pending[device->pending_count].input = *input;
    device->pending[device->pending_count].output = *output;
    device->pending[device->pending_count].swiglu_up = up;
    device->pending[device->pending_count].resadd = false;
    device->pending_count++;
    device->matvec_count += 2u; /* the gate and up projections it replaces */
    return 0;
}

/** Replay one FUSED projection on a re-armed residency, issued and awaited (the arm re-planted
 *  the members' replica table). A fused shape the re-armed group refuses is an error: the
 *  unfused replay would need the gate/up strips, which this level does not know. */
static int pn_tt_matvec_replay_swiglu(pn_tt_device *device, const pn_tt_weight *gate,
                                      const pn_tt_weight *up, const pn_tt_activation *input,
                                      const pn_tt_activation *output)
{
    if (pn_tt_matmul_group_flush(device) != FT_TT_OK ||
        pn_tt_matmul_group_enqueue_swiglu(device, gate, up, input, output) != FT_TT_OK ||
        pn_tt_matmul_group_flush(device) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: a fused SwiGLU projection could not be replayed after the "
                        "re-arm\n");
        return -1;
    }
    device->matvec_count += 2u;
    return 0;
}

int pn_tt_matvec_resadd(pn_tt_device *device, const pn_tt_weight *weight,
                        const pn_tt_activation *input, const pn_tt_activation *residual,
                        const pn_tt_activation *output)
{
    double started;
    int rc;

    if (!device || !weight || !input || !residual || !output)
        return -1;
    if (!pn_tt_fuse_resadd_mode() || !pn_tt_matvec_strips_fit(weight, input, output))
        return 1;
    /* THE WAITING FORM FLUSHES FIRST, as pn_tt_matvec_resident(): chained work may write the
     * input or the residual this projection reads. */
    if (pn_tt_matvec_flush(device) != 0)
        return -1;
    pn_tt_resident_idle_refresh(device);
    if (!device->resident_running)
        return 1;
    started = pn_tt_now_seconds();
    rc = pn_tt_matmul_group_enqueue_resadd(device, weight, input, residual, output);
    started = pn_tt_now_seconds() - started;
    device->matvec_seconds += started;
    device->dispatch_seconds += started;
    device->dispatch_issue_seconds += started;
    device->dispatch_issues++;
    /* REFUSED BEFORE ANYTHING WAS WRITTEN: the caller runs the projection and the add. */
    if (rc == FT_TT_EINVAL)
        return 1;
    if (rc != FT_TT_OK)
        return -1;
    device->pending[device->pending_count].weight = weight;
    device->pending[device->pending_count].input = *input;
    device->pending[device->pending_count].output = *output;
    device->pending[device->pending_count].swiglu_up = NULL;
    device->pending[device->pending_count].resadd = true;
    device->pending[device->pending_count].residual = *residual;
    device->pending_count++;
    device->matvec_count++;
    return pn_tt_matvec_flush(device) == 0 ? 0 : -1;
}

/** Replay one RESADD projection on a re-armed residency, issued and awaited. A shape the
 *  re-armed group refuses is an error: the unfused replay would need `project`, which this
 *  level does not know. */
static int pn_tt_matvec_replay_resadd(pn_tt_device *device, const pn_tt_weight *weight,
                                      const pn_tt_activation *input,
                                      const pn_tt_activation *residual,
                                      const pn_tt_activation *output)
{
    if (pn_tt_matmul_group_flush(device) != FT_TT_OK ||
        pn_tt_matmul_group_enqueue_resadd(device, weight, input, residual, output) != FT_TT_OK ||
        pn_tt_matmul_group_flush(device) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: a fused residual-add projection could not be replayed after "
                        "the re-arm\n");
        return -1;
    }
    device->matvec_count++;
    return 0;
}

int pn_tt_matvec_flush(pn_tt_device *device)
{
    double started;
    uint32_t pending;
    uint32_t index;
    int rc;

    if (!device || device->pending_count == 0u)
        return 0;
    pending = device->pending_count;
    started = pn_tt_now_seconds();
    rc = pn_tt_matmul_group_flush(device);
    started = pn_tt_now_seconds() - started;
    device->matvec_seconds += started;
    device->dispatch_seconds += started;
    /* The OTHER half: `pending` descriptors' worth of card work plus ONE completion poll.
     * The poll is the only host round trip in it, so this counter over `dispatch_waits` is
     * an upper bound on what Phase 3 can take out of a wait and the rest is the card. */
    device->dispatch_wait_seconds += started;
    device->dispatch_waits++;
    device->dispatch_wait_descriptors += pending;
    /* CLEARED BEFORE THE REPLAY, because the replay goes through the waiting form and that
     * form flushes on entry — a count still standing here would recurse. */
    device->pending_count = 0u;

    /* A HOST-SIDE REFUSAL IS NOT A CARD FAULT, and must not be treated as one. Retiring and
     * re-arming a residency because THIS side handed the accessor a bad length costs a
     * recovery cycle per projection and gets nothing back — measured 2026-08-16, when a
     * group-block read whose length missed the 16-byte L1 unit re-armed the group three
     * times in a row and left a core unbootable. The recovery below is for a card that did
     * not answer; FT_TT_EINVAL means it was never asked. */
    if (rc == FT_TT_EINVAL) {
        fprintf(stderr, "pn_tt: the completion read was REFUSED before it reached the card "
                        "(%s) — this is a host-side defect and the residency is left alone\n",
                ft_tt_error_string(rc));
        return -1;
    }
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: a chained projection did not complete (%s) — retiring the "
                        "residency by ASKING, re-arming, and replaying %u projection(s)\n",
                ft_tt_error_string(rc), pending);
        pn_tt_matmul_group_retire(device);
        device->resident_restarts++;
        pn_tt_arm_residency(device);
        for (index = 0; index < pending; index++)
            device->matvec_count -= device->pending[index].swiglu_up ? 2u : 1u;
        for (index = 0; index < pending; index++) {
            if (device->pending[index].swiglu_up) {
                if (pn_tt_matvec_replay_swiglu(device, device->pending[index].weight,
                                               device->pending[index].swiglu_up,
                                               &device->pending[index].input,
                                               &device->pending[index].output) != 0)
                    return -1;
                continue;
            }
            if (device->pending[index].resadd) {
                if (pn_tt_matvec_replay_resadd(device, device->pending[index].weight,
                                               &device->pending[index].input,
                                               &device->pending[index].residual,
                                               &device->pending[index].output) != 0)
                    return -1;
                continue;
            }
            if (pn_tt_matvec_resident(device, device->pending[index].weight,
                                      &device->pending[index].input,
                                      &device->pending[index].output) != 0)
                return -1;
        }
        return 0;
    }

    /* OUTSIDE the timed phase, and only now: the dump reads each projection's OUTPUT strip
     * off the card, which is not written until the member that owns those columns is done. */
    if (device->matvec_dump) {
        /* A fused entry's output is silu(gate) * up, not the gate's projection: not dumped. */
        for (index = 0; index < pending; index++)
            if (!device->pending[index].swiglu_up && !device->pending[index].resadd)
                pn_tt_dump_result_row(device, device->pending[index].weight,
                                      &device->pending[index].output);
    }
    return 0;
}

int pn_tt_matvec(pn_tt_device *device, const pn_tt_weight *weight, const float *x, float *y)
{
    pn_tt_activation input, output;
    const ft_tt_noc_access *bulk;
    size_t in_bytes, out_bytes;
    double started, phase;
    int rc;

    if (!device || !weight || !x || !y)
        return -1;
    if (weight->in_features > device->max_in_features ||
        weight->out_features > device->max_out_features)
        return -1;

    /* TWO accessors, and the split is not cosmetic. `runtime.noc` is UC/WORD32 and carries
     * run control ONLY — the go message and the launch args, where 4-byte width is a
     * correctness requirement. `bulk` is WC/BLOCK16 over ft_tt_{read,write}_dram and
     * carries the activation and the result. Routing data through the register accessor
     * was 71.1% of a 28 s token. */
    bulk = &device->runtime.bulk_dram;
    input.bank = device->activation.logical_bank;
    input.address = (uint32_t)device->activation.address;
    input.tile_count = device->max_tile_rows;
    output.bank = device->result.logical_bank;
    output.address = (uint32_t)device->result.address;
    output.tile_count = device->max_tile_cols;
    in_bytes = (size_t)weight->tile_rows * FT_TT_TILE_ROW0_CHUNKS * FT_TT_TILE_ROW0_HALF_BYTES;
    out_bytes = (size_t)weight->tile_cols * FT_TT_TILE_ROW0_CHUNKS * FT_TT_TILE_ROW0_READ_BYTES;
    started = pn_tt_now_seconds();

    /* One token is ONE row of a 32-row tile, and the matmul computes the other 31 anyway.
     * That waste is free ON the card — it is the tile quantum (ft_tt_op_matmul.h) — but it
     * is NOT free across the link, so only row 0 crosses it in either direction. Rows 1-31
     * of the activation were staged zero once, at load (pn_tt_stage_activation_zeros). */
    phase = started;
    ft_bf16_from_float_array(x, device->row_in, weight->in_features);
    rc = ft_tt_tile_row0_write(bulk, (uint16_t)device->dram[input.bank].x,
                               (uint16_t)device->dram[input.bank].y, input.address,
                               weight->tile_rows, device->row_in);
    if (rc != FT_TT_OK)
        return -1;
    device->bytes_written += in_bytes;
    device->upload_seconds += pn_tt_now_seconds() - phase;

    if (pn_tt_matvec_dispatch(device, weight, &input, &output) != 0)
        return -1;
    phase = pn_tt_now_seconds();

    /* 2 x 64 B per output tile instead of 2048. Not 2 x 32: a 32-byte DRAM read is
     * REFUSED — Blackhole's card->host alignment is 64 B — so half of each chunk is read
     * and discarded, which is why the recoverable factor is 16.00x and not 32x. */
    rc = ft_tt_tile_row0_read(bulk, (uint16_t)device->dram[output.bank].x,
                              (uint16_t)device->dram[output.bank].y, output.address,
                              weight->tile_cols, device->row_out);
    if (rc != FT_TT_OK)
        return -1;
    device->bytes_read += out_bytes;

    ft_float_from_bf16_array(device->row_out, y, weight->out_features);
    device->readback_seconds += pn_tt_now_seconds() - phase;
    device->matvec_seconds += pn_tt_now_seconds() - started;
    return 0;
}

uint64_t pn_tt_device_matvec_count(const pn_tt_device *device)
{
    return device ? device->matvec_count : 0u;
}

double pn_tt_device_matvec_seconds(const pn_tt_device *device)
{
    return device ? device->matvec_seconds : 0.0;
}

void pn_tt_device_phase_seconds(const pn_tt_device *device, double *out_upload,
                                double *out_dispatch, double *out_readback)
{
    if (out_upload)
        *out_upload = device ? device->upload_seconds : 0.0;
    if (out_dispatch)
        *out_dispatch = device ? device->dispatch_seconds : 0.0;
    if (out_readback)
        *out_readback = device ? device->readback_seconds : 0.0;
}

void pn_tt_device_oncard_stats(const pn_tt_device *device, double *out_seconds,
                               uint64_t *out_ops, uint64_t *out_restarts)
{
    if (out_seconds)
        *out_seconds = device ? device->oncard_seconds : 0.0;
    if (out_ops)
        *out_ops = device ? device->oncard_ops : 0u;
    if (out_restarts)
        *out_restarts = device ? device->oncard_restarts : 0u;
}

void pn_tt_oncard_account(pn_tt_device *device, const char *family, const char *function,
                          uint32_t tiles, double seconds)
{
    if (!device)
        return;
    device->oncard_seconds += seconds;
    device->oncard_ops++;
    /* A refusal here is NOT propagated to the caller: a full census table or an unexpected
     * width is a defect in the INSTRUMENT, and failing a token because the instrument ran out
     * of rows would be worse than the missing row. It is counted instead, and
     * `pn_tt_device_census()`'s reader prints `dropped` and `refused` beside every figure. */
    (void)pn_tt_descriptor_census_record(&device->census, family, function, tiles, seconds);
}

const pn_tt_descriptor_census *pn_tt_device_census(const pn_tt_device *device)
{
    return device ? &device->census : NULL;
}

/** Every residency this device holds, in ONE place — because two spellings of "the set of
 *  residencies" is how the strip bridge stayed out of the table for a whole round while its
 *  cost was being estimated by subtraction. The eltwise SPLIT MEMBERS are here and are not in
 *  the per-family report: that report keys on a name, and a member shares member 0's. */
#define PN_TT_RESIDENCY_TABLE_MAX                                                             \
    (9u + (PN_TT_REDUCE_MAX_MEMBERS - 2u) + (PN_TT_TOPK_MAX_MEMBERS - 1u) + PN_TT_RENDER_DIRECTIONS * PN_TT_RENDER_SPLIT_MAX_MEMBERS +                          \
     PN_TT_HEAD_NORM_MAX_WIDTHS * PN_TT_HEAD_NORM_MAX_MEMBERS +                              \
     PN_TT_ELTWISE_SPLIT_MAX_MEMBERS + PN_TT_ATTN_MAX_MEMBERS + PN_TT_ATTN_SPLITK_MAX_EXTRA +  \
     PN_TT_KV_APPEND_MAX_MEMBERS)

static unsigned residency_table(const pn_tt_device *device,
                                const pn_tt_residency *table[PN_TT_RESIDENCY_TABLE_MAX],
                                bool with_split_members)
{
    unsigned count = 0;
    uint32_t slot;

    table[count++] = &device->reduce;
    for (slot = 0; slot + 1u < PN_TT_REDUCE_MAX_MEMBERS; slot++)
        if (device->reduce_member[slot].running || device->reduce_member[slot].core_booted)
            table[count++] = &device->reduce_member[slot];
    table[count++] = &device->eltwise;
    table[count++] = &device->sample;
    {
        uint32_t m;

        for (m = 1u; m < device->sample_members; m++)
            table[count++] = &device->sample_member[m - 1u];
    }
    table[count++] = &device->rope;
    table[count++] = &device->memory;
    table[count++] = &device->attention;
    table[count++] = &device->gdn;
    /* THE STRIP BRIDGE WAS NOT IN THIS TABLE, and it is the second largest non-matmul term
     * in the token. Its ~0.022 s/token had only ever been a SUBTRACTION between two runs,
     * for the same reason the GDN phases were before they were counted directly. */
    for (slot = 0; slot < (uint32_t)PN_TT_RENDER_DIRECTIONS; slot++)
        table[count++] = &device->render[slot].residency;
    /* And the residency that DELETES the narrow half — counted from the first line it exists,
     * so its cost is never a subtraction between two runs the way the bridge's was. */
    table[count++] = &device->gate.residency;
    /* Only the LIVE head-norm slots, so a run with one head width reports one line and a
     * hybrid with two reports two — the count is part of the finding. */
    for (slot = 0; slot < device->head_norm_count; slot++)
        table[count++] = &device->head_norm[slot].residency;
    if (with_split_members) {
        uint32_t direction;

        for (slot = 1u; slot < device->eltwise_members; slot++)
            table[count++] = &device->eltwise_member[slot - 1u];
        for (slot = 1u; slot < device->attention_members; slot++)
            table[count++] = &device->attention_member[slot - 1u];
        for (slot = 0; slot < device->attention_splitk_extras; slot++)
            table[count++] = &device->attention_splitk_extra[slot];
        for (slot = 1u; slot < device->memory_members; slot++)
            table[count++] = &device->memory_member[slot - 1u];
        for (slot = 0; slot < device->head_norm_count; slot++)
            for (uint32_t m = 1u; m < device->head_norm[slot].members; m++)
                table[count++] = &device->head_norm[slot].member[m - 1u];
        /* AND THE STRIP BRIDGE'S. An await this process paid that no counter walks is an
         * arm reporting a saving the knob never armed — the failure `ft_tt_noc_path` was
         * taught to count ACCESSES to make visible. */
        for (direction = 0; direction < (uint32_t)PN_TT_RENDER_DIRECTIONS; direction++)
            for (slot = 1u; slot < device->render[direction].members; slot++)
                table[count++] = &device->render[direction].member[slot - 1u];
    }
    return count;
}

bool pn_tt_device_residency_stat(const pn_tt_device *device, unsigned index,
                                 const char **out_name, uint64_t *out_ops, double *out_seconds,
                                 uint32_t *out_core)
{
    const pn_tt_residency *table[PN_TT_RESIDENCY_TABLE_MAX];
    unsigned count;

    if (!device)
        return false;
    count = residency_table(device, table, false);
    if (index >= count)
        return false;
    if (out_name)
        *out_name = table[index]->name;
    if (out_ops)
        *out_ops = table[index]->ops;
    if (out_seconds)
        *out_seconds = table[index]->seconds;
    if (out_core)
        *out_core = table[index]->core_index;
    return true;
}

unsigned pn_tt_device_card_clocks(const pn_tt_device *device, const char **out_names,
                                  double *out_seconds, unsigned max)
{
    static const char *const gdnc_names[] = { "gdnc-phase0", "gdnc-conv", "gdnc-delta" };
    const pn_tt_residency *table[PN_TT_RESIDENCY_TABLE_MAX];
    unsigned count, index, written = 0;
    uint32_t phase;

    if (!device || !out_names || !out_seconds)
        return 0;
    count = residency_table(device, table, true);
    for (index = 0; index < count && written < max; index++, written++) {
        out_names[written] = table[index]->name;
        out_seconds[written] = table[index]->seconds;
    }
    for (phase = 0; phase < 3u && written < max; phase++, written++) {
        out_names[written] = gdnc_names[phase];
        out_seconds[written] = device->gdnc_phase_seconds[phase];
    }
    if (written < max) {
        out_names[written] = "gdnc-recovered";
        out_seconds[written++] = device->gdnc_phase_recovered_seconds[0] +
                                 device->gdnc_phase_recovered_seconds[1] +
                                 device->gdnc_phase_recovered_seconds[2];
    }
    if (written < max) {
        out_names[written] = "gdnc-stage";
        out_seconds[written++] = device->gdnc_stage_build_seconds +
                                 device->gdnc_stage_upload_seconds;
    }
    {
        static const char *const host_names[PN_TT_HOST_CLOCK_COUNT] = {
            "host-gdn-math", "host-gdn-consts", "host-rope-table" };
        unsigned clock;

        for (clock = 0; clock < (unsigned)PN_TT_HOST_CLOCK_COUNT && written < max; clock++) {
            out_names[written] = host_names[clock];
            out_seconds[written++] = device->host_clock_seconds[clock];
        }
    }
    return written;
}

void pn_tt_device_host_clock_add(pn_tt_device *device, pn_tt_host_clock clock, double seconds)
{
    if (device && (unsigned)clock < (unsigned)PN_TT_HOST_CLOCK_COUNT)
        device->host_clock_seconds[clock] += seconds;
}

void pn_tt_device_await_transactions(const pn_tt_device *device, uint64_t *out_awaits,
                                     uint64_t *out_poll, uint64_t *out_settle,
                                     unsigned *out_fast)
{
    const pn_tt_residency *table[PN_TT_RESIDENCY_TABLE_MAX];
    uint64_t awaits = 0, poll = 0, settle = 0;
    unsigned fast = 0, count, index;

    if (out_awaits)
        *out_awaits = 0;
    if (out_poll)
        *out_poll = 0;
    if (out_settle)
        *out_settle = 0;
    if (out_fast)
        *out_fast = 0;
    if (!device)
        return;
    count = residency_table(device, table, true);
    for (index = 0; index < count; index++) {
        awaits += table[index]->awaits;
        poll += table[index]->await_poll_transactions;
        settle += table[index]->await_settle_transactions;
        if (table[index]->awaits != 0u && table[index]->await_shape == PN_TT_AWAIT_LINE)
            fast++;
    }
    if (out_awaits)
        *out_awaits = awaits;
    if (out_poll)
        *out_poll = poll;
    if (out_settle)
        *out_settle = settle;
    if (out_fast)
        *out_fast = fast;
}

void pn_tt_device_issue_transactions(const pn_tt_device *device, uint64_t *out_issues,
                                     uint64_t *out_accesses, double *out_param_seconds,
                                     double *out_ring_seconds, double *out_head_seconds,
                                     unsigned *out_wide)
{
    const pn_tt_residency *table[PN_TT_RESIDENCY_TABLE_MAX];
    uint64_t issues = 0, accesses = 0;
    double param = 0.0, ring = 0.0, head = 0.0;
    unsigned wide = 0, count, index;

    if (out_issues)
        *out_issues = 0;
    if (out_accesses)
        *out_accesses = 0;
    if (out_param_seconds)
        *out_param_seconds = 0.0;
    if (out_ring_seconds)
        *out_ring_seconds = 0.0;
    if (out_head_seconds)
        *out_head_seconds = 0.0;
    if (out_wide)
        *out_wide = 0;
    if (!device)
        return;
    count = residency_table(device, table, true);
    for (index = 0; index < count; index++) {
        issues += table[index]->issues;
        accesses += table[index]->issue_accesses;
        param += table[index]->issue_param_seconds;
        ring += table[index]->issue_ring_seconds;
        head += table[index]->issue_head_seconds;
        if (table[index]->issues != 0u && table[index]->issue_shape == PN_TT_ISSUE_WIDE)
            wide++;
    }
    if (out_issues)
        *out_issues = issues;
    if (out_accesses)
        *out_accesses = accesses;
    if (out_param_seconds)
        *out_param_seconds = param;
    if (out_ring_seconds)
        *out_ring_seconds = ring;
    if (out_head_seconds)
        *out_head_seconds = head;
    if (out_wide)
        *out_wide = wide;
}

void pn_tt_device_dispatch_split(const pn_tt_device *device, double *out_issue_seconds,
                                 uint64_t *out_issues, double *out_wait_seconds,
                                 uint64_t *out_waits, uint64_t *out_wait_descriptors)
{
    if (out_issue_seconds)
        *out_issue_seconds = device ? device->dispatch_issue_seconds : 0.0;
    if (out_issues)
        *out_issues = device ? device->dispatch_issues : 0u;
    if (out_wait_seconds)
        *out_wait_seconds = device ? device->dispatch_wait_seconds : 0.0;
    if (out_waits)
        *out_waits = device ? device->dispatch_waits : 0u;
    if (out_wait_descriptors)
        *out_wait_descriptors = device ? device->dispatch_wait_descriptors : 0u;
}

void pn_tt_device_gdn_phase_stats(const pn_tt_device *device, unsigned phase,
                                  double *out_seconds, uint64_t *out_ops)
{
    const bool ok = device && phase < 3u;

    if (out_seconds)
        *out_seconds = ok ? device->gdn_phase_seconds[phase] : 0.0;
    if (out_ops)
        *out_ops = ok ? device->gdn_phase_ops[phase] : 0u;
}

void pn_tt_device_gdn_phase_recovery(const pn_tt_device *device, unsigned phase,
                                     double *out_seconds, uint64_t *out_ops,
                                     uint64_t *out_rearms)
{
    const bool ok = device && phase < 3u;

    if (out_seconds)
        *out_seconds = ok ? device->gdn_phase_recovered_seconds[phase] : 0.0;
    if (out_ops)
        *out_ops = ok ? device->gdn_phase_recovered_ops[phase] : 0u;
    if (out_rearms)
        *out_rearms = ok ? device->gdn_phase_rearms[phase] : 0u;
}

uint64_t pn_tt_device_bytes_read(const pn_tt_device *device)
{
    return device ? device->bytes_read : 0u;
}

uint64_t pn_tt_device_bytes_written(const pn_tt_device *device)
{
    return device ? device->bytes_written : 0u;
}

void pn_tt_device_dispatch_counts(const pn_tt_device *device, uint64_t *out_resident,
                                  uint64_t *out_oneshot, uint64_t *out_restarts)
{
    if (out_resident)
        *out_resident = device ? device->resident_ops : 0u;
    if (out_oneshot)
        *out_oneshot = device ? device->matvec_count - device->resident_ops : 0u;
    if (out_restarts)
        *out_restarts = device ? device->resident_restarts : 0u;
}

void pn_tt_device_window_traffic(const pn_tt_device *device, uint64_t *out_register_read,
                                 uint64_t *out_register_written, uint64_t *out_bulk_read,
                                 uint64_t *out_bulk_written)
{
    ft_tt_runtime_window_stats stats;

    /* Through the seam, NOT through the tt-kmd binding. Reading the binding directly is what
     * made this report 0 B on every window of a udrv run that had just served a token. */
    if (!device || ft_tt_runtime_window_stats_read(&device->runtime, &stats) != FT_TT_OK)
        memset(&stats, 0, sizeof(stats));

    if (out_register_read)
        *out_register_read = stats.register_read;
    if (out_register_written)
        *out_register_written = stats.register_written;
    if (out_bulk_read)
        *out_bulk_read = stats.bulk_read;
    if (out_bulk_written)
        *out_bulk_written = stats.bulk_written;
}

void pn_tt_device_window_aims(const pn_tt_device *device, uint64_t *out_register_aims,
                              uint64_t *out_bulk_aims, uint64_t *out_failures)
{
    ft_tt_runtime_window_stats stats;

    if (!device || ft_tt_runtime_window_stats_read(&device->runtime, &stats) != FT_TT_OK)
        memset(&stats, 0, sizeof(stats));

    if (out_register_aims)
        *out_register_aims = stats.register_aims;
    if (out_bulk_aims)
        *out_bulk_aims = stats.bulk_aims;
    if (out_failures)
        *out_failures = stats.aim_failures;
}

void pn_tt_device_window_aim_seconds(const pn_tt_device *device, double *out_seconds,
                                     double *out_verify_seconds)
{
    ft_tt_runtime_window_stats stats;

    if (!device || ft_tt_runtime_window_stats_read(&device->runtime, &stats) != FT_TT_OK)
        memset(&stats, 0, sizeof(stats));

    if (out_seconds)
        *out_seconds = stats.aim_seconds;
    if (out_verify_seconds)
        *out_verify_seconds = stats.aim_verify_seconds;
}

void pn_tt_device_window_bank(const pn_tt_device *device, uint32_t *out_register_windows,
                              uint32_t *out_bulk_windows, uint64_t *out_register_hits,
                              uint64_t *out_register_misses, uint64_t *out_bulk_hits,
                              uint64_t *out_bulk_misses, uint64_t *out_misaims,
                              uint64_t *out_audit_failures)
{
    ft_tt_runtime_window_stats stats;

    if (!device || ft_tt_runtime_window_stats_read(&device->runtime, &stats) != FT_TT_OK)
        memset(&stats, 0, sizeof(stats));

    if (out_register_windows)
        *out_register_windows = stats.register_windows;
    if (out_bulk_windows)
        *out_bulk_windows = stats.bulk_windows;
    if (out_register_hits)
        *out_register_hits = stats.register_hits;
    if (out_register_misses)
        *out_register_misses = stats.register_misses;
    if (out_bulk_hits)
        *out_bulk_hits = stats.bulk_hits;
    if (out_bulk_misses)
        *out_bulk_misses = stats.bulk_misses;
    if (out_misaims)
        *out_misaims = stats.misaims_injected;
    if (out_audit_failures)
        *out_audit_failures = stats.audit_failures;
}

/* Telemetry tag numbers and names follow tt-umd's TelemetryTag enumeration,
 * (c) 2025 Tenstorrent Inc., Apache License 2.0. See NOTICE. */
/**
 * MEASURE (`PN_TT_ARC_TELEMETRY=1`): one line of ARC telemetry — clocks, power, temperature,
 * limits and arbiter state (tt-umd TelemetryTag numbering). Reads only; no ARC message, no
 * clock or throttler is touched. udrv transport only; silent otherwise.
 */
void pn_tt_device_telemetry_log(pn_tt_device *device, const char *label)
{
    static const uint16_t tags[] = { 14, 15, 16, 6, 7, 8, 11, 54, 63, 64, 65, 66, 67, 68, 38,
                                     56, 32 };
    static const char *const names[] = { "AICLK", "AXICLK", "ARCCLK", "VCORE", "TDP", "TDC",
                                         "ASIC_TEMP", "INPUT_POWER", "AICLK_LIMIT_MAX",
                                         "TDP_LIMIT_MAX", "AICLK_ARB_MIN", "AICLK_ARB_MAX",
                                         "ENABLED_MIN_ARB", "ENABLED_MAX_ARB", "PCIE_USAGE",
                                         "THM_LIMIT_THROTTLE", "HEARTBEAT" };
    enum { kCount = sizeof(tags) / sizeof(tags[0]) };
    uint32_t values[kCount];
    bool found[kCount];
    unsigned i;
    int rc;

    if (!device || pn_tt_env_u32("PN_TT_ARC_TELEMETRY", 0u) == 0u || !device->runtime.udrv)
        return;
    rc = ft_tt_runtime_udrv_read_telemetry(device->runtime.udrv, tags, kCount, values, found);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "pn_tt: ARC TELEMETRY [%s] read failed rc=%d\n", label, rc);
        return;
    }
    fprintf(stderr, "pn_tt: ARC TELEMETRY [%s]", label);
    for (i = 0; i < kCount; i++) {
        if (!found[i])
            fprintf(stderr, " %s=-", names[i]);
        else if (tags[i] == 11)
            fprintf(stderr, " %s=%.2fC", names[i], (double)values[i] / 65536.0);
        else
            fprintf(stderr, " %s=%u", names[i], values[i]);
    }
    fprintf(stderr, "\n");
}

/**
 * STEP 3 GATE (`PN_TT_IDLE_CLOCK=1`, measure only): every resident writer adds the cycles it
 * spends waiting for a NEW descriptor to its control word 60. Two snapshots over the decode
 * window give each core's card-BUSY time (interval - idle). Families run one after another
 * on the host's schedule, so wall - (matmul busiest member + sum over the other families'
 * main residencies) is the card-idle time the host's issue/await chain puts in a token —
 * the time an on-card dispatcher can remove. `stage` 0 = start snapshot, 1 = report.
 */
/**
 * TTFT lane (`PN_TT_IDLE_CLOCK=2`, measure only): the cores the idle clock's table leaves out —
 * every split member (gdn, eltwise, attention, head-norm, render) and the two families with no
 * main in it (gdnc, gdngate). A member runs CONCURRENTLY with its family's main, so a family
 * costs the slowest of them: each extra core's busy is booked against its family's main busy
 * (`main_busy`, the same window) and only the excess is added. @return ms a token the base
 * clock did not see (0 at stage 0)
 */
static double idle_clock_members(pn_tt_device *device, int stage, double window, double tokens,
                                 const double *main_busy_by_family)
{
    enum { kFamGdn, kFamEltwise, kFamAttention, kFamHeadNorm, kFamRender, kFamGdnc,
           kFamGdngate, kFamCount, kMaxExtra = 128 };
    static const char *fam_name[kFamCount] = { "gdn", "eltwise", "attention", "head-norm",
                                               "render", "gdnc", "gdngate" };
    static uint32_t start_idle[kMaxExtra];
    const pn_tt_residency *extra[kMaxExtra];
    int fam_of[kMaxExtra];
    double fam_max[kFamCount] = { 0 }, unseen = 0.0;
    unsigned n = 0, i, f;
    uint32_t slot, m;

#define IDLE_EXTRA(r, fam) do { if (n < kMaxExtra && (r)->running) { extra[n] = (r); fam_of[n++] = (fam); } } while (0)
    for (m = 1u; m < device->gdn_members; m++) IDLE_EXTRA(&device->gdn_member[m - 1u], kFamGdn);
    for (m = 1u; m < device->eltwise_members; m++)
        IDLE_EXTRA(&device->eltwise_member[m - 1u], kFamEltwise);
    for (m = 1u; m < device->attention_members; m++)
        IDLE_EXTRA(&device->attention_member[m - 1u], kFamAttention);
    for (m = 0; m < device->attention_splitk_extras; m++)
        IDLE_EXTRA(&device->attention_splitk_extra[m], kFamAttention);
    for (slot = 0; slot < device->head_norm_count; slot++)
        for (m = 1u; m < device->head_norm[slot].members; m++)
            IDLE_EXTRA(&device->head_norm[slot].member[m - 1u], kFamHeadNorm);
    for (slot = 0; slot < (uint32_t)PN_TT_RENDER_DIRECTIONS; slot++)
        for (m = 1u; m < device->render[slot].members; m++)
            IDLE_EXTRA(&device->render[slot].member[m - 1u], kFamRender);
    IDLE_EXTRA(&device->gdnc, kFamGdnc);
    for (m = 1u; m < device->gdnc_members; m++) IDLE_EXTRA(&device->gdnc_member[m - 1u], kFamGdnc);
    for (m = 0u; m < device->gdngate_members; m++) IDLE_EXTRA(&device->gdngate[m], kFamGdngate);
#undef IDLE_EXTRA
    for (i = 0; i < n; i++) {
        const ft_tt_queue *queue = &extra[i]->queue;
        uint32_t idle = 0u;

        if (device->runtime.noc.read(device->runtime.noc.user, queue->x, queue->y,
                                     queue->control_address + 60u * 4u, &idle,
                                     sizeof(idle)) != FT_TT_OK)
            continue;
        /* A core whose clock word never moved keeps no clock (2026-10-22: attention members
         * and gdnc read as busy for the whole window) — skipped, not booked. */
        if (stage == 0) {
            start_idle[i] = idle;
        } else if (idle != 0u && idle != start_idle[i]) {
            const double idle_s = (double)(uint32_t)(idle - start_idle[i]) * 16.0 / 1.35e9;
            const double busy = (window - idle_s) / tokens;

            if (busy > fam_max[fam_of[i]])
                fam_max[fam_of[i]] = busy;
        }
    }
    if (stage == 0)
        return 0.0;
    fprintf(stderr, "pn_tt: IDLE CLOCK members (busiest extra core / its family's main, ms a "
                    "token):");
    for (f = 0; f < kFamCount; f++) {
        const double over = fam_max[f] - main_busy_by_family[f];

        if (fam_max[f] <= 0.0)
            continue;
        fprintf(stderr, " %s %.2f/%.2f", fam_name[f], fam_max[f] * 1e3,
                main_busy_by_family[f] * 1e3);
        if (over > 0.0)
            unseen += over;
    }
    fprintf(stderr, " | unseen by the base clock %.2f ms\n", unseen * 1e3);
    return unseen;
}

void pn_tt_device_idle_clock(pn_tt_device *device, int stage, double tokens)
{
    enum { kMax = PN_TT_RESIDENCY_TABLE_MAX + PN_TT_MATMUL_MAX_CORES };
    static uint32_t start_idle[kMax];
    static double start_wall;
    const pn_tt_residency *table[PN_TT_RESIDENCY_TABLE_MAX];
    unsigned count, index;
    double wall, matmul_max = 0.0, others = 0.0, last_main = 0.0;
    /* PN_TT_IDLE_CLOCK=2: the main busy of the families idle_clock_members() extends, in its
     * order (gdn, eltwise, attention, head-norm, render, gdnc, gdngate). */
    double main_busy_by_family[7] = { 0 };

    if (!device || pn_tt_env_u32("PN_TT_IDLE_CLOCK", 0u) == 0u)
        return;
    if (pn_tt_matvec_flush(device) != 0)
        return;
    wall = pn_tt_now_seconds();
    count = residency_table(device, table, false);
    if (stage == 0)
        start_wall = wall;
    else
        fprintf(stderr, "pn_tt: IDLE CLOCK over %.0f tokens, %.4f s/token wall:", tokens,
                (wall - start_wall) / tokens);
    for (index = 0; index < count + device->group.member_count && index < kMax; index++) {
        const ft_tt_queue *queue;
        const char *name;
        uint32_t idle = 0u;

        if (index < count) {
            if (!table[index]->running)
                continue;
            queue = &table[index]->queue;
            name = table[index]->name;
        } else {
            queue = device->group.member[index - count].queue;
            name = "matmul";
            if (!queue)
                continue;
        }
        if (device->runtime.noc.read(device->runtime.noc.user, queue->x, queue->y,
                                     queue->control_address + 60u * 4u, &idle,
                                     sizeof(idle)) != FT_TT_OK)
            continue;
        if (stage == 0) {
            start_idle[index] = idle;
        } else {
            const double idle_s = (double)(uint32_t)(idle - start_idle[index]) * 16.0 / 1.35e9;
            const double busy = ((wall - start_wall) - idle_s) / tokens;

            if (idle == 0u) {
                fprintf(stderr, " [%s: no clock]", name);
                continue;
            }
            if (index < count) {
                /* A SPLIT MEMBER runs CONCURRENTLY with its main (the table lists it right
                 * after): the family costs the slower of the two, not their sum. */
                if (strstr(name, "-member") != NULL) {
                    if (busy > last_main) {
                        others += busy - last_main;
                        last_main = busy;
                    }
                } else {
                    others += busy;
                    last_main = busy;
                }
                {
                    static const char *fam[5] = { "gdn", "eltwise", "attention", "head-norm",
                                                  "render" };
                    unsigned f;

                    for (f = 0; f < 5u; f++)
                        if ((f == 0u ? strcmp(name, fam[f]) == 0
                                     : strncmp(name, fam[f], strlen(fam[f])) == 0) &&
                            busy > main_busy_by_family[f])
                            main_busy_by_family[f] = busy;
                }
                fprintf(stderr, " %s %.2fms", name, busy * 1e3);
            } else if (busy > matmul_max) {
                matmul_max = busy;
            }
        }
    }
    if (stage != 0) {
        const double per = (wall - start_wall) / tokens;

        fprintf(stderr, " | matmul busiest member %.2fms\n", matmul_max * 1e3);
        fprintf(stderr, "pn_tt: IDLE CLOCK — card-busy sum %.2f ms (matmul %.2f + others %.2f) "
                        "of %.2f ms wall: HOST-ATTRIBUTABLE card idle %.2f ms/token\n",
                (matmul_max + others) * 1e3, matmul_max * 1e3, others * 1e3, per * 1e3,
                (per - matmul_max - others) * 1e3);
    }
    if (pn_tt_env_u32("PN_TT_IDLE_CLOCK", 0u) >= 2u) {
        const double unseen =
            idle_clock_members(device, stage, wall - start_wall, tokens, main_busy_by_family);

        if (stage != 0)
            fprintf(stderr, "pn_tt: IDLE CLOCK with members — card idle %.2f ms/token\n",
                    ((wall - start_wall) / tokens - matmul_max - others - unseen) * 1e3);
    }
}

/* ------------------------------------------------------------------------------------
 * M5 WAIT CLOCK REPORT. Every resident processor adds the wall-clock cycles (>> 4) it spent
 * spinning in a CB wait to its status word 6 and counts the waits in word 7
 * (resident_queue_protocol.h ftq_wait_account). The processor that waits LEAST on a busy
 * core is the one the other four wait for. Since the last launch of that core.
 * ---------------------------------------------------------------------------------- */
static void pn_tt_wait_line(pn_tt_device *device, const ft_tt_queue *queue, const char *label,
                            double tokens)
{
    static const char *names[5] = {"reader", "writer", "unpack", "math", "pack"};
    uint32_t words[5 * 8];
    uint32_t p;

    if (!queue || queue->control_address == 0u ||
        device->runtime.noc.read(device->runtime.noc.user, queue->x, queue->y,
                                 queue->control_address + FT_TT_QUEUE_STATUS_OFFSET, words,
                                 sizeof(words)) != FT_TT_OK)
        return;
    fprintf(stderr, "pn_tt: wait clock %-12s core (%u,%u):", label, queue->x, queue->y);
    for (p = 0; p < 5u; p++)
        fprintf(stderr, " %s %.2f ms (%u)", names[p],
                (double)words[p * 8u + 6u] * 16.0 / 1.35e9 * 1e3 / tokens, words[p * 8u + 7u]);
    fprintf(stderr, "  [ms/token since launch]\n");
}

/* M5: the gdn reader's own clocks, control words 12 and 57-60 (resident_gdn_reader.c). */
static void pn_tt_gdn_reader_clocks(pn_tt_device *device, const ft_tt_queue *queue,
                                    const char *label, double tokens)
{
    uint32_t w[8];
    const double k = 16.0 / 1.35e9 * 1e3 / tokens;

    if (!queue || queue->control_address == 0u ||
        device->runtime.noc.read(device->runtime.noc.user, queue->x, queue->y,
                                 queue->control_address + 56u * 4u, w, sizeof(w)) != FT_TT_OK ||
        device->runtime.noc.read(device->runtime.noc.user, queue->x, queue->y,
                                 queue->control_address + 12u * 4u, w, sizeof(w[0])) != FT_TT_OK)
        return;
    /* word 12 conv barrier (over 56, the S1 done-sequence); 57..63: delta barrier, build,
     * conv, delta (reader); CONV waits of the writer, unpack, and unpack on the reader's
     * buffers. */
    fprintf(stderr, "pn_tt: gdn reader clocks %-8s: DELTA barrier %.2f build %.2f all %.2f | "
                    "CONV barrier %.2f all %.2f | CONV waits: writer %.2f unpack %.2f (on "
                    "reader %.2f) ms/token\n",
            label, w[1] * k, w[2] * k, w[4] * k, w[0] * k, w[3] * k, w[5] * k, w[6] * k,
            w[7] * k);
}

void pn_tt_device_wait_report(pn_tt_device *device, double tokens)
{
    if (!device || tokens <= 0.0)
        return;
    if (device->group.member_count > 1u)
        pn_tt_wait_line(device, device->group.member[1].queue, "matmul m1", tokens);
    if (device->group.member_count > 0u)
        pn_tt_wait_line(device, device->group.member[0].queue, "matmul m0", tokens);
    if (device->gdn.running) {
        pn_tt_wait_line(device, &device->gdn.queue, "gdn", tokens);
        pn_tt_gdn_reader_clocks(device, &device->gdn.queue, "gdn", tokens);
    }
    if (device->gdn_members > 1u && device->gdn_member[0].running) {
        pn_tt_wait_line(device, &device->gdn_member[0].queue, "gdn m1", tokens);
        pn_tt_gdn_reader_clocks(device, &device->gdn_member[0].queue, "gdn m1", tokens);
    }
    if (device->reduce.running) {
        uint32_t w[3];

        pn_tt_wait_line(device, &device->reduce.queue, "reduce", tokens);
        if (device->runtime.noc.read(device->runtime.noc.user, device->reduce.queue.x,
                                     device->reduce.queue.y,
                                     device->reduce.queue.control_address + 57u * 4u, w,
                                     sizeof(w)) == FT_TT_OK)
            fprintf(stderr, "pn_tt: reduce phase clocks: map %.2f reduce %.2f final %.2f "
                            "ms/token\n",
                    (double)w[0] * 16.0 / 1.35e9 * 1e3 / tokens,
                    (double)w[1] * 16.0 / 1.35e9 * 1e3 / tokens,
                    (double)w[2] * 16.0 / 1.35e9 * 1e3 / tokens);
    }
    if (device->eltwise.running)
        pn_tt_wait_line(device, &device->eltwise.queue, "eltwise", tokens);
}

#endif /* USE_TT */
