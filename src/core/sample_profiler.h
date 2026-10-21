/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file sample_profiler.h
 * @brief A self-contained SAMPLING profiler: SIGPROF every millisecond of process CPU time,
 *        the interrupted instruction pointer and thread counted, dumped at exit.
 *
 * WHY. The TT host's kernel refuses unprivileged `perf` (perf_event_paranoid 4), and a root
 * run of the engine leaves the udrv claim files root-owned. This needs neither:
 * `PN_PROFILE=<file>` arms it, and the dump is `<offset-from-image-base> <thread> <count>`
 * lines plus the image base, resolved afterwards with `addr2line -f -e <binary>`
 * (tools/sample_profile_report.sh).
 *
 * @status Instrument only; off unless PN_PROFILE is set.
 * @issues Linux/x86-64 only (reads RIP from the ucontext); a no-op elsewhere.
 * @todo None
 */
#ifndef PN_SAMPLE_PROFILER_H
#define PN_SAMPLE_PROFILER_H

/** Arm the profiler when `PN_PROFILE` names an output file. Safe to call more than once. */
void pn_sample_profiler_start(void);

/** Record only inside marked windows from the first call on (1 = enter, 0 = leave): the
 *  prefill, not the 38 s weight upload around it. */
void pn_sample_profiler_window(int inside);

#endif /* PN_SAMPLE_PROFILER_H */
