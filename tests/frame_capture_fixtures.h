/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file frame_capture_fixtures.h
 * @brief Synthetic tool-call argument values reproducing the FRAME CAPTURE
 *        failure shapes: what lands in a parameter VALUE when a turn is
 *        force-closed mid-thought.
 *
 *        The content (project "lumen", its paths and its analysis text) is
 *        invented.  The shapes are not: each fixture keeps the structure,
 *        approximate length and escape pattern of a failure observed in real
 *        serving -- the leading "</think>", the prose lead-in, the call
 *        dialect, over-closed braces, shell metacharacters, code fences and a
 *        truncated SEARCH/REPLACE block.
 *
 *        WHAT THEY SHOW.  The forced tool grammar compels the XML frame; when a
 *        turn is force-closed mid-thought (the serve INJECTS "</think>" once the
 *        per-model reasoning_budget is spent -- batch_sched.c driver_reason_step,
 *        serve_generation.c sg_run_completion), the grammar cannot tell that
 *        close from the model's own, sets reasoning_ended, and compels the
 *        opener from the next char.  Whatever the model was still going to say
 *        then lands in the free-text parameter VALUE.  FX_NATIVE_XML_CAPTURE is
 *        the control: the captured text there is a NATIVE Qwen-XML call, the
 *        very dialect the grammar armed, so the capture is not dialect drift.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TEST_FRAME_CAPTURE_FIXTURES_H
#define PN_TEST_FRAME_CAPTURE_FIXTURES_H

/* A `grep` frame's `pattern` argument, ~60 chars.
 * THE FALSIFYING CASE.  The model's own </think> followed by a truncated
 * NATIVE Qwen-XML `read` call -- the exact dialect the grammar armed.  No
 * dialect drift happened here, so no dialect theory explains the capture; the
 * turn was force-closed mid-thought.  RECOVERABLE. */
static const char FX_NATIVE_XML_CAPTURE[] =
    "</think>\n\n<function=read>\n<parameter=path>\nlumen/serve.log";

/* A `read` frame's `path` argument, ~140 chars.
 * The model's own </think>, then a prose lead-in, then a complete call in the
 * Hermes `call_function` shape.  The leading </think> is the tell: the model
 * had NOT finished reasoning when the frame was compelled, so it wrote the
 * close it was always going to write INTO the value.  RECOVERABLE. */
static const char FX_HERMES_LEADIN_CAPTURE[] =
    "</think>\n\nLet me look at the Makefile to understand the build structur"
    "e:\n\n<tool_call>call_function{\"func\":\"read\",\"args\":{\"path\":\"lu"
    "men/Makefile\"}}";

/* A `read` frame's `path` argument, ~100 chars.
 * Bare `call_function` call, over-closed ("}}}" where two braces close the
 * object).  RECOVERABLE. */
static const char FX_OVERCLOSED_CAPTURE[] =
    "<tool_call>call_function{\"func\":\"read\",\"args\":{\"path\":\"lumen/in"
    "clude/lumen/settings_cache.h\"}}}";

/* A `read` frame's `path` argument, ~150 chars.
 * Prose lead-in then a complete `call_function` call, absolute path.
 * RECOVERABLE. */
static const char FX_ABSOLUTE_PATH_CAPTURE[] =
    "Let me find the correct path for settings_cache.h:\n\n<tool_call>call_fu"
    "nction{\"func\":\"read\",\"args\":{\"path\":\"/home/user/src/lumen/inclu"
    "de/settings_cache.h\"}}}";

/* A `glob` frame's `pattern` argument, ~190 chars.
 * A shell pipeline plus a dangling markdown fence in a `glob` pattern.  Not a
 * complete call in any shape: NOTHING TO RECOVER, and NO marker to key on.
 * The trailing token is cut mid-word, as a truncated emission is. */
static const char FX_SHELL_IN_GLOB[] =
    "lumen/include/**/*.h* | head -20; echo \"---\"; find lumen -name \"setti"
    "ngs_cache.h\" 2>/dev/null; echo \"---\"; ls lumen/src/settings/*.h 2>/de"
    "v/null; echo \"---\"; ls lume-4471093821\n```";

/* A `glob` frame's `pattern` argument, over 5,000 chars.
 * Mostly analysis prose, with C source (braces!) and a TRUNCATED SEARCH/REPLACE
 * edit block at the end.  Not a complete call, and the braces must not fool a
 * JSON scanner into manufacturing one.  NOTHING TO RECOVER, NO marker. */
static const char FX_ANALYSIS_DUMP[] =
    "lumen/include/lumen/settings_cache.h*lumen/src/settings/settings_cache.c"
    ":212` is the `log_warn(\"settings_cach**e: load failed...` line. The pro"
    "blem is that the loader reads a `version` field that older settings file"
    "s never wrote, and then bails out because it cannot find the key.\n\nLet"
    " me walk through what happens step by step:\n1. `settings_cache_open` re"
    "ads the whole file and hands it to `parse_settings`, which expects `vers"
    "ion` at the top level\n2. When a file written by an older build is opene"
    "d, the parser returns no error for the missing key, but the later lookup"
    " of `version` returns NULL and the caller treats that as corruption\n\nT"
    "he fix should be:\n1. Treat a missing `version` as version 1 instead of "
    "as a hard failure\n2. Run the upgrade steps in order, from whatever vers"
    "ion the file declares up to the current one\n\nLet me check how the othe"
    "r loaders deal with this - particularly anything named UPGRADE_STEPS or "
    "similar:\n*   Around line 140-151 there is already an UPGRADE_STEPS tabl"
    "e of functions that rewrite older layouts. The trouble is that step 2 re"
    "names `cache_dir` to `cache_root` unconditionally, which corrupts a file"
    " that already uses the new name.\n\nWhat actually needs to change:\n*   "
    "Check the declared version before running each step\n*   Make each step "
    "idempotent rather than assuming it runs exactly once\n\nLet me see what "
    "happens when step 2 runs on a file that already has `cache_root`. The ol"
    "d key is absent, so the rename copies a NULL value over the good one. Th"
    "at is the \"empty cache_root\" report.\n\nReading the flow again more ca"
    "refully:\n1. Line 205 calls parse_settings, which builds the tree - this"
    " succeeds even when keys are missing\n2. Line 209-210 loops over UPGRADE"
    "_STEPS and applies every one of them\n\nThe issue is that on a second st"
    "art the steps run again, and nothing records that they already ran. The "
    "return value is ignored too - looking at line 209-210:\n\n```c\nfor (siz"
    "e_t i = 0; i < sizeof(UPGRADE_STEPS) / sizeof(*UPGRADE_STEPS); i++)\n   "
    " UPGRADE_STEPS[i](cache->root);\n```\n\nNothing checks whether a step fa"
    "iled! So a failing rename is silently swallowed, and that is NOT the rea"
    "l issue on its own. The real issue shows up at line 205:\n\n```c\nif (pa"
    "rse_settings(cache->text, &cache->root, &err) != 0) {\n    log_warn(\"se"
    "ttings_cache: load failed: %s\", err ? err : \"?\");\n    free(err);\n``"
    "`\n\nHold on - parse_settings also validates that `cache_root` is a non-"
    "empty string. The validation runs BEFORE the upgrade steps, so on an old"
    " file the key is still named `cache_dir`, validation fails, and the upgr"
    "ade never gets a chance to run. Let me look at line 74:\n\n```c\n{ \"cac"
    "he_root\", KIND_STRING, REQUIRED },\n```\n\nThat entry will reject any f"
    "ile that predates the rename! And because validation and parsing happen "
    "in the same call, the whole load fails when that one entry fails.\n\nSo "
    "the fix is one of:\n1. Run validation separately, after the upgrade step"
    "s have rewritten the tree  \n2. Or mark `cache_root` optional and check "
    "it after upgrading\n\nThe cleanest fix is to make sure every upgrade ste"
    "p runs BEFORE validation, and to make each step check its own preconditi"
    "on so that running it twice is harmless.\n\nActually, looking again at l"
    "ine 146:\n```c\n{ 2, rename_cache_dir },\n```\n\nThis IS in UPGRADE_STEP"
    "S, which runs AFTER parse_settings. But parse_settings already applied t"
    "he schema table from line 74, and that happens INSIDE the same call that"
    " builds the tree. \n\nWhen parsing and validation share one call, a sing"
    "le failed rule fails the whole call and returns no tree at all. So when "
    "the `cache_root` rule runs against a file that still says `cache_dir` (b"
    "ecause the rename has not happened yet), it FAILS and throws away the pa"
    "rsed tree, including the parts that were perfectly fine.\n\nThe fix is t"
    "o move validation OUT of parse_settings, or split it so it only runs onc"
    "e the upgrade steps have renamed the key. Better approach: have parse_se"
    "ttings only build the tree, then run the upgrade steps, then validate th"
    "e result with a clear message for each rule that fails.\n\nI also want t"
    "o double-check the writer side, because if `settings_cache_save` writes "
    "the file before validation there is a window where a half-upgraded tree "
    "reaches disk. Looking at line 260-268, save is only called from `setting"
    "s_cache_commit`, and commit is only reachable after a successful open, s"
    "o that window does not exist today. Good - nothing to change there.\n\nO"
    "ne more thing worth confirming is the version bump itself. After the las"
    "t step runs, the loader must write `version` back as the current number,"
    " otherwise every start replays all of the steps from version 1. Line 214"
    " does set it, but only on the success path, which is exactly the path th"
    "at never runs for an old file. Once validation moves after the steps, th"
    "at line becomes reachable and the bump happens.\n\nLet me implement the "
    "fix:\n```c|CODE_EDIT|lumen/src/settings/settings_cache.c\n<<<<<<< SEARCH"
    "\n/* The full key table, in the order validate_settings walks it.\n * On"
    "e definition so a new key can never be validated in some loaders and not"
    "\n * others \xe2\x80\x94 that asymmetry is exactly how `cache_root` woul"
    "d have gone missing. */\n#define SETTINGS_KEYS                          "
    "                              \\";

/* A `read` frame's `path` argument, ~230 chars.
 * The ORIGINAL wedge shape: a bash pipeline captured into a `read` path.  The
 * model wanted `bash`, but nothing in the value says so.  NOTHING TO RECOVER,
 * and NO marker -- content sniffing cannot reach this family at all. */
static const char FX_SHELL_IN_READ_PATH[] =
    "/home/user/src/lumen/src/core/settings_cache.c.md 2>/dev/null; find /hom"
    "e/user/src -path \"*/src/settings_cache.c\" 2>/dev/null | head -5; find "
    "/home/user/sr -name \"settings_cache.c\" 2>/dev/null | grep lumen";

#endif /* PN_TEST_FRAME_CAPTURE_FIXTURES_H */
