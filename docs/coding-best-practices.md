# Coding Best Practices — C

The standard code in this repo is held to. PRs are reviewed against it, and review
comments cite it by rule number (`blocker: rule 3, this file is 1,400 lines`).

Rule numbers are frozen. New rules append; they never renumber.

The PR process itself — size, evidence, comment conventions — is in
[`../CONTRIBUTING.md`](../CONTRIBUTING.md).

---

## Rules

**1. Descriptive names.** Every name says what it is at a glance — files, functions,
types, variables. The test: would you know what it is from the tree alone, without
opening it? `pn_prefix_cache_lookup()`, not `lookup()`. Abbreviate only what is
universal (`id`, `url`, `http`, `kv`, `ctx`).

**2. Small functions — ~100 lines of body.** Doc comments don't count. Past ~100, look
for the extraction point. One blank line between logical chunks, never two.

**3. Small files — under 1,000 lines of code.** Comments and headers don't count. Prefer
more files with fewer functions. Every function in a file must belong there.

**4. Library-first.** Write every function as if it will be extracted into a standalone
library — self-contained, minimal dependencies, no hard-coded project values in reusable
logic. Headers define the contract, sources hold the implementation. Extraction should
mean moving files, not rewriting them.

**5. Folder nesting is navigation.** Group by domain, not by file type. Each directory
has one clear purpose. A directory holding one file is usually wrong.

**6. Error transparency.** Never swallow an error. Return a code or log it — never both
nothing. No empty error paths. Callers get a typed error, not a guess.

**7. Delete dead code.** Don't comment it out, don't leave unused functions or includes.
Git has it if you need it back.

**8. Prove a test can fail.** A test you have never seen fail is not evidence. Break the
code it covers, watch it go red, report the failure count.

> **Apple's `make` is 3.81 (2006)** and compares mtimes at one-second granularity. A
> source edited in the same second as its object is reported `up to date` and is **not
> rebuilt** — so the obvious mutation loop can be running the previous binary and
> reporting a vacuous gate that isn't. When proving a test red, `rm` the specific `.o`
> **and** the linked binary first. Switching build configs needs `rm -rf build` entirely.

---

## File headers

Every source file opens with the SPDX pair, then the header block:

```c
/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   prefix_cache.c
 * @brief  Longest-common-prefix KV reuse across turns.
 *
 * @status None
 * @issues None
 * @todo   None
 */
```

`@status` / `@issues` / `@todo` are always present, `None` when empty — the empty slot is
the reminder that the slot exists. When you work in a file, clear resolved entries and
reset to `None`.

**Required on every new file.** Existing coverage is partial and is being backfilled;
until that lands, CI enforces the rule on added files only.

**Exempt:** test files, generated sources, and third-party-derived files (which keep their
upstream attribution — a Pantheon block may go below it, never in place of it).

---

## Documentation

**Public functions are documented in the header**, which is the API contract:

```c
/**
 * Look up the longest cached prefix for a token sequence.
 *
 * @param cache   Prefix cache handle
 * @param tokens  Token sequence to match
 * @param n       Length of tokens
 * @return        Matched prefix length, or 0 if nothing matched
 *
 * Thread-safe: yes (internal locking)
 * Reentrant:   no
 * Signal-safe: no
 */
```

The three safety annotations are always present. `TBD` is an acceptable answer; omitting
them is not. Internal helpers need docs only when the name and signature don't carry it.

**Inline comments explain why, not what.** The code already says what it does.

```c
/* K80 has no __ldg for 8-byte types — split the load or it serialises */
```

---

## Layout

`include/` and `src/` mirror each other. `snake_case` filenames that name their contents
(`die_manager.c`, not `dm.c`). One header + one source per logical unit. Tests mirror the
source tree.

Every public symbol carries its module prefix — `pn_` for Pantheon, `fc_`/`tm_`/`ft_` for
Foundry. Anything not declared in a header is `static`. No exceptions.

---

## C conventions

**Formatting.** 4 spaces, no tabs. 100 columns. `char *ptr` — the star binds to the
variable. Opening brace on the **same line**, for functions and control flow alike.

```c
int pn_cache_lookup(pn_cache *cache, const uint32_t *tokens, size_t n) {
    if (!cache || !tokens)
        return -PN_EINVAL;

    for (size_t i = 0; i < cache->count; i++) {
        if (entry_matches(&cache->entries[i], tokens, n))
            return (int)cache->entries[i].len;
    }
    return 0;
}
```

Omit braces on single-statement bodies — unless any branch of an `if`/`else` chain needs
them, in which case every branch gets them.

**Includes**, four groups separated by blank lines: this module's own header first (it
catches missing dependencies in that header at compile time), then standard library, then
system/platform, then project headers.

**Errors.** Functions return `int` (0 success, negative error) or a pointer (`NULL` on
failure). Use goto-cleanup for anything acquiring more than one resource — one exit path,
deterministic teardown. Error codes are a prefixed enum with a string conversion function.

**Memory.** `sizeof(*ptr)`, never `sizeof(Type)`. `calloc` or explicit zeroing. Document
ownership in the function doc — who allocates, who frees. Every `_open`/`_create`/`_new`
has a matching `_close`/`_destroy`/`_free`. `free(p); p = NULL;`. Arena-allocate groups
sharing a lifetime. Never return a stack address.

**const.** Every pointer parameter is `const` unless the function must modify what it
points at.

**Strings and buffers.** `sprintf`, `strcpy`, `strcat`, `gets`, and unbounded `scanf("%s")`
are banned — use `snprintf`, `strlcpy`, `strlcat`, `fgets`, width-limited `scanf`. Always
pass buffer sizes explicitly; check the `snprintf` return for truncation.

**Initialize everything.** `int rc = 0;`, `char *buf = NULL;`, `struct foo f = {0};`.
Declare close to first use, except where goto-cleanup requires it earlier.

**Opaque types** at public API boundaries — the struct definition stays in the `.c`.
Callers go through the API, never into fields.

**Header guards** use the project prefix (`PANTHEON_PREFIX_CACHE_H`), with the guard name
repeated in the `#endif` comment, and `extern "C"` in every public header.

**Asserts** document invariants that must never be false if the code is correct. Don't
assert things that legitimately fail — I/O, allocation, user input — those get error codes.

**Compiler.** The standard, modelled on SQLite and comparable C projects:

```
-std=c11 -Wall -Wextra -Werror -pedantic
```

plus `-Wshadow`, `-Wstrict-prototypes`, `-Wformat=2`, `-Wnull-dereference`,
`-Wconversion`, `-fstack-protector-strong`. Zero warnings, always.

The build does not yet meet this — it is `-Wall -Wextra -std=c11` today. Closing that gap
is tracked in the conformance spec in `plans/`; the standard above is the target the code
is held to, not a description of the current Makefile.

---

## Logging

| Level | Use |
|---|---|
| `ERROR` | operation failed, cannot continue |
| `WARN` | unexpected but recoverable |
| `INFO` | major state transitions, operation boundaries |
| `DEBUG` | decision points — which branch, and why |
| `TRACE` | entry/exit, raw values, wire data |

`DEBUG` and `TRACE` compile out in release builds. Log at boundaries and decisions, never
inside hot loops at `INFO` or above. Never log secrets, raw binary, or the success of
trivial operations. One line per message, greppable, with timestamp, level, and source
location.

The test: if you would add a `printf` to debug it, a `log_debug` should already be there.
