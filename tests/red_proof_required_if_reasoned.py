#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""Red proof for the required_if_reasoned tests.

Applies one mutation at a time to the grammar sources, deletes the test binaries
(macOS make 3.81 has 1 s mtime granularity), rebuilds, runs both tests and counts
failed asserts, then restores the source.  Every mutation must turn at least one
test red.  Run from the repo root:  python3 tests/red_proof_required_if_reasoned.py
"""
import re
import subprocess
import sys

XML = "src/serve/tool_grammar.c"
JSON = "src/serve/tool_grammar_json.c"
TRACKER = "src/serve/reasoned_reply_tracker.h"
TESTS = ["tests/test_tool_grammar_reasoned", "tests/test_tool_grammar_json_reasoned"]

# (name, file, old, new)
MUTATIONS = [
    ("xml: reasoned-only keys never owed", XML,
     "if (reasoned_reply_call_named(&st->reply))",
     "if (reasoned_reply_call_named(&st->reply) && 0)"),
    ("xml: reasoned-only keys always owed", XML,
     "if (reasoned_reply_call_named(&st->reply))",
     "if (reasoned_reply_call_named(&st->reply) || 1)"),
    ("xml: required walk does not watch", XML,
     "    else reasoned_reply_begin(&st->reply);\n", ""),
    ("xml: close token not seen", XML,
     "        reasoned_reply_close(&st->reply);\n", ""),
    ("xml: AUTO watch is a no-op", XML,
     "    if (st) reasoned_reply_begin(&st->reply);", "    (void)st;"),
    ("xml: opener counted as reasoning", XML,
     "            reasoned_reply_call_opened(&st->reply, (int)strlen(OPEN_TC));\n", ""),
    ("xml: FREE chars not fed", XML,
     "        reasoned_reply_char(&st->reply, c, st->think_close_token);\n", ""),
    ("xml: injected close does not end", XML,
     "    st->reasoning_ended = true;\n    reasoned_reply_end(&st->reply);\n    /* Grant",
     "    st->reasoning_ended = true;\n    /* Grant"),
    ("xml: unknown name marks a key", XML,
     "if (idx >= 0 && idx < TG_REQ_MAX_KEYS)\n                g->tools[t].required_if_reasoned |= (uint64_t)1 << idx;",
     "g->tools[t].required_if_reasoned |= 1;"),
    ("json: reasoned-only keys never owed", JSON,
     "st->owes_reasoned_keys = reasoned_reply_call_named(&st->reply);",
     "st->owes_reasoned_keys = reasoned_reply_call_named(&st->reply) && 0;"),
    ("json: reasoned-only keys always owed", JSON,
     "st->owes_reasoned_keys = reasoned_reply_call_named(&st->reply);",
     "st->owes_reasoned_keys = reasoned_reply_call_named(&st->reply) || 1;"),
    ("json: required walk does not watch", JSON,
     "    else reasoned_reply_begin(&st->reply);\n", ""),
    ("json: close token not seen", JSON,
     "        reasoned_reply_close(&st->reply);\n", ""),
    ("json: AUTO watch is a no-op", JSON,
     "    if (st) reasoned_reply_begin(&st->reply);", "    (void)st;"),
    ("json: brace opener counted as reasoning", JSON,
     "            reasoned_reply_call_opened(&st->reply, 1);\n", ""),
    ("json: fence opener counted as reasoning", JSON,
     "                reasoned_reply_call_opened(&st->reply, (int)strlen(FENCE_OPEN));\n", ""),
    ("json: FREE chars not fed", JSON,
     "    reasoned_reply_char(&st->reply, c, st->think_close_token);\n", ""),
    ("json: reasoning end does not end", JSON,
     "    st->reasoning_ended = true;\n    reasoned_reply_end(&st->reply);\n}",
     "    st->reasoning_ended = true;\n}"),
    ("json: emitted reasoned key not counted", JSON,
     "(st->owes_reasoned_keys && t->required_if_reasoned &&\n                         t->required_if_reasoned[idx])",
     "false"),
    ("json: unknown name marks a key", JSON,
     "    int idx = set_exact_match(&t->keys, -1, key, (int)strlen(key));\n    if (idx < 0 || t->required[idx]) return true;",
     "    int idx = 0;\n    if (t->required[idx]) return true;"),
    ("tracker: whitespace counts", TRACKER,
     "    if (r->close_term == 0 && c != ' '", "    if (r->close_term == 0 && c != '\\x01'"),
    ("tracker: closer chars count", TRACKER,
     "        r->watching = false;   /* the closer's own chars never count */",
     "        r->watching = false; reasoned_reply_count(r, 8);"),
    ("tracker: broken partial close not counted", TRACKER,
     "    reasoned_reply_count(r, held);   /* a broken partial", "    (void)held;   /* a broken partial"),
    ("tracker: end drops a held partial", TRACKER,
     "    reasoned_reply_count(r, r->close_term);\n", ""),
    ("tracker: every call is the first", TRACKER,
     "    bool first = (r->calls_named == 0);", "    bool first = true;"),
    ("tracker: counter never advances", TRACKER,
     "    if (r->calls_named < 1 << 30) r->calls_named++;", ""),
]


def run_tests():
    """Rebuild both tests from scratch; return failed tests, or None when the build fails."""
    subprocess.run(["rm", "-f"] + TESTS)
    build = subprocess.run(["make", "USE_BLAS=1"] + TESTS, capture_output=True, text=True)
    if build.returncode != 0:
        return None
    failed = 0
    for t in TESTS:
        out = subprocess.run(["./" + t], capture_output=True, text=True).stdout
        m = re.search(r"(\d+) failed", re.sub(r"\x1b\[[0-9;]*m", "", out))
        failed += int(m.group(1)) if m else 1
    return failed


def main():
    base = run_tests()
    print(f"baseline: {base} failed")
    if base != 0:
        sys.exit("baseline must be green")
    survivors = 0
    for name, path, old, new in MUTATIONS:
        src = open(path).read()
        if src.count(old) != 1:
            sys.exit(f"mutation anchor not unique in {path}: {name}")
        open(path, "w").write(src.replace(old, new))
        try:
            failed = run_tests()
        finally:
            open(path, "w").write(src)
        verdict = "BUILD FAIL" if failed is None else f"{failed} failed"
        print(f"  {name:45s} {verdict}")
        if failed == 0:
            survivors += 1
    print(f"restored; final: {run_tests()} failed; survivors: {survivors}")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
