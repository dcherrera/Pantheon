#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""Red proof for tests/test_tool_grammar_no_repeat_keys.c.

Applies one mutation at a time to the sources (each a list of replacements),
deletes the test binaries (macOS make 3.81 has 1 s mtime granularity), rebuilds,
runs the tests and counts failed checks, then restores the sources.  Every
mutation must turn at least one check red.  The first mutation of each dialect
is the code before this change.  Run from the repo root (FOUNDRY as for
`make test`):
    FOUNDRY=... python3 tests/red_proof_no_repeat_keys.py
"""
import os
import re
import subprocess
import sys

XML = "src/serve/tool_grammar.c"
JSON = "src/serve/tool_grammar_json.c"
LOG = "src/serve/tool_raw_excerpt.h"
TESTS = ["tests/test_tool_grammar_no_repeat_keys", "tests/test_tool_grammar",
         "tests/test_tool_grammar_json", "tests/test_tool_grammar_field_patterns",
         "tests/test_tool_grammar_json_field_patterns", "tests/test_tool_call_truncation"]

XML_REOPEN = ("            if (kidx < 0 || !key_unseen(st, kidx)) { st->phase = TG_DEAD; return; }",
              "            if (kidx < 0) { st->phase = TG_DEAD; return; }")
XML_BODY = ("            if (!call_has_unseen_key(st)) { st->phase = TG_DEAD; return; }\n", "")
XML_LIVE = ("        } else if (key_char_live(st, c)) {",
            "        } else if (set_char_live(st->set, st->set_pos, c)) {")
JSON_REOPEN = ("    if (idx >= 0 && !arg_key_open(st, idx)) { st->phase = TGJ_DEAD; return -1; }\n", "")
JSON_COMMA = ("            if (!call_has_open_key(st)) { st->phase = TGJ_DEAD; return; }\n", "")
JSON_LIVE = ("        if (!arg_key_char_live(st, keys, c)) { st->phase = TGJ_DEAD; return -1; }",
             "        if (!set_char_live(keys, -1, st->acc_len, c)) { st->phase = TGJ_DEAD; return -1; }")

# (name, file, [(old, new), ...])
MUTATIONS = [
    ("xml: before this change", XML, [XML_REOPEN, XML_BODY, XML_LIVE]),
    ("xml: a written key reopens at '>'", XML, [XML_REOPEN, XML_LIVE]),
    ("xml: '<parameter=' with no key left", XML, [XML_BODY]),
    ("xml: key chars ignore written keys", XML,
     [("        if (key_unseen(st, i) && (int)strlen(m) > pos && m[pos] == c &&",
       "        if ((int)strlen(m) > pos && m[pos] == c &&")]),
    ("xml: key chars match one position only", XML,
     [("            strncmp(m, st->name_acc, cmp) == 0)\n            return true;",
       "            1)\n            return true;")]),
    ("json: before this change (repeat refused only at the quote)", JSON,
     [JSON_COMMA, JSON_LIVE]),
    ("json: a written key accepted at the quote", JSON, [JSON_REOPEN, JSON_LIVE]),
    ("json: ',' with no key left", JSON, [JSON_COMMA]),
    ("json: key chars ignore written keys", JSON,
     [("        if (arg_key_open(st, i) && (int)strlen(m) > pos && m[pos] == c &&",
       "        if ((int)strlen(m) > pos && m[pos] == c &&")]),
    ("json: key chars match one position only", JSON,
     [("            strncmp(m, st->acc, cmp) == 0)\n            return true;",
       "            1)\n            return true;")]),
    ("log: head only (before this change)", LOG,
     [('    fprintf(f, "raw_len=%zu raw=<<<%.*s…[truncated]>>> raw_tail=<<<%s>>>",\n'
       '            len, (int)head, raw, raw + from);',
       '    fprintf(f, "raw_len=%zu raw=<<<%.*s…[truncated]>>>", len, (int)head, raw);'
       ' (void)from;')]),
    ("log: tail may overlap the head", LOG,
     [("    size_t from = len - head > tail ? len - tail : head;",
       "    size_t from = len > tail ? len - tail : 0;")]),
    ("log: tail starts mid UTF-8 char", LOG,
     [("    while (from < len && ((unsigned char)raw[from] & 0xC0) == 0x80) from++;\n", "")]),
    ("log: short text gets a tail too", LOG,
     [("    if (len <= head) {", "    if (len < head) {")]),
]


def failed_checks(out):
    """Failed checks in one test's output (either report style); 1 when unreadable."""
    out = re.sub(r"\x1b\[[0-9;]*m", "", out)
    m = re.search(r"(\d+) failed", out)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)/(\d+) checks passed", out)
    return int(m.group(2)) - int(m.group(1)) if m else 1


def run_tests():
    """Rebuild the tests from scratch; return failed checks, or None when the build fails."""
    subprocess.run(["rm", "-f"] + TESTS)
    build = subprocess.run(["make", "USE_BLAS=1"] + TESTS, capture_output=True, text=True,
                           env=os.environ)
    if build.returncode != 0:
        return None
    return sum(failed_checks(subprocess.run(["./" + t], capture_output=True, text=True, errors="replace").stdout)
               for t in TESTS)


def main():
    base = run_tests()
    print(f"baseline: {base} failed")
    if base != 0:
        sys.exit("baseline must be green")
    survivors = 0
    for name, path, edits in MUTATIONS:
        src = open(path).read()
        mutated = src
        for old, new in edits:
            if mutated.count(old) != 1:
                sys.exit(f"mutation anchor not unique in {path}: {name}")
            mutated = mutated.replace(old, new)
        open(path, "w").write(mutated)
        try:
            failed = run_tests()
        finally:
            open(path, "w").write(src)
        verdict = "BUILD FAIL" if failed is None else f"{failed} failed"
        print(f"  {name:60s} {verdict}", flush=True)
        if not failed:
            survivors += 1
    print(f"restored; final: {run_tests()} failed; survivors: {survivors}")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
