#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""Red proof for the field_patterns tests.

Applies one mutation at a time to the sources, deletes the test binaries (macOS
make 3.81 has 1 s mtime granularity), rebuilds, runs the tests and counts failed
tests, then restores the source.  Every mutation must turn at least one test
red.  Run from the repo root:  python3 tests/red_proof_field_patterns.py
"""
import re
import subprocess
import sys

FP = "src/serve/field_pattern.c"
XML = "src/serve/tool_grammar.c"
JSON = "src/serve/tool_grammar_json.c"
REQ = "src/serve/field_patterns_request.c"
CHAT = "src/serve/openai_chat.c"
TESTS = ["tests/test_field_pattern", "tests/test_tool_grammar_field_patterns",
         "tests/test_tool_grammar_json_field_patterns", "tests/test_chat_request"]

# (name, file, old, new)
MUTATIONS = [
    ("regex: '.' takes a newline", FP,
     "        cls.w['\\n' >> 6] &= ~((uint64_t)1 << ('\\n' & 63));\n", ""),
    ("regex: negated class not negated", FP,
     "    if (negate) class_negate(&cls);\n", ""),
    ("regex: range is its low end only", FP,
     "        class_add_range(&cls, (unsigned char)lo, (unsigned char)hi);",
     "        class_add(&cls, (unsigned char)lo);"),
    ("regex: backwards range accepted", FP,
     "if (hi == -2 || hi < lo)", "if (hi == -2)"),
    ("regex: empty class accepted", FP,
     "    if (class_empty(&cls)) {", "    if (0) {"),
    ("regex: \\s misses \\v", FP,
     "class_add(out, '\\f'); class_add(out, '\\v');", "class_add(out, '\\f');"),
    ("regex: upper-case shorthand not negated", FP,
     "    if (letter >= 'A' && letter <= 'Z') class_negate(out);\n", ""),
    ("regex: stacked quantifier accepted", FP,
     "    if (ps->pos < ps->len && strchr(\"*+?{\", ps->src[ps->pos])) {",
     "    if (0) {"),
    ("regex: leading ^ is a literal", FP,
     "    if (len > 0 && src[0] == '^') ps.pos = 1;\n", ""),
    ("regex: trailing $ refused", FP,
     "if (c == '$' && ps->pos + 1 == ps->len && ps->depth == 0)", "if (0)"),
    ("regex: concat links nothing", FP,
     "    link_states(p, &acc->last, &next->first);\n", ""),
    ("regex: alternation drops its right side", FP,
     "        set_union(&out->first, &right.first);\n        set_union(&out->last, &right.last);",
     "        (void)right;"),
    ("regex: open repeat does not loop", FP,
     "        if (n->max < 0 && i == copies - 1) link_states(b->out, &copy.last, &copy.first);\n", ""),
    ("regex: optional copies required", FP,
     "        if (i >= n->min) copy.nullable = true;\n", ""),
    ("regex: nullable pattern rejects empty", FP,
     "        if (whole.nullable) set_add(&p->accept, 0);\n", ""),
    ("regex: state limit one short", FP,
     "if (q >= FP_MAX_STATES) {", "if (q >= FP_MAX_STATES - 1) {"),
    ("regex: step ignores the byte", FP,
     "        next.w[i] &= p->byte_mask[byte].w[i];\n", ""),
    ("xml: pattern never stashed", XML,
     "            st->val_pattern = meta ? meta->pattern : NULL;",
     "            st->val_pattern = NULL;"),
    ("xml: close not judged", XML,
     "                if (!value_pattern_closes(st)) { st->phase = TG_DEAD; return; }\n", ""),
    ("xml: no mid-value masking", XML,
     "    if (fp_set_empty(&st->pat_live) && !fp_accepting(st->val_pattern, closable))",
     "    if (0)"),
    ("xml: leading whitespace fed", XML,
     "    if (st->pat_begun || !tg_is_ws(c)) {", "    if (1) {"),
    ("xml: trailing whitespace not trimmed", XML,
     "        if (!tg_is_ws(c)) st->pat_trimmed = st->pat_live;",
     "        st->pat_trimmed = st->pat_live;"),
    ("xml: terminator start not frozen", XML,
     "            if (st->val_term == 0) value_pattern_mark_close(st);\n", ""),
    ("xml: terminator restart not frozen", XML,
     "        if (c == term[0]) value_pattern_mark_close(st);\n", ""),
    ("xml: terminator judged on the live walk", XML,
     "st->val_term > 0 ? &st->pat_at_close : &st->pat_trimmed", "&st->pat_trimmed"),
    ("xml: value start not reset", XML,
     "                st->pat_live = st->pat_trimmed = fp_start(st->val_pattern);\n", ""),
    ("xml: non-string keys take patterns", XML,
     "            if (!src || !key_takes_pattern(meta)) continue;",
     "            if (!src) continue;"),
    ("xml: unknown types count as string", XML,
     "                meta->string_typed = !type || strcmp(type, \"string\") == 0;",
     "                meta->string_typed = true;"),
    ("json: pattern never stashed", JSON,
     "            st->val_pattern = t->patterns ? t->patterns[idx] : NULL;",
     "            st->val_pattern = NULL;"),
    ("json: closing quote not judged", JSON,
     "    if (c == '\"') return fp_accepting(st->val_pattern, &st->pat_live);",
     "    if (c == '\"') return true;"),
    ("json: escapes fed undecoded", JSON,
     "               string_pattern_byte(st, (unsigned char)byte);",
     "               string_pattern_byte(st, (unsigned char)c);"),
    ("json: invalid escape accepted", JSON,
     "    default:  return -1;\n    }\n}", "    default:  return (unsigned char)c;\n    }\n}"),
    ("json: surrogate pair not joined", JSON,
     "    if (st->pat_high && unit >= 0xDC00 && unit <= 0xDFFF) {", "    if (0) {"),
    ("json: lone high surrogate dropped", JSON,
     "    return !high || string_pattern_code_point(st, high);", "    return true;"),
    ("json: non-hex \\u digit accepted", JSON,
     "        if (v < 0) return false;", "        if (v < 0) v = 0;"),
    ("json: 2-byte UTF-8 miscoded", JSON,
     "        bytes[0] = (unsigned char)(0xC0 | (cp >> 6));",
     "        bytes[0] = (unsigned char)(0xC0 | (cp >> 5));"),
    ("json: value start not reset", JSON,
     "        st->pat_live = fp_start(st->val_pattern);\n", ""),
    ("json: unknown types count as string", JSON,
     "        t->string_typed[idx] = !type || strcmp(type, \"string\") == 0;",
     "        t->string_typed[idx] = true;"),
    ("request: pattern not compiled", REQ,
     "            if (!compiled) {", "            if (!pattern) {"),
    ("request: tool entry shape unchecked", REQ,
     "        if (jr_kind(props) != JR_OBJECT) {", "        if (0) {"),
    ("parse: field not validated", CHAT,
     "    if (field_patterns_validate(field_patterns, fp_err, sizeof(fp_err)) != 0) {",
     "    if (0) {"),
    ("parse: field not forwarded", CHAT,
     "(req->tools && jr_kind(field_patterns) == JR_OBJECT) ? field_patterns : NULL;",
     "NULL;"),
]


def run_tests():
    """Rebuild the tests from scratch; return failed tests, or None when the build fails."""
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
