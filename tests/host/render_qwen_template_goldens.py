#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""render_qwen_template_goldens.py: write the template-parity fixtures and
render them with the models' own chat templates (reference Jinja).

Writes, under tests/data/template_parity_qwen/:
  fixtures/<name>.json          OpenAI wire request bodies (what a client sends;
                                tool_calls[].function.arguments is a JSON string)
  goldens/<name>.<mode>.txt     the template's rendering, add_generation_prompt

Modes:
  qwen36_default     Qwen3.6 chat_template.jinja, no extra variables
  qwen38_default     Qwen3.8 GGUF template, no extra variables
                     (preserve_thinking undefined, reasoning_effort xhigh)
  qwen38_ptfalse     Qwen3.8, preserve_thinking=false
  qwen38_medium      Qwen3.8, reasoning_effort="medium" (adds no system text:
                     isolates the think-block rules from the xhigh preamble)
  qwen38_medium_ptfalse  Qwen3.8, reasoning_effort="medium", preserve_thinking=false

The Jinja environment is the one transformers' apply_chat_template builds
(ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks, loopcontrols,
tojson = json.dumps(ensure_ascii=False) without key sorting).  Tool-call
arguments are parsed from their JSON string into a dict first, as transformers,
vLLM and llama.cpp do (the Qwen3.8 template refuses a string).

Usage (from the repo root): python3 tests/host/render_qwen_template_goldens.py
The C test (tests/test_template_parity_qwen.c) reads the committed outputs; it
never runs Python.
"""

import copy
import json
import os
import sys

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

DATA_DIR = os.path.join("tests", "data", "template_parity_qwen")
TEMPLATES = {
    "qwen36": os.path.join(DATA_DIR, "qwen36_chat_template.jinja"),
    "qwen38": os.path.join(DATA_DIR, "qwen38_chat_template.jinja"),
}
MODES = {
    "qwen36_default": ("qwen36", {}),
    "qwen38_default": ("qwen38", {}),
    "qwen38_ptfalse": ("qwen38", {"preserve_thinking": False}),
    "qwen38_medium": ("qwen38", {"reasoning_effort": "medium"}),
    "qwen38_medium_ptfalse": ("qwen38", {"reasoning_effort": "medium",
                                         "preserve_thinking": False}),
}

TOOLS = [
    {"type": "function", "function": {
        "name": "read",
        "description": "Read a file — returns numbered lines.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path"},
            "offset": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "todo",
        "description": "Replace the todo list.",
        "parameters": {"type": "object", "properties": {
            "items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"]}}},
]

SYSTEM = "You are a coding agent working in potluck/.\nBe precise.\n"
USER = "Find EVERY root cause of the RNG test failure in potluck."
BOUNDARY = ("[context checkpoint] Your earlier reasoning was cleared to make room. "
            "Every tool call above (with its concl and keep) and every tool result "
            "is still here: do not redo work they show.\n"
            "Task: find every root cause of the RNG failure\n"
            "Last conclusion: A1[4] overflow @ potluck/tests/test_pl_rng.c:54\n"
            "Continue with: look for a second cause")


def call(call_id, name, args):
    """One OpenAI wire tool call (arguments as a JSON string)."""
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def assistant(content, calls, reasoning=None):
    msg = {"role": "assistant", "content": content, "tool_calls": calls}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return msg


def tool(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def loop_full_reasoning():
    """System + tools, one user message, a tool loop (reasoning on some turns, a
    two-call turn whose results are consecutive tool messages), the §4.3
    boundary user message right after tool results, then more loop turns.
    Every assistant turn that had reasoning carries it."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
        assistant(None, [call("c1", "read", {"path": "potluck/tests/test_pl_rng.c"})],
                  "\nThe test file first; the failure is in test_pl_rng.\n\n"),
        tool("c1", "50\tfloat A1[4];\n54\tpl_rng_fill(A1, rank * din);\n"),
        assistant(None, [call("c2", "bash", {"command": "make -C potluck test 2>&1 | tail -3"})]),
        tool("c2", "FAIL test_pl_rng: buffer overflow"),
        assistant("Two reads at once.",
                  [call("c3", "read", {"path": "potluck/src/pl_rng.c", "offset": 40}),
                   call("c4", "todo", {"items": ["find cause 1", "find cause 2"]})],
                  "A1 holds 4 floats but rank*din = 256 are written.\nCheck pl_rng.c."),
        tool("c3", "60\tfor (i = 0; i < n; i++) out[i] = next();"),
        tool("c4", "ok"),
        {"role": "user", "content": BOUNDARY},
        assistant(None, [call("c5", "read", {"path": "potluck/src/pl_seed.c"})],
                  "Second cause: look at seeding."),
        tool("c5", "12\tseed = time(NULL);"),
        assistant(None, [call("c6", "bash", {"command": "grep -n seed potluck/src/*.c"})],
                  "Seed from time() makes the test nondeterministic."),
        tool("c6", "potluck/src/pl_seed.c:12:seed = time(NULL);"),
    ]
    return {"model": "qwen", "tools": TOOLS, "messages": messages}


def loop_boundary_dropped():
    """The same conversation as a boundary request would send it: no
    reasoning_content on any assistant turn before the boundary user message
    (spec §4.1/§4.5 `as_sent`), reasoning kept after it."""
    request = loop_full_reasoning()
    seen_boundary = False
    for msg in request["messages"]:
        if msg["role"] == "user" and msg["content"].startswith("[context checkpoint]"):
            seen_boundary = True
        if msg["role"] == "assistant" and not seen_boundary:
            msg.pop("reasoning_content", None)
    return request


def post_user_empty_reasoning():
    """Assistant turns AFTER the last user message with absent and with empty
    reasoning (spec §2.2: the template's empty think block = collapse shape)."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
        assistant(None, [call("c1", "read", {"path": "potluck/tests/test_pl_rng.c"})]),
        tool("c1", "54\tpl_rng_fill(A1, rank * din);"),
        assistant(None, [call("c2", "bash", {"command": "make -C potluck test"})], ""),
        tool("c2", "FAIL test_pl_rng"),
        assistant(None, [call("c3", "read", {"path": "potluck/src/pl_rng.c"})],
                  "Now the implementation."),
        tool("c3", "60\tout[i] = next();"),
    ]
    return {"model": "qwen", "tools": TOOLS, "messages": messages}


def content_whitespace():
    """Leading/trailing whitespace the template |trims: a user message ending in
    a newline (a boundary note written with one), assistant text around a tool
    call, and a tool result wrapped in newlines."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER + "\n"},
        assistant("  Reading the test.\n", [call("c1", "read", {"path": "potluck/tests/test_pl_rng.c"})],
                  "Read it."),
        tool("c1", "\n54\tpl_rng_fill(A1, rank * din);\n\n"),
        {"role": "user", "content": BOUNDARY + "\n"},
        assistant(None, [call("c2", "bash", {"command": "make -C potluck test"})], "Run it."),
        tool("c2", "FAIL test_pl_rng"),
    ]
    return {"model": "qwen", "tools": TOOLS, "messages": messages}


def tool_values_and_results():
    """Non-string argument values the template writes with tojson (integer,
    float, bool, null, nested object, array, non-ASCII string) and a three-call
    turn whose results are three consecutive tool messages (one user turn in the
    template), with results ending in punctuation, a newline, and whitespace."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
        assistant(None, [call("c1", "bash", {"command": "make test", "timeout": 30,
                                             "ratio": 0.1, "verbose": True, "cwd": None,
                                             "env": {"SEED": "1", "dims": [4, 256]},
                                             "note": "naïve — ok"}),
                         call("c2", "read", {"path": "potluck/src/pl_rng.c", "offset": 40}),
                         call("c3", "todo", {"items": ["cause 1", {"id": 2, "done": False}]})],
                  "Run, read and plan at once."),
        tool("c1", '{"exit": 1, "tail": "FAIL test_pl_rng"}\n'),
        tool("c2", "60\tout[i] = next();"),
        tool("c3", "  ok  "),
        assistant("Next.", [call("c4", "read", {"path": "potluck/src/pl_seed.c"})],
                  "Now the seed."),
        tool("c4", "12\tseed = time(NULL);"),
    ]
    return {"model": "qwen", "tools": TOOLS, "messages": messages}


def plain_chat_no_tools():
    """No tools: the plain system turn, and an assistant turn with reasoning
    before the last user message (dropped by Qwen3.6 / Qwen3.8 ptfalse)."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "What seeds pl_rng?"},
        {"role": "assistant", "content": "time(NULL), in pl_seed.c.",
         "reasoning_content": "pl_seed.c line 12 seeds from time()."},
        {"role": "user", "content": "Is that deterministic?"},
    ]
    return {"model": "qwen", "messages": messages}


FIXTURES = {
    "loop_full_reasoning": loop_full_reasoning,
    "loop_boundary_dropped": loop_boundary_dropped,
    "post_user_empty_reasoning": post_user_empty_reasoning,
    "content_whitespace": content_whitespace,
    "plain_chat_no_tools": plain_chat_no_tools,
    "tool_values_and_results": tool_values_and_results,
}


def make_environment():
    """The jinja environment transformers' apply_chat_template uses."""
    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def to_template_messages(messages):
    """OpenAI wire -> template input: tool-call arguments parsed to dicts."""
    converted = copy.deepcopy(messages)
    for msg in converted:
        for tool_call in msg.get("tool_calls") or []:
            fn = tool_call["function"]
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"])
    return converted


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as out:
        out.write(text)


def main():
    if not os.path.isdir(DATA_DIR):
        sys.exit("run from the repo root")
    env = make_environment()
    templates = {}
    for key, path in TEMPLATES.items():
        with open(path, encoding="utf-8") as handle:
            templates[key] = env.from_string(handle.read())
    for name, build in FIXTURES.items():
        request = build()
        write_text(os.path.join(DATA_DIR, "fixtures", name + ".json"),
                   json.dumps(request, indent=1, ensure_ascii=False) + "\n")
        for mode, (template_key, variables) in MODES.items():
            rendered = templates[template_key].render(
                messages=to_template_messages(request["messages"]),
                tools=request.get("tools"), add_generation_prompt=True, **variables)
            write_text(os.path.join(DATA_DIR, "goldens", f"{name}.{mode}.txt"), rendered)
            print(f"{name}.{mode}: {len(rendered.encode('utf-8'))} B")


if __name__ == "__main__":
    main()
