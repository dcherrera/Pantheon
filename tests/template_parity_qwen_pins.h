/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file template_parity_qwen_pins.h
 * @brief The pinned differences between Pantheon's prompt rendering and the
 *        Qwen chat templates, one row per (fixture, template mode).  Included
 *        into PARITY_CASES in tests/test_template_parity_qwen.c; see there for
 *        the signature format.  Update together with the spec (§2.2).
 *
 *        Since card 1.8 (D3 merged tool results, D4 content |trim, D5 tojson
 *        argument values, D7 <tool_response> ids) every row holds only the
 *        three differences kept on purpose: D1 (T at an assistant turn,
 *        Pantheon has a think block the template drops), D2 (T at an assistant
 *        turn, the template has an empty think block), D6 (T19/19, the
 *        Qwen3.8 reasoning_effort line).  "" = byte- and token-identical.
 *
 * @status Complete.
 * @issues None
 * @todo None
 */
{ "loop_full_reasoning", "qwen36_default", "qwen36",
  "T1767/1767 T2266/2336 " },
{ "loop_full_reasoning", "qwen38_default", "qwen38",
  "T19/19 T2297/2088 " },
{ "loop_full_reasoning", "qwen38_ptfalse", "qwen38",
  "T19/19 T1976/1767 T2475/2336 " },
{ "loop_full_reasoning", "qwen38_medium", "qwen38",
  "T2088/2088 " },
{ "loop_full_reasoning", "qwen38_medium_ptfalse", "qwen38",
  "T1767/1767 T2266/2336 " },
{ "loop_boundary_dropped", "qwen36_default", "qwen36",
  "" },
{ "loop_boundary_dropped", "qwen38_default", "qwen38",
  "T19/19 T1976/1767 T2246/2018 T2513/2266 " },
{ "loop_boundary_dropped", "qwen38_ptfalse", "qwen38",
  "T19/19 " },
{ "loop_boundary_dropped", "qwen38_medium", "qwen38",
  "T1767/1767 T2037/2018 T2304/2266 " },
{ "loop_boundary_dropped", "qwen38_medium_ptfalse", "qwen38",
  "" },
{ "post_user_empty_reasoning", "qwen36_default", "qwen36",
  "T1767/1767 T2021/2002 " },
{ "post_user_empty_reasoning", "qwen38_default", "qwen38",
  "T19/19 T1976/1767 T2230/2002 " },
{ "post_user_empty_reasoning", "qwen38_ptfalse", "qwen38",
  "T19/19 T1976/1767 T2230/2002 " },
{ "post_user_empty_reasoning", "qwen38_medium", "qwen38",
  "T1767/1767 T2021/2002 " },
{ "post_user_empty_reasoning", "qwen38_medium_ptfalse", "qwen38",
  "T1767/1767 T2021/2002 " },
{ "content_whitespace", "qwen36_default", "qwen36",
  "T1767/1767 " },
{ "content_whitespace", "qwen38_default", "qwen38",
  "T19/19 " },
{ "content_whitespace", "qwen38_ptfalse", "qwen38",
  "T19/19 T1976/1767 " },
{ "content_whitespace", "qwen38_medium", "qwen38",
  "" },
{ "content_whitespace", "qwen38_medium_ptfalse", "qwen38",
  "T1767/1767 " },
{ "plain_chat_no_tools", "qwen36_default", "qwen36",
  "T153/153 " },
{ "plain_chat_no_tools", "qwen38_default", "qwen38",
  "T19/19 " },
{ "plain_chat_no_tools", "qwen38_ptfalse", "qwen38",
  "T19/19 T362/153 " },
{ "plain_chat_no_tools", "qwen38_medium", "qwen38",
  "" },
{ "plain_chat_no_tools", "qwen38_medium_ptfalse", "qwen38",
  "T153/153 " },
{ "tool_values_and_results", "qwen36_default", "qwen36",
  "" },
{ "tool_values_and_results", "qwen38_default", "qwen38",
  "T19/19 " },
{ "tool_values_and_results", "qwen38_ptfalse", "qwen38",
  "T19/19 " },
{ "tool_values_and_results", "qwen38_medium", "qwen38",
  "" },
{ "tool_values_and_results", "qwen38_medium_ptfalse", "qwen38",
  "" },
