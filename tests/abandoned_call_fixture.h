/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file abandoned_call_fixture.h
 * @brief The raw tool text of an ABANDONED call: the shape the
 *        [tool_degenerate] diagnostic in openai_chat.c collect_tool_calls
 *        reports when a model opens a parameter and never fills or closes it.
 *        Reproduces an emission observed in real serving byte-for-byte; it
 *        carries no user content.
 *
 *        WHAT IT SHOWS.  The model opened a `read`, opened <parameter=path>,
 *        wrote NO value at all, and then jumped straight to </function> --
 *        skipping </parameter> entirely.  tc_parse_tool_calls found no closed
 *        parameter, dropped the empty value, and produced `{}`, which is
 *        indistinguishable at the client from a model that emitted an empty
 *        argument.  This is NOT the reasoning-budget force-close (zero FORCED
 *        close events in the same log), NOT string mangling, and NOT the
 *        frame-capture misroute: it was seen on a shallow prompt (~2,858
 *        tokens), mode=REQUIRED, dialect=XML.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TEST_ABANDONED_CALL_FIXTURE_H
#define PN_TEST_ABANDONED_CALL_FIXTURE_H

/* The raw tool channel text, 57 chars.  Logged as `tool=read args={}`. */
static const char FX_ABANDONED_READ_PATH[] =
    "<tool_call>\n<function=read>\n<parameter=path>\n\n</function>";

#endif /* PN_TEST_ABANDONED_CALL_FIXTURE_H */
