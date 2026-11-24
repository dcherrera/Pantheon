/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_emitter.c
 * @brief Implementation of the reasoning/content/tool-call stream splitter
 *        (see serve_emitter.h).  Depends only on libc so it is unit-testable in
 *        isolation from the engine/CUDA forward path.
 */

#include "serve_emitter.h"

#include <stdlib.h>
#include <string.h>

#include "think_close_rule.h"

static void tc_append(text_channel *tc, const char *s, size_t n)
{
    if (n == 0) return;
    if (tc->len + n + 1 > tc->cap) {
        size_t cap = tc->cap ? tc->cap : 256;
        while (cap < tc->len + n + 1) cap *= 2;
        char *grown = realloc(tc->data, cap);
        if (!grown) return;
        tc->data = grown;
        tc->cap = cap;
    }
    memcpy(tc->data + tc->len, s, n);
    tc->len += n;
    tc->data[tc->len] = '\0';
}

static const char TOOL_OPEN[]   = "<tool_call>";
static const char FN_OPEN[]     = "<function=";
/* MiniCPM5's bare opener: an attribute-style function tag "<function name=...>"
 * (SGLang's minicpm5 detector, bot_token "<function"). Distinct from the Qwen
 * "<function=" form above by the space after "function". Armed only when tools
 * are present, exactly like FN_OPEN. */
static const char FN_OPEN_ATTR[] = "<function ";
/* Ornith's fenced-JSON opener: a "```json" markdown fence.  Armed only for the
 * ORNITH_JSON dialect + tools; diverts the whole fenced block into the raw tool
 * channel so tc_parse_tool_calls sees the JSON object. */
static const char JSON_FENCE[]  = "```json";

void em_init(emitter *em, sg_delta_fn cb, void *ud,
             char **stops, int n_stop, bool has_tools, tc_dialect dialect)
{
    memset(em, 0, sizeof(*em));
    em->in_think = 1;
    em->has_tools = has_tools;
    em->dialect = dialect;
    em->cb = cb;
    em->ud = ud;
    em->stops = stops;
    em->n_stop = n_stop;
}

void em_set_think_close_token(emitter *em)
{
    em->think_close_token = true;
}

void em_set_tool_tap(emitter *em, em_tool_tap_fn tap, void *ud)
{
    em->tool_tap = tap;
    em->tool_tap_ud = ud;
}

/* Append to the raw tool channel and hand the same bytes to the tap. */
static void em_tool_append(emitter *em, const char *s, size_t n)
{
    tc_append(&em->tool, s, n);
    if (em->tool_tap && n > 0) em->tool_tap(s, n, em->tool_tap_ud);
}

void em_free(emitter *em)
{
    free(em->carry);
    free(em->reasoning.data);
    free(em->content.data);
    free(em->tool.data);
}

/* Longest k (k < strlen(needle)) such that the last k bytes of hay equal the
 * first k bytes of needle — i.e. how much tail could be the start of `needle`. */
static size_t suffix_prefix_overlap(const char *hay, size_t hay_len,
                                    const char *needle)
{
    size_t nlen = strlen(needle);
    size_t max = hay_len < nlen - 1 ? hay_len : nlen - 1;
    for (size_t k = max; k > 0; k--)
        if (memcmp(hay + hay_len - k, needle, k) == 0)
            return k;
    return 0;
}

/* True when the char immediately after "<function=" is a plausible function
 * name start (not '>', whitespace or end): guards against capturing a stray
 * "<function=" fragment (e.g. a literal mention in prose) as a tool call. */
static bool fn_open_is_plausible(const char *fn)
{
    const char *name = fn + strlen(FN_OPEN);
    char c = *name;
    return c != '\0' && c != '>' && c != ' ' && c != '\t' &&
           c != '\n' && c != '\r';
}

/* Bytes at the END of [buf,buf+len) that form a fully-present bare "<function="
 * opener which is NOT yet followed by a name char — i.e. the opener straddled a
 * token split right after the '=' (the live MiniCPM/grammar-forced case: the
 * model emits "<function" then "=" then "dispatch_agent" as separate tokens).
 * em_find_opener's fn_open_is_plausible gate rejects a "<function=" with nothing
 * after it, so without holding these bytes back the complete opener would flush
 * to content and the following "<name>" would never re-trigger the divert.  The
 * partial-prefix cases (buffer ends mid-"<function") are already covered by
 * suffix_prefix_overlap; this covers the complete-but-unfollowed tail.  Returns 0
 * when the tail is not such an opener (so the normal hold-back applies). */
static size_t trailing_incomplete_fn_open(const char *buf, size_t len)
{
    size_t flen = strlen(FN_OPEN);   /* "<function=" */
    if (len < flen) return 0;
    /* The buffer must END exactly with "<function=" (nothing after the '=', or
     * the plausibility gate would have accepted it as a real opener). */
    if (memcmp(buf + len - flen, FN_OPEN, flen) != 0) return 0;
    return flen;
}

/* True when the bytes at `p` (a '{') plausibly begin an Ornith JSON call object:
 * after the '{' and any whitespace, the next token is a "function" or "name"
 * key.  Guards against diverting a stray '{' in normal prose.  When the buffer
 * ends before the key is decided, returns false (the caller holds the '{' back
 * via the JSON-opener hold-back so the next feed re-decides). */
static bool json_brace_is_plausible(const char *p, const char *bufend)
{
    const char *q = p + 1;
    while (q < bufend && (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r'))
        q++;
    if (q >= bufend || *q != '"') return false;
    q++;
    if ((size_t)(bufend - q) >= strlen("function") &&
        memcmp(q, "function", strlen("function")) == 0)
        return true;
    if ((size_t)(bufend - q) >= strlen("name") &&
        memcmp(q, "name", strlen("name")) == 0)
        return true;
    return false;
}

/* Length of a trailing tail beginning at the LAST '{' in [buf,buf+len) whose
 * plausibility is still UNDECIDED — the buffer ends before we can see whether a
 * "function"/"name" key follows.  Such a '{' must be held back so the next feed
 * can re-decide (mirrors trailing_incomplete_fn_open for the XML path).  A '{'
 * already decided plausible is diverted by em_find_ornith_opener; one clearly
 * NOT a call (a non-quote, non-whitespace char follows) is not held.  Returns 0
 * when there is nothing to hold. */
static size_t trailing_undecided_json_brace(const char *buf, size_t len)
{
    const char *bufend = buf + len;
    /* Only the final '{' can straddle the end; earlier ones are fully present. */
    const char *brace = NULL;
    for (const char *p = memchr(buf, '{', len); p;
         p = memchr(p + 1, '{', (size_t)(bufend - (p + 1))))
        brace = p;
    if (!brace) return 0;
    const char *q = brace + 1;
    while (q < bufend && (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r'))
        q++;
    if (q >= bufend) return (size_t)(bufend - brace);   /* only whitespace so far */
    if (*q != '"') return 0;                            /* not a key → not a call */
    q++;
    /* q points just past the opening quote of the first key.  If the remaining
     * bytes are a strict prefix of "function"/"name", the key is not yet fully
     * present → hold.  A complete/decided key was handled by the opener finder. */
    size_t rem = (size_t)(bufend - q);
    if (rem == 0) return (size_t)(bufend - brace);
    if (rem < strlen("function") &&
        memcmp(q, "function", rem) == 0)
        return (size_t)(bufend - brace);
    if (rem < strlen("name") &&
        memcmp(q, "name", rem) == 0)
        return (size_t)(bufend - brace);
    return 0;
}

/* Find the Ornith JSON opener (```json fence, else a plausible bare '{') in
 * [buf,buf+len).  On a hit sets *at + *open_len and returns true.  The fence is
 * preferred; a bare '{' is only taken when json_brace_is_plausible. */
static bool em_find_ornith_opener(const char *buf, size_t len,
                                  size_t *at, size_t *open_len)
{
    char *fence = memmem(buf, len, JSON_FENCE, strlen(JSON_FENCE));
    if (fence) { *at = (size_t)(fence - buf); *open_len = strlen(JSON_FENCE); return true; }
    const char *bufend = buf + len;
    for (const char *p = memchr(buf, '{', len); p;
         p = memchr(p + 1, '{', (size_t)(bufend - (p + 1)))) {
        if (json_brace_is_plausible(p, bufend)) {
            *at = (size_t)(p - buf);
            *open_len = 1;
            return true;
        }
    }
    return false;
}

/* Find the earliest tool-call opener in [buf,buf+len).  "<tool_call>" is always
 * considered; "<function=" only when `has_tools` and the tag has a plausible
 * "<function=NAME>" shape.  For the ORNITH_JSON dialect the XML openers are NOT
 * watched — a ```json fence / plausible bare '{' is diverted instead.  On a hit
 * sets *at (offset) + *open_len (opener length) and returns true; whichever
 * opener appears first wins. */
static bool em_find_opener(const emitter *em, const char *buf, size_t len,
                           size_t *at, size_t *open_len)
{
    if (em->dialect == TC_DIALECT_ORNITH_JSON) {
        if (!em->has_tools) return false;
        return em_find_ornith_opener(buf, len, at, open_len);
    }

    bool found = false;
    size_t best = len;
    size_t best_len = 0;

    char *tc = memmem(buf, len, TOOL_OPEN, strlen(TOOL_OPEN));
    if (tc) {
        best = (size_t)(tc - buf);
        best_len = strlen(TOOL_OPEN);
        found = true;
    }

    if (em->has_tools) {
        /* Scan for the first plausibly-shaped "<function=" — skip stray
         * fragments (no name char after '=') and keep looking. */
        const char *p = buf;
        size_t rem = len;
        for (;;) {
            char *fn = memmem(p, rem, FN_OPEN, strlen(FN_OPEN));
            if (!fn) break;
            if (fn_open_is_plausible(fn)) {
                size_t off = (size_t)(fn - buf);
                if (!found || off < best) {
                    best = off;
                    best_len = strlen(FN_OPEN);
                    found = true;
                }
                break;
            }
            p = fn + strlen(FN_OPEN);
            rem = len - (size_t)(p - buf);
        }

        /* MiniCPM5's attribute-style "<function name=...>" opener. The tag is
         * self-identifying (a space then an attribute), so no plausibility gate
         * is needed beyond the literal. Diverts at the "<function" so the raw
         * `tool` channel captures the whole "<function ...>...</function>". */
        char *fa = memmem(buf, len, FN_OPEN_ATTR, strlen(FN_OPEN_ATTR));
        if (fa) {
            size_t off = (size_t)(fa - buf);
            if (!found || off < best) {
                best = off;
                best_len = strlen(FN_OPEN_ATTR);
                found = true;
            }
        }
    }

    if (found) { *at = best; *open_len = best_len; }
    return found;
}

/* Emit `n` bytes of `s` to a channel: left-trim leading whitespace until the
 * channel has real content, accumulate, and forward the delta to the callback. */
static void em_emit(emitter *em, int is_reasoning, const char *s, size_t n)
{
    text_channel *tc = is_reasoning ? &em->reasoning : &em->content;
    bool *opened = is_reasoning ? &em->reasoning_open : &em->content_open;
    if (!*opened) {
        while (n > 0 && (*s == '\n' || *s == '\r' || *s == ' ' || *s == '\t')) {
            s++;
            n--;
        }
        if (n == 0) return;
        *opened = true;
    }
    tc_append(tc, s, n);
    if (em->cb) {
        char *tmp = malloc(n + 1);
        if (tmp) {
            memcpy(tmp, s, n);
            tmp[n] = '\0';
            /* OpenAI reports logprobs for the ANSWER only, so a reasoning delta
             * carries none — but it still drains the queue, because the tokens
             * behind it were reasoning tokens and must not be re-attributed to
             * the next content delta. */
            if (is_reasoning)
                em->cb(tmp, 1, NULL, 0, em->ud);
            else
                em->cb(tmp, 0, em->lp_pending, em->n_lp_pending, em->ud);
            em->n_lp_pending = 0;
            free(tmp);
        }
    }
}

void em_note_token_logprob(emitter *em, const sg_token_logprob *lp)
{
    if (!em || !lp) return;
    if (em->n_lp_pending >= EM_LOGPROB_PENDING_MAX) {
        /* Drop the oldest: the queue only overflows inside a long diverted
         * region (a tool call), whose tokens OpenAI does not report anyway. */
        memmove(em->lp_pending, em->lp_pending + 1,
                (size_t)(EM_LOGPROB_PENDING_MAX - 1) * sizeof(em->lp_pending[0]));
        em->n_lp_pending = EM_LOGPROB_PENDING_MAX - 1;
    }
    em->lp_pending[em->n_lp_pending++] = lp;
}

static void em_carry_set(emitter *em, const char *s, size_t n)
{
    if (n + 1 > em->carry_cap) {
        char *grown = realloc(em->carry, n + 1);
        if (!grown) return;
        em->carry = grown;
        em->carry_cap = n + 1;
    }
    memmove(em->carry, s, n);
    em->carry_len = n;
    if (em->carry) em->carry[n] = '\0';
}

/* Find the earliest complete stop string in [buf,buf+len); returns its offset
 * and length, or sets *found=false. */
static bool find_stop(emitter *em, const char *buf, size_t len,
                      size_t *at, size_t *stop_len)
{
    bool found = false;
    size_t best = len;
    for (int i = 0; i < em->n_stop; i++) {
        const char *s = em->stops[i];
        if (!s || !*s) continue;
        size_t slen = strlen(s);
        for (size_t j = 0; j + slen <= len; j++) {
            if (memcmp(buf + j, s, slen) == 0) {
                if (j < best) { best = j; *stop_len = slen; found = true; }
                break;
            }
        }
    }
    if (found) *at = best;
    return found;
}

void em_feed(emitter *em, const char *piece)
{
    if (em->stopped) return;

    /* Token mode: the close is a whole piece, so nothing in <think> is held back
     * and a "</think>" spelled out of ordinary tokens is reasoning text. */
    if (em->in_think && em->think_close_token) {
        if (think_close_token_piece(piece, true))
            em->in_think = 0;
        else
            em_emit(em, 1, piece, strlen(piece));
        return;
    }

    /* Combine carry + new piece into a working buffer. */
    size_t plen = strlen(piece);
    size_t wlen = em->carry_len + plen;
    char *work = malloc(wlen + 1);
    if (!work) return;
    if (em->carry_len) memcpy(work, em->carry, em->carry_len);
    memcpy(work + em->carry_len, piece, plen);
    work[wlen] = '\0';

    size_t pos = 0;
    while (pos < wlen && !em->stopped) {
        if (em->in_think) {
            char *marker = memmem(work + pos, wlen - pos,
                                  THINK_CLOSE_TEXT, strlen(THINK_CLOSE_TEXT));
            if (marker) {
                size_t idx = (size_t)(marker - work);
                em_emit(em, 1, work + pos, idx - pos);
                pos = idx + strlen(THINK_CLOSE_TEXT);
                em->in_think = 0;
                continue;
            }
            size_t hold = suffix_prefix_overlap(work + pos, wlen - pos, THINK_CLOSE_TEXT);
            em_emit(em, 1, work + pos, (wlen - pos) - hold);
            em_carry_set(em, work + wlen - hold, hold);
            pos = wlen;
        } else if (em->in_tool) {
            /* Inside a tool call: capture everything raw for structured parsing;
             * never route it to the content channel or the delta callback (the
             * tap, when attached, streams it as tool-call deltas). */
            em_tool_append(em, work + pos, wlen - pos);
            pos = wlen;
        } else {
            /* Content channel: a tool-call opener ("<tool_call>" always, or a
             * bare "<function=" when tools are armed) switches to tool capture;
             * a stop string ends generation.  Whichever comes first wins. */
            size_t open_at = 0, open_len = 0;
            bool has_open = em_find_opener(em, work + pos, wlen - pos,
                                           &open_at, &open_len);
            size_t at = 0, slen = 0;
            bool has_stop = find_stop(em, work + pos, wlen - pos, &at, &slen);
            if (has_open && !(has_stop && at < open_at)) {
                size_t idx = pos + open_at;   /* offset of the opener in `work` */
                (void)open_len;
                em_emit(em, 0, work + pos, idx - pos);
                em_tool_append(em, work + idx, wlen - idx);
                em->in_tool = 1;
                em->carry_len = 0;   /* tail is now inside `work`; don't re-prepend */
                pos = wlen;
                continue;
            }
            if (has_stop) {
                em_emit(em, 0, work + pos, at);
                em->stopped = true;
                em->carry_len = 0;
                pos = wlen;
                continue;
            }
            /* Hold back the longest tail that could start a stop string or a
             * tool-call opener, so none is missed across a token split. The
             * bare "<function=" opener is only held back when tools are armed
             * (it is only ever acted on in that case). */
            size_t hold = 0;
            if (em->dialect == TC_DIALECT_ORNITH_JSON) {
                if (em->has_tools) {
                    /* Hold a partial "```json" fence prefix and a trailing '{'
                     * whose plausibility (its "function"/"name" key) has not yet
                     * arrived, so a fence/object split across tokens still
                     * diverts on the next feed. */
                    hold = suffix_prefix_overlap(work + pos, wlen - pos, JSON_FENCE);
                    size_t htj = trailing_undecided_json_brace(work + pos, wlen - pos);
                    if (htj > hold) hold = htj;
                }
            } else {
                hold = suffix_prefix_overlap(work + pos, wlen - pos, TOOL_OPEN);
                if (em->has_tools) {
                    size_t hfn = suffix_prefix_overlap(work + pos, wlen - pos, FN_OPEN);
                    if (hfn > hold) hold = hfn;
                    size_t hfa = suffix_prefix_overlap(work + pos, wlen - pos, FN_OPEN_ATTR);
                    if (hfa > hold) hold = hfa;
                    /* A fully-present but unfollowed "<function=" at the buffer end
                     * (opener split right after '=') must be held too — the
                     * plausibility gate needs the next name char to accept it. */
                    size_t htf = trailing_incomplete_fn_open(work + pos, wlen - pos);
                    if (htf > hold) hold = htf;
                }
            }
            for (int i = 0; i < em->n_stop; i++) {
                if (!em->stops[i] || !*em->stops[i]) continue;
                size_t h = suffix_prefix_overlap(work + pos, wlen - pos, em->stops[i]);
                if (h > hold) hold = h;
            }
            em_emit(em, 0, work + pos, (wlen - pos) - hold);
            em_carry_set(em, work + wlen - hold, hold);
            pos = wlen;
        }
    }
    free(work);
}

void em_flush(emitter *em)
{
    if (em->stopped || em->carry_len == 0) return;
    em_emit(em, em->in_think ? 1 : 0, em->carry, em->carry_len);
    em->carry_len = 0;
}
