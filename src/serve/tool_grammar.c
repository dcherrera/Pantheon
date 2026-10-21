/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Tool-call decoding grammar — a char-level FSM over the model's TRAINED XML
 * tool-call emission, with NAME/KEY sets drawn from the request's tools schema.
 * See tool_grammar.h for the contract and the reasoning/prose composition.
 *
 * Pure C, libc-only.  No engine/CUDA deps → unit-tested standalone.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * State model (char-consuming)
 * ────────────────────────────────────────────────────────────────────────────
 * The walk is a phase machine over a fixed XML skeleton.  Every phase consumes
 * either a fixed literal (matched char-by-char via `lit`/`lit_pos`), or a token
 * from a set (a tool NAME / a parameter KEY, accumulated into `name_acc` and
 * matched against the set), or free VALUE text terminated by "</parameter>".
 *
 *   TG_FREE      : arbitrary text before an opener.  The model reasons / answers
 *                  in prose here — UNCONSTRAINED in AUTO mode.  On recognizing a
 *                  full opener ("<tool_call>" or "<function=") it enters the
 *                  skeleton; any other text keeps it FREE, so a prose answer is
 *                  never blocked and grammar-on never forces a spurious call.
 *                  In FORCED mode, once reasoning has ended (tg_reasoning_end),
 *                  TG_FREE only accepts opener-leading chars — any other char
 *                  goes DEAD, so the opener is compelled (no prose/JSON escape).
 *   TG_TC_NL     : the "\n" right after "<tool_call>"        (wrapped form)
 *   TG_FUNC_OPEN : the literal "<function="
 *   TG_FUNC_NAME : a NAME from the tool-name set, then ">"
 *   TG_FUNC_NL   : the "\n" after "<function=NAME>"
 *   TG_BODY      : a body boundary — EITHER "<parameter=" OR "</function>",
 *                  with the "\n" separators between them skipped
 *   TG_PARAM_LIT : the remainder of the literal "<parameter="
 *   TG_PARAM_KEY : a KEY from this tool's param set, then ">"
 *   TG_PARAM_NL  : the "\n" after "<parameter=KEY>"
 *   TG_VALUE     : free VALUE text up to "</parameter>" — the SAME bytes the
 *                  parser ends a value on, so the two cannot disagree about
 *                  where a value ended.  DELIBERATELY free otherwise: see
 *                  "FRAME CAPTURE" below — a nested opener is NOT killed here,
 *                  it is read back by the parser.
 *   TG_VALUE_JSON: VALUE for a key whose schema `type` is "array"/"object" —
 *                  NOT free text: one well-formed JSON value of that container
 *                  kind (json_value_grammar.h), and only once it CLOSES does
 *                  "</parameter>" become legal.  These are the values
 *                  write_typed_value splices RAW into the call's `arguments`,
 *                  so a malformed one breaks the whole call, not one argument.
 *   TG_VALUE_ENUM: VALUE for a key with a schema `enum` — NOT free text: a
 *                  closed alternation over the listed literals, matched by the
 *                  same prefix walk as NAME/KEY, then the padded
 *                  "\n</parameter>\n" terminator (a closed set cannot desync)
 *   TG_FUNC_CLOSE: the literal "</function>\n"
 *   TG_TC_CLOSE  : the literal "</tool_call>"                (wrapped form)
 *   TG_DONE      : a complete call emitted
 *   TG_DEAD      : invalid prefix (only ever reached on a clone during a check)
 *
 * Set matching: as NAME/KEY chars arrive we accumulate them in name_acc and, on
 * each char, require that SOME set member has that char at the next position
 * (else DEAD).  The terminating '>' is accepted only if name_acc equals a
 * member EXACTLY — so unknown names/keys are impossible.  name_acc lives IN the
 * state, so a name split across two token boundaries still matches.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * FRAME CAPTURE — the injected-close hazard, and why TG_VALUE stays free text
 * ────────────────────────────────────────────────────────────────────────────
 * KNOWN DEFECT, cause side.  This grammar cannot tell a "</think>" the MODEL
 * chose from one the SERVE injected.  When a turn exhausts its per-model
 * `reasoning_budget`, the serve force-decodes the close itself — batch_sched.c
 * driver_reason_step → driver_grammar_advance_injected (batched) and
 * serve_generation.c sg_run_completion (single-user) — and that text arrives
 * here through tg_advance like any other.  TG_FREE's think_term matcher fires,
 * reasoning_ended is set, and from the very next char FORCED mode compels the
 * opener.  The model is mid-thought: it emits the compelled
 * "<function=NAME>\n<parameter=KEY>\n" and then writes what it was already going
 * to write — including the "</think>" it had not reached yet — into the
 * free-text VALUE:
 *
 *     <parameter=pattern>
 *     </think>\n\n<function=read>\n<parameter=path>\nlumen/serve.log
 *     </parameter>
 *
 * (a recorded native-XML capture, nex-n2-mini, a session that never wedged).  Note the
 * captured text is a NATIVE call in THIS grammar's own dialect — the capture is
 * about being cut off, not about dialect drift.
 *
 * The obvious patch is to make a nested "<tool_call>"/"</think>" inside TG_VALUE
 * off-grammar.  It is NOT done, for two reasons:
 *
 *   1. Masking those tokens does not restore the model's intent — it deletes the
 *      only evidence of it.  The model is still compelled and still mid-thought,
 *      so it just writes a differently-corrupted value; the parser then has
 *      nothing to read back (tool_calling.c recover_captured_call).
 *   2. Reaching TG_DEAD mid-VALUE is actively dangerous here: tg_mask_active()
 *      reports a DEAD walk INACTIVE, which RELEASES the constraint for the rest
 *      of the generation — the same failure mode the reasoning-budget path at
 *      TG_FREE goes out of its way to avoid.
 *
 * So this phase stays permissive on purpose, and the detector lives in the
 * parser, which sees whole values and can act on them.  The real repair belongs
 * at the injection point: an injected close should NOT compel the opener from
 * the next char, because the model has not finished its sentence.  See the
 * report accompanying this change — that fix is not built here.
 *
 * Perf: tg_token_allowed() copies the small POD state and walks the candidate.
 * In a fixed-literal phase the FSM expects one specific char, so almost every
 * one of the ~248k vocab tokens is rejected on its first char → ~O(1).  VALUE
 * accepts nearly everything with an early-accept O(len) scan.  NAME/KEY do a
 * per-char scan of a tiny set, entered only a few times per call.  A per-state
 * allowed-token cache (compute the mask once per skeleton state, reuse it across
 * steps in that state) is the next optimization — noted, not needed for
 * correctness or the current call rates.
 */

#include "tool_grammar.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json_value_grammar.h"
#include "tool_param_empty_value_rule.h"

/* ── compiled string set (tool names, or one tool's parameter keys) ── */
typedef struct {
    char **items;
    int    n;
} tg_set;

/* Required-key tracking is a bitmask over `keys` indices, so the walk state stays
 * a small POD (tg_token_allowed clones it per candidate token — see the perf note
 * above).  Tools with more than 64 parameters are beyond what any model emits;
 * a required key past bit 63 is simply not tracked, which degrades to the old
 * unenforced behavior for that key rather than rejecting a legal call. */
#define TG_REQ_MAX_KEYS 64

/* Largest `enum` compiled into a closed alternation — same degrade-don't-reject
 * rationale as TG_REQ_MAX_KEYS above: a schema listing more members than this
 * keeps a free-text VALUE rather than having legal calls rejected. */
#define TG_ENUM_MAX_MEMBERS 64

/* Everything the grammar compiled about ONE parameter of a tool.  Was a bare
 * `tg_set *enums` array; a second per-key fact (the JSON container its `type`
 * demands) would have made that two parallel arrays indexed the same way, which
 * is one array too many to keep straight at the three sites that touch them. */
typedef struct {
    tg_set        enum_values;      /* closed VALUE set; .n == 0 = not an enum */
    unsigned char json_container;   /* '[' array, '{' object, 0 = free-text VALUE */
    bool          forbids_empty;    /* schema says this value may not be empty —
                                     * tool_param_empty_value_rule.h */
} tg_key_meta;

typedef struct {
    char        *name;
    tg_set       keys;
    uint64_t     required;   /* bit i set ⇔ keys.items[i] is in the schema's `required` */
    tg_key_meta *key_meta;   /* [keys.n], parallel to keys.items */
} tg_tool;

struct tg_grammar {
    tg_tool *tools;
    int      n_tools;
    tg_set   names;   /* union of tool names (index i ↔ tools[i]) */
};

typedef enum {
    TG_FREE = 0,
    TG_TC_NL,
    TG_FUNC_OPEN,
    TG_FUNC_NAME,
    TG_FUNC_NL,
    TG_BODY,
    TG_PARAM_LIT,
    TG_PARAM_KEY,
    TG_PARAM_NL,
    TG_VALUE,
    TG_VALUE_ENUM,
    TG_VALUE_JSON,
    TG_FUNC_CLOSE,
    TG_AFTER_FUNC,   /* "</function>" consumed; optional "\n" then close/DONE */
    TG_TC_CLOSE,
    TG_DONE,
    TG_DEAD,
} tg_phase;

#define TG_NAME_MAX 256

struct tg_state {
    const tg_grammar *g;
    tg_phase phase;
    bool     wrapped;       /* saw "<tool_call>" → require "</tool_call>" close */
    int      cur_tool;      /* matched tool index (for its key set); -1 */

    /* Forced-call mode (Phase-5): compel a call after optional reasoning. */
    tg_mode  mode;          /* TG_MODE_AUTO | TG_MODE_FORCED */
    bool     reasoning_ended;  /* forced mode: "</think>" seen → opener required */
    /* The compelled region starts at the next TOKEN, not the next CHAR.
     *
     * Reasoning can end in the MIDDLE of a token: the budget trips on an
     * arbitrary char, and the model's own "</think>" can arrive glued to the text
     * after it.  tg_advance walks a whole token, so without this the very next
     * char of that same token is judged under the compelled rule — with no
     * sampling step, and therefore no mask, in between — and one ordinary prose
     * char there is TG_FREE_BREAK → TG_DEAD.  A DEAD walk reports its mask
     * INACTIVE, so the rest of the turn decodes unconstrained.
     *
     * Set by every reasoning-end that happens INSIDE the walk; cleared by
     * tg_advance once the token finishes.  So the token during which reasoning
     * ended stays unconstrained to its last char, and the constraint engages on
     * the next one — which IS a sampling step, and therefore is masked.
     *
     * Live: qwen36-bf16, one recorded session, three turns dead at
     * masked_steps=0 (dead_turns=1..3).  Not caught earlier because every test
     * fed this grammar one char at a time, the one granularity where the old
     * "arm cleanly, then mask the next step" comment was true. */
    bool     forced_pending;
    int      forced_tool;   /* forced-function pin (tool index), or -1 = any tool */
    /* THINK-GATED reasoning (tg_state_new_required).  The prompt left a <think>
     * block OPEN, so everything up to the model's own "</think>" is reasoning
     * and nothing in it is judged: no opener tracking (a "<tool_call>" the model
     * writes while thinking is thought, not a call) and no char budget (max_tokens
     * and the serve's reasoning_budget force-close bound it).  The opener is
     * compelled from the token after the close. */
    bool     think_gated;
    int      think_term;    /* forced mode: rolling-prefix length of "</think>" seen
                             * in the raw stream, so the grammar self-detects the
                             * reasoning close WITHOUT depending on the emitter's
                             * <think> flag (which the Q3 model doesn't reliably
                             * trigger — BUG 1). */

    const char *lit;        /* literal for a fixed-literal phase */
    int         lit_pos;

    const tg_set *set;      /* set for a NAME/KEY phase */
    int           set_pos;  /* chars matched so far */
    char          name_acc[TG_NAME_MAX];  /* accumulated NAME/KEY (survives tokens) */

    /* Closed VALUE set for the parameter currently being emitted, or NULL for a
     * free-text VALUE.  Borrowed from the grammar, which outlives every state, so
     * this stays a small POD — tg_token_allowed clones the whole struct per
     * candidate token (see the perf note at the top).  TG_VALUE_ENUM reuses
     * name_acc/set_pos to accumulate the value: the KEY that lived there has
     * already been matched and consumed, and the next TG_PARAM_KEY resets both. */
    const tg_set *enum_set;

    /* The JSON container this parameter's schema `type` demands of its VALUE
     * ('[' / '{'), or 0 for a free-text VALUE.  Stashed by TG_PARAM_KEY for the
     * same reason enum_set is: the key index is gone by the time the value
     * starts. */
    unsigned char json_container;

    /* Container-typed VALUE (TG_VALUE_JSON).  Embedded BY VALUE — 16 bytes of
     * POD, no pointers — precisely so tg_token_allowed can keep cloning the
     * whole tg_state per candidate token.  See json_value_grammar.h. */
    jvg_walk json_value;

    /* FREE-region opener tracker */
    int  free_open;         /* matched length of the opener literal */
    int  free_which;        /* 0 none, 1 <tool_call>, 2 <function=, 3 ambiguous '<' */

    /* TG_BODY boundary disambiguation ('<' seen, awaiting 'p' vs '/') */
    bool body_lt;

    /* VALUE terminator rolling-prefix length */
    int  val_term;

    /* EMPTY-VALUE FLOOR for the parameter currently being emitted.  `val_floor`
     * is this key's schema verdict, stashed by TG_PARAM_KEY for the same reason
     * enum_set/json_container are (the key index is gone once the value walk
     * reuses name_acc).  `val_content` is whether the value has yet carried a
     * character that SURVIVES the parser's dup_trim — the two together are what
     * TG_VALUE consults before it will let "</parameter>" close.
     *
     * Whitespace deliberately does not count.  The parser trims it away, so a
     * value of "\n" reaches the tool as "" — the grammar promising non-empty and
     * the host then deleting the only thing satisfying the promise is precisely
     * how {"path":""} shipped once already. */
    bool val_floor;
    bool val_content;

    /* Which of cur_tool's parameter keys this call has emitted so far (bit per
     * keys index).  TG_BODY refuses "</function>" until it covers the tool's
     * `required` mask, so a call MISSING a required argument cannot complete.
     * Reset when a NAME is matched, so each call in a multi-call stream is judged
     * on its own. */
    uint64_t seen_keys;

    /* Forced-mode reasoning budget (BUG: a model that degenerates in <think> and
     * never emits "</think>" streams unconstrained forever).  We count chars seen
     * while still in the reasoning grace (FREE, forced, !reasoning_ended, no opener
     * progress); once it exceeds TG_REASON_BUDGET we force reasoning_ended so the
     * opener is compelled — the degenerate ramble can't run past the budget. */
    int  reason_chars;

    /* LANDING WINDOW (see tg_reasoning_end_injected).  Chars of AUTO-like grace
     * still owed after an INJECTED "</think>", before the opener is compelled.
     * 0 = not in a landing window (the ordinary compelled behavior). */
    int  landing_chars;
};

/* Max chars a forced request may spend in the reasoning grace before the opener
 * is compelled.  Generous enough for a real <think> block, small enough that a
 * repetition loop is cut off in bounded time. */
#define TG_REASON_BUDGET 8192

/* Chars of AUTO-like grace granted after an INJECTED "</think>" before the
 * opener is compelled — the landing window (tg_reasoning_end_injected).
 *
 * Sized from recorded captures, which are the only measurements of how much
 * the model still had to say when it was cut off (the shapes are reproduced in
 * tests/frame_capture_fixtures.h):
 *
 *   native-XML capture —  ~10 chars ("</think>\n\n" then straight to the call)
 *   absolute-path      —  ~52 chars (one prose line, then the call)
 *   Hermes lead-in     —  ~90 chars (close + a prose lead-in, then the call)
 *
 * 512 clears the worst observed case by ~5x while staying negligible against the
 * 2000-TOKEN reasoning budget it backstops — so a model that is genuinely
 * degenerating is still cut off in bounded time, and the runaway this whole
 * mechanism exists to stop is not reopened.  The window also ends EARLY the
 * moment the model emits its own "</think>", which is what actually terminates
 * it in every recorded case; the char budget is only the backstop on the
 * backstop. */
#define TG_LANDING_BUDGET 512

/* Skeleton literals. */
static const char LIT_TC_NL[]      = "\n";
static const char LIT_FUNC_OPEN[]  = "<function=";
static const char LIT_FUNC_NL[]    = "\n";
static const char LIT_PARAM[]      = "<parameter=";
static const char LIT_PARAM_NL[]   = "\n";
static const char LIT_VALUE_TERM[] = "\n</parameter>\n";

/* FREE-TEXT VALUE terminator (TG_VALUE only; TG_VALUE_ENUM keeps the padded
 * LIT_VALUE_TERM above — see the DIVERGENCE note in the TG_VALUE case).
 *
 * This is deliberately the SAME string the parser searches for with
 * strstr(vstart, PARAM_CLOSE) (tool_calling.c parse_call_arguments).  The two
 * must be byte-identical or they disagree about where a value ended, and every
 * such disagreement is a call the grammar keeps open while the parser has
 * already closed it. */
static const char LIT_VALUE_CLOSE[] = "</parameter>";
static const char LIT_FUNC_CLOSE[] = "</function>";
static const char LIT_TC_CLOSE[]   = "</tool_call>";
/* FREE-region opener literals (indexed by free_which 1/2). */
static const char OPEN_TC[]   = "<tool_call>";
static const char OPEN_FUNC[] = "<function=";
/* Reasoning-close marker.  In FORCED mode the grammar watches the raw stream for
 * this itself (self-contained reasoning-end) — see BUG 1. */
static const char THINK_CLOSE[] = "</think>";

/* Whitespace, for TG_VALUE's empty-value floor.  This is exactly the set
 * dup_trim() (tool_calling.c) strips with isspace(), so the grammar and the
 * parser agree on what "empty" means — a disagreement here is what produced
 * {"path":""} from a value the grammar had certified non-empty.  Kept local and
 * explicit rather than calling isspace(), which is locale-dependent and is UB on
 * a negative char. */
static bool tg_is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' ||
           c == '\r' || c == '\f' || c == '\v';
}

/* ── set helpers ── */

static void set_free(tg_set *s) {
    if (s->items) {
        for (int i = 0; i < s->n; i++) free(s->items[i]);
        free(s->items);
    }
    s->items = NULL; s->n = 0;
}

static bool set_add(tg_set *s, const char *item) {
    char **grown = realloc(s->items, (size_t)(s->n + 1) * sizeof(char *));
    if (!grown) return false;
    s->items = grown;
    s->items[s->n] = strdup(item);
    if (!s->items[s->n]) return false;
    s->n++;
    return true;
}

/* Does SOME member have char `c` at position `pos`? */
static bool set_char_live(const tg_set *s, int pos, char c) {
    for (int i = 0; i < s->n; i++) {
        const char *m = s->items[i];
        if ((int)strlen(m) > pos && m[pos] == c) return true;
    }
    return false;
}

/* Index of the member exactly equal to `acc[0..len)`, or -1. */
static int set_exact_match(const tg_set *s, const char *acc, int len) {
    for (int i = 0; i < s->n; i++) {
        const char *m = s->items[i];
        if ((int)strlen(m) == len && strncmp(m, acc, (size_t)len) == 0) return i;
    }
    return -1;
}

/* Forced-function variants: only member `pin` (a single tool name) is live.
 * `pin < 0` means "any member" → falls back to the whole-set behavior. */
static bool set_char_live_pinned(const tg_set *s, int pin, int pos, char c) {
    if (pin < 0) return set_char_live(s, pos, c);
    if (pin >= s->n) return false;
    const char *m = s->items[pin];
    return (int)strlen(m) > pos && m[pos] == c;
}

static int set_exact_match_pinned(const tg_set *s, int pin, const char *acc, int len) {
    if (pin < 0) return set_exact_match(s, acc, len);
    if (pin >= s->n) return -1;
    const char *m = s->items[pin];
    return ((int)strlen(m) == len && strncmp(m, acc, (size_t)len) == 0) ? pin : -1;
}

/* ── build ── */

/* Compile a property's `enum` into a closed VALUE set, or leave `out` empty.
 *
 * Identical admission rules to the JSON dialect's build_key_enum
 * (tool_grammar_json.c) so the two grammars constrain the same schemas:
 *
 *   • the declared `type` must be absent or "string".  This grammar's VALUE is
 *     raw text between the <parameter=…> tags, so a string alternation is the
 *     only shape it can express; an enum on an integer/boolean/object type is a
 *     schema contradiction, and the declared type wins.  With NO type the enum
 *     still applies — it is self-describing.
 *   • every member is a non-empty string shorter than TG_NAME_MAX (the
 *     accumulator the walk matches against).  Numeric/boolean/mixed members are
 *     ignored wholesale: a number has no single spelling, so a literal set would
 *     reject legal emissions.
 *   • 1..TG_ENUM_MAX_MEMBERS members.
 *
 * Anything else degrades to a free-text VALUE — never to a rejected call.
 * Returns false only on OOM (tg_build then unwinds via tg_free). */
static bool build_key_enum(tg_set *out, const jr_value *prop) {
    const char *type = jr_as_string(jr_object_get(prop, "type"));
    if (type && strcmp(type, "string") != 0) return true;

    const jr_value *e = jr_object_get(prop, "enum");
    if (jr_kind(e) != JR_ARRAY) return true;
    int n = jr_array_len(e);
    if (n <= 0 || n > TG_ENUM_MAX_MEMBERS) return true;

    for (int i = 0; i < n; i++) {
        const char *m = jr_as_string(jr_array_get(e, i));
        if (!m || !*m || strlen(m) >= TG_NAME_MAX) { set_free(out); return true; }
        if (!set_add(out, m)) return false;
    }
    return true;
}

/* Which JSON container, if any, this parameter's schema `type` demands of its
 * VALUE: '[' for "array", '{' for "object", 0 for everything else.
 *
 * THE SET IS NOT ARBITRARY.  It is exactly the set of types whose value
 * tool_calling.c's write_typed_value SPLICES RAW into the call's `arguments`
 * object (`if (*val == '{' || *val == '[') jw_raw(...)`).  Every other type is
 * either JSON-escaped into a string (so a malformed value can only ever be a bad
 * ARGUMENT, never a broken arguments OBJECT) or re-validated there before being
 * spliced — "integer"/"number" through strtod, "boolean" through strcmp.  Those
 * two facts are the whole scope argument: constrain the values that are trusted
 * unchecked, and leave the ones that are already checked alone.
 *
 * A parameter with no declared `type` keeps a free-text VALUE.  The schema has
 * not said the value is structured, so the grammar does not get to decide that
 * it is — the same degrade-don't-reject rule build_key_enum follows. */
static unsigned char container_of_type(const jr_value *prop) {
    const char *type = jr_as_string(jr_object_get(prop, "type"));
    if (!type) return 0;
    if (strcmp(type, "array")  == 0) return '[';
    if (strcmp(type, "object") == 0) return '{';
    return 0;
}

tg_grammar *tg_build(const jr_value *tools) {
    if (jr_kind(tools) != JR_ARRAY) return NULL;
    int n = jr_array_len(tools);
    if (n <= 0) return NULL;

    tg_grammar *g = calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->tools = calloc((size_t)n, sizeof(tg_tool));
    if (!g->tools) { free(g); return NULL; }

    for (int i = 0; i < n; i++) {
        const jr_value *fn = jr_object_get(jr_array_get(tools, i), "function");
        const char *name = jr_as_string(jr_object_get(fn, "name"));
        if (!name || !*name) continue;   /* skip malformed entries */

        tg_tool *t = &g->tools[g->n_tools];
        t->name = strdup(name);
        if (!t->name) { tg_free(g); return NULL; }
        /* Count the tool NOW, before its sub-allocations: a later OOM in a set_add
         * below is then cleaned up by tg_free (which frees name + keys for tools in
         * [0, n_tools)) instead of leaking this partial tool. t->keys is zeroed, so
         * set_free on it is safe whether or not the keys loop has run. */
        g->n_tools++;
        if (!set_add(&g->names, name)) { tg_free(g); return NULL; }

        const jr_value *params = jr_object_get(fn, "parameters");
        const jr_value *props = jr_object_get(params, "properties");
        if (jr_kind(props) == JR_OBJECT) {
            int np = jr_object_member_count(props);
            if (np > 0) {
                /* Allocated BEFORE the key loop so a mid-loop OOM still unwinds
                 * through tg_free with a valid (zeroed) key_meta array. */
                t->key_meta = calloc((size_t)np, sizeof(tg_key_meta));
                if (!t->key_meta) { tg_free(g); return NULL; }
            }
            for (int k = 0; k < np; k++) {
                const char *key = jr_object_key_at(props, k);
                if (!key || !*key) continue;
                if (!set_add(&t->keys, key)) { tg_free(g); return NULL; }
                const jr_value *prop = jr_object_value_at(props, k);
                tg_key_meta *meta = &t->key_meta[t->keys.n - 1];
                if (!build_key_enum(&meta->enum_values, prop)) {
                    tg_free(g); return NULL;
                }
                /* Mutually exclusive by construction: build_key_enum only
                 * compiles an enum whose declared `type` is absent or "string",
                 * and container_of_type only fires on "array"/"object". */
                meta->json_container = container_of_type(prop);
                meta->forbids_empty = tool_param_forbids_empty_value(prop);
            }
        }

        /* The schema's `required` list → a bitmask over the key set built above.
         * A required name that is not among `properties` (malformed schema) has
         * no key the grammar could emit, so it is skipped rather than making
         * every call for this tool impossible. */
        const jr_value *req = jr_object_get(params, "required");
        if (jr_kind(req) == JR_ARRAY) {
            int nr = jr_array_len(req);
            for (int k = 0; k < nr; k++) {
                const char *key = jr_as_string(jr_array_get(req, k));
                if (!key || !*key) continue;
                int idx = set_exact_match(&t->keys, key, (int)strlen(key));
                if (idx >= 0 && idx < TG_REQ_MAX_KEYS)
                    t->required |= (uint64_t)1 << idx;
            }
        }
    }

    if (g->n_tools == 0) { tg_free(g); return NULL; }
    return g;
}

void tg_free(tg_grammar *g) {
    if (!g) return;
    for (int i = 0; i < g->n_tools; i++) {
        free(g->tools[i].name);
        if (g->tools[i].key_meta) {
            for (int k = 0; k < g->tools[i].keys.n; k++)
                set_free(&g->tools[i].key_meta[k].enum_values);
            free(g->tools[i].key_meta);
        }
        set_free(&g->tools[i].keys);
    }
    free(g->tools);
    set_free(&g->names);
    free(g);
}

/* ── state ── */

tg_state *tg_state_new(const tg_grammar *g) {
    if (!g) return NULL;
    tg_state *st = calloc(1, sizeof(*st));
    if (!st) return NULL;
    st->g = g;
    st->phase = TG_FREE;
    st->cur_tool = -1;
    st->mode = TG_MODE_AUTO;
    st->forced_tool = -1;
    return st;
}

tg_state *tg_state_new_forced(const tg_grammar *g, const char *forced_name) {
    if (!g) return NULL;
    int pin = -1;
    if (forced_name && *forced_name) {
        pin = set_exact_match(&g->names, forced_name, (int)strlen(forced_name));
        if (pin < 0) return NULL;   /* named function is not one of the tools */
    }
    tg_state *st = tg_state_new(g);
    if (!st) return NULL;
    st->mode = TG_MODE_FORCED;
    st->forced_tool = pin;   /* -1 = "required" (any tool), >=0 = pinned function */
    return st;
}

tg_state *tg_state_new_required(const tg_grammar *g, bool think_open) {
    tg_state *st = tg_state_new_forced(g, NULL);
    if (!st) return NULL;
    st->think_gated = think_open;
    /* No open think block: the first generated char is already the answer, so
     * the call is compelled from token 0 (nothing to wait for). */
    if (!think_open) st->reasoning_ended = true;
    return st;
}

tg_state *tg_state_clone(const tg_state *st) {
    if (!st) return NULL;
    tg_state *copy = malloc(sizeof(*copy));
    if (copy) *copy = *st;
    return copy;
}

void tg_state_copy(tg_state *dst, const tg_state *src) {
    if (dst && src) *dst = *src;
}

void tg_state_free(tg_state *st) { free(st); }

void tg_reasoning_end(tg_state *st) {
    if (st) st->reasoning_ended = true;
}

void tg_reasoning_end_injected(tg_state *st) {
    if (!st) return;
    st->reasoning_ended = true;
    /* Grant the landing window.  The reasoning region IS over (the close is in
     * the stream and the KV state), so every runaway guard stays armed — the
     * only thing deferred is COMPELLING the opener, and only for a bounded
     * number of chars.  Without this the model is made to emit
     * "<function=NAME>\n<parameter=KEY>\n" mid-sentence and its unfinished
     * thought lands in the free-text VALUE. */
    st->landing_chars = TG_LANDING_BUDGET;
    /* The injected close may have left a partial "</think>" match in the
     * rolling tracker; clear it so the window's own watch for the model's OWN
     * close starts from a clean slate. */
    st->think_term = 0;
    st->free_which = 0;
    st->free_open  = 0;
}

static void enter_literal(tg_state *st, tg_phase p, const char *lit) {
    st->phase = p; st->lit = lit; st->lit_pos = 0;
}

static void enter_set(tg_state *st, tg_phase p, const tg_set *set) {
    st->phase = p; st->set = set; st->set_pos = 0;
}

/* Advance a running fixed-literal; returns 1 when the literal is fully consumed
 * (caller transitions), 0 while still mid-literal, -1 on mismatch (DEAD). */
static int lit_step(tg_state *st, char c) {
    if (st->lit[st->lit_pos] != c) { st->phase = TG_DEAD; return -1; }
    st->lit_pos++;
    return st->lit[st->lit_pos] == '\0' ? 1 : 0;
}

/* Consume one char, mutating st (phase → TG_DEAD on any violation). */
static void step_char(tg_state *st, char c) {
    switch (st->phase) {

    case TG_FREE: {
        /* THINK-GATED (tg_state_new_required): inside the open think block only
         * the model's own "</think>" is watched for; nothing else is judged. */
        if (st->think_gated && !st->reasoning_ended) {
            if (THINK_CLOSE[st->think_term] == c) {
                st->think_term++;
                if (THINK_CLOSE[st->think_term] == '\0') {
                    st->reasoning_ended = true;
                    st->forced_pending  = true;   /* compelled from the next TOKEN */
                    st->think_term = 0;
                }
            } else {
                st->think_term = (c == THINK_CLOSE[0]) ? 1 : 0;
            }
            return;
        }

        /* FORCED mode, reasoning grace (before reasoning_ended): the model is
         * still inside its <think> block (the prompt prefilled "<think>", so the
         * generated stream opens mid-reasoning).  We watch the RAW stream for
         * "</think>" HERE, in the grammar itself, so forcing engages WITHOUT
         * depending on the emitter's in_think flag — which the Q3 model doesn't
         * reliably trigger (BUG 1: it sometimes reasons in the content channel /
         * emits no clean </think> the emitter recognizes, and forcing never fired).
         * Detecting the close in the grammar's own stream makes forcing
         * self-contained and reliable.  (The serve layer still calls
         * tg_reasoning_end() too — a harmless secondary trigger; whichever fires
         * first wins.)  Reasoning text is unconstrained until the close. */
        if (st->mode == TG_MODE_FORCED && !st->reasoning_ended) {
            /* Reasoning budget: a model that degenerates in <think> and never
             * emits "</think>" would otherwise stream unconstrained forever (the
             * live SABER/ornith degeneration).  Count EVERY grace char (the
             * degenerate ramble is often opener-rich — "<tools>", "<function=" in
             * its own reasoning — so gating the count on free_open==0 would starve
             * it and never trip); once the budget is spent, compel the opener even
             * without a clean "</think>".  Don't cut off a genuine opener that is
             * already underway (free_open>0): only revoke grace when not mid-opener,
             * so a real "<function=" the model just started still completes. */
            st->reason_chars++;
            if (st->reason_chars > TG_REASON_BUDGET && st->free_open == 0) {
                /* Budget spent → compel the opener FROM THE NEXT char.  Consume
                 * THIS char as the final grace char and return: if we instead fell
                 * through into the forced-opener path with this (prose) char, a
                 * non-opener char would drive the state to TG_DEAD — and a DEAD
                 * state reports the mask INACTIVE (tg_mask_active), silently
                 * disengaging the constraint so the ramble continues unmasked
                 * forever (a bug seen on K80 hardware).  Arm cleanly, then mask the next
                 * TOKEN — forced_pending is what makes "next step" true, since
                 * the rest of THIS token still arrives before the sampler runs
                 * again (see the field's note). */
                st->reasoning_ended = true;
                st->forced_pending  = true;
                st->think_term = 0;
                st->free_which = 0; st->free_open = 0;
                return;   /* armed; the next TOKEN is masked toward the opener */
            } else
            if (THINK_CLOSE[st->think_term] == c) {
                st->think_term++;
                if (THINK_CLOSE[st->think_term] == '\0') {
                    st->reasoning_ended = true;   /* "</think>" seen → compel opener */
                    st->forced_pending  = true;   /* … from the next TOKEN */
                    st->think_term = 0;
                    st->free_which = 0; st->free_open = 0;
                    return;   /* the close itself consumed; the rest of this
                               * token is still grace (the close can arrive
                               * glued to the text that follows it) */
                }
            } else {
                /* Broke the "</think>" match; re-seed if this char restarts it. */
                st->think_term = (c == THINK_CLOSE[0]) ? 1 : 0;
            }
            /* Still reasoning: fall through to the FREE opener tracker below so a
             * call opened DIRECTLY (mid-reasoning, no </think>) still engages the
             * skeleton (auto-like), while non-opener prose stays FREE (grace). */
        }

        /* LANDING WINDOW.  The reasoning close was INJECTED by the serve
         * (tg_reasoning_end_injected), not chosen by the model, so the model is
         * mid-sentence.  For a bounded number of chars behave exactly like AUTO
         * — prose is free, an opener still engages the skeleton — so the model
         * can finish its thought and start its call where it meant to, instead
         * of being forced to open a frame around it.
         *
         * The window ends on whichever comes FIRST:
         *   • the model emits its OWN "</think>" — it has now genuinely
         *     concluded, so compel the opener immediately (this is what actually
         *     ends the window in every recorded capture); or
         *   • the char budget runs out.
         * `tool_choice=required` is NOT weakened: the opener is still compelled,
         * just up to TG_LANDING_BUDGET chars later. */
        if (st->mode == TG_MODE_FORCED && st->landing_chars > 0) {
            if (THINK_CLOSE[st->think_term] == c) {
                st->think_term++;
                if (THINK_CLOSE[st->think_term] == '\0') {
                    /* The model's own close — the grace is no longer warranted.
                     * End the window HERE and consume this char, so the opener is
                     * compelled from the NEXT one (never from this one, which
                     * would drive a legal char to DEAD — and a DEAD walk reports
                     * its mask INACTIVE, releasing the constraint entirely). */
                    st->landing_chars = 0;
                    st->forced_pending = true;
                    st->think_term = 0;
                    st->free_which = 0; st->free_open = 0;
                    return;
                }
            } else {
                st->think_term = (c == THINK_CLOSE[0]) ? 1 : 0;
            }
            st->landing_chars--;
            if (st->landing_chars == 0) {
                /* Budget spent.  Same arm-cleanly-then-mask-next-TOKEN
                 * discipline as the reasoning-budget path above. */
                st->forced_pending = true;
                st->free_which = 0; st->free_open = 0;
                return;
            }
            /* Still landing: fall through to the opener tracker with `forced`
             * false, so prose stays FREE and a real opener still engages. */
        }

        /* Forced mode, reasoning over → the opener is REQUIRED: any char that is
         * not on an opener prefix is off-grammar (DEAD), so prose/JSON cannot be
         * emitted.  (In AUTO mode a non-opener char just keeps the walk FREE —
         * reasoning/prose is unconstrained.)  A live landing window suspends this
         * — and only this — for its duration. */
        bool forced = (st->mode == TG_MODE_FORCED && st->reasoning_ended &&
                       st->landing_chars == 0 && !st->forced_pending);

        /* Forced mode: the model's trained format puts whitespace/newline between
         * "</think>" and the "<tool_call>" opener (it emits "</think>\n<tool_call>").
         * BEFORE any opener progress (free_open == 0) tolerate leading whitespace
         * so that separator is allowed — otherwise the ONLY allowed first char is
         * '<', every whitespace-leading token is masked, the mask leaves ZERO
         * survivors, the sampler fails open, and a prose refusal slips through
         * (the observed no-force bug).  Whitespace only counts before the opener
         * starts; once '<' arrives the fixed skeleton takes over. */
        if (forced && st->free_open == 0 &&
            (c == '\n' || c == ' ' || c == '\t' || c == '\r'))
            return;   /* stay in TG_FREE awaiting the opener */

        /* On a broken opener prefix: DEAD in forced mode, else reset to FREE
         * (re-seeding on '<'). */
        #define TG_FREE_BREAK() do {                                        \
            if (forced) { st->phase = TG_DEAD; return; }                    \
            st->free_which = 0; st->free_open = 0;                          \
            if (c == '<') { st->free_which = 3; st->free_open = 1; }        \
            return;                                                         \
        } while (0)

        if (st->free_which == 0) {
            if (c == '<') { st->free_which = 3; st->free_open = 1; return; }
            TG_FREE_BREAK();
        }
        if (st->free_which == 3) {   /* ambiguous after '<' */
            bool tc = (st->free_open < (int)strlen(OPEN_TC)   && OPEN_TC[st->free_open]   == c);
            bool fn = (st->free_open < (int)strlen(OPEN_FUNC) && OPEN_FUNC[st->free_open] == c);
            if (tc && fn)      { st->free_open++; }
            else if (tc)       { st->free_which = 1; st->free_open++; }
            else if (fn)       { st->free_which = 2; st->free_open++; }
            else TG_FREE_BREAK();
        } else {
            const char *lit = (st->free_which == 1) ? OPEN_TC : OPEN_FUNC;
            if (st->free_open < (int)strlen(lit) && lit[st->free_open] == c) {
                st->free_open++;
            } else TG_FREE_BREAK();
        }
        #undef TG_FREE_BREAK
        /* completed an opener? */
        if (st->free_which == 1 && st->free_open == (int)strlen(OPEN_TC)) {
            st->wrapped = true;
            st->free_which = 0; st->free_open = 0;
            enter_literal(st, TG_TC_NL, LIT_TC_NL);
        } else if (st->free_which == 2 && st->free_open == (int)strlen(OPEN_FUNC)) {
            st->wrapped = false;
            st->free_which = 0; st->free_open = 0;
            enter_set(st, TG_FUNC_NAME, &st->g->names);
            st->set_pos = 0;
        }
        return;
    }

    case TG_TC_NL:
        if (lit_step(st, c) == 1) enter_literal(st, TG_FUNC_OPEN, LIT_FUNC_OPEN);
        return;

    case TG_FUNC_OPEN:
        if (lit_step(st, c) == 1) { enter_set(st, TG_FUNC_NAME, &st->g->names); st->set_pos = 0; }
        return;

    case TG_FUNC_NAME:
        /* Forced-function pins the NAME to a single tool (forced_tool); "required"
         * and AUTO allow any tool name (forced_tool == -1). */
        if (c == '>') {
            int idx = set_exact_match_pinned(st->set, st->forced_tool,
                                             st->name_acc, st->set_pos);
            if (idx < 0) { st->phase = TG_DEAD; return; }
            st->cur_tool = idx;   /* names set is 1:1 with tools */
            st->seen_keys = 0;    /* a fresh call owes this tool's required keys */
            enter_literal(st, TG_FUNC_NL, LIT_FUNC_NL);
        } else if (set_char_live_pinned(st->set, st->forced_tool, st->set_pos, c)) {
            if (st->set_pos < TG_NAME_MAX - 1) st->name_acc[st->set_pos] = c;
            st->set_pos++;
        } else st->phase = TG_DEAD;
        return;

    case TG_FUNC_NL:
        if (lit_step(st, c) == 1) { st->phase = TG_BODY; st->body_lt = false; }
        return;

    case TG_BODY:
        /* boundary: "<parameter=" (more) OR "</function>" (end); both start '<' */
        if (!st->body_lt) {
            /* Newlines between boundaries are SEPARATORS, not structure.  TG_VALUE
             * now hands over the instant "</parameter>" closes (it no longer eats
             * a trailing "\n" as part of the terminator), so the one the trained
             * format puts before the next "<parameter="/"</function>" arrives
             * here.  Accepting a run of them rather than exactly one also makes
             * "</parameter></function>" — emitted verbatim by the live fleet,
             * tool_arg_truncation.md log:274 — grammatical instead of a silent
             * swallow.  This is a blank line at a structural boundary; nothing a
             * tool argument can carry passes through it. */
            if (c == '\n') return;
            if (c == '<') { st->body_lt = true; return; }
            st->phase = TG_DEAD; return;
        }
        st->body_lt = false;
        if (c == 'p') {                       /* "<p" → parameter */
            enter_literal(st, TG_PARAM_LIT, LIT_PARAM);
            st->lit_pos = 2;                  /* "<p" already consumed */
        } else if (c == '/') {                /* "</" → function close */
            /* Missing-required-arg impossibility.  Closing the call is only a
             * legal continuation once every `required` key has been emitted;
             * otherwise the ONLY way forward is another "<parameter=".  Without
             * this the skeleton accepted "<function=NAME>\n</function>" — a
             * complete, well-formed call with NO arguments, which the parser
             * turns into `{}`.  That is what reached the client's orchestrator as
             * `dispatch_agent {}` and was refused for a missing "subtask"
             * (observed on the live fleet 2026-08-06, with the grammar ON): the
             * empty-VALUE guard in TG_VALUE only constrains a key that IS
             * emitted, never one that is left out.  The JSON dialect
             * (tool_grammar_json.c) has always required every `required` key;
             * this is the XML dialect catching up to the same contract. */
            const tg_tool *t = (st->cur_tool >= 0) ? &st->g->tools[st->cur_tool]
                                                   : NULL;
            if (t && (st->seen_keys & t->required) != t->required) {
                st->phase = TG_DEAD; return;
            }
            enter_literal(st, TG_FUNC_CLOSE, LIT_FUNC_CLOSE);
            st->lit_pos = 2;                  /* "</" already consumed */
        } else st->phase = TG_DEAD;
        return;

    case TG_PARAM_LIT:
        if (lit_step(st, c) == 1) {
            enter_set(st, TG_PARAM_KEY, &st->g->tools[st->cur_tool].keys);
            st->set_pos = 0;
        }
        return;

    case TG_PARAM_KEY:
        if (c == '>') {
            int kidx = set_exact_match(st->set, st->name_acc, st->set_pos);
            if (kidx < 0) { st->phase = TG_DEAD; return; }
            if (kidx < TG_REQ_MAX_KEYS)
                st->seen_keys |= (uint64_t)1 << kidx;   /* satisfies `required` */
            /* Stash this key's VALUE shape (closed enum set, or the JSON
             * container its `type` demands) for TG_PARAM_NL to route on — the
             * key index is not recoverable once name_acc is reused by the value
             * walk.  At most one of the two is ever set (see tg_build). */
            const tg_tool *kt = &st->g->tools[st->cur_tool];
            const tg_key_meta *meta = kt->key_meta ? &kt->key_meta[kidx] : NULL;
            st->enum_set = (meta && meta->enum_values.n > 0)
                               ? &meta->enum_values : NULL;
            st->json_container = meta ? meta->json_container : 0;
            /* No key_meta at all means no schema was compiled for this tool; a
             * floor we were never told about must not be invented. */
            st->val_floor = meta ? meta->forbids_empty : false;
            enter_literal(st, TG_PARAM_NL, LIT_PARAM_NL);
        } else if (set_char_live(st->set, st->set_pos, c)) {
            if (st->set_pos < TG_NAME_MAX - 1) st->name_acc[st->set_pos] = c;
            st->set_pos++;
        } else st->phase = TG_DEAD;
        return;

    case TG_PARAM_NL:
        if (lit_step(st, c) == 1) {
            st->phase = st->enum_set        ? TG_VALUE_ENUM
                      : st->json_container  ? TG_VALUE_JSON
                                            : TG_VALUE;
            st->val_term = 0;
            st->val_content = false;
            st->set_pos = 0;   /* TG_VALUE_ENUM accumulates into name_acc */
            if (st->phase == TG_VALUE_JSON)
                jvg_begin(&st->json_value, (char)st->json_container);
        }
        return;

    case TG_VALUE_ENUM: {
        /* Closed-alternation VALUE.  Two things must both hold before the walk
         * may leave: the accumulated text is EXACTLY one enum member, and the
         * "\n</parameter>\n" terminator then runs to completion.  Off-set chars
         * die at the first divergence, so "banana" is unreachable the moment it
         * stops being a prefix of any member.
         *
         * No empty-value guard is needed here (build_key_enum rejects empty
         * members, so no member is satisfiable by zero chars) and none is
         * applied.
         *
         * This phase KEEPS the padded "\n</parameter>\n" terminator that TG_VALUE
         * gave up, and that is not an oversight.  The divergence TG_VALUE fixes is
         * "the grammar swallowed a close the parser honoured"; it cannot happen
         * here, because an enum value is a closed set of literals and the walk
         * DIES on any char that leaves the set rather than absorbing it.  Matching
         * the bare "</parameter>" here would instead break the healthy emission:
         * after an exact member the next char is the padding "\n", which no member
         * continues, so the walk would die on a correct call.
         *
         * Ambiguity note: if a member contained the terminator's leading '\n' at
         * the current position AND the text so far were already an exact member,
         * the terminator wins.  Preferring the close is the only choice that
         * keeps the value finite; a member with an embedded newline is not
         * something a tool schema expresses. */
        const char *term = LIT_VALUE_TERM;
        if (st->val_term > 0) {
            /* Committed: the value already matched a member exactly, so the only
             * legal continuation is the rest of the terminator. */
            if (term[st->val_term] != c) { st->phase = TG_DEAD; return; }
            st->val_term++;
            if (term[st->val_term] == '\0') { st->phase = TG_BODY; st->body_lt = false; }
            return;
        }
        if (c == term[0] &&
            set_exact_match(st->enum_set, st->name_acc, st->set_pos) >= 0) {
            st->val_term = 1;
            return;
        }
        if (set_char_live(st->enum_set, st->set_pos, c)) {
            if (st->set_pos < TG_NAME_MAX - 1) st->name_acc[st->set_pos] = c;
            st->set_pos++;
            return;
        }
        st->phase = TG_DEAD;
        return;
    }

    case TG_VALUE_JSON: {
        /* CONTAINER-TYPED VALUE — the parameter's schema says "array"/"object",
         * so its bytes are not free text: they are a JSON value, and
         * write_typed_value will splice them into the call's `arguments` object
         * RAW, without looking at them.  That splice is why a malformed value
         * here is not a bad argument but a broken CALL: the arguments string
         * stops parsing, and every OTHER argument of that call dies with it.
         *
         * Live, qwen36-bf16, frame grammar armed and masking correctly — three
         * consecutive `todowrite` calls, same malformation, no recovery:
         *
         *     [{"content":"…","status":"in_progress"},{"content":"",""}]
         *
         * The second object opens a member, writes its key and value, then a
         * comma, then a BARE STRING where a key was owed.  JVG_P_KEY_REQ reads
         * that string as the key, JVG_P_COLON then requires a ':' — and the '}'
         * the model writes instead is off-grammar.  The malformation is not
         * merely detected here; the token carrying that '}' is masked, so it is
         * never drawn.
         *
         * THE TERMINATOR, AND THE ONE HAZARD IT CARRIES.  TG_VALUE's whole
         * safety property is that it ends on the exact bytes the parser ends on
         * ("</parameter>", strstr'd in parse_call_arguments), so the two cannot
         * disagree about where a value stopped.  A JSON value can CONTAIN those
         * bytes inside a string — `["</parameter>"]` — and the parser would cut
         * there, handing write_typed_value the un-parseable prefix `["`.  So the
         * rolling terminator match runs over the WHOLE value, and completing it
         * while the JSON value is still open is off-grammar.  Outside a string
         * '<' cannot appear in JSON at all, so in practice this only ever masks
         * the one token that would close that literal sequence inside a string —
         * and the value stays byte-identical to what the parser will read.
         *
         * AN EMPTY VALUE IS UNREACHABLE HERE, and unlike the free-text case that
         * is not a policy change smuggled in.  TG_VALUE keeps empty values legal
         * because no grammar can tell `edit`'s deliberate `new_string:""` from a
         * useless `path:""`.  For a container-typed key there is nothing to tell
         * apart: the empty LIST is `[]` and the empty OBJECT is `{}`, both
         * reachable, both two chars.  "" is not an empty array — it is a string,
         * which is the wrong type, which is the thing this phase exists to stop.
         *
         * Still NOT checked: the schema INSIDE the value (`items`, nested
         * `properties`/`required`, member types).  A well-formed `[{"x":1}]`
         * reaches the tool and is refused there, with the tool's own message. */
        const char *term = LIT_VALUE_CLOSE;

        if (jvg_is_complete(&st->json_value)) {
            /* The value is a finished JSON container.  Only padding whitespace
             * (which the parser's dup_trim removes) and the terminator remain. */
            if (term[st->val_term] == c) {
                st->val_term++;
                if (term[st->val_term] == '\0') {
                    st->phase = TG_BODY; st->body_lt = false;
                }
                return;
            }
            if (st->val_term == 0 &&
                (c == ' ' || c == '\t' || c == '\n' || c == '\r')) return;
            st->phase = TG_DEAD;
            return;
        }

        /* Still inside the value.  Track the terminator first: if these bytes
         * complete it, the parser would cut the value HERE, mid-JSON. */
        st->val_term = (term[st->val_term] == c) ? st->val_term + 1
                                                 : (c == term[0] ? 1 : 0);
        if (term[st->val_term] == '\0') { st->phase = TG_DEAD; return; }

        jvg_step(&st->json_value, c);
        if (jvg_is_dead(&st->json_value)) { st->phase = TG_DEAD; return; }
        if (jvg_is_complete(&st->json_value)) st->val_term = 0;
        return;
    }

    case TG_VALUE: {
        /* THE VALUE ENDS WHERE THE PARSER SAYS IT ENDS.
         *
         * The terminator is the bare "</parameter>" — the same bytes
         * parse_call_arguments looks for with strstr(vstart, PARAM_CLOSE).  Any
         * surrounding newlines are ordinary value padding here, exactly as they
         * are for the parser, which dup_trim()s them off.  So the two agree on
         * the value boundary BY CONSTRUCTION, for every emission, not just the
         * canonical one.
         *
         * WHAT THIS REPLACED, AND WHY IT HAD TO GO.  This phase used to terminate
         * on "\n</parameter>\n" AND carry an empty-value guard: while the value
         * held no non-whitespace char, a char that would start the terminator was
         * forced to be CONTENT instead (val_term reset to 0).  The stated
         * guarantee was that `path:""` could not be emitted.  It never held:
         *
         *   • The guard MASKS NOTHING.  Consuming a char as content is an ACCEPT,
         *     so tg_token_allowed("\n</parameter>\n") returned true at an empty
         *     value.  The model was free to emit the terminator and did.
         *   • Having eaten the leading "\n" as content, the guard then made
         *     "</parameter>" break the terminator match, so the close was
         *     swallowed as VALUE TEXT and the walk stayed in TG_VALUE for the rest
         *     of the turn — mask effectively off, call never closed.  The parser's
         *     strstr still found that same "</parameter>" and closed the value at
         *     it, yielding "" → the key dropped → `read {}`.
         *
         * So the guard did not prevent an empty value; it converted a VISIBLE
         * empty argument into an INVISIBLE missing one, and desynced the grammar
         * from the parser to do it.  Measured in real serving across 9 truncated
         * calls, all "grammar ACCEPTS char-for-char, tg_complete == NO".
         *
         * THE REPLACEMENT GUARD, AND WHAT MAKES IT A DIFFERENT THING.  An empty
         * value is not universally a defect — `edit`'s new_string:"" is a legal
         * deletion, and the client's own contract turns on the distinction
         * (argument_is_absent, the client's schema code: "A member present
         * with an EMPTY value is NOT absent").  What a grammar cannot decide is
         * whether THIS parameter means anything empty.  It does not have to
         * decide: the schema already says so, in `minLength`, and the client now
         * declares it on every string argument that cannot mean empty —
         * `bash.command`, `read.path`, `write.content`, and pointedly NOT
         * `edit.new_string`.  So the floor is per-key and derived
         * (tool_param_empty_value_rule.h), not a blanket rule this phase invents.
         *
         * The guard is applied HERE, at the closing terminator, and it DIES.
         * That distinction is the whole reason the previous attempt failed: it
         * consumed the terminator's first char AS CONTENT, and consuming is an
         * ACCEPT — tg_token_allowed returned true, the model emitted the close
         * anyway, and the walk was then stranded mid-value for the rest of the
         * turn.  Dying instead means the token carrying that close is MASKED and
         * is never drawn, and a walk that never enters TG_DEAD stays in lockstep
         * with the parser's strstr.  Live evidence for what the absence of any
         * guard costs: qwen36-bf16 with the grammar armed emitted
         * `bash {"command":""}` seven consecutive times — seven full decodes at
         * ~10 tok/s producing nothing — until the no-progress guard stopped the
         * run.  A model that emits an empty required argument cannot recover on
         * its own, which is why this is prevented at emission rather than
         * corrected afterwards.
         *
         * WHITESPACE IS NOT CONTENT (tg_is_ws).  The parser dup_trim()s the value,
         * so a value of "\n" arrives as "" — a floor satisfied by bytes the host
         * then deletes is not a floor.
         *
         * The parser is deliberately NOT changed to match.  An empty value that
         * arrives anyway is still reported as the empty value it is rather than
         * hidden as a missing member, which is the contract test_empty_argument.c
         * pins.  Three layers, each answering what it can see: the grammar stops
         * emission when the schema forbids it, the parser reports faithfully, the
         * client judges (the client's own may-be-empty check, which reads the
         * SAME minLength).
         *
         * KNOWN GAPS, all of them landing on that third layer rather than
         * silently.  This floor binds the bytes between "<parameter=KEY>" and
         * "</parameter>" and nothing else, so an empty value still reaches the
         * client when:
         *   • the grammar is off — PANTHEON_TOOL_GRAMMAR is off by default, and a
         *     request carrying no tools never builds one;
         *   • the schema declares no floor — a client that does not send
         *     minLength has not asked for one, and inventing it would forbid the
         *     `edit` deletion above;
         *   • the value arrives inside a "<value></value>" wrapper.  The
         *     MiniCPM-family trained format wraps each value, the grammar walks
         *     that wrapper as ordinary free text (7 non-whitespace chars, so the
         *     floor is met), and parse_call_arguments then PEELS it — leaving "".
         *     Teaching this phase the wrapper is what would fix it, and is
         *     declined: TG_VALUE's one safety property is that it ends on exactly
         *     the bytes the parser ends on, and every attempt to give it a second
         *     opinion about value structure has desynced it from the parser.
         *   • the dialect is MiniCPM's attribute form (`<param name="k">`), which
         *     this skeleton does not model at all.
         *
         * Still unreachable, unchanged: a call that OMITS a required key.  That
         * guard lives at TG_BODY (seen_keys vs the tool's required mask) and binds
         * "</function>", not the value. */
        const char *term = LIT_VALUE_CLOSE;
        if (term[st->val_term] == c) {
            st->val_term++;
            if (term[st->val_term] == '\0') {
                if (st->val_floor && !st->val_content) { st->phase = TG_DEAD; return; }
                st->phase = TG_BODY; st->body_lt = false;
            }
            return;
        }
        /* Terminator broke → everything the failed prefix consumed was value
         * content, and "</parameter>" holds no whitespace, so any progress at all
         * means real content preceded this char.  Crediting it retroactively is
         * what keeps a value that merely STARTS like the terminator ("</par…",
         * "<x>") legal under the floor.
         *
         * Re-seeding to 1 on '<' is exact, not an approximation: "</parameter>"
         * has no proper border, so the only suffix of a failed prefix that can
         * restart the match is the single '<'. */
        if (st->val_term > 0 || !tg_is_ws(c)) st->val_content = true;
        st->val_term = (c == term[0]) ? 1 : 0;
        return;
    }

    case TG_FUNC_CLOSE:
        if (lit_step(st, c) == 1) st->phase = TG_AFTER_FUNC;
        return;

    case TG_AFTER_FUNC:
        /* "</function>" is consumed → the call body is complete.  The trained
         * format has a "\n" before "</tool_call>" (wrapped) or ends here (bare).
         * Accept an optional "\n", then "</tool_call>" if wrapped, else DONE. */
        if (st->wrapped) {
            if (c == '\n') return;               /* optional separator */
            if (c == '<') { enter_literal(st, TG_TC_CLOSE, LIT_TC_CLOSE); st->lit_pos = 1; return; }
            st->phase = TG_DEAD;
        } else {
            /* bare form: complete. A trailing "\n" is tolerated; anything else
             * is off-grammar (the sampler stops at completion anyway). */
            if (c == '\n') return;
            st->phase = TG_DEAD;
        }
        return;

    case TG_TC_CLOSE:
        if (lit_step(st, c) == 1) st->phase = TG_DONE;
        return;

    case TG_DONE:
        st->phase = TG_DEAD;   /* nothing valid after a complete call */
        return;

    case TG_DEAD:
    default:
        return;
    }
}

static void walk(tg_state *st, const char *text) {
    for (const char *p = text; *p; p++) step_char(st, *p);
}

bool tg_token_allowed(const tg_state *st, const char *token_text) {
    if (!st) return false;
    if (!token_text || !*token_text) return true;   /* empty = no-op */
    tg_state clone = *st;
    walk(&clone, token_text);
    return clone.phase != TG_DEAD;
}

void tg_advance(tg_state *st, const char *token_text) {
    if (!st || !token_text) return;
    walk(st, token_text);
    /* The token during which reasoning ended is over.  From the NEXT one the
     * opener is compelled — and the next one is a sampling step, so the mask
     * runs and the model is steered into the opener instead of walking off the
     * grammar.  Clearing it HERE (not at the top) is what makes the compelled
     * region start at a token boundary; see tg_state.forced_pending. */
    st->forced_pending = false;
}

/* The rest of the fixed literal the walk is part-way through, so a closer can
 * finish it: a half-written "</function>" / "</tool_call>", or the value
 * terminator a VALUE has started.  "" when there is none. */
static const char *tg_pending_literal(const tg_state *st) {
    switch (st->phase) {
    case TG_FUNC_CLOSE:
    case TG_TC_CLOSE:
        return st->lit + st->lit_pos;
    case TG_VALUE:
    case TG_VALUE_JSON:
        return st->val_term > 0 ? LIT_VALUE_CLOSE + st->val_term : "";
    case TG_VALUE_ENUM:
        return st->val_term > 0 ? LIT_VALUE_TERM + st->val_term : "";
    default:
        return "";
    }
}

int tg_closing_text(const tg_state *st, char *out, int cap) {
    /* The trained format's own tail, shortest first.  Each is tried as it is
     * and then after the pending literal; the first that completes the call on
     * a copy of the walk wins. */
    static const char *const CLOSERS[] = {
        "", "\n", "\n</tool_call>", "\n</function>", "\n</function>\n</tool_call>",
        "\n</parameter>\n</function>", "\n</parameter>\n</function>\n</tool_call>",
    };
    const char *pending;
    char buf[96];

    if (!st || !out || cap < 1 || !tg_in_call(st) || tg_complete(st)) return -1;
    pending = tg_pending_literal(st);
    for (int with_pending = 0; with_pending < 2; with_pending++) {
        if (with_pending && !*pending) break;
        for (size_t i = 0; i < sizeof(CLOSERS) / sizeof(CLOSERS[0]); i++) {
            int n = snprintf(buf, sizeof(buf), "%s%s", with_pending ? pending : "",
                             CLOSERS[i]);
            if (n <= 0 || n >= (int)sizeof(buf) || n >= cap) continue;
            tg_state clone = *st;
            walk(&clone, buf);
            if (tg_complete(&clone)) {
                memcpy(out, buf, (size_t)n + 1);
                return n;
            }
        }
    }
    return -1;
}

bool tg_end_allowed(const tg_state *st) {
    char closing[TG_CLOSING_MAX];
    if (!st || !tg_mask_active(st)) return true;
    return tg_closing_text(st, closing, (int)sizeof(closing)) > 0;
}

bool tg_complete(const tg_state *st) {
    if (!st) return false;
    /* DONE = wrapped call fully closed.  For the bare form the call is complete
     * the moment "</function>" is consumed (TG_AFTER_FUNC, unwrapped) — there is
     * no wrapper to close. */
    return st->phase == TG_DONE ||
           (st->phase == TG_AFTER_FUNC && !st->wrapped);
}

bool tg_in_call(const tg_state *st) {
    return st && st->phase != TG_FREE && st->phase != TG_DEAD;
}

bool tg_is_dead(const tg_state *st) {
    return st && st->phase == TG_DEAD;
}

int tg_debug_reason_chars(const tg_state *st) {
    return st ? st->reason_chars : 0;
}

int tg_debug_landing_chars(const tg_state *st) {
    return st ? st->landing_chars : 0;
}

bool tg_forced_reasoning_ended(const tg_state *st) {
    /* A DEAD walk is excluded, and that exclusion is the whole point of this
     * being a function rather than a field read.
     *
     * The serve drives exactly one thing off this bit: it flips the emitter out
     * of its <think> channel so a COMPELLED call routes to CONTENT and is parsed
     * as a tool call (serve_generation.c, both decode loops).  A dead walk
     * compels nothing — tg_mask_active already reports it INACTIVE and every
     * remaining token of the turn is sampled unconstrained — so answering "yes"
     * here does not deliver a call to the content channel, it delivers the
     * model's ONGOING THOUGHT there, mid-sentence, and the transcript records it
     * as an ANSWER.  Live: qwen36-bf16 one recorded session, three turns,
     * 37,721 chars of think-block text filed as answers.
     *
     * Everything else in this module already declines to speak for a dead walk
     * (tg_mask_active; the serve's own sg_tg_report_tool_call, which refuses to
     * judge one).  This makes the reasoning verdict agree with them. */
    return st && st->mode == TG_MODE_FORCED && st->reasoning_ended &&
           st->phase != TG_DEAD;
}

bool tg_mask_active(const tg_state *st) {
    if (!st || st->phase == TG_DEAD) return false;
    if (tg_complete(st)) return false;   /* call done → release the constraint */
    /* AUTO: mask only inside a call (FREE prose unconstrained).
     * FORCED: also mask the post-reasoning FREE state to compel the opener. */
    if (tg_in_call(st)) return true;
    if (st->mode != TG_MODE_FORCED) return false;
    /* A live landing window masks exactly like AUTO — the grammar accepts any
     * prose during it, so computing a mask would be pure cost for zero
     * constraint.  Only an opener already in progress engages (below). */
    if (st->reasoning_ended && st->landing_chars == 0) return true;
    /* FORCED early-arm (BUG 1 hardening): the moment the model starts emitting a
     * tool-call opener DURING reasoning ("<", then "<function="/"<tool_call>…"),
     * engage the mask so the opener + NAME are constrained immediately — don't
     * wait for a "</think>" that a degenerating model may never emit.  free_open>0
     * means an opener prefix is in progress in the FREE tracker. */
    return st->free_open > 0;
}
