/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Tool-call decoding grammar (JSON dialect) — a char-level FSM over ornith's
 * RL-learned tool-call emission, with NAME/KEY sets + required-key sets + value
 * types drawn from the request's tools schema.  See tool_grammar_json.h for the
 * contract; this is the JSON-syntax analog of tool_grammar.c's XML skeleton.
 *
 * Pure C, libc-only.  No engine/CUDA deps → unit-tested standalone.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * The constrained emission (canonical)
 * ────────────────────────────────────────────────────────────────────────────
 *   ```json↵                     (OPTIONAL fence — the model usually emits it)
 *   {"function": "NAME", "arguments": {"KEY": <value>, "KEY": <value>}}
 *   ↵```                          (fence close, iff the fence was opened)
 *
 * with NAME ∈ the tool names, the "function" key spellable also as "name",
 * "arguments" also as "parameters", KEY ∈ that tool's parameter names, EVERY
 * `required` key present, and each <value> typed to its schema (string → a
 * quoted JSON string, integer/number → a numeric literal, boolean → true/false,
 * object/array → a balanced {…}/[…], else a quoted string).  An empty
 * `arguments` object ({}) is structurally impossible whenever the tool has ≥1
 * required key — the grammar cannot close `}` until every required key has been
 * emitted.  This is the JSON analog of the XML grammar's missing-key guard, and
 * it is what gives ornith the same `required` guarantee saber gets from the XML
 * grammar.
 *
 * A string value may be EMPTY only where the schema allows it.  `""` closes for
 * a key with no length floor (`edit`'s new_string, where empty IS the deletion)
 * and is off-grammar for one that has a floor (`bash`'s command, `read`'s path,
 * anything not declared a string).  The predicate is
 * tool_param_empty_value_rule.h, compiled into BOTH dialects from one function —
 * see the step_value comment for the two opposite ways this has been got wrong.
 *
 * A parameter whose schema carries a string `enum` is tighter still: its value
 * is a CLOSED ALTERNATION over the listed literals, matched by exactly the same
 * prefix machinery that constrains the tool NAME set and the parameter KEY set
 * (step_quoted_set).  The client's switch_mode declares
 * enum:["chat","plan","act","orchestrate"] — before this, "mode":"banana" was a
 * legal call.  See build_key_enum for which enums compile and which degrade.
 *
 * JSON whitespace: between structural tokens ({ } : , and around values) any run
 * of spaces / tabs / newlines / CRs is tolerated, matching what a model emits.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * Reasoning / AUTO / FORCED composition (identical policy to tool_grammar.c)
 * ────────────────────────────────────────────────────────────────────────────
 * The walk starts FREE (TGJ_FREE): arbitrary reasoning/prose text, UNCONSTRAINED
 * in AUTO mode.  It ENGAGES the JSON skeleton once a real opener appears — the
 * ```json fence OR a plausible bare `{` that begins a "function"/"name"-keyed
 * object (mirroring the serve_emitter's ornith opener rule).  FORCED mode
 * (tool_choice required / a named function) replaces the FREE grace with
 * "optional reasoning, then the opener is REQUIRED": after "</think>" (self-
 * detected in the raw stream, exactly like the XML grammar — BUG 1) or the
 * reasoning budget, only opener-leading tokens survive, so prose is impossible.
 * ornith runs reasoning_budget 0, so the grace is minimal; the budget + the
 * self-detected close both still arm the opener cleanly (see the 0-budget test).
 */

#include "tool_grammar_json.h"

#include <stdlib.h>
#include <string.h>

#include "tool_param_empty_value_rule.h"

/* ── compiled string set (tool names, or one tool's parameter keys) ── */
typedef struct {
    char **items;
    int    n;
} tgj_set;

/* Value type of a parameter, resolved from its schema at build time. */
typedef enum {
    TGJ_T_STRING = 0,   /* quoted JSON string (default / unknown) */
    TGJ_T_NUMBER,       /* integer or number → numeric literal */
    TGJ_T_BOOL,         /* boolean → true / false */
    TGJ_T_JSON,         /* object / array → a balanced {…} or […] */
    TGJ_T_ENUM,         /* schema `enum` → a quoted literal from a closed set */
} tgj_vtype;

typedef struct {
    char      *name;
    tgj_set    keys;        /* all parameter names (index i ↔ types[i]/required[i]) */
    tgj_vtype *types;       /* [keys.n] value type per key */
    tgj_set   *enums;       /* [keys.n] closed value set per key; .n == 0 = none */
    bool      *required;    /* [keys.n] true if the key is required */
    bool      *forbids_empty; /* [keys.n] schema forbids an empty value for this
                               * key — tool_param_empty_value_rule.h */
    int        n_required;  /* count of required keys */
} tgj_tool;

struct tgj_grammar {
    tgj_tool *tools;
    int       n_tools;
    tgj_set   names;        /* union of tool names (index i ↔ tools[i]) */
};

/* Phases of the JSON skeleton walk (char-consuming). */
typedef enum {
    TGJ_FREE = 0,       /* reasoning/prose before an opener (see composition note) */
    TGJ_FENCE_OPEN,     /* consuming the "```json\n" fence (optional) */
    TGJ_OBJ_OPEN,       /* expect '{' (whitespace ok) */
    TGJ_FN_KEY,         /* the "function"/"name" key (quoted), then ':' */
    TGJ_FN_COLON,       /* ':' after the function key (whitespace ok) */
    TGJ_FN_NAME,        /* the quoted NAME ∈ tool names, then ',' */
    TGJ_FN_COMMA,       /* ',' after the name (whitespace ok) */
    TGJ_ARGS_KEY,       /* the "arguments"/"parameters" key (quoted), then ':' */
    TGJ_ARGS_COLON,     /* ':' after the arguments key (whitespace ok) */
    TGJ_ARGS_OPEN,      /* expect '{' opening the arguments object (whitespace ok) */
    TGJ_ARG_KEY,        /* a parameter KEY (quoted) ∈ this tool's params, then ':' */
    TGJ_ARG_COLON,      /* ':' after a parameter key (whitespace ok) */
    TGJ_ARG_VALUE,      /* the typed value */
    TGJ_ARG_SEP,        /* after a value: ',' (more args) OR '}' (args done) */
    TGJ_OBJ_CLOSE,      /* expect '}' closing the outer object (whitespace ok) */
    TGJ_FENCE_CLOSE,    /* consuming the "\n```" fence close (iff fenced) */
    TGJ_DONE,           /* a complete call emitted */
    TGJ_DEAD,           /* invalid prefix (only ever reached on a clone) */
} tgj_phase;

#define TGJ_NAME_MAX 256
#define TGJ_MAX_ARGS 64      /* max distinct arg keys tracked per call */

/* Largest `enum` compiled into a closed alternation.  A schema listing more
 * members than this keeps its plain type instead of the enum — the same
 * degrade-don't-reject precedent as TG_REQ_MAX_KEYS in tool_grammar.c: a cap we
 * cannot represent must never turn a LEGAL call into an impossible one. */
#define TGJ_ENUM_MAX_MEMBERS 64

/* Reasoning grace budget — same rationale + value as the XML grammar. */
#define TGJ_REASON_BUDGET 8192

/* Literals. */
static const char FENCE_OPEN[]  = "```json";
static const char FENCE_CLOSE[] = "```";
static const char THINK_CLOSE[] = "</think>";
static const char KEY_FUNCTION[] = "function";
static const char KEY_NAME[]     = "name";
static const char KEY_ARGS[]     = "arguments";
static const char KEY_PARAMS[]   = "parameters";

/* Two accepted spellings for the function key and the arguments key. */
static const char *const FN_KEY_ALTS[]   = { KEY_FUNCTION, KEY_NAME };
static const char *const ARGS_KEY_ALTS[] = { KEY_ARGS, KEY_PARAMS };

struct tgj_state {
    const tgj_grammar *g;
    tgj_phase phase;
    bool      fenced;          /* opener was the ```json fence → require close */
    int       cur_tool;        /* matched tool index; -1 */

    /* Forced-call mode (mirrors tool_grammar.c). */
    tgj_mode  mode;
    bool      reasoning_ended;
    /* The compelled region starts at the next TOKEN, not the next CHAR — the
     * XML dialect's tg_state.forced_pending, for the same reason and with the
     * same live failure behind it (qwen36-bf16, one recorded session: three turns
     * DEAD at masked_steps=0).  tgj_advance walks a whole token, so a budget
     * that trips mid-token would judge that token's remaining chars under the
     * compelled rule with no mask in between, and one prose char there is
     * TGJ_DEAD — which reports the mask INACTIVE for the rest of the turn. */
    bool      forced_pending;
    int       forced_tool;     /* pinned tool index, or -1 = any */
    /* THINK-GATED reasoning (tgj_state_new_required).  The prompt left a
     * <think> block OPEN, so everything up to the model's own "</think>" is
     * reasoning and nothing in it is judged: no opener tracking (a '{' or a
     * ```json fence the model writes while thinking is just thought) and no
     * char budget (a long think is not a runaway — max_tokens and the
     * reasoning_budget force-close bound it).  The opener is compelled only
     * once the block has closed. */
    bool      think_gated;
    int       think_term;      /* rolling-prefix length of "</think>" seen */
    int       reason_chars;    /* grace char budget counter */

    /* FREE-region opener tracker: matching the ```json fence OR a bare '{'. */
    int  free_fence;           /* chars of FENCE_OPEN matched (>=0) */
    bool free_brace;           /* a bare '{' opener was taken (no fence) */

    /* Literal walk (for fence/think when a fixed literal is being consumed). */
    const char *lit;
    int         lit_pos;

    /* Quoted-token walk (function key / name / arg key / string value). */
    bool  in_quote;            /* inside the "…" of the current token */
    bool  quote_opened;        /* the opening '"' has been seen this token */
    bool  esc;                 /* previous char was a backslash (string value) */
    char  acc[TGJ_NAME_MAX];   /* accumulated key/name chars (survives tokens) */
    int   acc_len;

    /* Key-alternative matching: which of a small alt-set are still live, by the
     * accumulated prefix.  Used for the function key + the arguments key. */
    const char *const *alts;   /* candidate spellings */
    int                n_alts;

    /* Whitespace-or-token gate: the next non-ws char must be `expect`. */
    char expect;

    /* Value walk. */
    tgj_vtype vtype;           /* type of the value currently being emitted */
    const tgj_set *enum_set;   /* closed value set for TGJ_T_ENUM, else NULL.
                                * Borrowed from the grammar (outlives the state);
                                * keeps this struct a small POD, which matters
                                * because tgj_token_allowed copies it per token. */
    bool      val_content;     /* ≥1 non-whitespace char seen in the current value */
    bool      val_floor;       /* this key's schema forbids an empty value —
                                * stashed at TGJ_ARG_KEY, consumed at the closing
                                * quote (tool_param_empty_value_rule.h) */
    int       val_depth;       /* nesting depth for object/array values */
    bool      val_started;     /* the value's first structural char was seen */

    /* Arguments bookkeeping: which required keys are still outstanding. */
    bool key_done[TGJ_MAX_ARGS];   /* key i already emitted this call */
    int  required_left;            /* required keys not yet emitted */
};

/* ── set helpers ── */

static void set_free(tgj_set *s) {
    if (s->items) {
        for (int i = 0; i < s->n; i++) free(s->items[i]);
        free(s->items);
    }
    s->items = NULL; s->n = 0;
}

static bool set_add(tgj_set *s, const char *item) {
    char **grown = realloc(s->items, (size_t)(s->n + 1) * sizeof(char *));
    if (!grown) return false;
    s->items = grown;
    s->items[s->n] = strdup(item);
    if (!s->items[s->n]) return false;
    s->n++;
    return true;
}

/* Does SOME member have char `c` at position `pos`?  When `pin >= 0`, only that
 * one member is considered (forced-function pin). */
static bool set_char_live(const tgj_set *s, int pin, int pos, char c) {
    if (pin >= 0) {
        if (pin >= s->n) return false;
        const char *m = s->items[pin];
        return (int)strlen(m) > pos && m[pos] == c;
    }
    for (int i = 0; i < s->n; i++) {
        const char *m = s->items[i];
        if ((int)strlen(m) > pos && m[pos] == c) return true;
    }
    return false;
}

/* Index of the member exactly equal to acc[0..len), honoring a pin, or -1. */
static int set_exact_match(const tgj_set *s, int pin, const char *acc, int len) {
    if (pin >= 0) {
        if (pin >= s->n) return -1;
        const char *m = s->items[pin];
        return ((int)strlen(m) == len && strncmp(m, acc, (size_t)len) == 0)
                   ? pin : -1;
    }
    for (int i = 0; i < s->n; i++) {
        const char *m = s->items[i];
        if ((int)strlen(m) == len && strncmp(m, acc, (size_t)len) == 0) return i;
    }
    return -1;
}

/* Does SOME alt spelling have char `c` at position `pos`? */
static bool alts_char_live(const char *const *alts, int n, int pos, char c) {
    for (int i = 0; i < n; i++)
        if ((int)strlen(alts[i]) > pos && alts[i][pos] == c) return true;
    return false;
}

/* Is acc[0..len) an exact match of some alt spelling? */
static bool alts_exact(const char *const *alts, int n, const char *acc, int len) {
    for (int i = 0; i < n; i++)
        if ((int)strlen(alts[i]) == len && strncmp(alts[i], acc, (size_t)len) == 0)
            return true;
    return false;
}

/* ── build ── */

static tgj_vtype vtype_of(const char *type) {
    if (!type) return TGJ_T_STRING;
    if (strcmp(type, "integer") == 0 || strcmp(type, "number") == 0)
        return TGJ_T_NUMBER;
    if (strcmp(type, "boolean") == 0) return TGJ_T_BOOL;
    if (strcmp(type, "object") == 0 || strcmp(type, "array") == 0)
        return TGJ_T_JSON;
    return TGJ_T_STRING;
}

/* Compile a property's `enum` into a closed literal set, or leave `out` empty.
 *
 * An enum is compiled ONLY when every one of these holds; otherwise the key
 * keeps its plain type and the enum is dropped (degrade, never reject — a
 * constraint we cannot represent must not make a legal call impossible):
 *
 *   • the declared `type` is absent or "string".  A string alternation is the
 *     only shape this matcher can emit, so an enum sitting on an integer /
 *     boolean / object type is a schema contradiction we refuse to resolve —
 *     the declared type wins and the enum is ignored.  With NO type at all the
 *     enum still applies: it is self-describing (unlike llama.cpp, where a
 *     constraint with no sibling `type` is silently dropped by keyword order).
 *   • every member is a non-empty JSON string.  Numeric / boolean / null / mixed
 *     members are ignored wholesale: a number has no single spelling (1, 1.0,
 *     1e0 are the same value), so a literal set would reject legal emissions.
 *   • 1..TGJ_ENUM_MAX_MEMBERS members, each shorter than TGJ_NAME_MAX (the
 *     accumulator the walk matches against).
 *
 * Returns false only on OOM (the grammar build then unwinds via tgj_free). */
static bool build_key_enum(tgj_set *out, const jr_value *prop, const char *type) {
    if (type && strcmp(type, "string") != 0) return true;

    const jr_value *e = jr_object_get(prop, "enum");
    if (jr_kind(e) != JR_ARRAY) return true;
    int n = jr_array_len(e);
    if (n <= 0 || n > TGJ_ENUM_MAX_MEMBERS) return true;

    for (int i = 0; i < n; i++) {
        const char *m = jr_as_string(jr_array_get(e, i));
        if (!m || !*m || strlen(m) >= TGJ_NAME_MAX) { set_free(out); return true; }
        if (!set_add(out, m)) return false;
    }
    return true;
}

/* Is `key` listed in the tool's `required` array?  Absent array → not required. */
static bool key_is_required(const jr_value *required_arr, const char *key) {
    if (jr_kind(required_arr) != JR_ARRAY) return false;
    int n = jr_array_len(required_arr);
    for (int i = 0; i < n; i++) {
        const char *r = jr_as_string(jr_array_get(required_arr, i));
        if (r && strcmp(r, key) == 0) return true;
    }
    return false;
}

/* Populate one tool's keys/types/required from its parameters schema.  Returns
 * false only on OOM (the grammar build then unwinds via tgj_free). */
static bool build_tool_params(tgj_tool *t, const jr_value *fn) {
    const jr_value *params = jr_object_get(fn, "parameters");
    const jr_value *props = jr_object_get(params, "properties");
    if (jr_kind(props) != JR_OBJECT) return true;   /* no params → 0 keys, ok */
    const jr_value *required = jr_object_get(params, "required");

    int np = jr_object_member_count(props);
    if (np <= 0) return true;
    t->types = calloc((size_t)np, sizeof(tgj_vtype));
    t->enums = calloc((size_t)np, sizeof(tgj_set));
    t->required = calloc((size_t)np, sizeof(bool));
    t->forbids_empty = calloc((size_t)np, sizeof(bool));
    if (!t->types || !t->enums || !t->required || !t->forbids_empty) return false;

    for (int k = 0; k < np; k++) {
        const char *key = jr_object_key_at(props, k);
        if (!key || !*key) continue;
        const jr_value *prop = jr_object_value_at(props, k);
        if (!set_add(&t->keys, key)) return false;
        int idx = t->keys.n - 1;
        const char *type = jr_as_string(jr_object_get(prop, "type"));
        t->types[idx] = vtype_of(type);
        if (!build_key_enum(&t->enums[idx], prop, type)) return false;
        /* A compiled enum SUPERSEDES the type bucket: it is strictly the tighter
         * constraint, and build_key_enum only compiles one when the declared type
         * is absent or "string" — so this never contradicts the schema. */
        if (t->enums[idx].n > 0) t->types[idx] = TGJ_T_ENUM;
        t->forbids_empty[idx] = tool_param_forbids_empty_value(prop);
        t->required[idx] = key_is_required(required, key);
        if (t->required[idx]) t->n_required++;
    }
    return true;
}

tgj_grammar *tgj_build(const jr_value *tools) {
    if (jr_kind(tools) != JR_ARRAY) return NULL;
    int n = jr_array_len(tools);
    if (n <= 0) return NULL;

    tgj_grammar *g = calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->tools = calloc((size_t)n, sizeof(tgj_tool));
    if (!g->tools) { free(g); return NULL; }

    for (int i = 0; i < n; i++) {
        const jr_value *fn = jr_object_get(jr_array_get(tools, i), "function");
        const char *name = jr_as_string(jr_object_get(fn, "name"));
        if (!name || !*name) continue;   /* skip malformed entries */

        tgj_tool *t = &g->tools[g->n_tools];
        t->name = strdup(name);
        if (!t->name) { tgj_free(g); return NULL; }
        g->n_tools++;
        if (!set_add(&g->names, name)) { tgj_free(g); return NULL; }
        if (!build_tool_params(t, fn)) { tgj_free(g); return NULL; }
    }

    if (g->n_tools == 0) { tgj_free(g); return NULL; }
    return g;
}

void tgj_free(tgj_grammar *g) {
    if (!g) return;
    for (int i = 0; i < g->n_tools; i++) {
        free(g->tools[i].name);
        if (g->tools[i].enums) {
            for (int k = 0; k < g->tools[i].keys.n; k++)
                set_free(&g->tools[i].enums[k]);
            free(g->tools[i].enums);
        }
        set_free(&g->tools[i].keys);
        free(g->tools[i].types);
        free(g->tools[i].required);
        free(g->tools[i].forbids_empty);
    }
    free(g->tools);
    set_free(&g->names);
    free(g);
}

/* ── state ── */

tgj_state *tgj_state_new(const tgj_grammar *g) {
    if (!g) return NULL;
    tgj_state *st = calloc(1, sizeof(*st));
    if (!st) return NULL;
    st->g = g;
    st->phase = TGJ_FREE;
    st->cur_tool = -1;
    st->mode = TGJ_MODE_AUTO;
    st->forced_tool = -1;
    return st;
}

tgj_state *tgj_state_new_forced(const tgj_grammar *g, const char *forced_name) {
    if (!g) return NULL;
    int pin = -1;
    if (forced_name && *forced_name) {
        pin = set_exact_match(&g->names, -1, forced_name, (int)strlen(forced_name));
        if (pin < 0) return NULL;
    }
    tgj_state *st = tgj_state_new(g);
    if (!st) return NULL;
    st->mode = TGJ_MODE_FORCED;
    st->forced_tool = pin;
    return st;
}

tgj_state *tgj_state_new_required(const tgj_grammar *g, bool think_open) {
    tgj_state *st = tgj_state_new_forced(g, NULL);
    if (!st) return NULL;
    st->think_gated = think_open;
    /* No open think block: the first generated char is already the answer,
     * so the call is compelled from token 0 (nothing to wait for). */
    if (!think_open) st->reasoning_ended = true;
    return st;
}

tgj_state *tgj_state_clone(const tgj_state *st) {
    if (!st) return NULL;
    tgj_state *copy = malloc(sizeof(*copy));
    if (copy) *copy = *st;
    return copy;
}

void tgj_state_copy(tgj_state *dst, const tgj_state *src) {
    if (dst && src) *dst = *src;
}

void tgj_state_free(tgj_state *st) { free(st); }

void tgj_reasoning_end(tgj_state *st) {
    if (st) st->reasoning_ended = true;
}

/* ── phase helpers ── */

static bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

/* Enter a phase that gates on "optional whitespace, then the literal char
 * `expect`". */
static void enter_expect(tgj_state *st, tgj_phase p, char expect) {
    st->phase = p;
    st->expect = expect;
}

/* Enter a quoted-token phase (function key / name / arg key). */
static void enter_quote(tgj_state *st, tgj_phase p) {
    st->phase = p;
    st->in_quote = false;
    st->quote_opened = false;
    st->acc_len = 0;
}

/* Advance a running fixed-literal; 1 = fully consumed, 0 = mid-literal,
 * -1 = mismatch (DEAD). */
static int lit_step(tgj_state *st, char c) {
    if (st->lit[st->lit_pos] != c) { st->phase = TGJ_DEAD; return -1; }
    st->lit_pos++;
    return st->lit[st->lit_pos] == '\0' ? 1 : 0;
}

/* Begin the arguments object: reset the per-call key bookkeeping. */
static void enter_args_open(tgj_state *st) {
    memset(st->key_done, 0, sizeof(st->key_done));
    st->required_left = (st->cur_tool >= 0)
                            ? st->g->tools[st->cur_tool].n_required : 0;
    enter_expect(st, TGJ_ARGS_OPEN, '{');
}

/* After '{' of the arguments object: if the tool has ≥1 key, require a first
 * key (can't close empty while required keys remain); if it has NO keys at all,
 * an empty object is the only legal content → go straight to the separator so
 * '}' closes.  A tool with keys but zero required may also legally emit {}. */
static void enter_first_arg(tgj_state *st) {
    const tgj_tool *t = &st->g->tools[st->cur_tool];
    if (t->keys.n == 0) {
        /* No parameters: the only valid arguments object is {} → allow the close
         * (or trailing ws) immediately. */
        st->phase = TGJ_ARG_SEP;
        st->val_started = false;   /* signals "before any value": only '}' valid */
        return;
    }
    enter_quote(st, TGJ_ARG_KEY);
}

/* Begin a value of the given type.  acc_len is cleared because TGJ_T_ENUM
 * accumulates the value into acc[] (the key that was just matched lives there
 * until now) and matches it against the key's closed set. */
static void enter_value(tgj_state *st, tgj_vtype vt) {
    st->phase = TGJ_ARG_VALUE;
    st->vtype = vt;
    st->val_content = false;
    st->val_depth = 0;
    st->val_started = false;
    st->in_quote = false;
    st->quote_opened = false;
    st->esc = false;
    st->acc_len = 0;
}

/* ── the quoted-key / quoted-name / quoted-string steppers ── */

/* Step a quoted KEY/NAME phase matching against a set (arg keys / tool names).
 * `set`/`pin` select the live members; on the closing '"' the accumulated token
 * must exactly match a member.  `on_match(idx)` is invoked with the matched
 * member index; the caller transitions.  Returns the matched index or -1
 * (still in progress → phase unchanged, or DEAD). */
static int step_quoted_set(tgj_state *st, char c, const tgj_set *set, int pin) {
    if (!st->quote_opened) {
        if (is_ws(c)) return -1;                 /* leading ws before the quote */
        if (c == '"') { st->quote_opened = true; st->in_quote = true; return -1; }
        st->phase = TGJ_DEAD; return -1;
    }
    if (c == '"') {                              /* closing quote */
        int idx = set_exact_match(set, pin, st->acc, st->acc_len);
        if (idx < 0) { st->phase = TGJ_DEAD; return -1; }
        st->in_quote = false;
        return idx;                              /* caller transitions */
    }
    if (set_char_live(set, pin, st->acc_len, c)) {
        if (st->acc_len < TGJ_NAME_MAX - 1) st->acc[st->acc_len] = c;
        st->acc_len++;
        return -1;
    }
    st->phase = TGJ_DEAD;
    return -1;
}

/* Step a quoted KEY phase matching against an alt-spelling list (function key /
 * arguments key).  Returns 1 on the closing '"' with an exact match (caller
 * transitions), 0 in progress, -1 DEAD. */
static int step_quoted_alts(tgj_state *st, char c) {
    if (!st->quote_opened) {
        if (is_ws(c)) return 0;
        if (c == '"') { st->quote_opened = true; return 0; }
        st->phase = TGJ_DEAD; return -1;
    }
    if (c == '"') {
        if (!alts_exact(st->alts, st->n_alts, st->acc, st->acc_len)) {
            st->phase = TGJ_DEAD; return -1;
        }
        return 1;
    }
    if (alts_char_live(st->alts, st->n_alts, st->acc_len, c)) {
        if (st->acc_len < TGJ_NAME_MAX - 1) st->acc[st->acc_len] = c;
        st->acc_len++;
        return 0;
    }
    st->phase = TGJ_DEAD;
    return -1;
}

/* Step a typed VALUE.  Returns 1 when the value is COMPLETE (caller transitions
 * to the separator), 0 while still mid-value, -1 DEAD.  On completion the char
 * that terminated the value has NOT been consumed for strings (the closing '"'
 * IS consumed); for scalars/JSON the terminator is a structural char (',' '}'
 * or ws) that the SEPARATOR phase must re-see — so we signal completion and let
 * the caller re-dispatch that same char. */
static int step_value(tgj_state *st, char c, bool *reconsume) {
    *reconsume = false;
    switch (st->vtype) {

    case TGJ_T_STRING:
        if (!st->quote_opened) {
            if (is_ws(c)) return 0;              /* ws before the string */
            if (c == '"') { st->quote_opened = true; return 0; }
            st->phase = TGJ_DEAD; return -1;
        }
        if (st->esc) {
            /* The escape's DECODED value decides whether this counts as content:
             * \n \t \r \f are whitespace and must not satisfy the guard.  (\b is
             * a backspace, not whitespace; \" \\ \/ are real content.)
             *
             * KNOWN GAP: \uXXXX is not decoded here — this FSM has no hex
             * accumulator, so " " still counts as content.  Deliberate and
             * backstopped: the client's require_string() skips
             * ALL leading whitespace before testing for empty, so a
             * whitespace-only value fails at the tool however it was spelled. */
            st->esc = false;
            if (c != 'n' && c != 't' && c != 'r' && c != 'f')
                st->val_content = true;
            return 0;
        }
        /* A lone backslash is not content yet — the char it escapes decides. */
        if (c == '\\') { st->esc = true; return 0; }
        if (c == '"') {
            /* EMPTY-STRING FLOOR — per key, from the schema, not blanket.
             *
             * The closing quote may only fire once the value holds ≥1
             * NON-WHITESPACE char, and ONLY for a key whose schema forbids an
             * empty value (tool_param_empty_value_rule.h: a non-string type, or
             * `minLength` ≥ 1).  Both halves of that sentence were wrong here at
             * different times, in opposite directions:
             *
             *   • The guard first counted ANY in-quote char, so `" "` and `"\n"`
             *     satisfied it — the empty-path wedge, and a comment claiming
             *     `"path": ""` was "structurally impossible" while it was not.
             *   • Fixing that left the guard UNCONDITIONAL, which is the mirror
             *     defect: `edit`'s `new_string:""` is a legal deletion, and an
             *     ornith model could not emit one at all.  Under
             *     tool_choice=required an unemittable legal call is a wedge of
             *     its own, just a quieter one.
             *
             * The XML dialect runs the identical predicate at TG_VALUE, compiled
             * from the identical function, because these two grammars answering
             * one schema differently is this module's recurring defect rather
             * than a hypothetical.
             *
             * Whitespace still does not count. ornith's parser does not trim
             * (push_call_json forwards the value verbatim), so `{"path":" "}`
             * survives to the tool — where the client's require_string skips leading
             * whitespace before testing for empty and refuses it anyway.  A value
             * The client will certainly refuse is one this grammar should never
             * have let the model spend a turn on. */
            if (st->val_floor && !st->val_content) { st->phase = TGJ_DEAD; return -1; }
            return 1;                            /* closing quote consumed */
        }
        if (!is_ws(c)) st->val_content = true;
        return 0;

    case TGJ_T_NUMBER:
        if (!st->val_started) {
            if (is_ws(c)) return 0;
            if (c == '-' || (c >= '0' && c <= '9')) {
                st->val_started = true; st->val_content = true; return 0;
            }
            st->phase = TGJ_DEAD; return -1;
        }
        if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
            c == '+' || c == '-')
            return 0;                            /* still in the number */
        /* A structural char / ws ends the number → complete, re-see it. */
        *reconsume = true;
        return 1;

    case TGJ_T_BOOL:
        if (!st->val_started) {
            if (is_ws(c)) return 0;
            if (c == 't') { st->lit = "true";  st->lit_pos = 1; st->val_started = true; return 0; }
            if (c == 'f') { st->lit = "false"; st->lit_pos = 1; st->val_started = true; return 0; }
            st->phase = TGJ_DEAD; return -1;
        }
        if (st->lit[st->lit_pos] != c) { st->phase = TGJ_DEAD; return -1; }
        st->lit_pos++;
        if (st->lit[st->lit_pos] == '\0') { st->val_content = true; return 1; }
        return 0;

    case TGJ_T_ENUM:
        /* Closed alternation: the value is a quoted literal that must EXACTLY
         * equal one member.  Reuses step_quoted_set — the identical prefix walk
         * that constrains the tool NAME and the parameter KEY — so a member split
         * across token boundaries matches, an off-set char dies at the first
         * divergence, and an empty "" is impossible (no member is empty). */
        if (!st->enum_set) { st->phase = TGJ_DEAD; return -1; }
        if (step_quoted_set(st, c, st->enum_set, -1) >= 0) {
            st->val_content = true;
            return 1;                            /* closing quote consumed */
        }
        return (st->phase == TGJ_DEAD) ? -1 : 0;

    case TGJ_T_JSON:
    default:
        if (!st->val_started) {
            if (is_ws(c)) return 0;
            if (c == '{' || c == '[') {
                st->val_started = true; st->val_depth = 1;
                st->val_content = true; st->in_quote = false; st->esc = false;
                return 0;
            }
            st->phase = TGJ_DEAD; return -1;
        }
        /* Balanced-brace scan, string-aware (braces inside strings don't count). */
        if (st->in_quote) {
            if (st->esc) { st->esc = false; return 0; }
            if (c == '\\') { st->esc = true; return 0; }
            if (c == '"') st->in_quote = false;
            return 0;
        }
        if (c == '"') { st->in_quote = true; return 0; }
        if (c == '{' || c == '[') { st->val_depth++; return 0; }
        if (c == '}' || c == ']') {
            st->val_depth--;
            if (st->val_depth == 0) return 1;    /* balanced → complete */
            return 0;
        }
        return 0;
    }
}

/* ── FREE-region opener tracking (AUTO grace + forced arm) ── */

/* Forward declaration: transition into the function-key phase after '{'. */
static void enter_first_fn(tgj_state *st);

/* Consume one char while FREE.  Handles the reasoning grace, the </think> self-
 * detect, the forced budget, and the ```json-fence / bare-'{' opener trackers.
 * Sets phase past FREE once an opener completes. */
static void step_free(tgj_state *st, char c) {
    bool opener_progress = (st->free_fence > 0 || st->free_brace);

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

    if (st->mode == TGJ_MODE_FORCED && !st->reasoning_ended) {
        st->reason_chars++;
        if (st->reason_chars > TGJ_REASON_BUDGET && !opener_progress) {
            st->reasoning_ended = true;
            st->forced_pending  = true;
            st->think_term = 0;
            st->free_fence = 0; st->free_brace = false;
            return;                              /* armed; next TOKEN masked */
        }
        if (THINK_CLOSE[st->think_term] == c) {
            st->think_term++;
            if (THINK_CLOSE[st->think_term] == '\0') {
                st->reasoning_ended = true;
                st->forced_pending  = true;   /* … from the next TOKEN */
                st->think_term = 0;
                st->free_fence = 0; st->free_brace = false;
                return;
            }
        } else {
            st->think_term = (c == THINK_CLOSE[0]) ? 1 : 0;
        }
        /* fall through to the opener tracker so a call opened mid-reasoning
         * (no </think>) still engages the skeleton */
    }

    bool forced = (st->mode == TGJ_MODE_FORCED && st->reasoning_ended &&
                   !st->forced_pending);

    /* Forced grace: leading whitespace before the opener is tolerated so the
     * model's "</think>\n```json" separator survives (mirrors the XML grammar).
     * Only before any opener progress. */
    if (forced && !opener_progress && is_ws(c)) return;

    /* Try to extend the ```json fence match. */
    if (st->free_fence >= 0 && !st->free_brace) {
        if (FENCE_OPEN[st->free_fence] == c) {
            st->free_fence++;
            if (FENCE_OPEN[st->free_fence] == '\0') {
                /* fence body matched; consume the trailing newline(s) then '{' */
                st->fenced = true;
                st->free_fence = 0; st->free_brace = false;
                enter_expect(st, TGJ_OBJ_OPEN, '{');
            }
            return;
        }
        /* Not extending the fence.  A bare '{' opener is also valid. */
        if (st->free_fence == 0 && c == '{') {
            st->fenced = false;
            st->free_brace = false; st->free_fence = 0;
            enter_first_fn(st);                  /* fwd-declared below */
            return;
        }
        /* Broken opener prefix. */
        if (forced) { st->phase = TGJ_DEAD; return; }
        /* AUTO: reset; re-seed if this char restarts a fence. */
        st->free_fence = (c == FENCE_OPEN[0]) ? 1 : 0;
        return;
    }
}

/* ── the master char stepper ── */

static void step_char(tgj_state *st, char c) {
    switch (st->phase) {

    case TGJ_FREE:
        step_free(st, c);
        return;

    case TGJ_OBJ_OPEN:
        if (is_ws(c)) return;
        if (c == '{') { enter_first_fn(st); return; }
        st->phase = TGJ_DEAD;
        return;

    case TGJ_FN_KEY: {
        int r = step_quoted_alts(st, c);
        if (r == 1) enter_expect(st, TGJ_FN_COLON, ':');
        return;
    }

    case TGJ_FN_COLON:
        if (is_ws(c)) return;
        if (c == st->expect) { enter_quote(st, TGJ_FN_NAME); return; }
        st->phase = TGJ_DEAD;
        return;

    case TGJ_FN_NAME: {
        int idx = step_quoted_set(st, c, &st->g->names, st->forced_tool);
        if (idx >= 0) { st->cur_tool = idx; enter_expect(st, TGJ_FN_COMMA, ','); }
        return;
    }

    case TGJ_FN_COMMA:
        if (is_ws(c)) return;
        if (c == st->expect) {
            enter_quote(st, TGJ_ARGS_KEY);
            st->alts = ARGS_KEY_ALTS; st->n_alts = 2;
            return;
        }
        st->phase = TGJ_DEAD;
        return;

    case TGJ_ARGS_KEY: {
        int r = step_quoted_alts(st, c);
        if (r == 1) enter_expect(st, TGJ_ARGS_COLON, ':');
        return;
    }

    case TGJ_ARGS_COLON:
        if (is_ws(c)) return;
        if (c == st->expect) { enter_args_open(st); return; }
        st->phase = TGJ_DEAD;
        return;

    case TGJ_ARGS_OPEN:
        if (is_ws(c)) return;
        if (c == '{') { enter_first_arg(st); return; }
        st->phase = TGJ_DEAD;
        return;

    case TGJ_ARG_KEY: {
        const tgj_tool *t = &st->g->tools[st->cur_tool];
        int idx = step_quoted_set(st, c, &t->keys, -1);
        if (idx >= 0) {
            /* Reject a duplicate key (a completed call can't set one twice). */
            if (idx < TGJ_MAX_ARGS) {
                if (st->key_done[idx]) { st->phase = TGJ_DEAD; return; }
                st->key_done[idx] = true;
            }
            if (t->required[idx] && st->required_left > 0) st->required_left--;
            enter_expect(st, TGJ_ARG_COLON, ':');
            /* stash the type (and its closed value set, if any) for enter_value */
            st->vtype = t->types[idx];
            st->enum_set = (t->enums && t->enums[idx].n > 0) ? &t->enums[idx] : NULL;
            st->val_floor = t->forbids_empty ? t->forbids_empty[idx] : false;
        }
        return;
    }

    case TGJ_ARG_COLON:
        if (is_ws(c)) return;
        if (c == st->expect) { enter_value(st, st->vtype); return; }
        st->phase = TGJ_DEAD;
        return;

    case TGJ_ARG_VALUE: {
        bool reconsume = false;
        int r = step_value(st, c, &reconsume);
        if (r == 1) {
            st->phase = TGJ_ARG_SEP;
            st->val_started = true;              /* a value was emitted */
            if (reconsume) step_char(st, c);     /* re-dispatch the terminator */
        }
        return;
    }

    case TGJ_ARG_SEP:
        if (is_ws(c)) return;
        if (c == ',') {
            /* another key required */
            enter_quote(st, TGJ_ARG_KEY);
            return;
        }
        if (c == '}') {
            /* Close the arguments object ONLY if every required key was set. */
            if (st->required_left > 0) { st->phase = TGJ_DEAD; return; }
            enter_expect(st, TGJ_OBJ_CLOSE, '}');
            return;
        }
        st->phase = TGJ_DEAD;
        return;

    case TGJ_OBJ_CLOSE:
        if (is_ws(c)) return;
        if (c == '}') {
            if (st->fenced) {
                st->lit = FENCE_CLOSE; st->lit_pos = 0;
                st->phase = TGJ_FENCE_CLOSE;
            } else {
                st->phase = TGJ_DONE;
            }
            return;
        }
        st->phase = TGJ_DEAD;
        return;

    case TGJ_FENCE_CLOSE:
        /* Tolerate whitespace/newline before the closing ``` fence. */
        if (st->lit_pos == 0 && is_ws(c)) return;
        if (lit_step(st, c) == 1) st->phase = TGJ_DONE;
        return;

    case TGJ_DONE:
        st->phase = TGJ_DEAD;
        return;

    case TGJ_DEAD:
    case TGJ_FENCE_OPEN:   /* fence-open is handled inside step_free */
    default:
        return;
    }
}

/* '{' just consumed for the OUTER object → the function key must follow. */
static void enter_first_fn(tgj_state *st) {
    enter_quote(st, TGJ_FN_KEY);
    st->alts = FN_KEY_ALTS;
    st->n_alts = 2;
}

static void walk(tgj_state *st, const char *text) {
    for (const char *p = text; *p; p++) step_char(st, *p);
}

bool tgj_token_allowed(const tgj_state *st, const char *token_text) {
    if (!st) return false;
    if (!token_text || !*token_text) return true;
    tgj_state clone = *st;
    walk(&clone, token_text);
    return clone.phase != TGJ_DEAD;
}

void tgj_advance(tgj_state *st, const char *token_text) {
    if (!st || !token_text) return;
    walk(st, token_text);
    /* The token during which reasoning ended is over; the opener is compelled
     * from the next one, which IS a sampling step and therefore is masked.
     * See tgj_state.forced_pending. */
    st->forced_pending = false;
}

int tgj_closing_text(const tgj_state *st, char *out, int cap) {
    static const char *const CLOSERS[] = { "", "\n```", "}", "}\n```", "}}", "}}\n```",
                                           "\"}}", "\"}}\n```" };
    if (!st || !out || cap < 1 || !tgj_in_call(st) || tgj_complete(st)) return -1;
    for (size_t i = 0; i < sizeof(CLOSERS) / sizeof(CLOSERS[0]); i++) {
        int n = (int)strlen(CLOSERS[i]);
        if (n == 0 || n >= cap) continue;
        tgj_state clone = *st;
        walk(&clone, CLOSERS[i]);
        if (clone.phase == TGJ_DONE) {
            memcpy(out, CLOSERS[i], (size_t)n + 1);
            return n;
        }
    }
    return -1;
}

bool tgj_end_allowed(const tgj_state *st) {
    char closing[16];
    if (!st || !tgj_mask_active(st)) return true;
    return tgj_closing_text(st, closing, (int)sizeof(closing)) > 0;
}

bool tgj_complete(const tgj_state *st) {
    return st && st->phase == TGJ_DONE;
}

bool tgj_in_call(const tgj_state *st) {
    return st && st->phase != TGJ_FREE && st->phase != TGJ_DEAD;
}

bool tgj_is_dead(const tgj_state *st) {
    return st && st->phase == TGJ_DEAD;
}

int tgj_debug_reason_chars(const tgj_state *st) {
    return st ? st->reason_chars : 0;
}

bool tgj_forced_reasoning_ended(const tgj_state *st) {
    /* DEAD excluded, exactly as in tool_grammar.c: the serve drives the
     * emitter's reasoning/content channel split off this bit, and a dead walk
     * compels nothing — so saying "yes" routes the model's ongoing THOUGHT into
     * the content channel, where the transcript files it as an answer. */
    return st && st->mode == TGJ_MODE_FORCED && st->reasoning_ended &&
           st->phase != TGJ_DEAD;
}

bool tgj_mask_active(const tgj_state *st) {
    if (!st || st->phase == TGJ_DEAD) return false;
    if (tgj_complete(st)) return false;
    if (tgj_in_call(st)) return true;
    if (st->mode != TGJ_MODE_FORCED) return false;
    if (st->reasoning_ended) return true;
    /* Forced early-arm: opener progress (fence or bare '{') during reasoning. */
    return st->free_fence > 0 || st->free_brace;
}
