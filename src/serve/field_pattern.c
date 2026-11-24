/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * field_pattern — a small byte-level regex compiled to its position (Glushkov)
 * automaton.  See field_pattern.h for the syntax, the limits and why the
 * automaton is a fixed bitset.
 *
 * Two passes:
 *   1. parse  — recursive descent over the pattern into a pool of AST nodes
 *               (alt > concat > repeat > atom), every error reported with the
 *               byte offset it applies to;
 *   2. build  — the classic first/last/follow construction over that AST.  Each
 *               visit of an atom allocates a fresh state, so counted repetition
 *               is expanded by visiting its operand once per copy, and the state
 *               limit is enforced as the states are made.
 *
 * Pure C, libc-only.
 */

#include "field_pattern.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 256-bit byte class. */
typedef struct {
    uint64_t w[4];
} fp_class;

typedef enum {
    FPN_EMPTY = 0,   /* matches "" */
    FPN_ATOM,        /* one byte from `cls` */
    FPN_CONCAT,      /* left then right */
    FPN_ALT,         /* left or right */
    FPN_REPEAT,      /* left, min..max times (max < 0 = unbounded) */
} fp_node_kind;

typedef struct {
    fp_node_kind kind;
    int          left, right;
    int          min, max;
    fp_class     cls;
} fp_node;

struct field_pattern {
    int    n_states;                 /* start state 0 + one per atom copy */
    fp_set follow[FP_MAX_STATES];    /* states reachable from state q by one atom */
    fp_set byte_mask[256];           /* states whose atom accepts byte b */
    fp_set accept;                   /* states at which the whole pattern matched */
    int    dist[FP_MAX_STATES];      /* fewest bytes from state q to a match */
};

/* ── byte classes ── */

static void class_add(fp_class *c, unsigned char b) { c->w[b >> 6] |= (uint64_t)1 << (b & 63); }

static bool class_has(const fp_class *c, unsigned char b) {
    return (c->w[b >> 6] >> (b & 63)) & 1;
}

static void class_add_range(fp_class *c, unsigned char lo, unsigned char hi) {
    for (int b = lo; b <= hi; b++) class_add(c, (unsigned char)b);
}

static void class_union(fp_class *dst, const fp_class *src) {
    for (int i = 0; i < 4; i++) dst->w[i] |= src->w[i];
}

static void class_negate(fp_class *c) {
    for (int i = 0; i < 4; i++) c->w[i] = ~c->w[i];
}

static bool class_empty(const fp_class *c) {
    return !(c->w[0] | c->w[1] | c->w[2] | c->w[3]);
}

/* The set behind a \d \s \w letter (lower case), negated for upper case.
 * Returns false when `letter` is not one of the six. */
static bool class_shorthand(char letter, fp_class *out) {
    memset(out, 0, sizeof(*out));
    switch (letter | 0x20) {
    case 'd': class_add_range(out, '0', '9'); break;
    case 's':
        class_add(out, ' '); class_add(out, '\t'); class_add(out, '\n');
        class_add(out, '\r'); class_add(out, '\f'); class_add(out, '\v');
        break;
    case 'w':
        class_add_range(out, 'a', 'z'); class_add_range(out, 'A', 'Z');
        class_add_range(out, '0', '9'); class_add(out, '_');
        break;
    default: return false;
    }
    if (letter >= 'A' && letter <= 'Z') class_negate(out);
    return true;
}

/* The single byte a \n \t \r \f \v or \<punct> escape stands for; -1 if none. */
static int escape_byte(char c) {
    switch (c) {
    case 'n': return '\n';
    case 't': return '\t';
    case 'r': return '\r';
    case 'f': return '\f';
    case 'v': return '\v';
    default: break;
    }
    unsigned char u = (unsigned char)c;
    if (u >= 0x21 && u <= 0x7e && !((u >= '0' && u <= '9') || ((u | 0x20) >= 'a' && (u | 0x20) <= 'z')))
        return u;
    return -1;
}

/* ── parse ── */

typedef struct {
    const char *src;
    size_t      len, pos;
    int         depth;
    fp_node    *nodes;
    int         n_nodes, cap;
    const char *error;      /* first error, static text */
    size_t      error_pos;
} fp_parser;

static int parse_alt(fp_parser *ps);

static bool parse_fail(fp_parser *ps, const char *why) {
    if (!ps->error) { ps->error = why; ps->error_pos = ps->pos; }
    return false;
}

static int new_node(fp_parser *ps, fp_node_kind kind, int left, int right) {
    if (ps->n_nodes >= ps->cap) { parse_fail(ps, "pattern too complex"); return -1; }
    fp_node *n = &ps->nodes[ps->n_nodes];
    memset(n, 0, sizeof(*n));
    n->kind = kind; n->left = left; n->right = right;
    return ps->n_nodes++;
}

/* One class member at ps->pos: a byte (returned) or a shorthand set (-2, `set`
 * filled).  -1 on error. */
static int parse_class_member(fp_parser *ps, fp_class *set) {
    char c = ps->src[ps->pos];
    if (c != '\\') { ps->pos++; return (unsigned char)c; }
    if (ps->pos + 1 >= ps->len) { parse_fail(ps, "pattern ends in a lone '\\'"); return -1; }
    char e = ps->src[ps->pos + 1];
    if (class_shorthand(e, set)) { ps->pos += 2; return -2; }
    int b = escape_byte(e);
    if (b < 0) { parse_fail(ps, "unsupported escape"); return -1; }
    ps->pos += 2;
    return b;
}

/* "[...]" / "[^...]" starting at ps->pos (on the '['). */
static int parse_class(fp_parser *ps) {
    size_t open = ps->pos++;
    fp_class cls = { { 0 } }, set;
    bool negate = ps->pos < ps->len && ps->src[ps->pos] == '^';
    if (negate) ps->pos++;
    bool first = true;
    for (;;) {
        if (ps->pos >= ps->len) { ps->pos = open; parse_fail(ps, "unterminated '['"); return -1; }
        if (ps->src[ps->pos] == ']' && !first) { ps->pos++; break; }
        first = false;
        size_t at = ps->pos;
        int lo = parse_class_member(ps, &set);
        if (lo == -1) return -1;
        if (lo == -2) { class_union(&cls, &set); continue; }
        bool range = ps->pos + 1 < ps->len && ps->src[ps->pos] == '-' && ps->src[ps->pos + 1] != ']';
        if (!range) { class_add(&cls, (unsigned char)lo); continue; }
        ps->pos++;
        int hi = parse_class_member(ps, &set);
        if (hi == -1) return -1;
        if (hi == -2 || hi < lo) { ps->pos = at; parse_fail(ps, "bad range in class"); return -1; }
        class_add_range(&cls, (unsigned char)lo, (unsigned char)hi);
    }
    if (negate) class_negate(&cls);
    if (class_empty(&cls)) { ps->pos = open; parse_fail(ps, "class matches no byte"); return -1; }
    int n = new_node(ps, FPN_ATOM, -1, -1);
    if (n >= 0) ps->nodes[n].cls = cls;
    return n;
}

static int atom_node(fp_parser *ps, const fp_class *cls) {
    int n = new_node(ps, FPN_ATOM, -1, -1);
    if (n >= 0) ps->nodes[n].cls = *cls;
    return n;
}

static int parse_atom(fp_parser *ps) {
    char c = ps->src[ps->pos];
    fp_class cls = { { 0 } };
    switch (c) {
    case '(': {
        size_t open = ps->pos++;
        ps->depth++;
        int inner = parse_alt(ps);
        if (inner < 0) return -1;
        if (ps->pos >= ps->len || ps->src[ps->pos] != ')') {
            ps->pos = open;
            parse_fail(ps, "unbalanced '('");
            return -1;
        }
        ps->pos++;
        ps->depth--;
        return inner;
    }
    case '[':
        return parse_class(ps);
    case '.':
        ps->pos++;
        class_add_range(&cls, 0, 255);
        cls.w['\n' >> 6] &= ~((uint64_t)1 << ('\n' & 63));
        return atom_node(ps, &cls);
    case '\\': {
        int b = parse_class_member(ps, &cls);
        if (b == -1) return -1;
        if (b >= 0) class_add(&cls, (unsigned char)b);
        return atom_node(ps, &cls);
    }
    case '*': case '+': case '?': case '{':
        parse_fail(ps, "quantifier with nothing to repeat");
        return -1;
    case '^':
        parse_fail(ps, "'^' is only allowed at the start");
        return -1;
    case '$':
        parse_fail(ps, "'$' is only allowed at the end");
        return -1;
    default:
        ps->pos++;
        class_add(&cls, (unsigned char)c);
        return atom_node(ps, &cls);
    }
}

/* A decimal repeat count at ps->pos; -1 when there is none or it is too big. */
static int parse_count(fp_parser *ps) {
    int value = -1;
    while (ps->pos < ps->len && ps->src[ps->pos] >= '0' && ps->src[ps->pos] <= '9') {
        value = (value < 0 ? 0 : value) * 10 + (ps->src[ps->pos++] - '0');
        if (value > FP_MAX_REPEAT) return -2;
    }
    return value;
}

/* "{m}" "{m,}" "{m,n}" at ps->pos (on the '{') into min/max (max -1 = open). */
static bool parse_braces(fp_parser *ps, int *min, int *max) {
    size_t open = ps->pos++;
    int lo = parse_count(ps), hi = lo;
    if (lo >= 0 && ps->pos < ps->len && ps->src[ps->pos] == ',') {
        ps->pos++;
        hi = parse_count(ps);
        if (hi == -1) hi = -3;   /* "{m,}" */
    }
    if (lo == -2 || hi == -2) { ps->pos = open; return parse_fail(ps, "repeat count too large"); }
    if (lo < 0 || ps->pos >= ps->len || ps->src[ps->pos] != '}') {
        ps->pos = open;
        return parse_fail(ps, "bad '{' repeat");
    }
    ps->pos++;
    if (hi == -3) hi = -1;
    else if (hi < lo) { ps->pos = open; return parse_fail(ps, "repeat range is backwards"); }
    *min = lo; *max = hi;
    return true;
}

static int parse_repeat(fp_parser *ps) {
    int atom = parse_atom(ps);
    if (atom < 0 || ps->pos >= ps->len) return atom;
    int min, max;
    switch (ps->src[ps->pos]) {
    case '*': min = 0; max = -1; ps->pos++; break;
    case '+': min = 1; max = -1; ps->pos++; break;
    case '?': min = 0; max = 1;  ps->pos++; break;
    case '{': if (!parse_braces(ps, &min, &max)) return -1; break;
    default:  return atom;
    }
    if (ps->pos < ps->len && strchr("*+?{", ps->src[ps->pos])) {
        parse_fail(ps, "stacked or lazy quantifier");
        return -1;
    }
    int n = new_node(ps, FPN_REPEAT, atom, -1);
    if (n >= 0) { ps->nodes[n].min = min; ps->nodes[n].max = max; }
    return n;
}

static int parse_concat(fp_parser *ps) {
    int acc = -1;
    while (ps->pos < ps->len) {
        char c = ps->src[ps->pos];
        if (c == '|' || c == ')') break;
        if (c == '$' && ps->pos + 1 == ps->len && ps->depth == 0) { ps->pos++; break; }
        int item = parse_repeat(ps);
        if (item < 0) return -1;
        acc = (acc < 0) ? item : new_node(ps, FPN_CONCAT, acc, item);
        if (acc < 0) return -1;
    }
    return acc >= 0 ? acc : new_node(ps, FPN_EMPTY, -1, -1);
}

static int parse_alt(fp_parser *ps) {
    int left = parse_concat(ps);
    while (left >= 0 && ps->pos < ps->len && ps->src[ps->pos] == '|') {
        ps->pos++;
        int right = parse_concat(ps);
        if (right < 0) return -1;
        left = new_node(ps, FPN_ALT, left, right);
    }
    return left;
}

/* ── build (first / last / follow) ── */

typedef struct {
    bool   nullable;
    fp_set first, last;
} fp_frag;

/* Visits of the AST; bounds the time a pattern full of empty repeats can take. */
#define FP_MAX_BUILD_STEPS 200000

typedef struct {
    const fp_node *nodes;
    field_pattern *out;
    fp_class       state_class[FP_MAX_STATES];
    long           steps;
    const char    *error;
} fp_builder;

static void set_add(fp_set *s, int q) { s->w[q >> 6] |= (uint64_t)1 << (q & 63); }

static void set_union(fp_set *dst, const fp_set *src) {
    for (int i = 0; i < FP_SET_WORDS; i++) dst->w[i] |= src->w[i];
}

static int lowest_bit(uint64_t bits) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(bits);
#else
    int i = 0;
    while (!(bits & 1)) { bits >>= 1; i++; }
    return i;
#endif
}

/* follow(q) |= targets, for every state q in `from`. */
static void link_states(field_pattern *p, const fp_set *from, const fp_set *targets) {
    for (int w = 0; w < FP_SET_WORDS; w++)
        for (uint64_t bits = from->w[w]; bits; bits &= bits - 1)
            set_union(&p->follow[w * 64 + lowest_bit(bits)], targets);
}

/* `acc` := acc followed by `next`. */
static void frag_concat(field_pattern *p, fp_frag *acc, const fp_frag *next) {
    link_states(p, &acc->last, &next->first);
    if (acc->nullable) set_union(&acc->first, &next->first);
    fp_set last = next->last;
    if (next->nullable) set_union(&last, &acc->last);
    acc->last = last;
    acc->nullable = acc->nullable && next->nullable;
}

static bool build_node(fp_builder *b, int index, fp_frag *out);

/* x{min,max}: min copies of x then max-min optional ones; with max open, the
 * last copy loops back on itself (x* is one looping optional copy, x+ one
 * looping required copy). */
static bool build_repeat(fp_builder *b, const fp_node *n, fp_frag *out) {
    memset(out, 0, sizeof(*out));
    out->nullable = true;
    int copies = (n->max >= 0) ? n->max : (n->min > 0 ? n->min : 1);
    for (int i = 0; i < copies; i++) {
        fp_frag copy;
        if (!build_node(b, n->left, &copy)) return false;
        if (n->max < 0 && i == copies - 1) link_states(b->out, &copy.last, &copy.first);
        if (i >= n->min) copy.nullable = true;
        frag_concat(b->out, out, &copy);
    }
    return true;
}

static bool build_node(fp_builder *b, int index, fp_frag *out) {
    const fp_node *n = &b->nodes[index];
    if (++b->steps > FP_MAX_BUILD_STEPS) { b->error = "pattern too complex"; return false; }
    memset(out, 0, sizeof(*out));
    switch (n->kind) {
    case FPN_EMPTY:
        out->nullable = true;
        return true;
    case FPN_ATOM: {
        int q = b->out->n_states;
        if (q >= FP_MAX_STATES) { b->error = "pattern needs more than 128 states"; return false; }
        b->out->n_states++;
        b->state_class[q] = n->cls;
        set_add(&out->first, q);
        set_add(&out->last, q);
        return true;
    }
    case FPN_CONCAT: {
        fp_frag right;
        if (!build_node(b, n->left, out) || !build_node(b, n->right, &right)) return false;
        frag_concat(b->out, out, &right);
        return true;
    }
    case FPN_ALT: {
        fp_frag right;
        if (!build_node(b, n->left, out) || !build_node(b, n->right, &right)) return false;
        set_union(&out->first, &right.first);
        set_union(&out->last, &right.last);
        out->nullable = out->nullable || right.nullable;
        return true;
    }
    case FPN_REPEAT:
    default:
        return build_repeat(b, n, out);
    }
}

static bool set_has(const fp_set *s, int q) { return (s->w[q >> 6] >> (q & 63)) & 1; }

/* dist[q] = fewest bytes from q to a match: 0 on an accepting state, else one
 * more than the nearest state in follow(q).  Relaxed to a fixed point (every
 * state reaches acceptance, so every dist ends finite). */
static void compute_distances(field_pattern *p) {
    const int unreached = FP_MAX_STATES + 1;
    for (int q = 0; q < p->n_states; q++) p->dist[q] = set_has(&p->accept, q) ? 0 : unreached;
    for (bool changed = true; changed;) {
        changed = false;
        for (int q = 0; q < p->n_states; q++) {
            const fp_set *f = &p->follow[q];
            for (int w = 0; w < FP_SET_WORDS; w++)
                for (uint64_t bits = f->w[w]; bits; bits &= bits - 1) {
                    int d = p->dist[w * 64 + lowest_bit(bits)] + 1;
                    if (d < p->dist[q]) { p->dist[q] = d; changed = true; }
                }
        }
    }
}

/* Fill p's tables from the parsed AST rooted at `root`. */
static const char *build_automaton(field_pattern *p, const fp_node *nodes, int root) {
    fp_builder *b = calloc(1, sizeof(*b));
    if (!b) return "out of memory";
    b->nodes = nodes;
    b->out = p;
    p->n_states = 1;   /* state 0 = start */
    fp_frag whole;
    const char *error = NULL;
    if (!build_node(b, root, &whole)) {
        error = b->error;
    } else {
        p->follow[0] = whole.first;
        p->accept = whole.last;
        if (whole.nullable) set_add(&p->accept, 0);
        for (int q = 1; q < p->n_states; q++)
            for (int byte = 0; byte < 256; byte++)
                if (class_has(&b->state_class[q], (unsigned char)byte))
                    set_add(&p->byte_mask[byte], q);
        compute_distances(p);
    }
    free(b);
    return error;
}

/* ── public ── */

static field_pattern *compile_failed(char *err, size_t err_cap, const char *why, long at) {
    if (err && err_cap > 0) {
        if (at >= 0) snprintf(err, err_cap, "%s at byte %ld", why, at);
        else snprintf(err, err_cap, "%s", why);
    }
    return NULL;
}

field_pattern *fp_compile(const char *src, char *err, size_t err_cap) {
    if (!src) return compile_failed(err, err_cap, "pattern is not a string", -1);
    size_t len = strlen(src);
    if (len > FP_MAX_PATTERN_LEN)
        return compile_failed(err, err_cap, "pattern longer than 1024 bytes", -1);
    fp_parser ps = { .src = src, .len = len };
    if (len > 0 && src[0] == '^') ps.pos = 1;
    ps.cap = 3 * (int)len + 4;
    ps.nodes = calloc((size_t)ps.cap, sizeof(fp_node));
    if (!ps.nodes) return compile_failed(err, err_cap, "out of memory", -1);
    int root = parse_alt(&ps);
    if (root >= 0 && ps.pos < ps.len) { parse_fail(&ps, "unbalanced ')'"); root = -1; }
    if (root < 0) {
        compile_failed(err, err_cap, ps.error ? ps.error : "bad pattern", (long)ps.error_pos);
        free(ps.nodes);
        return NULL;
    }
    field_pattern *p = calloc(1, sizeof(*p));
    const char *error = p ? build_automaton(p, ps.nodes, root) : "out of memory";
    free(ps.nodes);
    if (error) { free(p); return compile_failed(err, err_cap, error, -1); }
    return p;
}

void fp_free(field_pattern *p) { free(p); }

fp_set fp_start(const field_pattern *p) {
    fp_set s = { { 0 } };
    (void)p;
    set_add(&s, 0);
    return s;
}

bool fp_step(const field_pattern *p, fp_set *s, unsigned char byte) {
    fp_set next = { { 0 } };
    for (int w = 0; w < FP_SET_WORDS; w++)
        for (uint64_t bits = s->w[w]; bits; bits &= bits - 1)
            set_union(&next, &p->follow[w * 64 + lowest_bit(bits)]);
    bool live = false;
    for (int i = 0; i < FP_SET_WORDS; i++) {
        next.w[i] &= p->byte_mask[byte].w[i];
        live = live || next.w[i];
    }
    *s = next;
    return live;
}

bool fp_accepting(const field_pattern *p, const fp_set *s) {
    for (int i = 0; i < FP_SET_WORDS; i++)
        if (s->w[i] & p->accept.w[i]) return true;
    return false;
}

bool fp_set_empty(const fp_set *s) {
    for (int i = 0; i < FP_SET_WORDS; i++)
        if (s->w[i]) return false;
    return true;
}

int fp_state_count(const field_pattern *p) { return p ? p->n_states : 0; }

/* Fewest bytes from `s` to a match; -1 for the empty set. */
static int set_distance(const field_pattern *p, const fp_set *s) {
    int best = -1;
    for (int w = 0; w < FP_SET_WORDS; w++)
        for (uint64_t bits = s->w[w]; bits; bits &= bits - 1) {
            int d = p->dist[w * 64 + lowest_bit(bits)];
            if (best < 0 || d < best) best = d;
        }
    return best;
}

int fp_forced_continuation(const field_pattern *p, const fp_set *s, char *out, int cap) {
    if (!p || !s || !out || cap < 1) return 0;
    fp_set walk = *s;
    int n = 0;
    out[0] = '\0';
    for (int d = set_distance(p, &walk); d > 0 && n < cap - 1; d--) {
        int forced = -1;
        fp_set next_of_forced;
        for (int byte = 0; byte < 256; byte++) {
            fp_set next = walk;
            if (!fp_step(p, &next, (unsigned char)byte) || set_distance(p, &next) != d - 1)
                continue;
            if (forced >= 0) { forced = -2; break; }   /* two ways on: not forced */
            forced = byte;
            next_of_forced = next;
        }
        if (forced < 0) break;
        out[n++] = (char)forced;
        walk = next_of_forced;
    }
    out[n] = '\0';
    return n;
}

bool fp_full_match(const field_pattern *p, const char *text, size_t len) {
    fp_set s = fp_start(p);
    for (size_t i = 0; i < len; i++)
        if (!fp_step(p, &s, (unsigned char)text[i])) return false;
    return fp_accepting(p, &s);
}
