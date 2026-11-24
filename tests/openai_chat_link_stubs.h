/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file openai_chat_link_stubs.h
 * @brief Link stubs for a test that #includes src/serve/openai_chat.c to reach
 *        its static request parse (parse_chat_request) and links PN_SRC (the
 *        engine core) but not the scheduler, HTTP server or model registry.
 *
 * Every stub here is a handler-side external the request-PARSE path never
 * calls; they only satisfy the linker for the handler functions compiled into
 * the including translation unit.  Include it once, after openai_chat.c.
 * tests/test_chat_request.c keeps its own copies because several of its stubs
 * capture output its response-shape tests read back.
 *
 * @status Complete.
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_TESTS_OPENAI_CHAT_LINK_STUBS_H
#define PANTHEON_TESTS_OPENAI_CHAT_LINK_STUBS_H

void bs_notify_pressure(pn_batch_sched *a, int b) { (void)a; (void)b; }
bs_acquire_result bs_slot_acquire_or_wait(pn_batch_sched *a, int b, int *c,
                                          int *d, const char *e)
{ (void)a; (void)b; (void)c; (void)d; (void)e; return BS_ACQUIRE_OK; }
int hs_begin_sse(int a) { (void)a; return 0; }
int hs_client_alive(int a) { (void)a; return 1; }
int hs_end_sse(int a) { (void)a; return 0; }
void hs_send_error(int a, int b, const char *c, const char *d)
{ (void)a; (void)b; (void)c; (void)d; }
void hs_send_error_ex(int a, int b, const char *c, const char *d, const char *e)
{ (void)a; (void)b; (void)c; (void)d; (void)e; }
void hs_send_json(int a, int b, const char *c) { (void)a; (void)b; (void)c; }
int hs_send_sse(int a, const char *b) { (void)a; (void)b; return 0; }
int hs_send_sse_comment(int a, const char *b) { (void)a; (void)b; return 0; }
mr_entry *mr_acquire(mr_registry *a, const char *b) { (void)a; (void)b; return NULL; }
mr_entry *mr_acquire_session(mr_registry *a, const char *b, const char *c)
{ (void)a; (void)b; (void)c; return NULL; }
void pn_replica_affinity_forget(const char *a) { (void)a; }
int pn_replica_pin_parse(const jr_value *a, const char *b, int *c, char *d, size_t e)
{ (void)a; (void)b; (void)d; (void)e; *c = 0; return 0; }
mr_entry *pn_replica_pin_acquire(int a, mr_registry *b, const char *c,
                                 const char *d, mr_route *e)
{ (void)a; (void)b; (void)c; (void)d; (void)e; return NULL; }
void pn_replica_pin_announce(const mr_entry *a) { (void)a; }
void pn_replica_pin_undo(const mr_route *a, const mr_entry *b, const char *c)
{ (void)a; (void)b; (void)c; }
void pn_replica_pin_send_full(int a, const mr_route *b, const mr_entry *c, int d)
{ (void)a; (void)b; (void)c; (void)d; }
int mr_count(mr_registry *a) { (void)a; return 0; }
void mr_release(mr_registry *a, mr_entry *b) { (void)a; (void)b; }
bool mr_turn_cancel_requested(const mr_entry *a) { (void)a; return false; }
int mr_snapshot(mr_registry *a, mr_info *b, int c) { (void)a; (void)b; (void)c; return 0; }
bool mr_model_is_minicpm(const pn_model *a) { (void)a; return false; }
int sg_run_completion(pn_model *a, pn_state *b, const sg_message *c, int d,
                      const sg_gen_params *e, sg_delta_fn f, void *g,
                      sg_result *h, const sg_prefix_ctx *i)
{ (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h; (void)i; return 0; }
int sg_run_completion_batched(pn_model *a, pn_batch_sched *b, int c,
                              const sg_message *d, int e, const sg_gen_params *f,
                              sg_delta_fn g, void *h, sg_result *i, const char *j)
{
    (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h;
    (void)i; (void)j; return 0;
}
void sg_result_free(sg_result *a) { (void)a; }
bool sg_tool_choice_required_with_thinking(tc_dialect d) { (void)d; return false; }
double sg_prefill_tok_per_s_measured(void) { return 0.0; }
uint64_t mr_weights_version(const mr_entry *a) { (void)a; return 0; }
unsigned char *pn_base64_decode(const char *a, size_t *b) { (void)a; if (b) *b = 0; return NULL; }
char *pn_base64_encode(const unsigned char *a, size_t b) { (void)a; (void)b; return NULL; }
void ar_write_summary(json_writer *a, const pn_model_attn_residency *b) { (void)a; (void)b; }
void gr_write_summary(json_writer *a, const pn_model_graph_residency *b) { (void)a; (void)b; }
const char *hs_thread_principal(void) { return NULL; }
int hs_thread_principal_admin(void) { return 0; }
mr_entry *mr_failover_session(mr_registry *a, const mr_entry *b, const char *c, mr_route *d)
{ (void)a; (void)b; (void)c; (void)d; return NULL; }
int mr_trylock_generation(mr_entry *a) { (void)a; return 0; }
uint64_t ms_now_ms(void) { return 0; }
bool ms_session_holds_seat(mr_entry *a, const char *b) { (void)a; (void)b; return false; }
uint64_t ms_session_idle_ttl_ms(void) { return 0; }
void ms_session_occupancy(const mr_entry *a, ms_occupancy *b) { (void)a; (void)b; }
ms_result ms_session_principal_check(mr_entry *a, const char *b, const char *c, bool d,
                                     ms_principal_use e)
{ (void)a; (void)b; (void)c; (void)d; (void)e; return (ms_result)0; }
void ms_session_release(mr_entry *a, const char *b) { (void)a; (void)b; }
ms_result ms_session_restore(mr_entry *a, const char *b, const void *c, const uint32_t *d, int e)
{ (void)a; (void)b; (void)c; (void)d; (void)e; return (ms_result)0; }

#endif /* PANTHEON_TESTS_OPENAI_CHAT_LINK_STUBS_H */
