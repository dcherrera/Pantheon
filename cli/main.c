/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/pantheon.h"
#include "../src/accel/backend_identity.h"
#include "../src/core/bench.h"
#include "../src/core/sample_profiler.h"
#include "../src/core/engine.h"
#include "../src/core/chat.h"
#include "../src/serve/serve_command.h"
#include "../src/serve/router_command.h"
#include "tensor/tensor.h"

#define PN_VERSION "0.4.0"

static void print_usage(const char *prog) {
    printf("Pantheon v%s — Pure C Model Hosting Engine\n\n", PN_VERSION);
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --model <path>        Model file (GGUF or safetensors)\n");
    printf("  --config <path>       HuggingFace config.json (for safetensors models)\n");
    printf("  --vocab <path>        RWKV World tokenizer vocab file\n");
    printf("  --prompt <text>       Input prompt (one-shot mode; chat template applied unless --raw)\n");
    printf("  --raw                 One-shot --prompt: send raw text, no chat template\n");
    printf("  --chat                Interactive chat mode\n");
    printf("  --system <text>       System prompt (chat mode only)\n");
    printf("  --max-tokens <n>      Maximum tokens to generate (default: 128)\n");
    printf("  --temperature <f>     Sampling temperature (default: 0.8)\n");
    printf("  --top-p <f>           Nucleus sampling threshold (default: 0.9)\n");
    printf("  --top-k <n>           Top-k sampling (default: 40)\n");
    printf("  --quantize            INT8 quantize weights at load time\n");
    printf("  --bench               Run benchmark (report tok/s and memory)\n");
    printf("  --help                Show this help\n");
    printf("  --version             Show version, compiled backend and build commit\n");
    printf("  --backend             Print ONLY the compiled backend token\n");
    printf("                        (cuda|tt|metal|cpu) and exit — the deploy-time\n");
    printf("                        assertion that this binary is the backend the\n");
    printf("                        fleet manifest expects\n");
}

static void stream_print_cb(const char *token, void *user_data) {
    (void)user_data;
    printf("%s", token);
    fflush(stdout);
}

int main(int argc, char **argv) {
    pn_sample_profiler_start(); /* PN_PROFILE=<file>: a SIGPROF sample profile, dumped at exit */
    /* Subcommand dispatch. `pantheon serve ...` starts the OpenAI HTTP server;
     * everything else is the original flag-driven CLI below. */
    if (argc > 1 && strcmp(argv[1], "serve") == 0)
        return pn_serve_main(argc - 1, argv + 1);
    /* `pantheon router ...` runs the dispatch layer (registry + LB router).
     * Pure host-side networking, no model load / CUDA. */
    if (argc > 1 && strcmp(argv[1], "router") == 0)
        return pn_router_main(argc - 1, argv + 1);

    const char *model_path = NULL;
    const char *config_path = NULL;
    const char *vocab_path = NULL;
    const char *prompt = NULL;
    const char *system_prompt = NULL;
    int max_tokens = 128;
    int bench = 0;
    int chat_mode = 0;
    int quantize = 0;
    int raw = 0;   /* --raw: bypass the chat template in one-shot --prompt mode */
    pn_sampling_params params = pn_sampling_defaults();

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        /* `--backend` prints ONE lowercase token and nothing else, so a deploy
         * script can assert on it with a plain string compare:
         *   [ "$("$BIN" --backend)" = cuda ] || exit 1
         * That is the whole contract — stdout is the token, exit status is 0.
         * Keep it the first thing parsed and never let anything else reach
         * stdout on this path. */
        if (strcmp(argv[i], "--backend") == 0) {
            printf("%s\n", pn_backend_token());
            return 0;
        }
        if (strcmp(argv[i], "--version") == 0) {
            /* First line byte-identical to what it always was — scripts parse
             * it. The backend goes on a SECOND line rather than into that one. */
            printf("Pantheon v%s\n", PN_VERSION);
            printf("backend=%s build=%s\n", pn_backend_token(),
                   pn_backend_build_commit());
            return 0;
        }
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (strcmp(argv[i], "--quantize") == 0) {
            quantize = 1;
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--chat") == 0) {
            chat_mode = 1;
        } else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            params.temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            params.top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            params.top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bench") == 0) {
            bench = 1;
        } else if (strcmp(argv[i], "--raw") == 0) {
            raw = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* If no model specified, just print version and verify Foundry links */
    if (!model_path) {
        printf("Pantheon v%s\n", PN_VERSION);
        printf("backend=%s build=%s\n", pn_backend_token(),
               pn_backend_build_commit());
        printf("Built on Foundry tensor library.\n\n");

        /* Verify Foundry links correctly */
        int64_t shape[] = {2, 3};
        tm_tensor *t = tm_ones(shape, 2, TM_F32);
        if (t) {
            printf("Foundry link check: OK (created %lldx%lld tensor)\n",
                   (long long)t->shape[0], (long long)t->shape[1]);
            tm_free(t);
        } else {
            printf("Foundry link check: FAILED\n");
            return 1;
        }

        if (!prompt && !chat_mode) {
            printf("\nRun with --model <path> --prompt <text> to generate text.\n");
            printf("Run with --model <path> --chat for interactive chat.\n");
            return 0;
        }
    }

    /* Load model */
    if (model_path) {
        double t0 = pn_time_ms();
        pn_load_opts opts = { .vocab_path = vocab_path, .config_path = config_path, .quantize = quantize };
        pn_model *model = pn_load_ex(model_path, &opts);
        if (!model) {
            fprintf(stderr, "Failed to load model: %s\n", model_path);
            return 1;
        }
        double load_ms = pn_time_ms() - t0;
        if (bench) fprintf(stderr, "  load: %.1f ms\n", load_ms);

        pn_state *state = pn_state_new(model);
        if (!state) {
            fprintf(stderr, "Failed to create inference state\n");
            pn_model_free(model);
            return 1;
        }

        if (chat_mode) {
            /* Interactive chat mode */
            pn_chat_style style = (pn_chat_style)model->chat_style;
            ft_tokenizer *tok = model->tokenizer;

            printf("Chat mode (%s). Type /quit to exit, /clear to reset.\n",
                   pn_chat_style_name(style));

            char line[4096];
            int turn = 0;

            while (1) {
                printf("> ");
                fflush(stdout);

                if (!fgets(line, sizeof(line), stdin)) break;

                /* Strip trailing newline */
                size_t len = strlen(line);
                if (len > 0 && line[len - 1] == '\n') line[len - 1] = '\0';

                /* Handle commands */
                if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0) break;
                if (strcmp(line, "/clear") == 0) {
                    pn_state_reset(state);
                    turn = 0;
                    printf("Conversation cleared.\n");
                    continue;
                }

                /* Skip empty lines */
                if (line[0] == '\0') continue;

                /* Format and generate */
                pn_token_seq seq;
                if (turn == 0)
                    seq = pn_chat_format_first(style, tok, system_prompt, line);
                else
                    seq = pn_chat_format_turn(style, tok, line);

                if (!seq.tokens || seq.n_tokens == 0) {
                    fprintf(stderr, "chat: tokenization failed\n");
                    free(seq.tokens);
                    continue;
                }

                /* The exact token ids this turn prefills with. A parity harness takes token
                 * ids, so without this the only way to re-run a chat prompt against the CPU
                 * oracle is to guess at the template's own tokenization. */
                if (getenv("PN_PROMPT_TOKEN_DUMP")) {
                    fprintf(stderr, "PROMPT_TOKENS %d:", seq.n_tokens);
                    for (int t = 0; t < seq.n_tokens; t++)
                        fprintf(stderr, " %d", seq.tokens[t]);
                    fprintf(stderr, "\n");
                }

                pn_stream_tokens(model, state, seq.tokens, seq.n_tokens,
                                 max_tokens, &params, stream_print_cb, NULL);
                printf("\n");
                free(seq.tokens);
                turn++;
            }
        } else if (bench) {
            /* Benchmark mode */
            pn_bench_result result = {0};
            result.memory_bytes = pn_memory_usage();

            double gen_start = pn_time_ms();
            char *output = pn_generate(model, state, prompt ? prompt : "Hello",
                                       max_tokens, &params);
            double gen_end = pn_time_ms();

            if (output) {
                printf("%s\n", output);
                free(output);
            }

            result.n_decode = max_tokens;
            result.decode_ms = gen_end - gen_start;
            result.memory_bytes = pn_memory_usage();
            pn_bench_report(&result);
        } else {
            /* One-shot generation. Apply the model's chat template (single
             * turn) so output matches the reference; --raw or a model with no
             * chat template falls back to raw completion. */
            const char *one_shot = prompt ? prompt : "Hello";
            pn_chat_style style = (pn_chat_style)model->chat_style;

            if (!raw && style != PN_CHAT_NONE) {
                pn_token_seq seq = pn_chat_format_first(style, model->tokenizer,
                                                        system_prompt, one_shot);
                if (seq.tokens && seq.n_tokens > 0) {
                    pn_stream_tokens(model, state, seq.tokens, seq.n_tokens,
                                     max_tokens, &params, stream_print_cb, NULL);
                    printf("\n");
                    free(seq.tokens);
                } else {
                    fprintf(stderr, "one-shot: chat formatting failed\n");
                    free(seq.tokens);
                }
            } else {
                char *output = pn_generate(model, state, one_shot,
                                           max_tokens, &params);
                if (output) {
                    printf("%s\n", output);
                    free(output);
                }
            }
        }

        pn_state_free(state);
        pn_model_free(model);
    }

    return 0;
}
