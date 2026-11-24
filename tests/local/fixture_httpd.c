/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * fixture_httpd.c — standalone static file server for the Pantheon Local
 * on-hardware scripts (stands in for Hugging Face: serve a directory laid out as
 * <org>/<repo>/resolve/<revision>/<file>). Test infrastructure only.
 *
 *   build/fixture_httpd <port> <root dir> [cut_after_bytes]
 */

#include "../local_fixture_httpd.h"

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: fixture_httpd <port> <root> [cut_after_bytes]\n");
        return 2;
    }
    /* Loopback only: the box downloads from it on the same host. */
    return fx_httpd_run(atoi(argv[1]), argv[2], argc > 3 ? atol(argv[3]) : 0, 0);
}
