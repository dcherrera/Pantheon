/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_pick_port — lt_pick_port skips a port another program already listens
 * on (a stray stub server in 18400-18999 once made test_local_hook_adopt connect to it).
 */
#include "local_test_support.h"

int main(void)
{
    int first = lt_pick_port(5), fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a;

    memset(&a, 0, sizeof(a));
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)first);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    printf("pick_port: a held port is skipped\n");
    CHECK(fd >= 0 && bind(fd, (struct sockaddr *)&a, sizeof(a)) == 0 && listen(fd, 1) == 0,
          "hold the first pick");
    int second = lt_pick_port(5);
    CHECK(second != first, "the held port is not picked again");
    CHECK(second >= 18400 && second < 19000, "the pick stays in 18400-18999");
    CHECK(lt_port_free(second), "the new pick is free");
    close(fd);
    CHECK(lt_pick_port(5) == first, "a released port is picked again");
    return lt_report("test_local_pick_port") ? 1 : 0;
}
