/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   mdns_responder.c
 * @brief  Implementation of the mDNS responder thread (see mdns_responder.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "mdns_responder.h"

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include "local_random.h"

#define MAX_IFACES      16
#define PROBE_ATTEMPTS  10
#define IFACE_REFRESH_S 30

typedef struct {
    struct in_addr addr, mask;
} iface;

struct pn_mdns_responder {
    pthread_t       thread;
    pthread_mutex_t lock;            /* guards label */
    int             sock;
    int             wake[2];         /* stop pipe */
    char            base[PN_MDNS_LABEL_MAX + 1];
    char            label[PN_MDNS_LABEL_MAX + 1];
    uint16_t        port;
    pn_mdns_txt_fn  txt_fn;
    void           *txt_user;
    iface           ifaces[MAX_IFACES];
    int             n_ifaces;
    time_t          ifaces_at;
};

/* Re-read the IPv4 multicast interfaces and join the group on new ones. */
static void refresh_ifaces(pn_mdns_responder *r)
{
    struct ifaddrs *list = NULL;

    if (getifaddrs(&list) != 0)
        return;
    r->n_ifaces = 0;
    for (struct ifaddrs *a = list; a && r->n_ifaces < MAX_IFACES; a = a->ifa_next) {
        if (!a->ifa_addr || a->ifa_addr->sa_family != AF_INET || !a->ifa_netmask)
            continue;
        if (!(a->ifa_flags & IFF_UP) || !(a->ifa_flags & IFF_MULTICAST) ||
            (a->ifa_flags & (IFF_LOOPBACK | IFF_POINTOPOINT)))
            continue;
        iface *f = &r->ifaces[r->n_ifaces++];
        f->addr = ((struct sockaddr_in *)a->ifa_addr)->sin_addr;
        f->mask = ((struct sockaddr_in *)a->ifa_netmask)->sin_addr;
        struct ip_mreq mr;
        mr.imr_multiaddr.s_addr = inet_addr(PN_MDNS_GROUP);
        mr.imr_interface = f->addr;
        /* EADDRINUSE = already joined on this interface: fine. */
        setsockopt(r->sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mr, sizeof(mr));
    }
    freeifaddrs(list);
    r->ifaces_at = time(NULL);
}

static int open_socket(void)
{
    int fd = socket(AF_INET, SOCK_DGRAM, 0), one = 1;
    unsigned char ttl = 255, loop = 1;
    struct sockaddr_in a = {0};

    if (fd < 0)
        return -1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#ifdef SO_REUSEPORT
    setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one));
#endif
    a.sin_family = AF_INET;
    a.sin_port = htons(PN_MDNS_PORT);
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(fd, (struct sockaddr *)&a, sizeof(a)) != 0) {
        fprintf(stderr, "local: mdns bind :5353 failed: %s\n", strerror(errno));
        close(fd);
        return -1;
    }
    setsockopt(fd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl));
    setsockopt(fd, IPPROTO_IP, IP_MULTICAST_LOOP, &loop, sizeof(loop));
    return fd;
}

/* The service as seen on interface f (its address in the A record). */
static void make_service(pn_mdns_responder *r, const iface *f, pn_mdns_service *s,
                         char txt[][PN_MDNS_TXT_ITEM])
{
    memset(s, 0, sizeof(*s));
    pthread_mutex_lock(&r->lock);
    snprintf(s->label, sizeof(s->label), "%s", r->label);
    pthread_mutex_unlock(&r->lock);
    s->port = r->port;
    s->ipv4 = f ? f->addr.s_addr : 0;
    s->n_txt = r->txt_fn ? r->txt_fn(r->txt_user, txt, PN_MDNS_MAX_TXT) : 0;
    for (int i = 0; i < s->n_txt; i++)
        s->txt[i] = txt[i];
}

static void send_to(pn_mdns_responder *r, const iface *f, const uint8_t *pkt, size_t len,
                    const struct sockaddr_in *dst)
{
    struct sockaddr_in group = {0};

    if (!dst) {
        group.sin_family = AF_INET;
        group.sin_port = htons(PN_MDNS_PORT);
        group.sin_addr.s_addr = inet_addr(PN_MDNS_GROUP);
        dst = &group;
    }
    if (f)
        setsockopt(r->sock, IPPROTO_IP, IP_MULTICAST_IF, &f->addr, sizeof(f->addr));
    if (sendto(r->sock, pkt, len, 0, (const struct sockaddr *)dst, sizeof(*dst)) < 0)
        fprintf(stderr, "local: mdns send failed: %s\n", strerror(errno));
}

/* Announce (or say goodbye) on every interface. */
static void announce_all(pn_mdns_responder *r, bool goodbye)
{
    uint8_t pkt[1500];
    char txt[PN_MDNS_MAX_TXT][PN_MDNS_TXT_ITEM];
    pn_mdns_service s;

    for (int i = 0; i < r->n_ifaces; i++) {
        make_service(r, &r->ifaces[i], &s, txt);
        size_t n = pn_mdns_build_announce(&s, goodbye, pkt, sizeof(pkt));
        if (n)
            send_to(r, &r->ifaces[i], pkt, n, NULL);
    }
}

/* The interface whose subnet holds `src`, else the first one. */
static const iface *iface_for(const pn_mdns_responder *r, struct in_addr src)
{
    for (int i = 0; i < r->n_ifaces; i++)
        if ((src.s_addr & r->ifaces[i].mask.s_addr) ==
            (r->ifaces[i].addr.s_addr & r->ifaces[i].mask.s_addr))
            return &r->ifaces[i];
    return r->n_ifaces ? &r->ifaces[0] : NULL;
}

static long monotonic_ms(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/* Wait up to ms for packets; returns true if one conflicts with `s`. */
static bool listen_for_conflict(pn_mdns_responder *r, const pn_mdns_service *s, int ms)
{
    uint8_t pkt[9000];
    pn_mdns_message m;
    struct pollfd pf = { r->sock, POLLIN, 0 };
    long deadline = monotonic_ms() + ms;

    for (long left = ms; left > 0; left = deadline - monotonic_ms()) {
        if (poll(&pf, 1, (int)left) <= 0)
            break;
        ssize_t n = recv(r->sock, pkt, sizeof(pkt), 0);
        if (n > 0 && pn_mdns_parse(pkt, (size_t)n, &m) == 0 && pn_mdns_conflicts(&m, s))
            return true;
    }
    return false;
}

/* RFC 6762 §8.1: three probes 250 ms apart; rename on a conflict. */
static void probe_name(pn_mdns_responder *r)
{
    uint8_t pkt[1500];
    char txt[PN_MDNS_MAX_TXT][PN_MDNS_TXT_ITEM];
    uint32_t jitter = 0;

    pn_local_random_below(250, &jitter);
    usleep(jitter * 1000);
    for (int attempt = 1; attempt <= PROBE_ATTEMPTS; attempt++) {
        pthread_mutex_lock(&r->lock);
        pn_mdns_conflict_label(r->base, attempt, r->label, sizeof(r->label));
        pthread_mutex_unlock(&r->lock);
        bool conflict = false;
        for (int round = 0; round < 3 && !conflict; round++) {
            pn_mdns_service s;
            for (int i = 0; i < r->n_ifaces; i++) {
                make_service(r, &r->ifaces[i], &s, txt);
                size_t n = pn_mdns_build_probe(&s, pkt, sizeof(pkt));
                if (n)
                    send_to(r, &r->ifaces[i], pkt, n, NULL);
            }
            make_service(r, NULL, &s, txt);
            conflict = listen_for_conflict(r, &s, 250);
        }
        if (!conflict)
            return;
        fprintf(stderr, "local: mdns name %s.local is taken; trying the next one\n", r->label);
    }
}

/* Answer one received packet. */
static void handle_packet(pn_mdns_responder *r, const uint8_t *pkt, size_t len,
                          const struct sockaddr_in *src)
{
    pn_mdns_message q;
    pn_mdns_service s;
    char txt[PN_MDNS_MAX_TXT][PN_MDNS_TXT_ITEM];
    uint8_t out[1500];

    if (pn_mdns_parse(pkt, len, &q) != 0 || q.is_response || q.n_questions == 0)
        return;
    if (time(NULL) - r->ifaces_at > IFACE_REFRESH_S)
        refresh_ifaces(r);
    const iface *f = iface_for(r, src->sin_addr);
    const bool legacy = ntohs(src->sin_port) != PN_MDNS_PORT;
    make_service(r, f, &s, txt);
    size_t n = pn_mdns_build_answer(&s, &q, legacy, out, sizeof(out));
    if (n)
        send_to(r, f, out, n, legacy ? src : NULL);
}

static void *responder_main(void *arg)
{
    pn_mdns_responder *r = arg;
    uint8_t pkt[9000];
    struct pollfd pf[2] = { { r->sock, POLLIN, 0 }, { r->wake[0], POLLIN, 0 } };

    probe_name(r);
    fprintf(stderr, "local: discoverable as %s.local (_pantheon._tcp port %u)\n", r->label,
            (unsigned)r->port);
    announce_all(r, false);
    for (int second = 0;; second++) {
        if (second == 1)
            announce_all(r, false);   /* RFC 6762 §8.3: announce twice, 1 s apart */
        if (poll(pf, 2, 1000) < 0 && errno != EINTR)
            break;
        if (pf[1].revents & POLLIN)
            break;
        if (!(pf[0].revents & POLLIN))
            continue;
        struct sockaddr_in src;
        socklen_t sl = sizeof(src);
        ssize_t n = recvfrom(r->sock, pkt, sizeof(pkt), 0, (struct sockaddr *)&src, &sl);
        if (n > 0)
            handle_packet(r, pkt, (size_t)n, &src);
    }
    announce_all(r, true);
    return NULL;
}

pn_mdns_responder *pn_mdns_start(const char *hostname, uint16_t port, pn_mdns_txt_fn txt,
                                 void *user)
{
    pn_mdns_responder *r = calloc(1, sizeof(*r));

    if (!r)
        return NULL;
    pthread_mutex_init(&r->lock, NULL);
    r->wake[0] = r->wake[1] = -1;
    pn_mdns_label_for_host(hostname, r->base, sizeof(r->base));
    snprintf(r->label, sizeof(r->label), "%s", r->base);
    r->port = port;
    r->txt_fn = txt;
    r->txt_user = user;
    r->sock = open_socket();
    if (r->sock < 0 || pipe(r->wake) != 0)
        goto fail;
    refresh_ifaces(r);
    if (r->n_ifaces == 0)
        fprintf(stderr, "local: mdns: no multicast IPv4 interface yet; answering when one is up\n");
    if (pthread_create(&r->thread, NULL, responder_main, r) != 0)
        goto fail;
    return r;
fail:
    if (r->sock >= 0) close(r->sock);
    if (r->wake[0] >= 0) { close(r->wake[0]); close(r->wake[1]); }
    pthread_mutex_destroy(&r->lock);
    free(r);
    return NULL;
}

void pn_mdns_stop(pn_mdns_responder *r)
{
    if (!r)
        return;
    if (write(r->wake[1], "x", 1) != 1)
        fprintf(stderr, "local: mdns stop signal failed\n");
    pthread_join(r->thread, NULL);
    close(r->sock);
    close(r->wake[0]);
    close(r->wake[1]);
    pthread_mutex_destroy(&r->lock);
    free(r);
}

void pn_mdns_current_label(pn_mdns_responder *r, char *out, size_t cap)
{
    pthread_mutex_lock(&r->lock);
    snprintf(out, cap, "%s", r->label);
    pthread_mutex_unlock(&r->lock);
}
