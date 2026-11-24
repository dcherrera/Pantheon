/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   key_migration.c
 * @brief  Implementation of the keys.json v1 → v2 migration (see key_migration.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "key_migration.h"

#include <stdio.h>

int pn_key_migrate_v1(pn_key_store *ks, pn_people *people, const char *backup_path)
{
    pn_person owner = {0};
    int made = 0, n = 0;

    if (!ks || !people || !backup_path)
        return -1;
    if (!pn_key_store_is_v1(ks))
        return 0;
    n = pn_key_store_count(ks);
    /* An empty v1 file (never paired) gets no person: the first pair makes one. */
    if (n > 0 && pn_people_owner_or_create(people, "owner", "Owner", &owner, &made) != 0) {
        fprintf(stderr, "local: keys.json v1 -> v2: cannot find or create the owner\n");
        return -1;
    }
    if (pn_key_store_migrate_v1(ks, owner.id, backup_path) != PN_KEY_OK) {
        fprintf(stderr, "local: keys.json v1 -> v2: cannot write %s or keys.json\n",
                backup_path);
        return -1;
    }
    fprintf(stderr, "local: keys.json v1 -> v2: %d key(s) now belong to %s (%s%s); "
            "the v1 file is kept as %s\n", n, n ? owner.handle : "nobody", n ? owner.id : "-",
            made ? ", created" : "", backup_path);
    return 0;
}
