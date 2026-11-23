/*
 * prep.c — funzioni ausiliarie di preprocessamento (vedi prep.h).
 */
#define _POSIX_C_SOURCE 200809L   /* posix_memalign con -std=c11 */
#include <stdio.h>
#include <stdlib.h>
#include "prep.h"

real_t *prep_alloc_matrix(int nrows, int ld, int align)
{
    if (align < 16) align = 16;                   /* come malloc */
    size_t bytes = (size_t)nrows * ld * sizeof(real_t);
    if (bytes == 0) bytes = (size_t)align;        /* blocco vuoto: puntatore comunque valido */

    void *p = NULL;
    /* posix_memalign: come malloc, ma con l'indirizzo multiplo
       dell'allineamento richiesto (malloc garantisce solo 16 byte). */
    if (posix_memalign(&p, (size_t)align, bytes) != 0) {
        fprintf(stderr, "errore: allocazione di %zu byte fallita\n", bytes);
        exit(EXIT_FAILURE);
    }
    return p;
}
