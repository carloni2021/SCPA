#ifndef PREP_H
#define PREP_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Leading dimension per righe di ncols elementi, allineate a align byte
   (align = 0: nessun padding, ld = ncols). */
static inline int prep_leading_dim(int ncols, int align)
{
    const int per_align = align / (int)sizeof(real_t);   /* elementi per unità */
    if (per_align <= 1) return ncols;
    return (ncols + per_align - 1) / per_align * per_align;   /* per eccesso */
}

/* Alloca una matrice di nrows righe con leading dimension ld, a un
   indirizzo multiplo di align byte (potenza di 2, almeno 16).
   Si libera con free(). Non inizializza: la riempie chi la usa. */
real_t *prep_alloc_matrix(int nrows, int ld, int align);

#ifdef __cplusplus
}
#endif

#endif /* PREP_H */
