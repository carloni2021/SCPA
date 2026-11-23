/*
 * prep.h — funzioni ausiliarie di preprocessamento dei dati della matrice.
 *
 * Preparano la memoria di A prima del calcolo (fuori dal tempo misurato):
 *
 *  1. Padding delle righe. La leading dimension lda (distanza in elementi
 *     tra l'inizio di due righe consecutive) viene arrotondata per eccesso a
 *     un multiplo di "align" byte. Le colonne in più ("padding") non
 *     contengono dati e non vengono mai lette.
 *
 *  2. Allocazione allineata. Il primo elemento di A sta a un indirizzo
 *     multiplo di "align".
 *
 * Insieme garantiscono che OGNI riga inizi a un indirizzo allineato.
 *
 * L'allineamento dipende da chi legge, quindi lo sceglie il backend
 * (be_align_bytes in backend.h), in base alla propria unità di lettura:
 *  - GPU: 128 byte = una linea della cache L1 (4 settori da 32 byte). Un
 *    warp legge 32 elementi consecutivi di una riga (256 byte in double):
 *    con la riga allineata tocca il minimo di settori (8) e di linee (2);
 *    disallineata, un settore e una linea in più;
 *  - CPU: 64 byte = una linea di cache: le righe iniziano a inizio linea.
 * Senza padding (opzione -p 0) lda = numero di colonne: se la larghezza del
 * blocco non è multipla dell'allineamento, la maggior parte delle righe
 * inizia disallineata.
 *
 * Costo: al massimo align - 1 byte in più per riga.
 */
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
