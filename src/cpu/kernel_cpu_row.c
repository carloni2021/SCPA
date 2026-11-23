/*
 * kernel_cpu_row.c — una riga alla volta (baseline CPU).
 *
 * È l'ordine "riuso di A" (i -> j -> c), lo stesso del riferimento seriale:
 * per ogni riga i si scorre la riga i di A, ogni a_ij viene letto una volta
 * e moltiplicato per l'intera riga j di X, aggiornando i k accumulatori di
 * y_i. A attraversa la memoria una volta sola.
 *
 * Limiti voluti, punto di partenza dell'analisi:
 *  - k è noto solo a runtime, quindi gli accumulatori stanno in un array in
 *    memoria (cache L1), non in registri: a ogni prodotto un load e uno
 *    store dell'accumulatore;
 *  - ogni valore di X letto viene usato per UNA sola riga di A: per ogni
 *    operazione (FMA) servono due letture (x e accumulatore) e una
 *    scrittura, e il core aspetta la memoria più che calcolare.
 * Il kernel cpu_block risolve entrambi i punti.
 */
#include <stdlib.h>
#include "kernels_cpu.h"

void cpu_row(int M, int N, int k, const real_t *restrict A, int lda,
             const real_t *restrict X, real_t *restrict Y)
{
    real_t *acc = malloc((size_t)k * sizeof *acc);   /* k accumulatori, allocati una volta */

    for (int i = 0; i < M; i++) {
        for (int c = 0; c < k; c++) acc[c] = 0;

        const real_t *arow = A + (size_t)i * lda;
        for (int j = 0; j < N; j++) {
            const real_t a = arow[j];                    /* letto una volta, in registro */
            const real_t *xrow = X + (size_t)j * k;      /* riga j di X: k valori contigui */
            for (int c = 0; c < k; c++)
                acc[c] += a * xrow[c];
        }

        for (int c = 0; c < k; c++) Y[IDX(i, c, k)] = acc[c];
    }
    free(acc);
}
