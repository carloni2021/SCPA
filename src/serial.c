/*
 * serial.c — riferimento seriale di Y = A * X.
 *
 * Ordine dei cicli i -> j -> c ("riuso di A"):
 *   per ogni riga i, si scorre la riga i di A; ogni a_ij viene letto UNA
 *   volta e moltiplicato per l'intera riga j di X, aggiornando i k
 *   accumulatori di y_i. Così A (la struttura grande, M x N) attraversa la
 *   memoria una sola volta, mentre X (piccola, N x k) viene riletta per ogni
 *   riga ma resta in cache.
 *
 * L'alternativa "prodotto scalare" (i -> c -> j) rileggerebbe la riga i di A
 * k volte, una per colonna di Y.
 */
#include "serial.h"

void serial_mxm(int M, int N, int k,
                const real_t *A, int lda,
                const real_t *X, int ldx,
                double *Y, int ldy)
{
    for (int i = 0; i < M; i++) {
        /* La riga i di Y fa da accumulatore: in double per precisione. */
        double *acc = Y + (size_t)i * ldy;
        for (int c = 0; c < k; c++) acc[c] = 0.0;

        for (int j = 0; j < N; j++) {
            /* a_ij letto una volta dalla memoria, riusato per le k colonne */
            double a = A[IDX(i, j, lda)];

            /* Riga j di X: in row-major i suoi k valori sono contigui. */
            for (int c = 0; c < k; c++)
                acc[c] += a * X[IDX(j, c, ldx)];
        }
    }
}
