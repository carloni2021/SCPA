#include "serial.h"

/// Y = A * X, tutte le matrici in row-major. Versione seriale per il calcolo di riferimento e la verifica.
void serial_mxm(int M, int N, int k,const real_t *A, int lda,const real_t *X, int ldx,double *Y, int ldy){

    for (int i = 0; i < M; i++) {
        /* La riga i di Y fa da accumulatore: in double per precisione. */
        double *acc = Y + (size_t)i * ldy;
        for (int c = 0; c < k; c++) acc[c] = 0.0; //azzero

        for (int j = 0; j < N; j++) {
            /* a_ij letto una volta dalla memoria, riusato per le k colonne */
            double a = A[IDX(i, j, lda)];

            /* Riga j di X: in row-major i suoi k valori sono contigui. */
            for (int c = 0; c < k; c++)
                acc[c] += a * X[IDX(j, c, ldx)];
        }

    }

}
