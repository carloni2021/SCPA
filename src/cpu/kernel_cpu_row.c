#include <stdlib.h>
#include "kernels_cpu.h"

//implementazione del kernel cpu_row: calcolo riga per riga, con k accumulatori in registro
void cpu_row(int M, int N, int k, const real_t *restrict A, int lda,const real_t *restrict X, real_t *restrict Y){

    real_t *acc = malloc((size_t)k * sizeof *acc);   /* k accumulatori, allocati una volta */

    for (int i = 0; i < M; i++) {
        for (int c = 0; c < k; c++) acc[c] = 0; //azzero gli accumulatori

        const real_t *arow = A + (size_t)i * lda; // riga i-esima di A N valori contigui
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
