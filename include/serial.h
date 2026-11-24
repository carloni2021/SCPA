#ifndef SERIAL_H
#define SERIAL_H

#include "common.h"

/*
 * Y = A * X, tutte le matrici in row-major.
 * Versione seriale per il calcolo di riferimento e la verifica.
 */
void serial_mxm(int M, int N, int k,
                const real_t *A, int lda,
                const real_t *X, int ldx,
                double *Y, int ldy);

#endif /* SERIAL_H */
