/*
 * serial.h — riferimento seriale del prodotto Y = A * X.
 *
 * È l'"implementazione seriale minimale" richiesta dalla traccia: ogni
 * versione parallela (GPU, CPU, MPI) viene validata confrontandola con
 * questa (vedi verify() in main.c). Non è ottimizzata: deve essere semplice
 * e corretta, non veloce.
 */
#ifndef SERIAL_H
#define SERIAL_H

#include "common.h"

/*
 * Y = A * X, tutte le matrici in row-major.
 *
 *   A : M x N, leading dimension lda (distanza in elementi tra due righe)
 *   X : N x k, leading dimension ldx
 *   Y : M x k, leading dimension ldy   -- in DOUBLE anche se real_t è float
 *
 * Accumula e restituisce in double: il riferimento deve essere più
 * accurato del codice che deve giudicare, altrimenti una differenza nel
 * confronto non direbbe chi dei due ha sbagliato.
 * Le leading dimension separate permettono di operare su sotto-blocchi.
 * A e X sono const: la funzione le legge soltanto, Y è l'uscita.
 */
void serial_mxm(int M, int N, int k,
                const real_t *A, int lda,
                const real_t *X, int ldx,
                double *Y, int ldy);

#endif /* SERIAL_H */
