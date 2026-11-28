/*
 * kernels_cpu.h — elenco dei kernel CPU (solo per i file di src/cpu/).
 *
 * Ogni kernel calcola il prodotto LOCALE di un processo MPI, su un solo core:
 * il parallelismo della versione CPU è interamente dato da MPI (un processo
 * per core). Ogni kernel sta in un file proprio, src/cpu/kernel_cpu_<nome>.c.
 *
 * Convenzioni comuni (tutto in memoria host, row-major):
 *   A : M x N  (leading dimension lda)   X : N x k  (ld = k)   Y : M x k  (ld = k)
 */
#ifndef KERNELS_CPU_H
#define KERNELS_CPU_H

#include "common.h"

/* Identificativi dei kernel: è il valore dell'opzione -v del programma. */
enum {
    KERNEL_CPU_ROW   = 0,   /* una riga alla volta (baseline)                 */
    KERNEL_CPU_ROW_REG = 1  /* come cpu_row, accumulatori in registri          */
};

void cpu_row(int M, int N, int k, const real_t *A, int lda,
             const real_t *X, real_t *Y);

void cpu_row_reg(int M, int N, int k, const real_t *A, int lda,
                 const real_t *X, real_t *Y);

#endif /* KERNELS_CPU_H */
