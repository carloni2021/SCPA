#ifndef KERNELS_H
#define KERNELS_H

#include <cublas_v2.h>
#include "common.h"

/* Identificativi dei kernel: è il valore dell'opzione -v del programma. */
enum {
    KERNEL_CUBLAS      = -1,  /* riferimento esterno, non fa parte della soluzione */
    KERNEL_THREAD_ROW  =  0,  /* un thread per riga di Y (baseline)                */
    KERNEL_WARP_ROW    =  1,  /* un warp per riga di A, X letta row-major          */
    KERNEL_WARP_ROW_XT =  2,  /* come warp_row, X trasposta: letture coalescenti   */
    KERNEL_WARP_ROW_SMEM = 3  /* come warp_row_xt, X condivisa dal blocco in shared */
};

void launch_thread_row(int M, int N, int k, const real_t *A, int lda, const real_t *X, real_t *Y);

void launch_warp_row(int M, int N, int k, const real_t *A, int lda, const real_t *X, real_t *Y);

void launch_transpose_x(int N, int k, const real_t *X, real_t *Xt, int ldxt);

/* Prodotto con X già trasposta (Xt, k x N, leading dimension ldxt). */
void launch_warp_row_xt(int M, int N, int k, const real_t *A, int lda, const real_t *Xt, int ldxt, real_t *Y);

/* Come warp_row_xt (stessi argomenti, Xt già trasposta), ma ogni blocco
   carica X a pezzi in memoria shared e la condivide tra i suoi 8 warp. */
void launch_warp_row_smem(int M, int N, int k, const real_t *A, int lda, const real_t *Xt, int ldxt, real_t *Y);

void launch_cublas(cublasHandle_t h, int M, int N, int k, const real_t *A, int lda, const real_t *X, real_t *Y);

#endif /* KERNELS_H */
