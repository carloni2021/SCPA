/*
 * kernels.h — elenco dei kernel CUDA e dei loro lanciatori (solo per i .cu).
 *
 * Ogni approccio di calcolo sta in un file proprio, src/cuda/kernel_<nome>.cu, che
 * contiene il kernel (__global__) e una funzione host launch_<nome>() che
 * sceglie griglia e blocchi e lo lancia. backend_cuda.cu gestisce memoria,
 * tempi e scelta del kernel, e chiama solo questi lanciatori.
 *
 * Convenzioni comuni a tutti i lanciatori: puntatori in memoria DEVICE,
 * matrici row-major.
 *   A : M x N  (leading dimension lda >= N: le righe possono avere padding)
 *   X : N x k  (ld = k)      Y : M x k  (ld = k)
 * I lanci sono asincroni: chi chiama si occupa di sincronizzazione e tempi.
 */
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

void launch_thread_row(int M, int N, int k, const real_t *A, int lda,
                       const real_t *X, real_t *Y);

void launch_warp_row(int M, int N, int k, const real_t *A, int lda,
                     const real_t *X, real_t *Y);

/* Trasposizione di X (N x k, row-major) in Xt (k x N, cioè X column-major,
   leading dimension ldxt >= N: le righe di Xt possono avere padding come
   quelle di A). È la preparazione dei dati per warp_row_xt: il backend la
   esegue in be_set_x, fuori dal tempo del nucleo. */
void launch_transpose_x(int N, int k, const real_t *X, real_t *Xt, int ldxt);

/* Prodotto con X già trasposta (Xt, k x N, leading dimension ldxt). */
void launch_warp_row_xt(int M, int N, int k, const real_t *A, int lda,
                        const real_t *Xt, int ldxt, real_t *Y);

/* Come warp_row_xt (stessi argomenti, Xt già trasposta), ma ogni blocco
   carica X a pezzi in memoria shared e la condivide tra i suoi 8 warp. */
void launch_warp_row_smem(int M, int N, int k, const real_t *A, int lda,
                          const real_t *Xt, int ldxt, real_t *Y);

void launch_cublas(cublasHandle_t h, int M, int N, int k, const real_t *A, int lda,
                   const real_t *X, real_t *Y);

#endif /* KERNELS_H */
