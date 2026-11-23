/*
 * kernel_thread_row.cu — un thread per riga di Y (baseline).
 *
 * Il thread i calcola da solo l'intera riga i di Y, con k prodotti scalari
 * tra la riga i di A e le colonne di X (ordine "prodotto scalare").
 *
 * Difetti voluti, punto di partenza dell'analisi:
 *  - A riletta k volte: la riga i viene scorsa una volta per ogni colonna c;
 *  - accessi ad A NON coalescenti: nello stesso istante i 32 thread di un
 *    warp leggono A[i][j], A[i+1][j], ..., indirizzi distanti N elementi:
 *    ogni thread tocca un settore di memoria diverso e ne usa 8 byte su 32;
 *  - con M piccolo i thread sono pochi e la GPU resta in gran parte ferma.
 */
#include "kernels.h"

#define THREADS_PER_BLOCK 128   /* multiplo di 32 (dimensione del warp) */

__global__ void k_thread_row(int M, int N, int k,
                             const real_t *__restrict__ A, int lda,
                             const real_t *__restrict__ X,
                             real_t *__restrict__ Y)
{
    /* indice globale del thread = riga di cui si occupa */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;              /* thread in eccesso nell'ultimo blocco */

    for (int c = 0; c < k; c++) {
        real_t s = 0;
        for (int j = 0; j < N; j++)
            s += A[IDX(i, j, lda)] * X[IDX(j, c, k)];
        Y[IDX(i, c, k)] = s;
    }
}

void launch_thread_row(int M, int N, int k, const real_t *A, int lda,
                       const real_t *X, real_t *Y)
{
    int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;   /* ceil(M / 128) */
    k_thread_row<<<blocks, THREADS_PER_BLOCK>>>(M, N, k, A, lda, X, Y);
}
