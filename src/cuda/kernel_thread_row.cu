#include "kernels.h"

#define THREADS_PER_BLOCK 128   /* multiplo di 32 (dimensione del warp) */

__global__ void k_thread_row(int M, int N, int k,const real_t *__restrict__ A, int lda,const real_t *__restrict__ X,real_t *__restrict__ Y){
    /* indice globale del thread = riga di cui si occupa */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;              /* thread in eccesso nell'ultimo blocco */

    for (int c = 0; c < k; c++) {
        real_t s = 0;
        for (int j = 0; j < N; j++)
            s += A[IDX(i, j, lda)] * X[IDX(j, c, k)]; // calcola il prodotto riga-colonna come A[i, j] * X[j, c]
        Y[IDX(i, c, k)] = s; //accumula
    }
}

void launch_thread_row(int M, int N, int k, const real_t *A, int lda,const real_t *X, real_t *Y){
    int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;   /* ceil(M / 128) */
    k_thread_row<<<blocks, THREADS_PER_BLOCK>>>(M, N, k, A, lda, X, Y);
}
