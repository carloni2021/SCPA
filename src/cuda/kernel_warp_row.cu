#include "kernels.h"
#include "k_dispatch.h"
#define WARP_SIZE        32
#define WARPS_PER_BLOCK  8
#define FULL_MASK        0xffffffffu       /* tutte le 32 lane partecipano allo shuffle */

template <int KC>
__global__ void k_warp_row(int M, int N, int k, int c0,const real_t *__restrict__ A, int lda,const real_t *__restrict__ X,real_t *__restrict__ Y){
    const int lane = threadIdx.x % WARP_SIZE;                               /* posizione nel warp */
    const int row  = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE; /* riga del warp     */


    if (row >= M) return;

    const real_t *arow = A + (size_t)row * lda; /* inizio della riga di A   */
    const real_t *xcol = X + c0;                /* prima colonna della fetta */

    real_t acc[KC];                             /* in registri: KC è costante */
#pragma unroll
    for (int c = 0; c < KC; c++) acc[c] = 0;

    /* La lane (thread singolo) scorre la riga con passo 32. Se N non è multiplo di 32, le lane
       oltre la fine fanno un giro in meno: i loro accumulatori restano validi. */
    for (int j = lane; j < N; j += WARP_SIZE) {
        const real_t a = arow[j];                        /* lettura coalescente  */
        const real_t *xrow = xcol + (size_t)j * k;       /* riga j di X (non coalescente)  */
#pragma unroll
        for (int c = 0; c < KC; c++)
            acc[c] += a * xrow[c];                       /* a riusato KC volte  */
    }

    /* Riduzione ad albero: a ogni passo la lane l somma il valore della lane
       l + off. Dopo 5 passi la lane 0 contiene la somma di tutte le 32. */
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off /= 2) {
#pragma unroll
        for (int c = 0; c < KC; c++)
            acc[c] += __shfl_down_sync(FULL_MASK, acc[c], off);
    }

    if (lane == 0) {
        real_t *yrow = Y + (size_t)row * k + c0;
#pragma unroll
        for (int c = 0; c < KC; c++) yrow[c] = acc[c];
    }
}

/* Lancia il kernel per la fetta di colonne [c0, c0 + KC). */
template <int KC>
static void launch_slice(int M, int N, int k, int c0,const real_t *A, int lda, const real_t *X, real_t *Y){
    const int threads = WARPS_PER_BLOCK * WARP_SIZE;
    const int blocks  = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;   /* ceil(M / 8) */
    k_warp_row<KC><<<blocks, threads>>>(M, N, k, c0, A, lda, X, Y);
}

void launch_warp_row(int M, int N, int k, const real_t *A, int lda,const real_t *X, real_t *Y){
    /* Sceglie le fette in base a k (k_dispatch.h): un lancio per fetta. */
#define SLICE(KC, c0) launch_slice<KC>(M, N, k, c0, A, lda, X, Y)
    K_DISPATCH(k, SLICE);
#undef SLICE
}
