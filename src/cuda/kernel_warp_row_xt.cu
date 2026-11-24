#include "kernels.h"
#include "k_dispatch.h"

#define WARP_SIZE        32
#define WARPS_PER_BLOCK  8                 /* 8 warp (righe) per blocco: 256 thread */
#define FULL_MASK        0xffffffffu
#define TRANSPOSE_THREADS 256

/* Trasposizione:*/
__global__ void k_transpose_x(int N, int k,const real_t *__restrict__ X,real_t *__restrict__ Xt, int ldxt){
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)N * k) return;
    int c = (int)(idx / N);
    int j = (int)(idx % N);
    Xt[(size_t)c * ldxt + j] = X[(size_t)j * k + c];
}

template <int KC>
__global__ void k_warp_row_xt(int M, int N, int c0,const real_t *__restrict__ A, int lda,const real_t *__restrict__ Xt, int ldxt,int k, real_t *__restrict__ Y){
    const int lane = threadIdx.x % WARP_SIZE;
    const int row  = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE;
    if (row >= M) return;                       /* uscita uniforme per warp */

    const real_t *arow = A + (size_t)row * lda;
    const real_t *xt0  = Xt + (size_t)c0 * ldxt; /* prima riga di Xt della fetta */

    real_t acc[KC];
#pragma unroll
    for (int c = 0; c < KC; c++) acc[c] = 0;

    for (int j = lane; j < N; j += WARP_SIZE) {
        const real_t a = arow[j];                       /* coalescente               */
#pragma unroll
        for (int c = 0; c < KC; c++)
            acc[c] += a * xt0[(size_t)c * ldxt + j];    /* coalescente: j consecutivi */
    }

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

template <int KC>
static void launch_slice(int M, int N, int k, int c0, const real_t *A, int lda,
                         const real_t *Xt, int ldxt, real_t *Y)
{
    const int threads = WARPS_PER_BLOCK * WARP_SIZE;
    const int blocks  = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    k_warp_row_xt<KC><<<blocks, threads>>>(M, N, c0, A, lda, Xt, ldxt, k, Y);
}

void launch_transpose_x(int N, int k, const real_t *X, real_t *Xt, int ldxt){
    size_t total = (size_t)N * k;
    if (total == 0) return;
    unsigned blocks = (unsigned)((total + TRANSPOSE_THREADS - 1) / TRANSPOSE_THREADS);
    k_transpose_x<<<blocks, TRANSPOSE_THREADS>>>(N, k, X, Xt, ldxt);
}

void launch_warp_row_xt(int M, int N, int k, const real_t *A, int lda,const real_t *Xt, int ldxt, real_t *Y){
    /* Xt è già stata preparata da launch_transpose_x (in be_set_x).
       Sceglie le fette in base a k (k_dispatch.h): un lancio per fetta. */
#define SLICE(KC, c0) launch_slice<KC>(M, N, k, c0, A, lda, Xt, ldxt, Y)
    K_DISPATCH(k, SLICE);
#undef SLICE
}
