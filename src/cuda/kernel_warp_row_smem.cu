#include "kernels.h"
#include "k_dispatch.h"

#define WARP_SIZE        32
#define WARPS_PER_BLOCK  8                 /* 8 warp (righe) per blocco */
#define THREADS          (WARPS_PER_BLOCK * WARP_SIZE)   /* 256 */
#define FULL_MASK        0xffffffffu
#define SMEM_BUDGET      16384             /* byte di shared per blocco */

/* Colonne per tile: la più grande potenza di 2, almeno 32, tale che KC x TC elementi stiano in SMEM_BUDGET byte (calcolata in compilazione). */
template <int KC>
__host__ __device__ constexpr int tile_cols()
{
    int tc = 32;
    while (2 * tc * KC * (int)sizeof(real_t) <= SMEM_BUDGET) tc *= 2;
    return tc;
}

template <int KC>
__global__ void __launch_bounds__(THREADS)
k_warp_row_smem(int M, int N, int c0,
                const real_t *__restrict__ A, int lda,
                const real_t *__restrict__ Xt, int ldxt,
                int k, real_t *__restrict__ Y)
{
    constexpr int TC = tile_cols<KC>();
    __shared__ real_t xs[KC][TC];                 /* tile di Xt: KC righe x TC colonne */

    const int  tid    = threadIdx.x;
    const int  lane   = tid % WARP_SIZE;
    const int  row    = blockIdx.x * WARPS_PER_BLOCK + tid / WARP_SIZE;
    const bool active = row < M;                  /* niente return: vedi le barriere */

    const real_t *arow = A + (size_t)(active ? row : 0) * lda;
    const real_t *xt0  = Xt + (size_t)c0 * ldxt;  /* prima riga di Xt della fetta */

    real_t acc[KC];
#pragma unroll
    for (int c = 0; c < KC; c++) acc[c] = 0;

    for (int j0 = 0; j0 < N; j0 += TC) {
        const int w = min(TC, N - j0);            /* colonne valide nel tile (l'ultimo può essere corto) */

        // lancio del tile: ogni thread carica 0 o più elementi di Xt in shared memory, con accesso coalescente
        constexpr int LOADS = (KC * TC + THREADS - 1) / THREADS;
#pragma unroll 4                                  /* vedi "Registri" in testa al file */
        for (int it = 0; it < LOADS; it++) {
            const int e = it * THREADS + tid;
            const int c = e / TC, jj = e % TC;
            if (e < KC * TC && jj < w) xs[c][jj] = xt0[(size_t)c * ldxt + j0 + jj];
        }
        __syncthreads();                          /* 2. tile completo */

        /* 3. calcolo sul tile: come in warp_row_xt, con X letta dalla shared */
        if (active) {
#pragma unroll 2                                  /* vedi "Registri" in testa al file */
            for (int jj = lane; jj < w; jj += WARP_SIZE) {
                const real_t a = arow[j0 + jj];   /* coalescente */
#pragma unroll
                for (int c = 0; c < KC; c++)
                    acc[c] += a * xs[c][jj];      /* shared: lane su elementi consecutivi */
            }
        }
        __syncthreads();                          /* 4. xs libera per il tile successivo */
    }

    /* Da qui non ci sono più barriere: i warp senza riga possono uscire.
       L'uscita è uniforme per warp, quindi lo shuffle resta tra 32 lane. */
    if (!active) return;

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
// Lancia il kernel per la fetta di colonne [c0, c0 + KC). KC è costante in compilazione.
static void launch_slice(int M, int N, int k, int c0, const real_t *A, int lda,const real_t *Xt, int ldxt, real_t *Y){
    const int blocks = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    k_warp_row_smem<KC><<<blocks, THREADS>>>(M, N, c0, A, lda, Xt, ldxt, k, Y);
}

void launch_warp_row_smem(int M, int N, int k, const real_t *A, int lda,const real_t *Xt, int ldxt, real_t *Y){
    /* Xt è già stata preparata da launch_transpose_x (in be_set_x).
       Sceglie le fette in base a k (k_dispatch.h): un lancio per fetta. */
#define SLICE(KC, c0) launch_slice<KC>(M, N, k, c0, A, lda, Xt, ldxt, Y)
    K_DISPATCH(k, SLICE);
#undef SLICE
}
