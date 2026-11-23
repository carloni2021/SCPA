/*
 * kernel_warp_row_xt.cu — un warp per riga di A, con X trasposta.
 *
 * Stesso schema di warp_row (un warp per riga, lane lungo j, A letta una
 * volta e coalescente, k accumulatori in registri, riduzione con shuffle),
 * con UNA differenza: come si legge X.
 *
 * Il problema di warp_row. Al giro t la lane l elabora j = l + 32t e legge
 * la riga j di X. Per una colonna c fissata, le 32 lane leggono
 *     X[j][c], X[j+1][c], ..., X[j+31][c]
 * che in X row-major (N x k) stanno a distanza k elementi l'una dall'altra:
 * ogni lane tocca una linea di memoria diversa e ne usa pochi byte.
 *
 * La soluzione. Prima del calcolo si traspone X in Xt (k x N, cioè X
 * column-major): Xt[c][j] = X[j][c]. Ora per una colonna c fissata le 32
 * lane leggono
 *     Xt[c][j], Xt[c][j+1], ..., Xt[c][j+31]
 * che sono CONSECUTIVI: una sola lettura coalescente, come per A.
 *
 *   X row-major (N x k)                   Xt = X column-major (k x N)
 *   riga j:   x(j,0) x(j,1) ... x(j,k-1)  riga c:  x(0,c) x(1,c) ... x(N-1,c)
 *   le lane leggono "in verticale" ->     le lane leggono "in orizzontale" ->
 *   indirizzi distanti k                  indirizzi contigui
 *
 * Perché non tenere X column-major ovunque: la parte MPI e la CPU usano X
 * row-major (blocchi di righe contigui per il broadcast, righe contigue per
 * i registri vettoriali della CPU). Il layout migliore dipende da chi legge:
 * qui lo si adatta sulla GPU, dove serve.
 *
 * Allineamento di Xt: le righe di Xt sono lunghe N, esattamente come
 * quelle di A, quindi hanno lo stesso problema di allineamento. Il backend
 * le alloca con la stessa leading dimension allineata (ldxt, padding a 128
 * byte, vedi prep.h) quando il padding è attivo.
 *
 * Dove si traspone: la trasposizione è una preparazione dei dati di
 * ingresso, come la copia di X sulla GPU. Per questo il backend la esegue in
 * be_set_x (launch_transpose_x), subito dopo la copia host -> device: è
 * ESCLUSA dal tempo del nucleo e misurata a parte insieme alla copia
 * (colonna t_h2d). Costa poco: legge e scrive N x k elementi, contro gli
 * M x N di A letti dal calcolo.
 *
 * k a runtime: stesso schema di warp_row (template su KC, una fetta per i k
 * della traccia, fette da 8 più il resto per ogni altro k).
 */
#include "kernels.h"
#include "k_dispatch.h"

#define WARP_SIZE        32
#define WARPS_PER_BLOCK  8                 /* 8 warp (righe) per blocco: 256 thread */
#define FULL_MASK        0xffffffffu
#define TRANSPOSE_THREADS 256

/* Trasposizione: Xt[c][j] = X[j][c]. Un thread per elemento; thread
   consecutivi hanno j consecutivi (stessa c), quindi le SCRITTURE in Xt
   sono coalescenti; le letture da X sono a passo k (piccolo) e restano
   in poche linee di cache. Le colonne di padding di Xt (j >= N) non
   vengono scritte né lette. */
__global__ void k_transpose_x(int N, int k,
                              const real_t *__restrict__ X,
                              real_t *__restrict__ Xt, int ldxt)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)N * k) return;
    int c = (int)(idx / N);
    int j = (int)(idx % N);
    Xt[(size_t)c * ldxt + j] = X[(size_t)j * k + c];
}

template <int KC>
__global__ void k_warp_row_xt(int M, int N, int c0,
                              const real_t *__restrict__ A, int lda,
                              const real_t *__restrict__ Xt, int ldxt,
                              int k, real_t *__restrict__ Y)
{
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

void launch_transpose_x(int N, int k, const real_t *X, real_t *Xt, int ldxt)
{
    size_t total = (size_t)N * k;
    if (total == 0) return;
    unsigned blocks = (unsigned)((total + TRANSPOSE_THREADS - 1) / TRANSPOSE_THREADS);
    k_transpose_x<<<blocks, TRANSPOSE_THREADS>>>(N, k, X, Xt, ldxt);
}

void launch_warp_row_xt(int M, int N, int k, const real_t *A, int lda,
                        const real_t *Xt, int ldxt, real_t *Y)
{
    /* Xt è già stata preparata da launch_transpose_x (in be_set_x).
       Sceglie le fette in base a k (k_dispatch.h): un lancio per fetta. */
#define SLICE(KC, c0) launch_slice<KC>(M, N, k, c0, A, lda, Xt, ldxt, Y)
    K_DISPATCH(k, SLICE);
#undef SLICE
}
