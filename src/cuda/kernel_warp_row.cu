/*
 * kernel_warp_row.cu — un warp per riga di A.
 *
 * Schema (per la riga i, assegnata a un warp di 32 thread, "lane" 0..31):
 *
 *   riga i di A:  [ a0  a1 ... a31 | a32 ... a63 | ... ]
 *   lane:            0   1 ...  31 |  0  ...  31 | ...
 *
 *  1. lettura coalescente: al giro t la lane l legge A[i][l + 32t]; i 32
 *     indirizzi sono contigui -> una sola transazione per l'intero warp;
 *  2. riuso di A: ogni elemento letto finisce in un registro e aggiorna i k
 *     accumulatori della lane con la riga corrispondente di X (A letta una
 *     volta sola, invece di k volte come nel kernel thread_row);
 *  3. riduzione nel warp: ogni lane ha somme parziali (colonne l, l+32, ...);
 *     si sommano tra le lane con __shfl_down_sync, che scambia registri
 *     senza passare dalla memoria (5 passi: offset 16, 8, 4, 2, 1);
 *  4. la lane 0 scrive la riga i di Y.
 *
 * Limite (misurato sul server): ogni lane legge la PROPRIA riga di X,
 * X[j][0..KC-1]. Per una colonna c fissata le 32 lane leggono X[j][c],
 * X[j+1][c], ... a distanza k elementi: la lettura non è coalescente e,
 * con k grande, ogni istruzione tocca 32 linee di memoria diverse. Le
 * prestazioni calano al crescere di k (299 -> 181 -> 92 GFLOPS in double a
 * k = 3, 8, 32). kernel_warp_row_xt.cu risolve il problema trasponendo X.
 *
 * k NOTO SOLO A RUNTIME. Gli accumulatori devono stare nei registri, ma un
 * array indicizzato con una dimensione variabile non può: il compilatore lo
 * sposterebbe in "local memory" (fisicamente memoria globale, lenta). Perciò
 * il kernel è un template su KC = numero di colonne di Y calcolate (costante
 * di compilazione) e lavora su una "fetta" di colonne [c0, c0 + KC):
 *  - per i k richiesti dalla traccia (3, 6, 8, 20, 32) basta una fetta, KC = k:
 *    A viene letta una volta sola;
 *  - per ogni altro k si usano fette da 8 colonne più una fetta finale per il
 *    resto (1..7): il codice funziona per k generico, al costo di leggere A
 *    una volta per fetta.
 * La scelta delle fette è comune a tutti i kernel a fette: k_dispatch.h.
 */
#include "kernels.h"
#include "k_dispatch.h"

/* Un warp ha sempre 32 thread, perché è una caratteristica fissa
   dell'hardware. WARPS_PER_BLOCK = 8 non significa "warp da 8", ma
   "8 warp per blocco", cioè 8 x 32 = 256 thread (e 8 righe per blocco).
   Il numero di thread per blocco è calcolato come WARPS_PER_BLOCK *
   WARP_SIZE, quindi è sempre un multiplo di 32: i warp sono completi per
   costruzione. Con un numero di thread non multiplo di 32 l'ultimo warp
   sarebbe incompleto: alcune colonne della sua riga non verrebbero lette e
   lo shuffle con FULL_MASK sarebbe indefinito. */
#define WARP_SIZE        32
#define WARPS_PER_BLOCK  8
#define FULL_MASK        0xffffffffu       /* tutte le 32 lane partecipano allo shuffle */

template <int KC>
__global__ void k_warp_row(int M, int N, int k, int c0,
                           const real_t *__restrict__ A, int lda,
                           const real_t *__restrict__ X,
                           real_t *__restrict__ Y)
{
    const int lane = threadIdx.x % WARP_SIZE;                               /* posizione nel warp */
    const int row  = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE; /* riga del warp     */

    /* Tutte le lane di un warp hanno la stessa riga: l'uscita è uniforme per
       warp, quindi i warp che restano attivi sono completi (32 lane) e lo
       shuffle con FULL_MASK è corretto. */
    if (row >= M) return;

    const real_t *arow = A + (size_t)row * lda; /* inizio della riga di A   */
    const real_t *xcol = X + c0;                /* prima colonna della fetta */

    real_t acc[KC];                             /* in registri: KC è costante */
#pragma unroll
    for (int c = 0; c < KC; c++) acc[c] = 0;

    /* La lane scorre la riga con passo 32. Se N non è multiplo di 32, le lane
       oltre la fine fanno un giro in meno: i loro accumulatori restano validi. */
    for (int j = lane; j < N; j += WARP_SIZE) {
        const real_t a = arow[j];                        /* lettura coalescente  */
        const real_t *xrow = xcol + (size_t)j * k;       /* riga j di X (fetta)  */
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
static void launch_slice(int M, int N, int k, int c0,
                         const real_t *A, int lda, const real_t *X, real_t *Y)
{
    const int threads = WARPS_PER_BLOCK * WARP_SIZE;
    const int blocks  = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;   /* ceil(M / 8) */
    k_warp_row<KC><<<blocks, threads>>>(M, N, k, c0, A, lda, X, Y);
}

void launch_warp_row(int M, int N, int k, const real_t *A, int lda,
                     const real_t *X, real_t *Y)
{
    /* Sceglie le fette in base a k (k_dispatch.h): un lancio per fetta. */
#define SLICE(KC, c0) launch_slice<KC>(M, N, k, c0, A, lda, X, Y)
    K_DISPATCH(k, SLICE);
#undef SLICE
}
