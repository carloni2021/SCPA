/*
 * kernel_warp_row_smem.cu — un warp per riga di A, X trasposta, X condivisa
 * dal blocco in memoria shared.
 *
 * Stesso schema di warp_row_xt (un warp per riga, lane lungo j, A e Xt lette
 * in modo coalescente, k accumulatori in registri, riduzione con shuffle),
 * con UNA differenza: da dove i warp leggono X.
 *
 * Il problema di warp_row_xt. Un blocco ha 8 warp, cioè 8 righe di A
 * DIVERSE, ma la X che leggono è la STESSA: al giro j ognuno degli 8 warp
 * chiede Xt[0..KC-1][j..j+31]. Gli stessi valori vengono richiesti 8 volte e
 * li serve la cache (L1/L2), se li trova. Il traffico è grande: a ogni giro
 * un warp legge 32 elementi di A e 32 * KC elementi di Xt, cioè le richieste
 * per X sono KC volte quelle per A. Con k = 32 in double Xt occupa 4 MB
 * (N = 16384), quanto tutta la L2.
 *
 * La soluzione. Il blocco carica X UNA volta in memoria shared (on-chip,
 * condivisa dai thread del blocco, latenza simile alla L1) e gli 8 warp la
 * leggono da lì. La riga è divisa in "tile" di TC colonne:
 *
 *   per ogni tile [j0, j0 + TC):
 *     1. i 256 thread copiano Xt[0..KC-1][j0 .. j0+TC-1] in xs (shared),
 *        un elemento a testa per giro: letture coalescenti, una sola volta
 *        per blocco invece che una per warp;
 *     2. __syncthreads(): nessuno usa xs prima che sia tutta caricata;
 *     3. ogni warp calcola la sua riga sul tile, leggendo A dalla memoria
 *        globale e X da xs;
 *     4. __syncthreads(): nessuno sovrascrive xs mentre altri la leggono.
 *
 * Le richieste di X verso L1/L2 scendono di 8 volte (warp per blocco).
 * È lo stesso principio di cpu_block sulla CPU: lì X si riusa tra RB righe
 * tenendola nei registri, qui tra 8 righe tenendola in shared.
 *
 * Dimensione del tile. La shared per blocco è SMEM_BUDGET = 16 KB. Su Turing
 * un SM ha 64 KB di shared e ospita al più 1024 thread, cioè 4 blocchi da
 * 256: con 16 KB a blocco la shared non riduce il numero di blocchi
 * residenti (l'occupazione). TC è la più grande potenza di 2 tale che
 * KC * TC elementi stiano nel budget (almeno 32, un giro di warp); potenza
 * di 2 perché divisioni e resti sugli indici diventino shift.
 *   double: KC = 3 -> TC = 512, KC = 8 -> 256, KC = 20 e 32 -> 64
 *   float : il doppio delle colonne, stessi byte.
 *
 * Accessi alla shared. In xs le lane di un warp leggono xs[c][jj..jj+31],
 * elementi consecutivi: banchi diversi, nessun conflitto (in double il warp
 * chiede 256 byte, serviti in 2 passate: è il minimo).
 *
 * Registri. Il kernel è limitato dalla memoria: servono molti warp
 * residenti per SM, così mentre alcuni aspettano i dati altri lavorano
 * (occupazione). Un SM ha 65536 registri: 4 blocchi da 256 thread
 * richiedono al più 64 registri per thread. Lasciando che il compilatore
 * srotoli del tutto i due cicli interni, i registri salgono fino a 190
 * (k = 32, double: 1 solo blocco per SM). La causa principale è il ciclo di
 * caricamento: gli indirizzi delle LOADS letture di un thread sono gli
 * stessi a ogni tile (cambia solo j0), quindi il compilatore li calcola una
 * volta fuori dal ciclo e li tiene tutti in registri. Con
 * "#pragma unroll 4" (4 letture in volo per thread) sul caricamento e
 * "#pragma unroll 2" sul calcolo si resta entro 64 registri, cioè 4 blocchi
 * per SM, per ogni k tranne k = 32 in double (124 registri, 2 blocchi, come
 * warp_row_xt a k = 32). Valori misurati con il compilatore di CUDA 12.8
 * (ptxas -v, sm_75); `make NVFLAGS_EXTRA=-Xptxas=-v` li stampa sul server.
 *
 * Attenzione: questo kernel NON può far uscire subito i warp senza riga
 * (row >= M, nell'ultimo blocco) come fanno gli altri. __syncthreads()
 * aspetta TUTTI i thread del blocco: se alcuni fossero già usciti il
 * comportamento sarebbe indefinito. I warp senza riga partecipano quindi al
 * caricamento dei tile e alle barriere, e saltano solo il calcolo.
 *
 * Cosa ci si aspetta: su Turing L1 e shared sono la stessa memoria fisica.
 * Se la L1 cattura già il riuso tra warp il guadagno è piccolo, e le
 * barriere hanno un costo. Il beneficio atteso cresce con k (più traffico
 * di X); va misurato (gruppo B delle misure).
 *
 * k a runtime: stesso schema di warp_row (template su KC, una fetta per i k
 * della traccia, fette da 8 più il resto per ogni altro k).
 */
#include "kernels.h"
#include "k_dispatch.h"

#define WARP_SIZE        32
#define WARPS_PER_BLOCK  8                 /* 8 warp (righe) per blocco */
#define THREADS          (WARPS_PER_BLOCK * WARP_SIZE)   /* 256 */
#define FULL_MASK        0xffffffffu
#define SMEM_BUDGET      16384             /* byte di shared per blocco */

/* Colonne per tile: la più grande potenza di 2, almeno 32, tale che
   KC x TC elementi stiano in SMEM_BUDGET byte (calcolata in compilazione). */
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

        /* 1. caricamento cooperativo: il tile ha KC * TC elementi, ogni
              thread ne carica LOADS (giri da 256). L'elemento e è
              (c, jj) = (e / TC, e % TC): thread consecutivi hanno jj
              consecutivi nella stessa riga c, quindi letture coalescenti da
              Xt e scritture in xs senza conflitti. Oltre w non si carica
              nulla (e non si leggerà). */
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
static void launch_slice(int M, int N, int k, int c0, const real_t *A, int lda,
                         const real_t *Xt, int ldxt, real_t *Y)
{
    const int blocks = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    k_warp_row_smem<KC><<<blocks, THREADS>>>(M, N, c0, A, lda, Xt, ldxt, k, Y);
}

void launch_warp_row_smem(int M, int N, int k, const real_t *A, int lda,
                          const real_t *Xt, int ldxt, real_t *Y)
{
    /* Xt è già stata preparata da launch_transpose_x (in be_set_x).
       Sceglie le fette in base a k (k_dispatch.h): un lancio per fetta. */
#define SLICE(KC, c0) launch_slice<KC>(M, N, k, c0, A, lda, Xt, ldxt, Y)
    K_DISPATCH(k, SLICE);
#undef SLICE
}
