/*
 * kernel_cpu_block.c — blocchi di RB righe calcolate insieme, accumulatori
 * in registri.
 *
 * Idea: cpu_row riusa A (ogni a_ij serve k colonne) ma non X (ogni valore
 * di X letto serve UNA sola riga). Qui si calcolano RB righe di Y insieme:
 *
 *   per j = 0 .. N-1:
 *       x = X[j][0 .. k-1]               <- k valori letti UNA volta
 *       per ognuna delle RB righe r:
 *           a = A[r][j]                   <- letto una volta
 *           acc[r][0..k-1] += a * x       <- k FMA, tutto in registri
 *
 * Due forme di riuso, una per operando:
 *  - riuso di A: ogni a viene usato k volte (come in cpu_row);
 *  - riuso di X: ogni x viene usato RB volte. Le letture per operazione
 *    scendono di un fattore RB.
 * A viene comunque letta UNA sola volta: le RB righe si scorrono insieme.
 *
 * Accumulatori in registri. Servono RB x k accumulatori con dimensioni
 * note al compilatore, altrimenti finiscono in memoria come in cpu_row.
 * C non ha i template: si usa una funzione "always_inline" chiamata con
 * argomenti costanti, così il compilatore ne genera una copia specializzata
 * per ogni coppia (RB, k), con i cicli su r e c completamente svolti e
 * trasformati in istruzioni SIMD.
 *
 * Scelta di RB: gli accumulatori devono stare nei registri vettoriali, che
 * sono pochi. Lo Xeon del server ha 32 registri (AVX-512) e il compilatore
 * li usa da 256 bit, cioè 4 double l'uno: 128 double in tutto. RB x k deve
 * quindi restare entro un budget: ACC_BUDGET = 128 accumulatori in double
 * (1024 byte di registri; in float 256); RB = ACC_BUDGET / k, limitato a [1, 8]:
 *     double: k=3,6,8 -> RB=8   k=20 -> RB=6   k=32 -> RB=4
 *     float : RB=8 per tutti i k della traccia
 * Il valore è stato scelto misurando sul server i budget 32, 64 e 128
 * (bench/budget_cpu.sh, double, 16384 x 16384):
 *   - con 20 processi 64 e 128 si equivalgono (k=20: ~295 GFLOPS, k=32:
 *     ~264): appena ogni x serve almeno 2 righe il limite è la banda di
 *     memoria condivisa dai core, e righe in più non aiutano;
 *   - con 1 processo 128 è il migliore (k=20: 18 GFLOPS contro 13 con 64;
 *     k=32: 15 contro 10): un core da solo ha banda in abbondanza e più
 *     riuso di X significa più velocità. Conta per la scalabilità (gruppo
 *     D): il riferimento a 1 processo deve usare il kernel al meglio;
 *   - a k=32 i 128 accumulatori riempiono tutti i registri e il compilatore
 *     ne tiene qualcuno in memoria (cache L1); misurato, conviene comunque,
 *     perché leggere X la metà delle volte fa risparmiare di più.
 * Il budget (numero di accumulatori) si può cambiare in compilazione per
 * esperimenti, per esempio RB=1 a k=20 e 32 in double:
 *     make clean && make BACKEND=cpu CFLAGS_EXTRA=-DACC_BUDGET=32
 *
 * k qualunque: i k della traccia (3, 6, 8, 20, 32) hanno una versione
 * dedicata, con A letta una volta. Per ogni altro k le colonne si elaborano
 * a fette di 8 (più una fetta finale per il resto): funziona per ogni k, al
 * costo di leggere A una volta per fetta. La regola è la stessa dei kernel
 * GPU warp_row* e sta in un posto solo, k_dispatch.h.
 *
 * Nota sul compilatore. Dopo aver svolto i cicli su r e c, il ciclo su j
 * resta il più interno, e gcc tende a vettorizzare QUELLO (A è contigua
 * lungo j): carica 4 valori di a consecutivi e poi deve rimescolarli con
 * permutazioni per combinarli con x, esaurendo i registri e spostando gli
 * accumulatori in memoria (misurato: 4 GFLOPS invece di 18 in alcuni casi).
 * La vettorizzazione giusta è lungo c: un registro contiene più colonne di
 * x, a viene replicato (broadcast) e ogni riga costa kc/larghezza FMA.
 * Si disattiva quindi la vettorizzazione "di ciclo" per questo file,
 * lasciando attiva quella dei blocchi di istruzioni (SLP), che produce
 * esattamente lo schema broadcast + FMA. (gcc 14 avrebbe "#pragma GCC
 * novector" sul singolo ciclo; il server ha gcc 13.)
 */
#pragma GCC optimize ("no-tree-loop-vectorize")

#include "kernels_cpu.h"
#include "k_dispatch.h"

/* Numero massimo di accumulatori (vedi sopra): 1024 byte di registri,
   cioè 128 double o 256 float. */
#ifndef ACC_BUDGET
#define ACC_BUDGET (1024 / (int)sizeof(real_t))
#endif
#define RB_MAX 8
/* righe per blocco per un dato numero di colonne kc (costante) */
#define RB_FOR(kc) ((ACC_BUDGET / (kc)) > RB_MAX ? RB_MAX : \
                    (ACC_BUDGET / (kc)) < 1 ? 1 : (ACC_BUDGET / (kc)))

/* forza l'espansione nel chiamante: senza, la specializzazione non avviene */
#define ALWAYS_INLINE static inline __attribute__((always_inline))

/*
 * Blocco di rb righe e kc colonne di Y (rb <= RB_MAX, kc <= 32).
 *   A punta alla prima riga del blocco, X alla prima colonna della fetta,
 *   Y all'elemento (prima riga, prima colonna) del blocco; ldx = ldy = k.
 */
ALWAYS_INLINE void block(int rb, int kc, int N, int k,
                         const real_t *restrict A, int lda,
                         const real_t *restrict X, real_t *restrict Y)
{
    /* Inizializzato per intero: il compilatore elimina la parte non usata
       (rb e kc sono costanti) e gli accumulatori restano in registri. */
    real_t acc[RB_MAX][32] = { { 0 } };

    for (int j = 0; j < N; j++) {
        const real_t *xrow = X + (size_t)j * k;          /* riga j di X (fetta) */
        /* svolgimento completo dei cicli su r e c (al massimo 8 x 32): solo
           così gli accumulatori diventano registri e il vettorizzatore SLP
           può raggruppare le FMA lungo c */
#pragma GCC unroll 8
        for (int r = 0; r < rb; r++) {
            const real_t a = A[(size_t)r * lda + j];     /* a riusato kc volte  */
#pragma GCC unroll 32
            for (int c = 0; c < kc; c++)
                acc[r][c] += a * xrow[c];                /* x riusato rb volte  */
        }
    }

    for (int r = 0; r < rb; r++)
        for (int c = 0; c < kc; c++)
            Y[(size_t)r * k + c] = acc[r][c];
}

/*
 * Tutte le M righe per la fetta di colonne [c0, c0 + KC): blocchi pieni da
 * RB_FOR(KC) righe, poi le righe rimanenti una alla volta (rb = 1).
 * Chiamata sempre con KC costante: ogni chiamata genera codice dedicato.
 */
ALWAYS_INLINE void slice(int KC, int M, int N, int k, int c0,
                         const real_t *restrict A, int lda,
                         const real_t *restrict X, real_t *restrict Y)
{
    const int RB = RB_FOR(KC);
    int i = 0;
    for (; i + RB <= M; i += RB)
        block(RB, KC, N, k, A + (size_t)i * lda, lda, X + c0, Y + (size_t)i * k + c0);
    for (; i < M; i++)
        block(1, KC, N, k, A + (size_t)i * lda, lda, X + c0, Y + (size_t)i * k + c0);
}

void cpu_block(int M, int N, int k, const real_t *restrict A, int lda,
               const real_t *restrict X, real_t *restrict Y)
{
    /* Sceglie le fette in base a k (k_dispatch.h); ogni fetta è una
       chiamata a slice() con KC costante, quindi con codice dedicato. */
#define SLICE(KC, c0) slice(KC, M, N, k, c0, A, lda, X, Y)
    K_DISPATCH(k, SLICE);
#undef SLICE
}
