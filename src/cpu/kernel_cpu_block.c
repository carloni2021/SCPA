#pragma GCC optimize ("no-tree-loop-vectorize")

#include "kernels_cpu.h"
#include "k_dispatch.h"

/* Numero massimo di accumulatori: 1024 byte di registri, cioè 128 double o 256 float. */
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
