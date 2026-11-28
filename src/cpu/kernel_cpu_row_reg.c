/*
 * kernel_cpu_row_reg.c: una riga alla volta, come cpu_row, ma con gli
 * accumulatori in registri. Una versione per ogni k della traccia
 * (k_dispatch.h), così il numero di accumulatori è noto in compilazione.
 * La vettorizzazione di ciclo è disattivata: gcc vettorizzerebbe lungo j
 * invece che lungo c, e gli accumulatori finirebbero in memoria.
 */
#pragma GCC optimize ("no-tree-loop-vectorize")

#include "kernels_cpu.h"
#include "k_dispatch.h"

/* forza l'espansione nel chiamante: senza, la specializzazione non avviene */
#define ALWAYS_INLINE static inline __attribute__((always_inline))

/*
 * Tutte le M righe per la fetta di colonne [c0, c0 + KC) di Y.
 * Chiamata sempre con KC costante: ogni chiamata genera codice dedicato,
 * con il ciclo su c svolto e gli accumulatori in registri.
 */
ALWAYS_INLINE void slice(int KC, int M, int N, int k, int c0,
                         const real_t *restrict A, int lda,
                         const real_t *restrict X, real_t *restrict Y)
{
    for (int i = 0; i < M; i++) {
        const real_t *arow = A + (size_t)i * lda;
        real_t acc[32] = { 0 };                          /* al massimo 32 colonne */

        for (int j = 0; j < N; j++) {
            const real_t a = arow[j];                    /* letto una volta       */
            const real_t *xrow = X + (size_t)j * k + c0; /* riga j di X (fetta)   */
#pragma GCC unroll 32
            for (int c = 0; c < KC; c++)
                acc[c] += a * xrow[c];                   /* a riusato KC volte    */
        }

        for (int c = 0; c < KC; c++)
            Y[(size_t)i * k + c0 + c] = acc[c];
    }
}

void cpu_row_reg(int M, int N, int k, const real_t *restrict A, int lda,
                 const real_t *restrict X, real_t *restrict Y)
{
#define SLICE(KC, c0) slice(KC, M, N, k, c0, A, lda, X, Y)
    K_DISPATCH(k, SLICE);
#undef SLICE
}
