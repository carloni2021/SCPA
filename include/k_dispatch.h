/*
 * k_dispatch.h — da k (noto solo a runtime) a kernel con numero di colonne
 * costante. Usato da cpu_block e dai kernel GPU warp_row*.
 *
 * Questi kernel tengono gli accumulatori nei registri, e per farlo il
 * numero di colonne di Y che calcolano (KC) deve essere una costante di
 * compilazione. Ognuno sa quindi elaborare una "fetta" di colonne
 * [c0, c0 + KC) con KC fissato. Qui sta, una volta sola, la regola che
 * copre un k qualunque con le fette:
 *
 *   - k della traccia (3, 6, 8, 20, 32): una sola fetta con KC = k, quindi
 *     A viene letta una volta;
 *   - ogni altro k: fette da 8 colonne, più una fetta finale per le colonne
 *     rimanenti (1..7). Funziona per ogni k, al costo di leggere A una
 *     volta per fetta.
 *
 * È una macro e non una funzione perché KC deve arrivare al kernel scritto
 * nel codice come numero: in C è l'argomento costante di una funzione
 * always_inline, in CUDA il parametro di un template. Il kernel dice come
 * si elabora una fetta con una macro a due argomenti (KC, c0), ad esempio:
 *
 *     #define SLICE(KC, c0)  launch_slice<KC>(M, N, k, c0, A, lda, X, Y)
 *     K_DISPATCH(k, SLICE);
 *     #undef SLICE
 */
#ifndef K_DISPATCH_H
#define K_DISPATCH_H

#define K_DISPATCH(k, SLICE)                                   \
    do {                                                       \
        int c0_ = 0;                                           \
        switch (k) {                                           \
        /* k della traccia: una sola fetta */                  \
        case 3:  SLICE(3,  0); break;                          \
        case 6:  SLICE(6,  0); break;                          \
        case 8:  SLICE(8,  0); break;                          \
        case 20: SLICE(20, 0); break;                          \
        case 32: SLICE(32, 0); break;                          \
        default:                                               \
            /* k generico: fette piene da 8 colonne... */      \
            for (; c0_ + 8 <= (k); c0_ += 8)                   \
                SLICE(8, c0_);                                 \
            /* ...più una fetta con le colonne rimanenti */    \
            switch ((k) - c0_) {                               \
            case 1: SLICE(1, c0_); break;                      \
            case 2: SLICE(2, c0_); break;                      \
            case 3: SLICE(3, c0_); break;                      \
            case 4: SLICE(4, c0_); break;                      \
            case 5: SLICE(5, c0_); break;                      \
            case 6: SLICE(6, c0_); break;                      \
            case 7: SLICE(7, c0_); break;                      \
            default: break;              /* 0: nessun resto */ \
            }                                                  \
        }                                                      \
    } while (0)

#endif /* K_DISPATCH_H */
