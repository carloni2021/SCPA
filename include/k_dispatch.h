#ifndef K_DISPATCH_H
#define K_DISPATCH_H
/* K_DISPATCH: divide le k colonne in fette con KC costante, per i kernel CPU e GPU */
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
