#include "matgen.h"

static inline uint64_t mix64(uint64_t z)
{
    /* "Golden gamma" */
    z += 0x9E3779B97F4A7C15ULL;

    /* Mix13 di Stafford:
       https://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html
       * 64 bit di input -> 64 bit di output, con buona distribuzione dei bit*/
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double gen_value(uint64_t seed, long i, long j)
{
    uint64_t h = mix64(mix64(seed ^ (uint64_t)i) + (uint64_t)j);
    return (double)(h >> 11) * 0x1.0p-53 * 2.0 - 1.0;
}

/*
 riempie un blocco row-major nrows x ncols (leading dim ld)
 che corrisponde alla porzione della matrice globale che parte da
 (row0, col0).
*/
void fill_block(real_t *B, int ld, long row0, int nrows,
                long col0, int ncols, uint64_t seed)
{
    for (int r = 0; r < nrows; r++)
        for (int c = 0; c < ncols; c++)
            B[IDX(r, c, ld)] = (real_t)gen_value(seed, row0 + r, col0 + c);
}
