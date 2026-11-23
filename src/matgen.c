/*
 * matgen.c — generazione deterministica delle matrici di test.
 *
 * Ogni elemento è una funzione pura della posizione globale:
 *
 *      a_ij = gen_value(seed, i, j)
 *
 * Non si usa un generatore sequenziale (rand()), il cui valore dipenderebbe
 * dall'ordine di generazione e quindi dalla griglia di processi. Qui ogni
 * processo genera da solo il proprio blocco e la matrice è identica per
 * qualunque decomposizione. Questa idea (generatore "counter-based": l'uscita
 * è funzione di un contatore/indice invece che di uno stato) è descritta in:
 *
 *   J. K. Salmon, M. A. Moraes, R. O. Dror, D. E. Shaw,
 *   "Parallel random numbers: as easy as 1, 2, 3", SC '11, 2011.
 *
 * ---------------------------------------------------------------------------
 * ALGORITMO RIPRESO: SplitMix64
 *
 * La funzione mix64() qui sotto è il passo di uscita del generatore
 * SplitMix64, ripreso senza modifiche:
 *
 *   G. L. Steele Jr., D. Lea, C. H. Flood,
 *   "Fast splittable pseudorandom number generators",
 *   OOPSLA '14, ACM, pp. 453-472, 2014.  doi:10.1145/2660193.2660195
 *
 * La parte di mescolamento (moltiplicatori e shift) è la variante "Mix13" di
 * David Stafford, ottenuta con una ricerca al computer per massimizzare
 * l'effetto valanga del finalizzatore a 64 bit di MurmurHash3:
 *
 *   D. Stafford, "Better Bit Mixing - Improving on MurmurHash3's 64-bit
 *   Finalizer", 2011.
 *   https://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html
 *
 * La conversione da intero a 64 bit a double in [0,1) con (x >> 11) * 2^-53
 * è la tecnica standard raccomandata da S. Vigna (https://prng.di.unimi.it/).
 * ---------------------------------------------------------------------------
 */
#include "matgen.h"

/*
 * mix64 — rimescola i 64 bit di z (passo di uscita di SplitMix64).
 *
 * Proprietà:
 *  - biiettiva: ogni passo è invertibile (XOR-shift e moltiplicazione per una
 *    costante dispari modulo 2^64), quindi ingressi diversi danno sempre
 *    uscite diverse;
 *  - effetto valanga: invertendo un solo bit d'ingresso cambiano in media
 *    32 dei 64 bit d'uscita (misurato su 6.4 milioni di prove: 32.00).
 *
 * static: visibile solo in questo file (dettaglio interno).
 * inline: suggerisce al compilatore di evitare il costo della chiamata.
 */
static inline uint64_t mix64(uint64_t z)
{
    /* "Golden gamma" = 2^64 / phi (phi = rapporto aureo), reso dispari.
       Essendo dispari, sommarlo ripetutamente modulo 2^64 visita tutti i
       valori prima di ripetersi; il rapporto aureo distribuisce i multipli
       nel modo più uniforme possibile (Fibonacci hashing, Knuth). */
    z += 0x9E3779B97F4A7C15ULL;

    /* Mix13 di Stafford: ogni riga alterna
       - z ^ (z >> s): porta i bit alti verso quelli bassi;
       - * costante dispari: porta i bit bassi verso quelli alti.
       Moltiplicatori e shift (30, 27, 31) sono quelli trovati dalla ricerca
       di Stafford: non derivano da una formula. */
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/*
 * gen_value — valore deterministico in [-1, 1) per la posizione (i, j).
 *
 * Dipende SOLO da (seed, i, j): non dalle dimensioni della matrice, né dal
 * numero di processi, né dall'ordine delle chiamate.
 */
double gen_value(uint64_t seed, long i, long j)
{
    /* Due mescolamenti in cascata: prima seed e riga, poi si aggiunge la
       colonna. Così (i, j) e (j, i) danno valori diversi, cosa che non
       accadrebbe con una combinazione simmetrica come i + j. */
    uint64_t h = mix64(mix64(seed ^ (uint64_t)i) + (uint64_t)j);

    /* Da 64 bit casuali a double in [-1, 1):
       - h >> 11      tiene i 53 bit alti (un double ha 53 bit di mantissa,
                      64 - 53 = 11): intero in [0, 2^53);
       - * 0x1.0p-53  moltiplica per 2^-53 (letterale esadecimale del C,
                      esatto): valore in [0, 1);
       - * 2 - 1      porta l'intervallo in [-1, 1).
       Valori positivi e negativi a media nulla: le somme non crescono
       sistematicamente con N e le cancellazioni mettono alla prova la
       precisione del calcolo. */
    return (double)(h >> 11) * 0x1.0p-53 * 2.0 - 1.0;
}

/*
 * fill_block — riempie un blocco row-major nrows x ncols (leading dim ld)
 * che corrisponde alla porzione della matrice globale che parte da
 * (row0, col0).
 *
 *   r, c                 indici LOCALI  -> dove scrivere in B
 *   row0 + r, col0 + c   indici GLOBALI -> quale valore generare
 *
 * Il cast a real_t arrotonda al momento della memorizzazione (rilevante in
 * float): il riferimento seriale deve usare lo stesso arrotondamento.
 */
void fill_block(real_t *B, int ld, long row0, int nrows,
                long col0, int ncols, uint64_t seed)
{
    for (int r = 0; r < nrows; r++)
        for (int c = 0; c < ncols; c++)
            B[IDX(r, c, ld)] = (real_t)gen_value(seed, row0 + r, col0 + c);
}
