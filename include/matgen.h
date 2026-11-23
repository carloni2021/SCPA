/*
 * matgen.h — generazione ripetibile e indipendente dalla griglia.
 *
 * Ogni elemento è funzione pura di (seed, i_globale, j_globale): qualunque
 * processo può generare qualunque sotto-blocco senza comunicare, e il risultato
 * è identico per ogni griglia Pr x Pc e ogni numero di processi.
 * Questo rende possibile anche la verifica distribuita (vedi main.c).
 */
#ifndef MATGEN_H
#define MATGEN_H

#include <stdint.h>
#include "common.h"

#define SEED_A 0x5CA0A11CEULL
#define SEED_X 0x5CA0B0B0EULL

/* Valore pseudocasuale in [-1, 1) per la posizione globale (i, j). */
double gen_value(uint64_t seed, long i, long j);

/* Riempie un blocco row-major nrows x ncols (leading dim ld) che parte
   dalla posizione globale (row0, col0). */
void fill_block(real_t *B, int ld, long row0, int nrows,
                long col0, int ncols, uint64_t seed);

#endif /* MATGEN_H */
