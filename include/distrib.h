#ifndef DISTRIB_H
#define DISTRIB_H

#include <mpi.h>

/* Numero di elementi della parte r. */
static inline int blk_size(int n, int p, int r)
{
    return n / p + (r < n % p ? 1 : 0);
}

/* Indice globale del primo elemento della parte r: le r parti precedenti
   contengono r*q elementi, più uno extra per ciascuna di quelle (al massimo
   rem) che hanno ricevuto un avanzo. */
static inline int blk_off(int n, int p, int r)
{
    int q = n / p, rem = n % p;
    return r * q + (r < rem ? r : rem);
}

// Descrittore della griglia di processi.
typedef struct {
    MPI_Comm grid;   /* comunicatore cartesiano Pr x Pc          */
    MPI_Comm row;    /* processi della stessa riga della griglia   */
    MPI_Comm col;    /* processi della stessa colonna della griglia */
    int rank, size;  /* rank e numero di processi nella griglia    */
    int Pr, Pc;      /* estensione della griglia                   */
    int myr, myc;    /* coordinate (riga, colonna) del processo    */
} grid_t;


int  grid_create(MPI_Comm comm, grid_t *g, int Pr, int Pc);

/* Libera i comunicatori creati da grid_create. */
void grid_free(grid_t *g);

#endif /* DISTRIB_H */
