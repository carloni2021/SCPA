/*
 * distrib.h — griglia di processi 2D e distribuzione a blocchi degli indici.
 *
 * La matrice A (M x N) è divisa su una griglia Pr x Pc di processi: il
 * processo di coordinate (r, c) possiede il blocco formato dalle righe del
 * r-esimo gruppo e dalle colonne del c-esimo gruppo.
 *
 *              colonne di A ->
 *            c=0      c=1      c=2
 *   r=0   [ A_00  |  A_01  |  A_02 ]     X_c (righe di X corrispondenti
 *   r=1   [ A_10  |  A_11  |  A_12 ]          alle colonne del gruppo c)
 *                                        Y_r (righe di Y del gruppo r)
 */
#ifndef DISTRIB_H
#define DISTRIB_H

#include <mpi.h>

/*
 * Distribuzione bilanciata di n elementi su p parti.
 *
 * Ogni parte riceve q = n / p elementi; i rem = n % p elementi avanzati
 * vanno uno ciascuno alle prime rem parti. Così le parti differiscono al
 * massimo di 1 elemento e la divisione non esatta non richiede padding.
 *
 *   esempio n = 10, p = 3  ->  q = 3, rem = 1
 *     parte 0: 4 elementi, offset 0   (indici 0..3)
 *     parte 1: 3 elementi, offset 4   (indici 4..6)
 *     parte 2: 3 elementi, offset 7   (indici 7..9)
 */

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

/*
 * Descrittore della griglia di processi.
 *
 * Un comunicatore MPI è un gruppo di processi che possono comunicare tra
 * loro; ogni processo vi ha un rank (0..size-1). Oltre al comunicatore
 * dell'intera griglia ne creiamo due "parziali", usati dall'algoritmo:
 *
 *   row : i processi della MIA riga (stesso r, c variabile).
 *         Il rank in row coincide con la mia colonna c.
 *         Serve per la reduce di Y: si sommano i contributi di tutti i
 *         blocchi della riga r.
 *   col : i processi della MIA colonna (stesso c, r variabile).
 *         Il rank in col coincide con la mia riga r.
 *         Serve per il broadcast di X: la riga 0 invia X_c a tutta la
 *         colonna c.
 */
typedef struct {
    MPI_Comm grid;   /* comunicatore cartesiano Pr x Pc          */
    MPI_Comm row;    /* processi della stessa riga della griglia   */
    MPI_Comm col;    /* processi della stessa colonna della griglia */
    int rank, size;  /* rank e numero di processi nella griglia    */
    int Pr, Pc;      /* estensione della griglia                   */
    int myr, myc;    /* coordinate (riga, colonna) del processo    */
} grid_t;

/* Crea la griglia Pr x Pc sui processi di comm (di solito MPI_COMM_WORLD;
   un'applicazione più grande può passare un sottoinsieme dei suoi processi)
   e i comunicatori di riga e di colonna. È una chiamata collettiva su comm.
   Ritorna 0 se ok, -1 se Pr*Pc non coincide con il numero di processi di
   comm (stesso esito su tutti i processi). */
int  grid_create(MPI_Comm comm, grid_t *g, int Pr, int Pc);

/* Libera i comunicatori creati da grid_create. */
void grid_free(grid_t *g);

#endif /* DISTRIB_H */
