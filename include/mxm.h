#ifndef MXM_H
#define MXM_H

#include "common.h"
#include "distrib.h"

/* Il blocco di A di un processo: quali righe e colonne globali possiede e
   come va disposto in memoria. */
typedef struct {
    int  Mloc, Nloc;     /* righe e colonne del blocco (0 se il blocco è vuoto) */
    long m_off, n_off;   /* prima riga e prima colonna globale                  */
    int  lda;            /* leading dimension di A: >= Nloc (padding, prep.h)   */
    int  align;          /* allineamento delle righe in byte; 0 = senza padding */
} mxm_block_t;

// Tempi di una invocazione, in secondi, misurati su QUESTO processo.
enum {
    MXM_T_BCAST,        /* MPI_Bcast di X lungo la colonna                      */
    MXM_T_H2D,          /* preparazione di X sul device: copia host -> device
                           e, per warp_row_xt, trasposizione          (esclusa) */
    MXM_T_COMPUTE,      /* calcolo locale, a orologio                (nel nucleo) */
    MXM_T_D2H,          /* copia di Y device -> host                  (esclusa) */
    MXM_T_REDUCE,       /* MPI_Reduce di Y lungo la riga                        */
    MXM_T_CORE,         /* nucleo = bcast + calcolo + reduce                    */
    MXM_T_LOCAL,        /* fase locale a orologio (set_x + calcolo + get_y; a
                           turni comprende l'attesa del turno degli altri)      */
    MXM_T_COMPUTE_DEV,  /* calcolo misurato dal backend            (informativo) */
    MXM_NTIMES
};

/* Tipo opaco: lo stato di un prodotto pronto per essere invocato. */
typedef struct mxm mxm_t;

//GPU: sceglie la scheda in base alla posizione del processo sul suo nodo, prima di mxm_create.
void mxm_init_device(MPI_Comm comm);

// Blocco di A del processo chiamante, per una matrice M x N sulla griglia g
mxm_block_t mxm_block(const grid_t *g, int M, int N, int pad);

// Crea lo stato del prodotto distribuito, pronto per essere invocato.
mxm_t *mxm_create(const grid_t *g, const mxm_block_t *b, int k,
                  const real_t *A, int kernel);

// Invoca il prodotto distribuito Y = A * X, con X e Y come descritto sopra.
void mxm_apply(mxm_t *h, const real_t *X, real_t *Y, double *times);

//distinzione tra calcolo seriale e parallelo
void mxm_set_serial(mxm_t *h, int on);

/* Libera lo stato del prodotto (non A, che è del chiamante). */
void mxm_destroy(mxm_t *h);

#endif /* MXM_H */
