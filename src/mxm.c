/*
 * mxm.c — prodotto distribuito Y = A * X (interfaccia e algoritmo in mxm.h).
 *
 * Qui c'è solo la parte MPI: chi comunica cosa e in che ordine. Il prodotto
 * del singolo blocco è del backend (backend.h), che non sa nulla di MPI.
 */
#include <stdlib.h>
#include <mpi.h>

#include "mxm.h"
#include "prep.h"
#include "backend.h"

/* Tipo MPI corrispondente a real_t, per le comunicazioni. */
#ifdef USE_FLOAT
#define MPI_REAL_T MPI_FLOAT
#else
#define MPI_REAL_T MPI_DOUBLE
#endif

//MPI_Init -> grid_create -> mxm_init_device -> mxm_block
//-> (alloca e riempie A, X) -> mxm_create -> [mxm_apply ripetuta] -> mxm_destroy

struct mxm {
    const grid_t *g;
    backend_t *be;       /* prodotto locale su A_rc                          */
    int Mloc, Nloc, k;
    int serial;          /* 1 = fase locale a turno (mxm_set_serial)         */
    real_t *Xrecv;       /* X_c ricevuta dalla riga 0 (sulla riga 0 non
                            serve: lì X è quella del chiamante)              */
    real_t *Ypart;       /* Y_rc: il contributo di questo processo a Y_r     */
};

void mxm_init_device(MPI_Comm comm) //assegna device a processo
{
    // Ottengo il rank locale del processo sul nodo, per inizializzare il device corretto.
    MPI_Comm node;
    int local_rank;
    //splita processi in base al nodo fisico
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    // Ottengo il rank locale del processo sul nodo
    MPI_Comm_rank(node, &local_rank);
    // init
    MPI_Comm_free(&node);
    be_init_device(local_rank);
}

mxm_block_t mxm_block(const grid_t *g, int M, int N, int pad){
    //assegna blocco di a a processo
    mxm_block_t b;
    b.Mloc  = blk_size(M, g->Pr, g->myr); //gestione +1
    b.Nloc  = blk_size(N, g->Pc, g->myc); 
    b.m_off = blk_off(M, g->Pr, g->myr);
    b.n_off = blk_off(N, g->Pc, g->myc);
    b.align = pad ? be_align_bytes() : 0; //padding calcolato
    b.lda   = prep_leading_dim(b.Nloc, b.align); // padding applicato
    return b;
}

mxm_t *mxm_create(const grid_t *g, const mxm_block_t *b, int k,const real_t *A, int kernel){
    //prepara struttura dati per il prodotto distribuito
    mxm_t *h = malloc(sizeof *h);
    h->g = g;
    h->Mloc = b->Mloc;
    h->Nloc = b->Nloc;
    h->k = k;
    h->serial = 0;

    /* +1: un blocco può essere vuoto (più processi che righe o colonne). */
    h->Xrecv = (g->myr == 0) ? NULL
             : malloc((size_t)b->Nloc * k * sizeof(real_t) + 1);
    h->Ypart = malloc((size_t)b->Mloc * k * sizeof(real_t) + 1);
    //passaggio (eventual) a gpu
    h->be = be_create(b->Mloc, b->Nloc, k, A, b->lda, b->align != 0, kernel);
    return h;
}

void mxm_set_serial(mxm_t *h, int on)
{ //seriale (-s) o parallelo (default)
    h->serial = on;
}

/* Prodotto locale Y_rc = A_rc * X_c, con i tempi delle sue tre fasi. */
static void local_product(mxm_t *h, const real_t *Xc, double *t)
{
    const double t0 = MPI_Wtime();
    be_set_x(h->be, Xc);
    const double t1 = MPI_Wtime();
    t[MXM_T_COMPUTE_DEV] = be_compute(h->be);    /* tempo visto dal device */
    const double t2 = MPI_Wtime();
    be_get_y(h->be, h->Ypart);
    const double t3 = MPI_Wtime();

    t[MXM_T_H2D]     = t1 - t0;
    t[MXM_T_COMPUTE] = t2 - t1;                  /* a orologio: attesa della GPU compresa */
    t[MXM_T_D2H]     = t3 - t2;
}

void mxm_apply(mxm_t *h, const real_t *X, real_t *Y, double *times)
{ //invocazione del prodotto, cioè broadcast di X, prodotto locale e riduzione di Y, e restituisce i tempi;
    const grid_t *g = h->g;
    double t[MXM_NTIMES];

    // a seconda del rank del processo, Xc punta a X (riga 0) o a Xrecv (altrove)
    real_t *Xc = (g->myr == 0) ? (real_t *)X : h->Xrecv;

    const double t0 = MPI_Wtime();
    MPI_Bcast(Xc, h->Nloc * h->k, MPI_REAL_T, 0, g->col);    /* root = riga 0 */
    const double t1 = MPI_Wtime();

    if (!h->serial) {
        local_product(h, Xc, t);                 /* tutti insieme */
    } else {
        /* A turno: calcola il processo di rank "turn", gli altri lo
           aspettano alla barriera; poi tocca al successivo. */
        for (int turn = 0; turn < g->size; turn++) {
            if (g->rank == turn)
                local_product(h, Xc, t);
            MPI_Barrier(g->grid);
        }
    }
    const double t2 = MPI_Wtime();

    MPI_Reduce(h->Ypart, Y, h->Mloc * h->k, MPI_REAL_T, MPI_SUM, 0, g->row);  /* root = colonna 0 */
    const double t3 = MPI_Wtime();

    if (times == NULL) return;

    t[MXM_T_BCAST]  = t1 - t0;
    t[MXM_T_LOCAL]  = t2 - t1;
    t[MXM_T_REDUCE] = t3 - t2;
    /* Tempo ufficiale del nucleo: comunicazioni + calcolo, senza i
       trasferimenti host <-> device. A turni il calcolo è quello del
       proprio turno, misurato in isolamento. */
    t[MXM_T_CORE]   = t[MXM_T_BCAST] + t[MXM_T_COMPUTE] + t[MXM_T_REDUCE];
    for (int p = 0; p < MXM_NTIMES; p++)
        times[p] = t[p];
}

void mxm_destroy(mxm_t *h) // libera la struttura dati del prodotto distribuito
{
    be_destroy(h->be);
    free(h->Xrecv);
    free(h->Ypart);
    free(h);
}
