/*
 * mxm.h — il nucleo: prodotto distribuito Y = A * X su una griglia Pr x Pc.
 *
 * È la parte riutilizzabile del progetto. Chi ha una griglia di processi
 * (distrib.h) e il proprio blocco di A calcola il prodotto con tre chiamate,
 * senza sapere nulla di opzioni, ripetizioni o statistiche: quelle restano
 * a chi misura (main.c, che è solo uno dei possibili chiamanti).
 *
 * Dove stanno i dati (vedi distrib.h):
 *   A_rc  sul processo (r, c): il suo blocco di righe e colonne di A;
 *   X_c   sulla riga 0 della griglia, nel processo (0, c);
 *   Y_r   sulla colonna 0 della griglia, nel processo (r, 0).
 *   La traccia permette di assumere che X e Y occupino una sola riga o
 *   colonna della griglia.
 *
 * Una invocazione (mxm_apply):
 *   1. MPI_Bcast di X_c lungo la colonna c  (dalla riga 0 a tutta la colonna)
 *   2. Y_rc = A_rc * X_c sul backend         (prodotto locale: GPU o CPU)
 *   3. MPI_Reduce (somma) degli Y_rc lungo la riga r  (verso la colonna 0)
 *   Il passo 3 serve perché ogni processo ha usato solo le SUE colonne di A:
 *   y_i = somma sui gruppi di colonne c dei contributi A_rc * X_c.
 *
 * Uso, su ogni processo della griglia g:
 *
 *   mxm_init_device(MPI_COMM_WORLD);                  una volta per processo
 *   mxm_block_t b = mxm_block(&g, M, N, 1);           il mio blocco di A
 *   real_t *A = prep_alloc_matrix(b.Mloc, b.lda, b.align);
 *   ... riempire A: elemento (i, j) del blocco in A[IDX(i, j, b.lda)] ...
 *   mxm_t *h = mxm_create(&g, &b, k, A, be_default_kernel());
 *   mxm_apply(h, X, Y, NULL);                         quante volte si vuole
 *   mxm_destroy(h);
 */
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

/* Tempi di una invocazione, in secondi, misurati su QUESTO processo.
   Il tempo del nucleo (CORE) è bcast + calcolo + reduce: la traccia chiede
   di escludere i trasferimenti host <-> device di X e Y, riportati a parte.
   Il calcolo si misura a orologio prima e dopo be_compute, che ritorna solo
   a calcolo finito: l'intervallo comprende tutto ciò che il processo
   aspetta per avere il risultato, compresa l'eventuale attesa della GPU
   occupata da altri processi. COMPUTE_DEV è invece il tempo visto dal
   dispositivo (GPU: eventi CUDA): con più processi sulla stessa GPU non
   vede l'attesa del turno, perché il cronometro parte solo quando la GPU
   arriva al processo. Con la GPU in uso esclusivo le due misure
   differiscono solo per il costo di lancio e sincronizzazione (pochi
   microsecondi). */
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

/* Prepara il dispositivo di calcolo di questo processo (GPU: sceglie la
   scheda in base alla posizione del processo sul suo nodo; CPU: nulla).
   Chiamata collettiva su comm, una volta per processo, prima di mxm_create. */
void mxm_init_device(MPI_Comm comm);

/* Blocco di A del processo chiamante, per una matrice M x N sulla griglia g.
   pad = 1: righe allineate all'unità di lettura del backend (prep.h). */
mxm_block_t mxm_block(const grid_t *g, int M, int N, int pad);

/* Prepara il prodotto con multivettori di k colonne. A è il blocco locale,
   già riempito, con la disposizione descritta da b (GPU: viene caricato
   sulla scheda qui, una volta sola). kernel: vedi be_kernel_list in
   backend.h. È il preprocessamento: va fatto fuori dal tempo del nucleo.
   g, b e A restano del chiamante e devono vivere fino a mxm_destroy. */
mxm_t *mxm_create(const grid_t *g, const mxm_block_t *b, int k,
                  const real_t *A, int kernel);

/* Una invocazione: Y = A * X. Chiamata collettiva sulla griglia.
     X      ingresso, Nloc x k row-major (ld = k): letto solo sulla riga 0
            della griglia, altrove può essere NULL;
     Y      uscita, Mloc x k row-major (ld = k): scritto solo sulla colonna
            0 della griglia, altrove può essere NULL;
     times  se non NULL, riceve gli MXM_NTIMES tempi di questo processo. */
void mxm_apply(mxm_t *h, const real_t *X, real_t *Y, double *times);

/* Strumento di misura: con on = 1 i processi eseguono la fase locale a
   turno, uno alla volta, invece che tutti insieme. Ognuno ha la macchina
   per sé, come se fosse su un nodo separato ("nodi emulati"). Il risultato
   non cambia, cambiano i tempi. Stesso valore su tutti i processi. */
void mxm_set_serial(mxm_t *h, int on);

/* Libera lo stato del prodotto (non A, che è del chiamante). */
void mxm_destroy(mxm_t *h);

#endif /* MXM_H */
