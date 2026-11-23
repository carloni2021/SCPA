/*
 * backend.h — interfaccia del prodotto LOCALE  Y_loc = A_loc * X_loc.
 *
 * È il "contratto" tra la parte MPI e la parte di calcolo:
 *   - mxm.c (MPI) decide QUALI dati ha ogni processo e li fa comunicare;
 *   - il backend decide COME calcolare il prodotto del proprio blocco.
 * mxm.c non sa nulla di CUDA né dei kernel: chiama solo le funzioni qui
 * sotto. Cambiare backend o kernel non tocca la parte MPI.
 *
 * Implementazioni (se ne compila una sola, scelta con make BACKEND=...):
 *   src/cuda/ : GPU   (un processo MPI usa la GPU per il suo blocco)
 *   src/cpu/  : CPU   (un processo MPI usa un core per il suo blocco)
 * Ogni backend ha più kernel, scelti a runtime con l'opzione -v.
 *
 * Ciclo di vita e misure. Le fasi sono separate perché la traccia chiede di
 * escludere dal tempo del nucleo preprocessamento e trasferimenti da/verso
 * la scheda, misurandoli a parte:
 *
 *   be_create   una volta:       prepara A (GPU: upload)        (preprocessamento, ESCLUSO)
 *   be_set_x    per invocazione: rende X disponibile al calcolo (GPU: host -> device
 *                                e, per warp_row_xt e warp_row_smem, trasposizione
 *                                di X; ESCLUSO)
 *   be_compute  per invocazione: solo il calcolo                (INCLUSO nel tempo)
 *   be_get_y    per invocazione: rende Y disponibile a MPI      (GPU: device -> host, ESCLUSO)
 *   be_destroy  una volta:       libera tutto
 *
 * Sulla GPU A viene caricata una volta sola, mentre X e Y viaggiano a ogni
 * invocazione perché MPI le comunica in memoria host (MPI sul server non è
 * CUDA-aware). Sulla CPU i dati sono già in memoria host: set_x e get_y
 * costano poco o nulla.
 */
#ifndef BACKEND_H
#define BACKEND_H

#include "common.h"

/* Il backend CUDA è compilato da nvcc come C++: extern "C" impedisce il
   name mangling, così i simboli hanno lo stesso nome che si aspetta il
   linker quando li chiama mxm.c (compilato come C). */
#ifdef __cplusplus
extern "C" {
#endif

/* Tipo opaco: mxm.c maneggia solo un puntatore, i campi (puntatori device,
   eventi CUDA, ...) sono definiti nell'implementazione e restano nascosti. */
typedef struct backend backend_t;

/* Prepara il dispositivo di calcolo. Backend CUDA: con più processi sullo
   stesso nodo, local_rank % numero_di_GPU li distribuisce sulle GPU (sul
   server ce n'è una sola: tutti la condividono). Backend CPU: nulla. */
void        be_init_device(int local_rank);

/* Allineamento (in byte) delle righe di A preferito dal backend, cioè la
   sua unità di lettura dalla memoria: GPU 128 (linea della cache L1),
   CPU 64 (linea di cache). mxm.c lo usa per il padding (vedi prep.h). */
int         be_align_bytes(void);

/* Crea il backend per un blocco M x N con multivettore di k colonne e
   prepara A (row-major, leading dimension lda >= N: le righe possono avere
   colonne di padding per l'allineamento, vedi prep.h). pad = 1 se il
   padding è attivo (opzione -p): il backend lo applica anche alle proprie
   strutture interne (GPU: X trasposta). kernel seleziona la versione da
   usare (vedi be_kernel_list). Un kernel inesistente termina il programma
   con un errore. */
backend_t  *be_create(int M, int N, int k, const real_t *A, int lda, int pad, int kernel);

/* Rende X (N x k, row-major, ld = k) disponibile al calcolo, comprese
   eventuali trasformazioni richieste dal kernel (GPU, warp_row_xt e
   warp_row_smem: trasposizione). Ritorna a operazione completata. */
void        be_set_x(backend_t *b, const real_t *X);

/* Esegue Y = A * X e ritorna il tempo di solo calcolo in secondi.
   Ritorna solo a calcolo terminato (sincrono). */
double      be_compute(backend_t *b);

/* Copia Y (M x k, row-major, ld = k) nel buffer host indicato. */
void        be_get_y(backend_t *b, real_t *Y);

/* Libera memoria e risorse del backend (A appartiene al chiamante). */
void        be_destroy(backend_t *b);

/* Nome del backend ("cuda" / "cpu"), per l'output. */
const char *be_name(void);

/* Nome di un kernel, per l'output (NULL se il kernel non esiste). */
const char *be_kernel_name(int kernel);

/* Kernel usato se l'utente non specifica -v: il migliore del backend. */
int         be_default_kernel(void);

/* Elenco leggibile dei kernel disponibili, per il messaggio d'uso. */
const char *be_kernel_list(void);

#ifdef __cplusplus
}
#endif

#endif /* BACKEND_H */
