#ifndef BACKEND_H
#define BACKEND_H

#include "common.h"

// per evitare name mangling in C++ (vedi https://gcc.gnu.org/onlinedocs/gcc-3.0.1/gcc_3.html#SEC3)
#ifdef __cplusplus
extern "C" {
#endif

// Tipo opaco: mxm.c maneggia solo un puntatore.
typedef struct backend backend_t;

// Prepara il dispositivo di calcolo.
void be_init_device(int local_rank);

// Allineamento (in byte) delle righe di A preferito dal backend.
int be_align_bytes(void);

// Crea il backend per un blocco M x N con multivettore di k colonne e prepara A
backend_t  *be_create(int M, int N, int k, const real_t *A, int lda, int pad, int kernel);

//Rende X disponibile al calcolo con eventuali trasformazioni
void        be_set_x(backend_t *b, const real_t *X);

/* Esegue Y = A * X e ritorna il tempo di solo calcolo in secondi.*/
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
