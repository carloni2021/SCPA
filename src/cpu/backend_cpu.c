#define _POSIX_C_SOURCE 200809L   /* clock_gettime con -std=c11 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "backend.h"
#include "kernels_cpu.h"

struct backend {
    int M, N, k, lda, kernel;
    const real_t *A;     /* blocco di A del chiamante (non copiato)   */
    const real_t *X;     /* X del chiamante, impostata da be_set_x    */
    real_t *Y;           /* risultato locale, copiato da be_get_y      */
};

/* Tempo monotono in secondi: non risente di correzioni dell'orologio. */
static double now(void)
{
    struct timespec ts;
    //orario attuale
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void be_init_device(int local_rank) { (void)local_rank; }   /* nessun dispositivo */

const char *be_name(void) { return "cpu"; }

const char *be_kernel_name(int kernel)
{
    switch (kernel) {
    case KERNEL_CPU_ROW:   return "cpu_row";
    case KERNEL_CPU_BLOCK: return "cpu_block";
    default:               return NULL;
    }
}

int be_default_kernel(void) { return KERNEL_CPU_BLOCK; }

/* Linea di cache della CPU: righe di A allineate a inizio linea. */
int be_align_bytes(void) { return 64; }

const char *be_kernel_list(void)
{
    return "0 = cpu_row, 1 = cpu_block (default)";
}

backend_t *be_create(int M, int N, int k, const real_t *A, int lda, int pad, int kernel)
{
    (void)pad;           /* la CPU non ha strutture interne da allineare */
    if (be_kernel_name(kernel) == NULL) {
        fprintf(stderr, "kernel %d inesistente\n", kernel);
        exit(EXIT_FAILURE);
    }
    backend_t *b = malloc(sizeof *b);
    b->M = M; b->N = N; b->k = k; b->lda = lda; b->kernel = kernel;
    b->A = A;
    b->X = NULL;
    /* +1: un processo può avere un blocco vuoto (più processi che righe) */
    b->Y = malloc((size_t)M * k * sizeof(real_t) + 1);
    return b;
}

void be_set_x(backend_t *b, const real_t *X){
    b->X = X;            /*semplice setter */
}

double be_compute(backend_t *b){
    //lancia il kernel scelto e ritorna il tempo di calcolo in secondi
    double t0 = now();
    switch (b->kernel) {
    case KERNEL_CPU_ROW:   cpu_row  (b->M, b->N, b->k, b->A, b->lda, b->X, b->Y); break;
    case KERNEL_CPU_BLOCK: cpu_block(b->M, b->N, b->k, b->A, b->lda, b->X, b->Y); break;
    }
    return now() - t0;
}

void be_get_y(backend_t *b, real_t *Y){
    //getter
    memcpy(Y, b->Y, (size_t)b->M * b->k * sizeof(real_t));
}

void be_destroy(backend_t *b)
{
    free(b->Y);          /* A e X appartengono al chiamante */
    free(b);
}
