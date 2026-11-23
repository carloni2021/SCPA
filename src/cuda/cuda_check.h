/*
 * cuda_check.h — controllo degli errori CUDA e cuBLAS (solo per i file .cu).
 *
 * Ogni funzione del runtime CUDA restituisce un codice di errore. Se lo si
 * ignora, un errore (memoria esaurita, accesso illegale, configurazione di
 * lancio non valida...) passa inosservato e si manifesta più avanti come
 * risultato sbagliato. Le macro controllano il codice e terminano indicando
 * file e riga in cui l'errore si è verificato.
 *
 * do { ... } while (0): fa sì che la macro si comporti come una singola
 * istruzione anche dentro un if senza graffe.
 */
#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do {                                              \
    cudaError_t e_ = (call);                                               \
    if (e_ != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(e_));                                   \
        exit(EXIT_FAILURE);                                                \
    } } while (0)

#define CUBLAS_CHECK(call) do {                                            \
    cublasStatus_t s_ = (call);                                            \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                     \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,    \
                (int)s_);                                                  \
        exit(EXIT_FAILURE);                                                \
    } } while (0)

#endif /* CUDA_CHECK_H */
