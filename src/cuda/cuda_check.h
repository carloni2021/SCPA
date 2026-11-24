#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
// controllo degli errori CUDA e cuBLAS: se il codice ritorna un errore, stampa il messaggio e termina
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
