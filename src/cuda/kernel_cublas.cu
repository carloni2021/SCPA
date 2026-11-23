/*
 * kernel_cublas.cu — prodotto con cuBLAS (gemm), usato SOLO come termine di
 * confronto: non fa parte della soluzione.
 *
 * cuBLAS lavora in column-major. Una matrice row-major R x C, letta come
 * column-major, è la sua trasposta C x R. Quindi
 *     Y = A X   (row-major)   <=>   Y^T = X^T A^T   (column-major)
 * cioè una gemm senza trasposizioni con m = k, n = M, dimensione interna N,
 * passando X prima di A. Nessuna copia: cambia solo l'interpretazione.
 * La leading dimension di A^T (column-major) è lda: il padding è gestito.
 */
#include "cuda_check.h"
#include "kernels.h"

void launch_cublas(cublasHandle_t h, int M, int N, int k, const real_t *A, int lda,
                   const real_t *X, real_t *Y)
{
    const real_t one = 1, zero = 0;          /* Y = 1 * (X^T A^T) + 0 * Y */
#ifdef USE_FLOAT
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, k, M, N,
                             &one, X, k, A, lda, &zero, Y, k));
#else
    CUBLAS_CHECK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, k, M, N,
                             &one, X, k, A, lda, &zero, Y, k));
#endif
}
