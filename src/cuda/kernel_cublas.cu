#include "cuda_check.h"
#include "kernels.h"

void launch_cublas(cublasHandle_t h, int M, int N, int k, const real_t *A, int lda,const real_t *X, real_t *Y){
    const real_t one = 1, zero = 0;          /* Y = 1 * (X^T A^T) + 0 * Y */
#ifdef USE_FLOAT
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, k, M, N,&one, X, k, A, lda, &zero, Y, k));
#else
    CUBLAS_CHECK(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, k, M, N,&one, X, k, A, lda, &zero, Y, k));
#endif
}
