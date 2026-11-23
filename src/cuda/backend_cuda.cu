/*
 * backend_cuda.cu — implementazione GPU dell'interfaccia backend.h.
 *
 * Questo file NON contiene kernel: gestisce memoria del device, tempi e
 * scelta del kernel. I kernel stanno in src/cuda/kernel_<nome>.cu (vedi kernels.h).
 *
 * Modello di esecuzione CUDA in breve:
 *   - host = CPU e sua RAM; device = GPU e sua memoria (separate: i dati
 *     vanno copiati esplicitamente da una all'altra);
 *   - un kernel è una funzione __global__ eseguita sul device da molti
 *     thread in parallelo; si lancia con  kernel<<<grid, block>>>(args);
 *   - i thread sono raggruppati in blocchi (fino a 1024 thread), i blocchi
 *     formano la griglia; l'hardware esegue i thread a gruppi di 32 (warp);
 *   - il lancio è asincrono: la CPU prosegue subito, la GPU lavora in
 *     parallelo. Per sapere quando il kernel è finito serve sincronizzarsi.
 */
#include <cstdio>
#include <cstdlib>
#include "backend.h"
#include "prep.h"
#include "cuda_check.h"
#include "kernels.h"

/* Definizione del tipo opaco per la GPU. I puntatori dA, dX, dY puntano
   alla memoria del DEVICE: non si possono dereferenziare dalla CPU. */
struct backend {
    int M, N, k, lda, kernel;
    real_t *dA, *dX, *dY;     /* A: M x N (ld = lda), X: N x k, Y: M x k, row-major */
    real_t *dXt;              /* X trasposta (k x N, ld = ldxt), solo per i kernel che la usano */
    int ldxt;                 /* leading dimension di dXt (padding come per A) */
    cudaEvent_t ev0, ev1;     /* eventi per cronometrare il kernel sulla GPU */
    cublasHandle_t blas;      /* contesto cuBLAS (solo per KERNEL_CUBLAS) */
};

extern "C" void be_init_device(int local_rank)
{
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    CUDA_CHECK(cudaSetDevice(local_rank % ndev));
    /* Il contesto CUDA si crea pigramente alla prima chiamata e costa
       decine o centinaia di ms: lo forziamo qui, fuori da ogni misura. */
    CUDA_CHECK(cudaFree(0));
}

/* I kernel che leggono X trasposta (Xt) invece di X. */
static bool uses_xt(int kernel)
{
    return kernel == KERNEL_WARP_ROW_XT || kernel == KERNEL_WARP_ROW_SMEM;
}

extern "C" const char *be_name(void) { return "cuda"; }

extern "C" const char *be_kernel_name(int kernel)
{
    switch (kernel) {
    case KERNEL_THREAD_ROW: return "thread_row";
    case KERNEL_WARP_ROW:   return "warp_row";
    case KERNEL_WARP_ROW_XT: return "warp_row_xt";
    case KERNEL_WARP_ROW_SMEM: return "warp_row_smem";
    case KERNEL_CUBLAS:     return "cublas";
    default:                return NULL;          /* kernel inesistente */
    }
}

extern "C" int be_default_kernel(void) { return KERNEL_WARP_ROW_XT; }

/* Linea della cache L1 della GPU (128 byte = 4 settori da 32): con le righe
   allineate a 128 byte un warp tocca il minimo sia di settori sia di linee. */
extern "C" int be_align_bytes(void) { return 128; }

extern "C" const char *be_kernel_list(void)
{
    return "0 = thread_row, 1 = warp_row, 2 = warp_row_xt (default), "
           "3 = warp_row_smem, -1 = cublas (solo riferimento)";
}

extern "C" backend_t *be_create(int M, int N, int k, const real_t *A, int lda, int pad, int kernel)
{
    if (be_kernel_name(kernel) == NULL) {         /* errore subito, prima di allocare */
        fprintf(stderr, "kernel %d inesistente\n", kernel);
        exit(EXIT_FAILURE);
    }

    backend_t *b = (backend_t *)calloc(1, sizeof *b);
    b->M = M; b->N = N; b->k = k; b->lda = lda; b->kernel = kernel;

    /* Allocazione in memoria device (globale). "? : 1" evita allocazioni di
       0 byte per i blocchi vuoti. A mantiene sul device la stessa leading
       dimension dell'host: se le righe hanno padding (prep.h), restano
       allineate anche sulla GPU (cudaMalloc restituisce indirizzi
       allineati a 256 byte). */
    size_t szA = (size_t)M * lda * sizeof(real_t);
    size_t szX = (size_t)N * k * sizeof(real_t);
    size_t szY = (size_t)M * k * sizeof(real_t);
    CUDA_CHECK(cudaMalloc(&b->dA, szA ? szA : 1));
    CUDA_CHECK(cudaMalloc(&b->dX, szX ? szX : 1));
    CUDA_CHECK(cudaMalloc(&b->dY, szY ? szY : 1));
    if (uses_xt(kernel)) {
        /* Le righe di Xt sono lunghe N come quelle di A: stesso padding.
           cudaMalloc restituisce indirizzi allineati a 256 byte, quindi
           con ldxt multiplo di 128 byte ogni riga di Xt parte allineata. */
        b->ldxt = prep_leading_dim(N, pad ? be_align_bytes() : 0);
        size_t szXt = (size_t)k * b->ldxt * sizeof(real_t);
        CUDA_CHECK(cudaMalloc(&b->dXt, szXt ? szXt : 1));
    }

    /* Upload di A (preprocessamento, escluso dal tempo del nucleo).
       cudaMemcpy2D copia M righe di N elementi con passo lda su entrambi i
       lati: si trasferiscono solo i dati, non le colonne di padding (che
       non vengono mai lette). */
    if (szA)
        CUDA_CHECK(cudaMemcpy2D(b->dA, (size_t)lda * sizeof(real_t),   /* dst, passo dst */
                                A,     (size_t)lda * sizeof(real_t),   /* src, passo src */
                                (size_t)N * sizeof(real_t), M,         /* byte per riga, righe */
                                cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventCreate(&b->ev0));
    CUDA_CHECK(cudaEventCreate(&b->ev1));
    if (kernel == KERNEL_CUBLAS) CUBLAS_CHECK(cublasCreate(&b->blas));
    return b;
}

extern "C" void be_set_x(backend_t *b, const real_t *X)
{
    /* cudaMemcpy è sincrona: ritorna a copia completata. */
    CUDA_CHECK(cudaMemcpy(b->dX, X, (size_t)b->N * b->k * sizeof(real_t),
                          cudaMemcpyHostToDevice));

    /* warp_row_xt e warp_row_smem leggono X trasposta: la si prepara qui,
       come parte della messa a disposizione di X, quindi fuori dal tempo
       del nucleo. Il lancio è asincrono: si attende la fine, così il tempo
       misurato da mxm.c per questa fase (t_h2d) comprende anche la
       trasposizione e il calcolo successivo parte con Xt già pronta. */
    if (uses_xt(b->kernel)) {
        launch_transpose_x(b->N, b->k, b->dX, b->dXt, b->ldxt);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

extern "C" double be_compute(backend_t *b)
{
    if (b->M == 0) return 0.0;                    /* blocco vuoto: niente da calcolare */

    /* Misura con eventi CUDA: ev0 ed ev1 vengono inseriti nella coda della
       GPU prima e dopo il lancio, e la GPU annota l'istante in cui li
       raggiunge. Un timer CPU misurerebbe solo il tempo di accodamento,
       perché il lancio è asincrono. Se un lanciatore esegue più kernel
       (k generico in warp_row), l'intervallo li comprende tutti. */
    CUDA_CHECK(cudaEventRecord(b->ev0));
    switch (b->kernel) {
    case KERNEL_THREAD_ROW:
        launch_thread_row(b->M, b->N, b->k, b->dA, b->lda, b->dX, b->dY); break;
    case KERNEL_WARP_ROW:
        launch_warp_row(b->M, b->N, b->k, b->dA, b->lda, b->dX, b->dY); break;
    case KERNEL_WARP_ROW_XT:      /* X già trasposta in be_set_x */
        launch_warp_row_xt(b->M, b->N, b->k, b->dA, b->lda, b->dXt, b->ldxt, b->dY); break;
    case KERNEL_WARP_ROW_SMEM:    /* idem */
        launch_warp_row_smem(b->M, b->N, b->k, b->dA, b->lda, b->dXt, b->ldxt, b->dY); break;
    case KERNEL_CUBLAS:
        launch_cublas(b->blas, b->M, b->N, b->k, b->dA, b->lda, b->dX, b->dY); break;
    }
    /* Il lancio di un kernel non restituisce un codice d'errore:
       cudaGetLastError recupera eventuali errori di configurazione. */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(b->ev1));
    CUDA_CHECK(cudaEventSynchronize(b->ev1));     /* attende la fine del calcolo */
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, b->ev0, b->ev1));
    return ms * 1e-3;
}

extern "C" void be_get_y(backend_t *b, real_t *Y)
{
    CUDA_CHECK(cudaMemcpy(Y, b->dY, (size_t)b->M * b->k * sizeof(real_t),
                          cudaMemcpyDeviceToHost));
}

extern "C" void be_destroy(backend_t *b)
{
    if (b->kernel == KERNEL_CUBLAS) cublasDestroy(b->blas);
    cudaEventDestroy(b->ev0);
    cudaEventDestroy(b->ev1);
    cudaFree(b->dA); cudaFree(b->dX); cudaFree(b->dY);
    if (b->dXt) cudaFree(b->dXt);
    free(b);
}
