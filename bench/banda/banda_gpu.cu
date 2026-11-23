/*
 * banda_gpu.cu — banda di LETTURA dalla memoria globale della GPU.
 *
 * Serve al roofline (gruppo C delle misure): il nucleo Y = A * X legge A una
 * volta sola e la sua velocità, a k piccolo, è limitata da quanti byte al
 * secondo la GPU riesce a leggere. Il valore teorico (448 GB/s per la Quadro
 * RTX 5000) non è raggiungibile in pratica: qui si misura il tetto REALE,
 * con lo stesso tipo di accesso del nucleo (letture coalescenti da 8 byte).
 *
 * Il kernel legge un vettore di double molto più grande della cache L2
 * (4 MB) e ne somma gli elementi; non scrive nulla di significativo, così i
 * byte trasferiti sono solo quelli letti.
 *
 * Uso:  bin/banda_gpu [-s MiB (default 2048)] [-R ripetizioni (10)] [-W riscaldamento (2)]
 * Stampa una riga CSV (con -H anche l'intestazione).
 */
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include "../../src/cuda/cuda_check.h"

#define THREADS 256

/* Ogni thread somma gli elementi i, i + stride, i + 2 stride, ... (stride =
   numero totale di thread): a ogni passo un warp legge 32 double contigui,
   una lettura coalescente come quelle di A nel nucleo. Quattro accumulatori
   indipendenti tengono più letture in volo per thread. */
__global__ void k_read(const double *__restrict__ p, size_t n, double *out)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (; i + 3 * stride < n; i += 4 * stride) {
        s0 += p[i];
        s1 += p[i + stride];
        s2 += p[i + 2 * stride];
        s3 += p[i + 3 * stride];
    }
    for (; i < n; i += stride) s0 += p[i];
    const double s = s0 + s1 + s2 + s3;
    /* Condizione mai vera (i dati valgono 1): impedisce al compilatore di
       eliminare le letture, senza aggiungere scritture in memoria. */
    if (s < 0) out[0] = s;
}

__global__ void k_fill(double *p, size_t n)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        p[i] = 1.0;
}

int main(int argc, char **argv)
{
    long mib = 2048; int reps = 10, warm = 2, header = 0, opt;
    while ((opt = getopt(argc, argv, "s:R:W:H")) != -1) {
        switch (opt) {
        case 's': mib = atol(optarg); break;
        case 'R': reps = atoi(optarg); break;
        case 'W': warm = atoi(optarg); break;
        case 'H': header = 1; break;
        default:
            fprintf(stderr, "uso: %s [-s MiB] [-R rip] [-W risc] [-H]\n", argv[0]);
            return 1;
        }
    }
    const size_t n = (size_t)mib * 1024 * 1024 / sizeof(double);
    const double bytes = (double)n * sizeof(double);

    double *dp, *dout;
    CUDA_CHECK(cudaMalloc(&dp, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dout, sizeof(double)));
    /* Riempimento con 1.0 direttamente sulla GPU (niente copie dall'host). */
    k_fill<<<1024, THREADS>>>(dp, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Griglia che riempie esattamente la GPU: tanti blocchi quanti ne
       stanno contemporaneamente su tutti gli SM. */
    int dev, nsm, per_sm;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_read, THREADS, 0));
    const int blocks = nsm * per_sm;

    /* Banda teorica dal driver: 2 trasferimenti per ciclo di clock (DDR) x
       larghezza del bus in byte. */
    int clk_khz, bus_bits;
    CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
    const double theo = 2.0 * clk_khz * 1e3 * (bus_bits / 8) / 1e9;

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    double tmin = 1e30, tsum = 0;
    for (int r = 0; r < warm + reps; r++) {
        CUDA_CHECK(cudaEventRecord(e0));
        k_read<<<blocks, THREADS>>>(dp, n, dout);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (r < warm) continue;                   /* riscaldamento: non misurato */
        const double t = ms * 1e-3;
        tsum += t;
        if (t < tmin) tmin = t;
    }
    const double tmean = tsum / reps;

    if (header) printf("dispositivo,processi,mib,reps,t_min,t_mean,gbs_max,gbs_mean,gbs_teorica\n");
    /* gbs_max dal tempo minimo, gbs_mean dal tempo medio (come i GFLOPS del nucleo) */
    printf("gpu,1,%ld,%d,%.6e,%.6e,%.1f,%.1f,%.1f\n", mib, reps, tmin, tmean,
           bytes / tmin / 1e9, bytes / tmean / 1e9, theo);

    cudaFree(dp); cudaFree(dout);
    return 0;
}
