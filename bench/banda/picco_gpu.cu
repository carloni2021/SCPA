/*
 * picco_gpu.cu — picco di calcolo REALE della GPU, in double e in float.
 *
 * Serve al roofline come banda_gpu.cu: banda_gpu misura il tetto della
 * memoria, questo programma il tetto del calcolo. Il picco teorico si
 * ricava dalla frequenza dichiarata (Boost, 1815 MHz per la Quadro RTX
 * 5000), ma la GPU può salire oltre quel valore se temperatura e consumo
 * lo permettono: il nucleo misurato in double (357 GFLOPS) supera infatti
 * il picco calcolato così (348). Qui si misura il picco effettivo.
 *
 * Il kernel non legge né scrive memoria durante il calcolo: ogni thread
 * esegue solo FMA (a = a * b + c) sui propri registri. CHAINS catene
 * indipendenti per thread tengono occupate le unità di calcolo: con una
 * sola catena ogni FMA aspetterebbe il risultato della precedente e si
 * misurerebbe la latenza dell'istruzione, non la velocità di esecuzione.
 * b e c vengono lette dalla memoria globale all'inizio del kernel: il
 * compilatore non può calcolare il risultato in anticipo e, soprattutto,
 * le tiene in registri. Passate come parametri, finivano nella memoria delle
 * costanti e ogni DFMA leggeva un operando da lì: in double il benchmark si
 * fermava a 315 GFLOPS, sotto i 357 misurati dal nucleo, che usa solo
 * registri. Il risultato viene scritto solo se
 * vale un valore impossibile: il calcolo non può essere eliminato, ma non
 * c'è traffico verso la memoria.
 *
 * Operazioni contate: ogni FMA vale 2 operazioni (una moltiplicazione e
 * una somma), come nella formula dei FLOPS del nucleo.
 *
 * Riscaldamento: come nel nucleo, almeno W lanci E almeno T secondi, così
 * la GPU raggiunge la sua frequenza di regime prima delle misure.
 *
 * Uso:  bin/picco_gpu [-R ripetizioni (10)] [-W lanci di riscaldamento (2)]
 *                     [-T secondi di riscaldamento (2)] [-H]
 * Stampa una riga CSV per precisione (con -H anche l'intestazione).
 */
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include "../../src/cuda/cuda_check.h"

#define THREADS 256
/* Modificabili in compilazione per esperimenti, es. -DCHAINS=16 */
#ifndef CHAINS
#define CHAINS  8       /* catene di FMA indipendenti per thread */
#endif
#ifndef ITERS
#define ITERS   8192    /* giri del ciclo interno per lancio     */
#endif

template <typename T>
__global__ void k_fma(T *out, const T *in)
{
    const T b = in[0], c = in[1];          /* in registri, vedi sopra */
    T a[CHAINS];
#pragma unroll
    for (int i = 0; i < CHAINS; i++) a[i] = (T)(threadIdx.x + i);

    for (int it = 0; it < ITERS; it++) {
#pragma unroll
        for (int i = 0; i < CHAINS; i++) a[i] = fma(a[i], b, c);   /* 2 operazioni */
    }

    T s = 0;
#pragma unroll
    for (int i = 0; i < CHAINS; i++) s += a[i];
    if (s == (T)-12345.0) out[0] = s;      /* mai vero in pratica: vedi sopra */
}

/* Misura il picco per il tipo T e stampa una riga CSV. */
template <typename T>
static void misura(const char *nome, int blocks, int reps, int warm, double warm_s)
{
    T *dout, *din;
    CUDA_CHECK(cudaMalloc(&dout, sizeof(T)));
    CUDA_CHECK(cudaMalloc(&din, 2 * sizeof(T)));
    /* b < 1 e c piccolo: i valori restano limitati, niente overflow. */
    const T bc[2] = { (T)0.999999, (T)1e-6 };
    CUDA_CHECK(cudaMemcpy(din, bc, sizeof bc, cudaMemcpyHostToDevice));
    const double flop = 2.0 * (double)blocks * THREADS * ITERS * CHAINS;

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    float ms;

    /* riscaldamento: almeno warm lanci e almeno warm_s secondi */
    double speso = 0.0;
    for (int w = 0; w < warm || speso < warm_s; w++) {
        CUDA_CHECK(cudaEventRecord(e0));
        k_fma<T><<<blocks, THREADS>>>(dout, din);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        speso += ms * 1e-3;
    }

    double tmin = 1e30, tsum = 0.0;
    for (int r = 0; r < reps; r++) {
        CUDA_CHECK(cudaEventRecord(e0));
        k_fma<T><<<blocks, THREADS>>>(dout, din);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        const double t = ms * 1e-3;
        tsum += t;
        if (t < tmin) tmin = t;
    }
    /* gflops_max dal tempo minimo, gflops_mean dal tempo medio */
    printf("gpu,%s,%d,%.6e,%.6e,%.1f,%.1f\n", nome, reps, tmin, tsum / reps,
           flop / tmin / 1e9, flop / (tsum / reps) / 1e9);

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(dout));
    CUDA_CHECK(cudaFree(din));
}

int main(int argc, char **argv)
{
    int reps = 10, warm = 2, header = 0;
    double warm_s = 2.0;
    int opt;
    while ((opt = getopt(argc, argv, "R:W:T:H")) != -1) {
        switch (opt) {
        case 'R': reps = atoi(optarg); break;
        case 'W': warm = atoi(optarg); break;
        case 'T': warm_s = atof(optarg); break;
        case 'H': header = 1; break;
        default:
            fprintf(stderr, "uso: %s [-R rip] [-W lanci] [-T secondi] [-H]\n", argv[0]);
            return 1;
        }
    }

    /* Molti blocchi per SM: la GPU resta piena per tutto il lancio. */
    int dev, nsm;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    const int blocks = nsm * 32;

    if (header) printf("dispositivo,prec,reps,t_min,t_mean,gflops_max,gflops_mean\n");
    misura<double>("double", blocks, reps, warm, warm_s);
    misura<float> ("float",  blocks, reps, warm, warm_s);
    return 0;
}
