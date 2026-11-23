/*
 * banda_cpu.c — banda di LETTURA dalla memoria principale, con P processi MPI.
 *
 * Serve al roofline (gruppo C delle misure) come banda_gpu.cu, ma per la
 * versione CPU. Si lancia come il nucleo, un processo per core legato al
 * proprio core:
 *     mpirun -np 20 --map-by core --bind-to core bin/banda_cpu
 * Ogni processo riempie e poi legge un proprio vettore (i dati finiscono
 * nella memoria del socket del processo, come il blocco di A nel nucleo).
 * La banda complessiva è: byte letti da tutti / tempo del processo più lento
 * (lo stesso criterio del massimo sui processi usato per il nucleo).
 *
 * Il vettore complessivo (default 2048 MiB, come A a 16384 x 16384 in
 * double) è diviso tra i processi ed è molto più grande delle cache L3.
 *
 * Uso:  mpirun ... bin/banda_cpu [-s MiB totali] [-R ripetizioni] [-W riscaldamento] [-H]
 */
#define _POSIX_C_SOURCE 200809L   /* getopt e posix_memalign con -std=c11 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

#define NACC 16                   /* accumulatori indipendenti */

/* Somma con NACC accumulatori: il ciclo interno su u è vettorizzato dal
   compilatore (4 registri da 4 double con AVX2) e le NACC catene di somme
   sono indipendenti, quindi più letture in volo. Senza accumulatori
   multipli ogni somma aspetterebbe la precedente e il ciclo misurerebbe la
   latenza dell'addizione, non la banda. */
static double read_sum(const double *p, size_t n)
{
    double acc[NACC] = { 0 };
    size_t i = 0;
    for (; i + NACC <= n; i += NACC)
        for (int u = 0; u < NACC; u++) acc[u] += p[i + u];
    for (; i < n; i++) acc[0] += p[i];
    double s = 0;
    for (int u = 0; u < NACC; u++) s += acc[u];
    return s;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    long mib = 2048; int reps = 10, warm = 2, header = 0, opt;
    while ((opt = getopt(argc, argv, "s:R:W:H")) != -1) {
        switch (opt) {
        case 's': mib = atol(optarg); break;
        case 'R': reps = atoi(optarg); break;
        case 'W': warm = atoi(optarg); break;
        case 'H': header = 1; break;
        default:
            if (rank == 0) fprintf(stderr, "uso: %s [-s MiB] [-R rip] [-W risc] [-H]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* Divisione bilanciata degli elementi tra i processi */
    const size_t ntot = (size_t)mib * 1024 * 1024 / sizeof(double);
    const size_t n = ntot / P + ((size_t)rank < ntot % P ? 1 : 0);

    double *p = NULL;
    if (posix_memalign((void **)&p, 64, (n ? n : 1) * sizeof(double)) != 0) {
        fprintf(stderr, "allocazione fallita\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    /* Primo accesso (first touch) dal processo stesso: le pagine vengono
       allocate nella memoria del suo socket. */
    for (size_t i = 0; i < n; i++) p[i] = 1.0;

    double tmin = 1e30, tsum = 0, sink = 0;
    for (int r = 0; r < warm + reps; r++) {
        MPI_Barrier(MPI_COMM_WORLD);              /* partenza comune */
        const double t0 = MPI_Wtime();
        sink += read_sum(p, n);
        double t = MPI_Wtime() - t0, tmax;
        /* tempo della ripetizione = quello del processo più lento */
        MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (r < warm || rank != 0) continue;
        tsum += tmax;
        if (tmax < tmin) tmin = tmax;
    }

    if (rank == 0) {
        const double bytes = (double)ntot * sizeof(double);
        const double tmean = tsum / reps;
        if (header) printf("dispositivo,processi,mib,reps,t_min,t_mean,gbs_max,gbs_mean,gbs_teorica\n");
        /* Banda teorica: non ricavabile da qui. Dipende da QUANTI canali di
           memoria hanno un modulo installato: 2400 MT/s x 8 byte = 19,2 GB/s
           per canale, fino a 6 canali per socket (230,4 GB/s con tutti e 12
           occupati). banda.sh stampa i moduli visibili al sistema. */
        printf("cpu,%d,%ld,%d,%.6e,%.6e,%.1f,%.1f,nd\n", P, mib, reps, tmin, tmean,
               bytes / tmin / 1e9, bytes / tmean / 1e9);
    }
    /* sink usato: il compilatore non può eliminare le letture */
    if (sink < 0) printf("%g\n", sink);

    free(p);
    MPI_Finalize();
    return 0;
}
