/*
 * main.c — programma di misura del prodotto distribuito Y = A * X.
 *
 * Il prodotto è in mxm.h / mxm.c (griglia, bcast, prodotto locale, reduce).
 * Qui c'è solo ciò che serve a misurarlo: opzioni, generazione di A e X,
 * riscaldamento, ripetizioni, statistiche, verifica e riga CSV.
 *
 * Uso:
 *   mpirun -np P bin/mxm_<backend>_<prec> -m M -n N -k K -r Pr -c Pc
 *          [-R ripetizioni] [-W warmup] [-T secondi] [-v kernel] [-p 0|1]
 *          [-S] [-C] [-H]
 *   Senza -v si usa il kernel migliore del backend (be_default_kernel).
 */
/* getopt/optarg sono funzioni POSIX, non ISO C: con -std=c11 la libreria C
   le dichiara solo se lo standard POSIX viene richiesto esplicitamente.
   Va definito PRIMA di qualunque #include. */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>      /* getopt */
#include <math.h>
#include <mpi.h>

#include "common.h"
#include "distrib.h"
#include "matgen.h"
#include "serial.h"
#include "prep.h"
#include "backend.h"
#include "mxm.h"
#include "stats.h"

typedef struct {
    int M, N, k;        /* dimensioni del problema globale            */
    int Pr, Pc;         /* griglia di processi                        */
    int reps, warmup;   /* invocazioni misurate e di riscaldamento (minimo) */
    double warm_time;   /* durata minima del riscaldamento, in secondi */
    int kernel;         /* versione del kernel (vedi be_kernel_list)   */
    int serial;         /* 1 = fase locale a turno (nodi emulati, -S)  */
    int pad;            /* 1 = righe di A allineate (padding, prep.h)  */
    int check;          /* 1 = verifica contro il riferimento seriale  */
    int header;         /* 1 = stampa l'intestazione CSV               */
} opts_t;

static void usage(const char *p)
{
    fprintf(stderr,
        "uso: %s -m M -n N -k K [-r Pr] [-c Pc] [-R reps] [-W warmup] [-T secondi]\n"
        "        [-v kernel] [-p 0|1 (righe di A allineate: GPU 128 B, CPU 64 B; default 1)]\n"
        "        [-S (calcolo a turno: nodi emulati)]\n"
        "        [-T durata minima del riscaldamento in secondi (default 1; 0 = solo -W)]\n"
        "        [-C (verifica)] [-H (stampa header CSV)]\n"
        "  backend %s, kernel: %s\n", p, be_name(), be_kernel_list());
}

/* Lettura delle opzioni con getopt: ogni lettera seguita da ':' richiede un
   valore (optarg). Dimensioni scelte a runtime, come chiede la traccia. */
static int parse(int argc, char **argv, opts_t *o)
{
    *o = (opts_t){ .Pr = 1, .Pc = 1, .reps = 20, .warmup = 3, .warm_time = 1.0,
                   .kernel = be_default_kernel(), .pad = 1 };
    int c;
    // while per leggere le opzioni della riga di comando
    while ((c = getopt(argc, argv, "m:n:k:r:c:R:W:T:v:p:SCH")) != -1) {
        switch (c) {
        case 'm': o->M = atoi(optarg); break;
        case 'n': o->N = atoi(optarg); break;
        case 'k': o->k = atoi(optarg); break;
        case 'r': o->Pr = atoi(optarg); break;
        case 'c': o->Pc = atoi(optarg); break;
        case 'R': o->reps = atoi(optarg); break;
        case 'W': o->warmup = atoi(optarg); break;
        case 'T': o->warm_time = atof(optarg); break;
        case 'v': o->kernel = atoi(optarg); break;
        case 'p': o->pad = atoi(optarg) != 0; break;
        case 'S': o->serial = 1; break;
        case 'C': o->check = 1; break;
        case 'H': o->header = 1; break;
        default:  return -1;
        }
    }
    // Controllo dei valori letti: devono essere positivi Pr*Pc == numero di processi controllato a riga numero 149 circa.
    if (o->M <= 0 || o->N <= 0 || o->k <= 0 || o->Pr <= 0 || o->Pc <= 0 ||
        o->reps <= 0 || o->warmup < 0 || o->warm_time < 0)
        return -1;
    return 0;
}

/* Verifica distribuita. Ogni processo della colonna 0 possiede le righe
 * finali Y_r e le ricalcola con il riferimento seriale (serial.c),
 * rigenerando A una riga alla volta.
 * NB: il generatore è funzione degli indici globali. 
 * In memoria solo una riga di A per volta, più X intera.
 *  
 * errore calcolato come ||Y - Yref||_F / ||Yref||_F, Yref calcolato in seriale.
 */

static double verify(const grid_t *g, const opts_t *o, const mxm_block_t *b, const real_t *Y) {
    //in input verify vuole: la griglia di processi, le opzioni, il blocco locale di A e la matrice Y calcolata
    double loc[2] = { 0.0, 0.0 };   /* [0] = sum (y - yref)^2,  [1] = sum yref^2 */

    if (g->myc == 0) {
        real_t *Xfull = malloc((size_t)o->N * o->k * sizeof(real_t));
        real_t *arow  = malloc((size_t)o->N * sizeof(real_t));   // una riga di A
        double *yref  = malloc((size_t)o->k * sizeof(double));   // una riga di Y 
        fill_block(Xfull, o->k, 0, o->N, 0, o->k, SEED_X);       // X intera 

        for (int r = 0; r < b->Mloc; r++) {
            // riga globale (m_off + r)-esima di A, 
            //con lo stesso arrotondamento a real_t usato per riempire A nel calcolo vero
            fill_block(arow, o->N, b->m_off + r, 1, 0, o->N, SEED_A);
            serial_mxm(1, o->N, o->k, arow, o->N, Xfull, o->k, yref, o->k);
            for (int c = 0; c < o->k; c++) {
                double d = (double)Y[IDX(r, c, o->k)] - yref[c];
                loc[0] += d * d;
                loc[1] += yref[c] * yref[c];
            }
        }
        free(yref);
        free(arow);
        free(Xfull);
    }

    /* Somme parziali di tutti i processi (quelli fuori dalla colonna 0
       contribuiscono con zero). */
    double tot[2];
    MPI_Allreduce(loc, tot, 2, MPI_DOUBLE, MPI_SUM, g->grid);
    return tot[1] > 0 ? sqrt(tot[0] / tot[1]) : sqrt(tot[0]);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    opts_t o;
    int wrank;
    //controllo che le opzioni siano corrette, altrimenti stampo l'uso e termino
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    if (parse(argc, argv, &o) != 0) {
        if (wrank == 0) usage(argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    grid_t g;
    // Creazione della griglia di processi. Se Pr*Pc != numero di processi, errore e termina.
    if (grid_create(MPI_COMM_WORLD, &g, o.Pr, o.Pc) != 0) { // in distrib.h
        if (wrank == 0)
            fprintf(stderr, "errore: Pr*Pc = %d diverso dal numero di processi\n",
                    o.Pr * o.Pc);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    //init del backend 
    mxm_init_device(MPI_COMM_WORLD);

    //inizia il tempo di preparazione
    double t_pre0 = MPI_Wtime();

    const int k = o.k;
    const mxm_block_t b = mxm_block(&g, o.M, o.N, o.pad);   // blocco A locale 

    // Allocazione di A, X e Y. A è sempre allocata, X solo sulla riga 0, Y solo sulla colonna 0.
    real_t *A = prep_alloc_matrix(b.Mloc, b.lda, b.align);        /* A_rc                 */
    real_t *X = (g.myr == 0)                                      /* X_c, solo sulla      */
              ? malloc((size_t)b.Nloc * k * sizeof(real_t) + 1)   /* riga 0               */
              : NULL;
    real_t *Y = (g.myc == 0)                                      /* Y_r, solo sulla      */
              ? malloc((size_t)b.Mloc * k * sizeof(real_t) + 1)   /* colonna 0            */
              : NULL;
    //piu 1 necessario non togliere!!!

    fill_block(A, b.lda, b.m_off, b.Mloc, b.n_off, b.Nloc, SEED_A);
    if (g.myr == 0)
        fill_block(X, k, b.n_off, b.Nloc, 0, k, SEED_X);

    mxm_t *mxm = mxm_create(&g, &b, k, A, o.kernel);        /* GPU: upload di A */
    //scelta del seriale
    mxm_set_serial(mxm, o.serial);
    //fine del tempo di preparazione
    MPI_Barrier(g.grid);
    double t_pre = MPI_Wtime() - t_pre0;

    // Tabella dei tempi: per ogni fase e ogni ripetizione, il massimo tra i processi.
    double *ph = malloc((size_t)MXM_NTIMES * o.reps * sizeof *ph);
    int warming = o.warmup > 0 || o.warm_time > 0;   /* -W 0 -T 0: nessun riscaldamento */
    int n_warm = 0, r = 0;                   /* r = ripetizioni misurate finora */
    const double tw0 = MPI_Wtime();

    while (r < o.reps) {
        double t[MXM_NTIMES];
        MPI_Barrier(g.grid);     /* tutti partono insieme: tempi confrontabili */
        mxm_apply(mxm, X, Y, t);

        if (warming) {
            /* fine del riscaldamento? decide il processo 0 */
            n_warm++;
            int more = n_warm < o.warmup || MPI_Wtime() - tw0 < o.warm_time;
            MPI_Bcast(&more, 1, MPI_INT, 0, g.grid);
            warming = more;
            continue;
        }
        for (int p = 0; p < MXM_NTIMES; p++)
            ph[IDX(p, r, o.reps)] = t[p];
        r++;
    }

    //reduce per ottenere i tempi massimi per ogni fase e ogni ripetizione
    MPI_Reduce(g.rank == 0 ? MPI_IN_PLACE : ph, g.rank == 0 ? ph : NULL,MXM_NTIMES * o.reps, MPI_DOUBLE, MPI_MAX, 0, g.grid);

    //verifica 
    double relerr = -1.0;
    if (o.check) relerr = verify(&g, &o, &b, Y);

    // output solo processo 0
    if (g.rank == 0) {
        // Statistiche complete per il nucleo e per il solo calcolo; per le singole fasi basta la media.
        stats_t core = stats_compute(&ph[IDX(MXM_T_CORE,    0, o.reps)], o.reps);
        stats_t comp = stats_compute(&ph[IDX(MXM_T_COMPUTE, 0, o.reps)], o.reps);
        double  mean[MXM_NTIMES];
        for (int p = 0; p < MXM_NTIMES; p++)
            mean[p] = stats_compute(&ph[IDX(p, 0, o.reps)], o.reps).mean;

        /* FLOPS = 2 M N k / T, con T tempo MEDIO per invocazione, in double senno i bit non bastano. */
        double flop = 2.0 * o.M * (double)o.N * k;
        // stampa CSV
        if (o.header)
            printf("backend,kernel,prec,mode,pad,M,N,k,Pr,Pc,reps,"
                   "t_core_mean,t_core_median,t_core_min,t_core_std,t_core_cv,"
                   "t_compute_mean,t_compute_min,t_compute_cv,"
                   "t_bcast,t_reduce,t_h2d,t_d2h,t_local_wall,t_pre,"
                   "gflops_core,gflops_compute,relerr,t_compute_dev,n_warmup\n");
        printf("%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,"
               "%.6e,%.6e,%.6e,%.6e,%.2f,"
               "%.6e,%.6e,%.2f,"
               "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,"
               "%.3f,%.3f,%.3e,%.6e,%d\n",
               be_name(), be_kernel_name(o.kernel), PREC_NAME,
               o.serial ? "serial" : "concurrent", o.pad,
               o.M, o.N, k, o.Pr, o.Pc, o.reps,
               core.mean, core.median, core.min, core.std, core.cv,
               comp.mean, comp.min, comp.cv,
               mean[MXM_T_BCAST], mean[MXM_T_REDUCE], mean[MXM_T_H2D], mean[MXM_T_D2H],
               mean[MXM_T_LOCAL], t_pre,
               flop / core.mean * 1e-9, flop / comp.mean * 1e-9, relerr,
               mean[MXM_T_COMPUTE_DEV], n_warm);

        if (o.check) {
            const double scale = REAL_EPS * sqrt((double)o.N);   /* eps * sqrt(N) */
            const double tol   = 10.0 * scale;
            fprintf(stderr, "verifica: errore relativo %.3e (tolleranza %.1e, "
                            "errore/(eps*sqrt(N)) = %.3f) -> %s\n",
                    relerr, tol, relerr / scale, relerr <= tol ? "OK" : "FALLITA");
        }
    }

    mxm_destroy(mxm);
    free(ph);
    free(A); free(X); free(Y);
    grid_free(&g);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
