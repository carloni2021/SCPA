/* test_grid.c — stampa coordinate, rank nei comunicatori e blocco di A
   di ogni processo.   uso: mpirun -np P ./test_grid Pr Pc M N          */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "distrib.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int Pr = atoi(argv[1]), Pc = atoi(argv[2]), M = atoi(argv[3]), N = atoi(argv[4]);
    grid_t g;
    if (grid_create(MPI_COMM_WORLD, &g, Pr, Pc) != 0) { MPI_Finalize(); return 1; }
    int rr, rc; MPI_Comm_rank(g.row, &rr); MPI_Comm_rank(g.col, &rc);

    /* stampa a turno, così le righe non si mescolano */
    for (int p = 0; p < g.size; p++) {
        if (p == g.rank)
            printf("rank %d -> (r=%d, c=%d) | rank in row=%d, in col=%d | "
                   "righe di A %d..%d, colonne di A %d..%d\n",
                   g.rank, g.myr, g.myc, rr, rc,
                   blk_off(M, Pr, g.myr), blk_off(M, Pr, g.myr) + blk_size(M, Pr, g.myr) - 1,
                   blk_off(N, Pc, g.myc), blk_off(N, Pc, g.myc) + blk_size(N, Pc, g.myc) - 1);
        fflush(stdout);
        MPI_Barrier(g.grid);
    }
    grid_free(&g);
    MPI_Finalize();
    return 0;
}
