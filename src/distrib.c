#include "distrib.h"

int grid_create(MPI_Comm comm, grid_t *g, int Pr, int Pc){
    int nprocs;
    MPI_Comm_size(comm, &nprocs);

    if (Pr * Pc != nprocs) return -1;    /* la griglia deve usare tutti i processi di comm */

    /* Griglia 2D non periodica: il processo sul bordo non ha come "vicino"
       quello sul bordo opposto (non ci servono comunicazioni tra vicini). */
    int dims[2]    = { Pr, Pc };
    int periods[2] = { 0, 0 };

    /* reorder = 0: MPI non rinumera i processi, quindi il rank nella griglia
       coincide con quello in comm. Il binding dei processi ai core/socket
       deciso da mpirun resta così prevedibile. */
    MPI_Cart_create(comm, 2, dims, periods, 0, &g->grid);

    MPI_Comm_rank(g->grid, &g->rank);
    MPI_Comm_size(g->grid, &g->size);

    /* Coordinate del processo. L'ordinamento è row-major: il rank cresce
       prima lungo le colonne (rank = r * Pc + c). */
    int coords[2];
    MPI_Cart_coords(g->grid, g->rank, 2, coords);
    g->myr = coords[0];
    g->myc = coords[1];
    g->Pr  = Pr;
    g->Pc  = Pc;


//un comunicatore per ogni riga della griglia.
//un comunicatore per ogni colonna della griglia. */
    int keep_cols[2] = { 0, 1 };
    int keep_rows[2] = { 1, 0 };
    MPI_Cart_sub(g->grid, keep_cols, &g->row);
    MPI_Cart_sub(g->grid, keep_rows, &g->col);
    return 0;
}

void grid_free(grid_t *g)
{
    /* I comunicatori sono risorse allocate da MPI: vanno liberati prima di
       MPI_Finalize. */
    MPI_Comm_free(&g->row);
    MPI_Comm_free(&g->col);
    MPI_Comm_free(&g->grid);
}
