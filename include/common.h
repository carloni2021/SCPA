/*
 * common.h — tipo numerico e utility condivise da C e CUDA.
 *
 * La precisione si sceglie a compile-time:  make PREC=double | PREC=float
 * Questo header NON include mpi.h, così può essere incluso anche dai .cu
 * (nvcc non deve vedere MPI: la parte MPI resta confinata nei .c).
 */
#ifndef COMMON_H
#define COMMON_H

#include <stddef.h>

#ifdef USE_FLOAT
typedef float real_t;
#define REAL_EPS   1.1920929e-07
#define PREC_NAME  "float"
#else
typedef double real_t;
#define REAL_EPS   2.220446049250313e-16
#define PREC_NAME  "double"
#endif

/* Indice lineare row-major con leading dimension esplicita. */
#define IDX(i, j, ld) ((size_t)(i) * (size_t)(ld) + (size_t)(j))

#endif /* COMMON_H */
