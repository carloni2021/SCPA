// per non includere più volte common.h
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

#endif 
