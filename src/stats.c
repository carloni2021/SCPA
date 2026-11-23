/*
 * stats.c — statistiche descrittive su una serie di tempi (vedi stats.h).
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "stats.h"

/* Confronto per qsort: ordine crescente di double. */
static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

stats_t stats_compute(const double *v, int n)
{
    stats_t s = { 0 };
    if (n <= 0) return s;

    /* Media, minimo e massimo in una passata. */
    double sum = 0.0;
    s.min = s.max = v[0];
    for (int i = 0; i < n; i++) {
        sum += v[i];
        if (v[i] < s.min) s.min = v[i];
        if (v[i] > s.max) s.max = v[i];
    }
    s.mean = sum / n;

    /* Deviazione standard campionaria: divisore n - 1, perché le R
       ripetizioni sono un campione delle possibili esecuzioni e la media
       usata è stimata dagli stessi dati (correzione di Bessel). */
    double sq = 0.0;
    for (int i = 0; i < n; i++) sq += (v[i] - s.mean) * (v[i] - s.mean);
    s.std = (n > 1) ? sqrt(sq / (n - 1)) : 0.0;
    s.cv  = (s.mean > 0.0) ? 100.0 * s.std / s.mean : 0.0;

    /* Mediana: si ordina una copia, per non alterare l'ordine originale. */
    double *tmp = malloc((size_t)n * sizeof *tmp);
    memcpy(tmp, v, (size_t)n * sizeof *tmp);
    qsort(tmp, (size_t)n, sizeof *tmp, cmp_double);
    s.median = (n % 2) ? tmp[n / 2] : 0.5 * (tmp[n / 2 - 1] + tmp[n / 2]);
    free(tmp);

    return s;
}
