/*
 * stats.h — statistiche descrittive su una serie di tempi.
 *
 * Usate per riassumere le R invocazioni misurate del nucleo. Oltre alla
 * media (richiesta dalla traccia: FLOPS = 2MNk / T medio) servono a
 * valutare l'affidabilità della misura:
 *   - median : robusta ai valori anomali (un'invocazione disturbata da un
 *              altro utente sul server sposta la media, non la mediana);
 *   - min    : il caso migliore, vicino al costo "intrinseco" del codice;
 *   - std/cv : dispersione; il coefficiente di variazione cv = std / mean
 *              in percentuale dice se le ripetizioni sono coerenti
 *              (pochi % = misura stabile, decine di % = misura rumorosa).
 * Codice C puro, nessuna dipendenza da MPI o CUDA.
 */
#ifndef STATS_H
#define STATS_H

typedef struct {
    double mean;     /* media aritmetica                                */
    double median;   /* valore centrale (media dei due centrali se n pari) */
    double min;      /* minimo                                           */
    double max;      /* massimo                                          */
    double std;      /* deviazione standard campionaria (divisore n - 1) */
    double cv;       /* coefficiente di variazione, in % (100 std / mean) */
} stats_t;

/* Calcola le statistiche dei n valori di v (n >= 1). v non viene modificato. */
stats_t stats_compute(const double *v, int n);

#endif /* STATS_H */
