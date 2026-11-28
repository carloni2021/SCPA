#!/bin/bash
# banda.sh — gruppo C delle misure: banda di lettura reale di GPU e CPU,
# il "tetto" orizzontale del roofline (da eseguire SUL SERVER).
#
# Uso, dalla cartella del progetto:   bash bench/banda.sh
#
# Compila i due programmi di bench/banda/ in bin/ e misura:
#   1. GPU: 1 processo, letture coalescenti da 8 byte (come A nel nucleo);
#   2. CPU: 20 processi MPI, uno per core (come la versione CPU del nucleo).
# Per vedere anche la banda di un solo core:
#   PROCS_CPU="1 20" bash bench/banda.sh
# Risultati in results/banda_<data>_<ora>.csv (le righe con # sono commenti).

cd "$(dirname "$0")/.." || exit 1

if ! command -v mpicc >/dev/null 2>&1 || ! command -v nvcc >/dev/null 2>&1; then
    [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
    module purge && module load gnu/13.3.0 openmpi/5.0.9 cuda/12.8 >/dev/null 2>&1
fi

mkdir -p bin results
# Stesse opzioni del progetto (vedi Makefile)
nvcc -O3 -arch=sm_75 -o bin/banda_gpu bench/banda/banda_gpu.cu \
    || { echo "errore di compilazione: banda_gpu"; exit 1; }
mpicc -O3 -march=native -Wall -Wextra -std=c11 -o bin/banda_cpu bench/banda/banda_cpu.c \
    || { echo "errore di compilazione: banda_cpu"; exit 1; }

OUT="results/banda_$(date +%Y%m%d_%H%M).csv"
{
    echo "# banda di lettura, $(date), $(hostname)"
    # Moduli di memoria visti dal sistema (se il driver EDAC li espone):
    # servono a calcolare la banda teorica della CPU (19,2 GB/s per canale occupato).
    for d in /sys/devices/system/edac/mc/mc*/dimm*; do
        [ -f "$d/size" ] && echo "# modulo $(cat "$d/dimm_label" 2>/dev/null): $(cat "$d/size") MB"
    done
    bin/banda_gpu -H
    for p in ${PROCS_CPU:-20}; do
        # memoria vincolata al socket di ogni processo (vedi bench/mem_locale.sh)
        mpirun -np "$p" --map-by core --bind-to core bash bench/mem_locale.sh bin/banda_cpu
    done
} 2>/dev/null | tee "$OUT"

echo
echo "salvato in $OUT"
