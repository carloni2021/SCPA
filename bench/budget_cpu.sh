#!/bin/bash
# budget_cpu.sh — scelta del budget di registri di cpu_block (da eseguire SUL SERVER).
#
# cpu_block calcola RB righe insieme, con RB x k accumulatori in registri:
# RB = ACC_BUDGET / k, limitato a [1, 8]. Con 32 (il valore sbagliato usato
# all'inizio) RB = 1 a k = 20 e 32: nessun riuso di X proprio dove servirebbe.
# Il default attuale (128, scelto con questo script) è in kernel_cpu_block.c;
# lo script resta per ripetere il confronto e per la run "RB = 1" (budget 32).
#
#   budget  k=3  k=6  k=8  k=20  k=32     (RB in double)
#     32     8    5    4    1     1
#     64     8    8    8    3     2
#    128     8    8    8    6     4
#
# Per ogni budget: ricompila, controlla la correttezza su una matrice
# piccola, poi misura a 16384 x 16384 con 1 processo e con 20 processi
# (griglia 4 x 5, uno per core), k = 3, 6, 8, 20, 32. Durata: pochi minuti.
# Alla fine cancella la compilazione di prova: la successiva "make" torna
# al budget di default.
#
# Uso, dalla cartella del progetto:   bash bench/budget_cpu.sh

cd "$(dirname "$0")/.." || exit 1
if ! command -v mpicc >/dev/null 2>&1; then
    [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
    module purge && module load gnu/13.3.0 openmpi/5.0.9 cuda/12.8 >/dev/null 2>&1
fi

mkdir -p results
OUT="results/budget_cpu_$(date +%Y%m%d_%H%M).csv"
BIND="--map-by core --bind-to core"
WRAP="bash bench/mem_locale.sh"     # memoria vincolata al socket (vedi mem_locale.sh)
header=0
printf "%-7s %-9s %-4s %s\n" budget processi k GFLOPS
for B in 32 64 128; do
    make clean >/dev/null
    make BACKEND=cpu PREC=double CFLAGS_EXTRA=-DACC_BUDGET=$B >/dev/null 2>&1 \
        || { echo "errore di compilazione con ACC_BUDGET=$B"; exit 1; }

    # correttezza: ogni budget cambia RB, cioè il codice che viene generato
    for k in 3 8 13 20 32; do
        v=$(mpirun -np 4 bin/mxm_cpu_double -m 301 -n 257 -k $k -r 2 -c 2 -R 1 -W 0 -T 0 -C 2>&1 >/dev/null | grep verifica)
        [[ "$v" == *"-> OK"* ]] || { echo "budget $B, k=$k: verifica FALLITA ($v)"; exit 1; }
    done

    for launch in "1 1 1" "20 4 5"; do
        set -- $launch
        for k in 3 6 8 20 32; do
            res=$(mpirun -np $1 $BIND $WRAP bin/mxm_cpu_double -m 16384 -n 16384 -k $k -r $2 -c $3 \
                         -R 10 -W 2 -H 2>/dev/null)
            if [ "$header" -eq 0 ]; then echo "budget,$(head -1 <<< "$res")" > "$OUT"; header=1; fi
            echo "$B,$(tail -1 <<< "$res")" >> "$OUT"
            printf "%-7s %-9s %-4s %s\n" "$B" "$1" "$k" "$(tail -1 <<< "$res" | cut -d, -f26)"
        done
    done
done
make clean >/dev/null
echo
echo "salvato in $OUT"
