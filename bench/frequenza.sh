#!/bin/bash
# frequenza.sh — verifica dell'ipotesi "frequenza della GPU" (da eseguire SUL SERVER).
#
# Nel gruppo A, a dimensioni medie, la stessa configurazione dava a volte
# ~285 e a volte ~340 GFLOPS, con variabilità quasi nulla dentro la singola
# esecuzione. Ipotesi: la frequenza della GPU cambia da un'esecuzione
# all'altra (si adatta a carico, temperatura e consumo) e le esecuzioni
# brevi non le danno il tempo di salire.
#
# Il test ripete 3 volte la stessa matrice (8192 x 8192, k = 6, 8, 20):
#   - con riscaldamento breve (2 invocazioni, come nelle campagne);
#   - con riscaldamento lungo (200 invocazioni, circa 1-2 secondi di calcolo).
# Come leggerlo:
#   - se con il riscaldamento breve la STESSA configurazione cambia tra un
#     giro e l'altro e con quello lungo resta stabile al valore alto, è la
#     frequenza: nelle campagne serve un riscaldamento più lungo;
#   - se un valore resta basso anche con il riscaldamento lungo, la causa
#     è legata alla configurazione e va cercata lì.
#
# Esito (30/10/2026): confermata. Da allora il programma fa un riscaldamento
# di almeno 1 secondo (opzione -T); qui si usa -T 0 per riprodurre il test.
#
# Uso, dalla cartella del progetto:   bash bench/frequenza.sh

cd "$(dirname "$0")/.." || exit 1
if ! command -v mpicc >/dev/null 2>&1 || ! command -v nvcc >/dev/null 2>&1; then
    [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
    module purge && module load gnu/13.3.0 openmpi/5.0.9 cuda/12.8 >/dev/null 2>&1
fi
make BACKEND=cuda PREC=double >/dev/null || { echo "errore di compilazione"; exit 1; }

mkdir -p results
OUT="results/frequenza_$(date +%Y%m%d_%H%M).csv"
header=0
printf "%-14s %-5s %-4s %s\n" riscaldamento giro k GFLOPS
for W in 2 200; do
    for giro in 1 2 3; do
        for k in 6 8 20; do
            res=$(mpirun -np 1 bin/mxm_cuda_double -m 8192 -n 8192 -k "$k" -R 10 -W "$W" -T 0 -H 2>/dev/null)
            if [ "$header" -eq 0 ]; then echo "warmup,giro,$(head -1 <<< "$res")" > "$OUT"; header=1; fi
            echo "$W,$giro,$(tail -1 <<< "$res")" >> "$OUT"
            printf "%-14s %-5s %-4s %s\n" "$W" "$giro" "$k" "$(tail -1 <<< "$res" | cut -d, -f26)"
        done
    done
done
echo
echo "salvato in $OUT"
