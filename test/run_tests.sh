#!/bin/bash
# run_tests.sh — collaudo funzionale completo del progetto.
#
# Uso (dalla cartella del progetto, sul server):   bash test/run_tests.sh
#
# Fasi:
#   1. compilazione delle 4 varianti (cpu/cuda x double/float)
#   2. versione CPU (solo MPI): griglie diverse, divisioni non esatte,
#      entrambi i kernel, double e float
#   3. versione GPU con un processo: ogni kernel x precisione x k
#   4. versione GPU + MPI: warp_row, warp_row_xt e warp_row_smem su griglie diverse
# Le fasi 2 e 4 provano anche la modalità a turno (-S, nodi emulati),
# le fasi 2 e 3 anche le righe di A senza padding (-p 0).
# Ogni prova stampa una riga; alla fine il riepilogo OK / FALLITE.
# Le prove verificano la correttezza, non i tempi: -T 0 toglie il
# riscaldamento a tempo (vedi main.c), che qui allungherebbe solo il collaudo.
# Dimensioni volutamente NON multiple di 32, del numero di processi o della
# dimensione dei blocchi dei kernel, per esercitare i casi limite degli indici.

cd "$(dirname "$0")/.." || exit 1

# Ambiente del server: se compilatori e MPI non sono nel PATH (moduli non
# caricati nella shell da cui si lancia lo script), li carica qui.
if ! command -v mpicc >/dev/null 2>&1 || ! command -v nvcc >/dev/null 2>&1; then
    [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
    if command -v module >/dev/null 2>&1; then
        echo "== caricamento moduli: gnu/13.3.0 openmpi/5.0.9 cuda/12.8"
        module purge && module load gnu/13.3.0 openmpi/5.0.9 cuda/12.8 >/dev/null 2>&1
    fi
    if ! command -v mpicc >/dev/null 2>&1; then
        echo "errore: mpicc non trovato; caricare i moduli a mano e riprovare"; exit 1
    fi
fi

ok=0; fail=0
# Massimo del rapporto errore / (eps * sqrt(N)) sulle prove riuscite, per
# precisione: è la costante misurata del modello d'errore. Diviso per il 10
# della tolleranza (main.c) dà il margine effettivo della verifica.
max_ratio_double=0; max_ratio_float=0
check() {               # $1 = descrizione, resto = comando mpirun
    local desc="$1"; shift
    local err out ratio
    err=$("$@" 2>&1 >/dev/null)                 # solo stderr: riga di verifica o errori
    out=$(grep "verifica" <<< "$err")
    if [[ "$out" == *"-> OK"* ]]; then
        ok=$((ok + 1)); printf "  OK       %-40s %s\n" "$desc" "${out#verifica: }"
        ratio=$(sed -n 's/.*sqrt(N)) = \([0-9.]*\)).*/\1/p' <<< "$out")
        if [ -n "$ratio" ]; then
            if [[ "$*" == *_float* ]]; then
                max_ratio_float=$(awk -v a="$max_ratio_float" -v b="$ratio" 'BEGIN { print (b > a) ? b : a }')
            else
                max_ratio_double=$(awk -v a="$max_ratio_double" -v b="$ratio" 'BEGIN { print (b > a) ? b : a }')
            fi
        fi
    else
        fail=$((fail + 1))
        # senza riga di verifica: mostra il primo messaggio d'errore utile
        # (ignorando l'avviso innocuo di hwloc sui dispositivi PCI)
        [ -z "$out" ] && out=$(grep -v -i -E "hwloc|enable-32bits|ABI|^-+$|^\s*$" <<< "$err" | head -1)
        printf "  FALLITA  %-40s %s\n" "$desc" "${out:-nessun output}"
    fi
}

echo "== 1. compilazione"
make clean >/dev/null
for target in "BACKEND=cpu PREC=double" "BACKEND=cpu PREC=float" \
              "BACKEND=cuda PREC=double" "BACKEND=cuda PREC=float"; do
    if make $target >/dev/null 2>build_errors.txt; then
        echo "  OK       make $target"
    else
        echo "  FALLITA  make $target  (dettagli in build_errors.txt)"; exit 1
    fi
done
rm -f build_errors.txt

echo "== 2. versione CPU, solo MPI (M=301, N=257)"
for v in 0 1; do
    for cfg in "1 1 1" "2 2 1" "3 1 3" "4 2 2" "6 3 2" "6 2 3" "4 4 1" "4 1 4"; do
        set -- $cfg
        for k in 3 8 13 32; do
            check "kernel=$v P=$1 griglia $2x$3 k=$k" \
                mpirun -np "$1" bin/mxm_cpu_double -m 301 -n 257 -k "$k" -v "$v" \
                       -r "$2" -c "$3" -R 2 -W 1 -T 0 -C
        done
    done
done
for v in 0 1; do
    for k in 3 6 8 13 20 32; do
        check "float kernel=$v P=4 griglia 2x2 k=$k" \
            mpirun -np 4 bin/mxm_cpu_float -m 301 -n 257 -k "$k" -v "$v" -r 2 -c 2 -R 2 -W 1 -T 0 -C
    done
done
for cfg in "2 2 1" "4 2 2" "6 2 3"; do
    set -- $cfg
    check "modalità -S P=$1 griglia $2x$3 k=13" \
        mpirun -np "$1" bin/mxm_cpu_double -m 301 -n 257 -k 13 -r "$2" -c "$3" -R 2 -W 1 -T 0 -S -C
done
for v in 0 1; do
    check "senza padding (-p 0) kernel=$v P=4 2x2 k=13" \
        mpirun -np 4 bin/mxm_cpu_double -m 301 -n 257 -k 13 -v "$v" -r 2 -c 2 -R 2 -W 1 -T 0 -p 0 -C
done

echo "== 3. versione GPU, 1 processo (M=2000, N=1500)"
for prec in double float; do
    for v in 0 1 2 3 -1; do
        for k in 3 6 8 13 20 32; do
            check "$prec kernel=$v k=$k" \
                mpirun -np 1 "bin/mxm_cuda_$prec" -m 2000 -n 1500 -k "$k" -v "$v" -R 2 -W 1 -T 0 -C
        done
    done
done

for v in 0 1 2 3 -1; do
    check "senza padding (-p 0) double kernel=$v k=13" \
        mpirun -np 1 bin/mxm_cuda_double -m 2000 -n 1500 -k 13 -v "$v" -p 0 -R 2 -W 1 -T 0 -C
done

echo "== 4. versione GPU + MPI, warp_row, warp_row_xt, warp_row_smem (M=3001, N=2003, k=8)"
for v in 1 2 3; do
    for cfg in "2 2 1" "2 1 2" "4 2 2" "4 1 4" "4 4 1" "6 3 2"; do
        set -- $cfg
        check "kernel=$v P=$1 griglia $2x$3" \
            mpirun -np "$1" bin/mxm_cuda_double -m 3001 -n 2003 -k 8 -v "$v" -r "$2" -c "$3" -R 2 -W 1 -T 0 -C
    done
done
for cfg in "2 2 1" "4 2 2"; do
    set -- $cfg
    check "modalità -S P=$1 griglia $2x$3" \
        mpirun -np "$1" bin/mxm_cuda_double -m 3001 -n 2003 -k 8 -r "$2" -c "$3" -R 2 -W 1 -T 0 -S -C
done

echo
echo "== riepilogo: $ok OK, $fail FALLITE"
echo "   errore/(eps*sqrt(N)) massimo: double $max_ratio_double, float $max_ratio_float (tolleranza: 10)"
[ "$fail" -eq 0 ]
