#!/bin/bash
# campagna.sh — esegue una campagna di misure descritta da un file di
# configurazione e raccoglie i risultati in un unico CSV.
#
# Uso (sul server, dalla cartella del progetto):
#     bash bench/campagna.sh bench/config/<gruppo>.conf
# Per campagne lunghe, in modo che non si interrompano se la connessione
# SSH cade:
#     nohup bash bench/campagna.sh bench/config/A_finale.conf > campagna.out 2>&1 &
#     tail -f campagna.out          # per seguirne l'avanzamento
# Solo l'elenco delle esecuzioni, senza compilare né eseguire nulla:
#     ELENCO=1 bash bench/campagna.sh bench/config/<gruppo>.conf
#
# Un file di configurazione per gruppo di misure (vedi bench/config/,
# A_finale.conf è commentato per esteso). Lo script:
#   1. costruisce l'elenco di TUTTE le esecuzioni (prodotto cartesiano dei
#      parametri), così sa quante sono;
#   2. compila le versioni che servono;
#   3. le esegue una alla volta; un'esecuzione che fallisce (per esempio
#      memoria insufficiente) viene annotata nel log e la campagna prosegue;
#   4. scrive in results/<nome>_<data>_<ora>.csv una riga per esecuzione
#      (intestazione una volta sola), più una copia della configurazione
#      (.conf) e un log con eventuali errori (.log).

set -u
cd "$(dirname "$0")/.." || exit 1

CONF="${1:-}"
if [ -z "$CONF" ] || [ ! -f "$CONF" ]; then
    echo "uso: bash bench/campagna.sh bench/config/<nome>.conf"; exit 1
fi
# shellcheck disable=SC1090
source "$CONF"

# Moduli del server, se non già caricati nella shell.
if ! command -v mpicc >/dev/null 2>&1; then
    [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
    module purge && module load gnu/13.3.0 openmpi/5.0.9 cuda/12.8 >/dev/null 2>&1
fi

# --------------------------------------------- elenco di tutte le esecuzioni
# Forme della matrice a partire dalla dimensione di base S (la traccia
# chiede almeno 3 rapporti M/N, compreso il caso quadrato):
#   square: M = S,   N = S        m3n: M = 3S, N = S        n2m: M = S, N = 2S
# Kernel per versione e precisione: KERNELS_<versione>_<precisione>, se
# definita (anche vuota: nessuna esecuzione), altrimenti KERNELS_<versione>.
kernels_for() {
    local kv="KERNELS_$1_$2"
    [ -n "${!kv+x}" ] || kv="KERNELS_$1"
    echo "${!kv}"
}

RUNS=()
for b in $BACKENDS; do
    lv="LAUNCH_$b"
    for prec in $PRECS; do
        klist=$(kernels_for "$b" "$prec")
        for S in $SIZES; do
            for shape in $SHAPES; do
                case "$shape" in
                    square) M=$S;         N=$S ;;
                    m3n)    M=$((3 * S)); N=$S ;;
                    n2m)    M=$S;         N=$((2 * S)) ;;
                    *) echo "forma sconosciuta: $shape"; exit 1 ;;
                esac
                for k in $KS; do
                    for v in $klist; do
                        for launch in ${!lv}; do
                            for mode in $MODES; do
                                # -S con un solo processo non ha senso: è identico al concorrente
                                [ "$mode" = serial ] && [ "${launch%%:*}" = 1 ] && continue
                                for pad in $PADS; do
                                    RUNS+=("$b $prec $shape $M $N $k $v $launch $mode $pad")
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

TOTAL=${#RUNS[@]}

if [ "${ELENCO:-0}" = 1 ]; then          # solo elenco: niente compilazione né esecuzione
    printf '%s\n' "${RUNS[@]}"
    echo "campagna '$NAME': $TOTAL esecuzioni"
    exit 0
fi

# ---------------------------------------------------------------- compilazione
# Solo le combinazioni versione/precisione che hanno almeno un kernel.
for b in $BACKENDS; do
    for prec in $PRECS; do
        [ -n "$(kernels_for "$b" "$prec")" ] || continue
        make BACKEND="$b" PREC="$prec" >/dev/null 2>&1 \
            || { echo "errore di compilazione: make BACKEND=$b PREC=$prec"; exit 1; }
    done
done

mkdir -p results
STAMP="$(date +%Y%m%d_%H%M)"
OUT="results/${NAME}_${STAMP}.csv"
LOG="results/${NAME}_${STAMP}.log"
# Copia della configurazione, per riprodurre la campagna, con in fondo i
# kernel effettivamente usati (la configurazione può leggerli da un altro
# file, bench/config/migliore.conf, che nel frattempo potrebbe cambiare).
{
    cat "$CONF"
    echo
    for b in $BACKENDS; do for prec in $PRECS; do
        echo "# kernel usati per $b $prec: $(kernels_for "$b" "$prec")"
    done; done
} > "results/${NAME}_${STAMP}.conf"
echo "campagna '$NAME': $TOTAL esecuzioni -> $OUT"
echo "# campagna $NAME, $(date), $(hostname)" > "$LOG"

# ---------------------------------------------------------------- esecuzione
header_written=0; fail=0; i=0; t_start=$(date +%s)
for run in "${RUNS[@]}"; do
    read -r b prec shape M N k v launch mode pad <<< "$run"
    IFS=: read -r P Pr Pc <<< "$launch"
    i=$((i + 1))
    mv="MPIRUN_$b"; wv="WRAP_$b"; flags=""
    [ "$mode" = serial ] && flags="-S"

    printf "[%d/%d] %s %s %s M=%d N=%d k=%d v=%s P=%s mode=%s pad=%s\n" \
        "$i" "$TOTAL" "$b" "$prec" "$shape" "$M" "$N" "$k" "$v" "$P" "$mode" "$pad"

    # shellcheck disable=SC2086
    # WRAP_<versione>: comando che lancia il programma (per la CPU
    # bench/mem_locale.sh, che vincola la memoria al socket del processo)
    res=$(mpirun -np "$P" ${!mv:-} ${!wv:-} "bin/mxm_${b}_${prec}" -m "$M" -n "$N" -k "$k" -v "$v" \
              -r "$Pr" -c "$Pc" -p "$pad" -R "$REPS" -W "$WARMUP" -T "${WARMUP_TIME:-1}" \
              $flags -H 2>>"$LOG")
    if [ "$(wc -l <<< "$res")" -ge 2 ]; then
        # con -H il programma stampa intestazione + riga di dati
        if [ "$header_written" -eq 0 ]; then head -1 <<< "$res" > "$OUT"; header_written=1; fi
        tail -1 <<< "$res" >> "$OUT"
    else
        fail=$((fail + 1))
        echo "FALLITA: $run" >> "$LOG"
        echo "   -> FALLITA (vedi $LOG)"
    fi
done

echo
echo "fatto: $((TOTAL - fail)) esecuzioni riuscite, $fail fallite, in $(( ($(date +%s) - t_start) / 60 )) min"
echo "risultati: $OUT"
