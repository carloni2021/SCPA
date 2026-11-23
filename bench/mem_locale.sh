#!/bin/bash
# mem_locale.sh — esegue un comando con la memoria vincolata al socket (nodo
# NUMA) del core a cui il processo è legato.
#
# Uso (dentro mpirun, dopo il binding ai core):
#   mpirun -np 20 --map-by core --bind-to core bash bench/mem_locale.sh bin/mxm_cpu_double ...
#
# Perché serve. Ogni socket del server ha i propri canali di memoria. Un
# processo dovrebbe usare la memoria del proprio socket (vicina e veloce).
# Linux però, se quella memoria è occupata dalla cache dei file (lasciata
# dal lavoro degli altri utenti), non la libera: alloca le pagine sull'altro
# socket (vm.zone_reclaim_mode = 0). Il processo legge allora da lontano e
# la banda complessiva cala. Misurato sul server con 20 processi: 71 GB/s
# con la memoria libera, 52 GB/s con il socket 0 pieno di cache, circa la
# metà nel caso peggiore; con questo vincolo, 70,8 GB/s anche con il socket
# 0 pieno.
#
# Come funziona. "numactl --show" dice su quale nodo è legato il processo
# (riga nodebind); "numactl --membind" lo obbliga ad allocare solo lì, e
# Linux deve liberare la cache locale (recuperabile) invece di spostarsi
# sull'altro socket. Se numactl manca o il processo non è legato a un solo
# nodo, il comando parte senza vincolo e lo segnala.

if command -v numactl >/dev/null 2>&1; then
    # nodi a cui è legato il processo, per esempio "0" oppure "0 1"
    read -r -a nodes <<< "$(numactl --show | awk '/^nodebind/ { $1 = ""; print }')"
    if [ "${#nodes[@]}" -eq 1 ]; then
        exec numactl --membind="${nodes[0]}" "$@"
    fi
fi
echo "mem_locale.sh: memoria NON vincolata (numactl assente o processo non legato a un solo nodo)" >&2
exec "$@"
