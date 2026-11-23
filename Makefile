# Makefile — progetto SCPA: prodotto matrice densa x multivettore
#
# Due versioni dello stesso programma, scelte con BACKEND:
#   BACKEND=cuda  MPI + CUDA: ogni processo calcola il suo blocco sulla GPU
#   BACKEND=cpu   solo MPI:   ogni processo calcola il suo blocco su un core
#
# Uso:
#   make                          CUDA, double  -> bin/mxm_cuda_double
#   make PREC=float               CUDA, float   -> bin/mxm_cuda_float
#   make BACKEND=cpu              CPU,  double  -> bin/mxm_cpu_double
#   make BACKEND=cpu PREC=float   CPU,  float   -> bin/mxm_cpu_float
#   make clean                    rimuove build/ e bin/
#
# Sul server, prima di compilare (se non già in ~/.bashrc):
#   module purge && module load gnu/13.3.0 openmpi/5.0.9 cuda/12.8
#
# Organizzazione dei sorgenti:
#   src/*.c       parte comune: main e MPI, generazione, verifica, statistiche
#   src/cuda/     backend GPU: memoria del device e un file per ogni kernel
#   src/cpu/      backend CPU: un file per ogni kernel
# Si compilano sempre i sorgenti comuni più quelli del backend scelto.
# Aggiungere un file in una di queste cartelle non richiede modifiche qui.

PREC    ?= double
BACKEND ?= cuda
# Quadro RTX 5000: Turing, compute capability 7.5
ARCH    ?= sm_75

MPICC   ?= mpicc
NVCC    ?= nvcc
# Radice dell'installazione CUDA, ricavata dal percorso di nvcc
# (/opt/cuda/12.8/bin/nvcc -> /opt/cuda/12.8).
CUDA_HOME ?= $(patsubst %/bin/nvcc,%,$(shell which $(NVCC) 2>/dev/null))

DEFS :=
ifeq ($(PREC),float)
  DEFS += -DUSE_FLOAT
endif

# -march=native: codice ottimizzato per la CPU su cui si compila (sul server:
# Xeon Cascade Lake, AVX-512). Per questo si compila sul server stesso.
# -ffp-contract=fast: permette di fondere a*b + c in una sola istruzione FMA
# (fused multiply-add). Con -std=c11 gcc la disattiva per default, e il
# picco di calcolo della CPU si dimezza. nvcc la usa già per default sulla
# GPU (-fmad=true): così le due versioni sono confrontabili. L'FMA arrotonda
# una volta sola invece di due: il risultato è leggermente PIÙ accurato.
# CFLAGS_EXTRA / NVFLAGS_EXTRA: opzioni aggiuntive da riga di comando, per
# esperimenti senza modificare il Makefile (eseguire prima make clean,
# altrimenti gli oggetti già compilati non vengono rifatti). Esempi:
#   make BACKEND=cpu CFLAGS_EXTRA=-DACC_BUDGET=32  (blocchi CPU più piccoli)
#   make NVFLAGS_EXTRA="-Xptxas -v"                (registri usati dai kernel)
CFLAGS_EXTRA  ?=
NVFLAGS_EXTRA ?=
CFLAGS  := -O3 -march=native -ffp-contract=fast -Wall -Wextra -std=c11 -Iinclude $(DEFS) $(CFLAGS_EXTRA)
NVFLAGS := -O3 -arch=$(ARCH) -lineinfo -Iinclude $(DEFS) $(NVFLAGS_EXTRA)
LDLIBS  := -lm

# Una cartella di oggetti per ogni combinazione backend/precisione: cambiando
# PREC o BACKEND non si mescolano oggetti compilati con opzioni diverse.
BUILD := build/$(BACKEND)_$(PREC)
BIN   := bin/mxm_$(BACKEND)_$(PREC)

# Parte comune.
OBJS := $(patsubst src/%.c,$(BUILD)/%.o,$(wildcard src/*.c))

# Parte specifica del backend.
ifeq ($(BACKEND),cuda)
  OBJS   += $(patsubst src/%.cu,$(BUILD)/%.o,$(wildcard src/cuda/*.cu))
  # -rpath: il programma trova libcudart/libcublas a runtime anche senza
  # LD_LIBRARY_PATH. -lstdc++: il codice compilato da nvcc è C++.
  LDLIBS += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -lcublas -lstdc++
else ifeq ($(BACKEND),cpu)
  OBJS   += $(patsubst src/%.c,$(BUILD)/%.o,$(wildcard src/cpu/*.c))
else
  $(error BACKEND deve essere cuda oppure cpu)
endif

# Ogni oggetto dipende da tutti gli header: modificarne uno ricompila tutto
# (semplice e sicuro, il progetto è piccolo).
HDRS := $(wildcard include/*.h src/cpu/*.h src/cuda/*.h)

.PHONY: all clean
all: $(BIN)

# Link finale.  $@ = bersaglio, $^ = tutti i prerequisiti.
$(BIN): $(OBJS)
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Sorgenti C (comuni e backend CPU), compilati da mpicc.  $< = sorgente.
$(BUILD)/%.o: src/%.c $(HDRS)
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) -c -o $@ $<

# Sorgenti CUDA, compilati da nvcc (che NON vede MPI).
$(BUILD)/%.o: src/%.cu $(HDRS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -c -o $@ $<

clean:
	rm -rf build bin
