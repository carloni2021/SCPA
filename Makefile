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

PREC    ?= double
BACKEND ?= cuda
ARCH    ?= sm_75
# scelta compilatori
MPICC   ?= mpicc
NVCC    ?= nvcc
# Radice dell'installazione CUDA, ricavata dal percorso di nvcc
# (/opt/cuda/12.8/bin/nvcc -> /opt/cuda/12.8).
# serve per trovare librerie cuda
CUDA_HOME ?= $(patsubst %/bin/nvcc,%,$(shell which $(NVCC) 2>/dev/null))

#Scelta precisione 
DEFS :=
ifeq ($(PREC),float)
  DEFS += -DUSE_FLOAT
endif

#Opzioni aggiuntive 
# -O3: ottimizzazione 
# -march=native: ottimizza per l'architettura della CPU locale all'esecuzione
# -ffp-contract=fast: permette di fondere a*b + c in una sola istruzione FMA
# -Wall -Wextra: attiva tutti i warning del compilatore
# -std=c11: standard C11
# -Iinclude: aggiunge la cartella include al percorso di ricerca degli header

# CFLAGS_EXTRA e NVFLAGS_EXTRA possono essere definite dall'utente per aggiungere altre opzioni di compilazione.
CFLAGS_EXTRA  ?=
NVFLAGS_EXTRA ?=
CFLAGS  := -O3 -march=native -ffp-contract=fast -Wall -Wextra -std=c11 -Iinclude $(DEFS) $(CFLAGS_EXTRA)
NVFLAGS := -O3 -arch=$(ARCH) -lineinfo -Iinclude $(DEFS) $(NVFLAGS_EXTRA)
LDLIBS  := -lm

# Una cartella di oggetti per ogni combinazione backend/precisione: cambiando
# PREC o BACKEND non si mescolano oggetti compilati con opzioni diverse.
BUILD := build/$(BACKEND)_$(PREC)
BIN   := bin/mxm_$(BACKEND)_$(PREC)

# Parte comune alke build
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
HDRS := $(wildcard include/*.h src/cpu/*.h src/cuda/*.h)

.PHONY: all clean
all: $(BIN)

# Link finale.  $@ = bersaglio, $^ = tutti i prerequisiti.
$(BIN): $(OBJS)
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Compilazione Sorgenti C (comuni e backend CPU), compilati da mpicc.  $< = sorgente.
$(BUILD)/%.o: src/%.c $(HDRS)
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) -c -o $@ $<

# Compilazione Sorgenti CUDA, compilati da nvcc (che NON vede MPI).
$(BUILD)/%.o: src/%.cu $(HDRS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -c -o $@ $<

clean:
	rm -rf build bin
