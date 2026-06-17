# Makefile for running the Lit2Vec full pipeline on the litsync corpus.
#
# Usage:
#   make -f pipeline.mk run
#   make -f pipeline.mk run-bg          # run under nohup with logging
#   make -f pipeline.mk OUTPUT=/path/to/out run
#
# The default input is the litsync corpus directory:
#   /mnt/parallax/SynologyDrive/DEV/LitSync/data/corpus

.PHONY: help run run-bg check clean status

# ---------------------------------------------------------------------------
# Paths (override on the command line if needed)
# ---------------------------------------------------------------------------
INPUT      ?= /mnt/parallax/SynologyDrive/DEV/LitSync/data/corpus
OUTPUT     ?= /mnt/parallax/SynologyDrive/DEV/LitSync/data/lit2vec_indices
LOG_DIR    ?= $(OUTPUT)/logs
LOG_FILE   ?= $(LOG_DIR)/pipeline_$(shell date +%Y%m%d_%H%M%S).log
ZIP_FILE   ?= $(OUTPUT)/data.zip

# ---------------------------------------------------------------------------
# Pipeline settings (override on the command line if needed)
# ---------------------------------------------------------------------------
MODEL      ?= dunzhang/stella_en_400M_v5
VEC_DIM    ?= 1024
BATCH_SIZE ?= 512
ANN_TREES  ?= 100
N_RESULTS  ?= 5
BM25_THR   ?= 12.85
TEXT_FIELD ?= auto
LITSYNC  ?= true

# ---------------------------------------------------------------------------
# Command assembly
# ---------------------------------------------------------------------------
LIT2VEC_RUN := python3 -m lit2vec run \
	-i $(INPUT) \
	-o $(OUTPUT) \
	$(if $(filter true,$(LITSYNC)),--litsync) \
	--model-name $(MODEL) \
	--vec-dim $(VEC_DIM) \
	--batch-size $(BATCH_SIZE) \
	--annoy-trees $(ANN_TREES) \
	--n-results $(N_RESULTS) \
	--bm25-threshold $(BM25_THR) \
	--text-field $(TEXT_FIELD) \
	--zip

help:
	@echo "Lit2Vec pipeline Makefile"
	@echo ""
	@echo "Variables (override with make -f pipeline.mk VAR=value):"
	@echo "  INPUT      = $(INPUT)"
	@echo "  OUTPUT     = $(OUTPUT)"
	@echo "  MODEL      = $(MODEL)"
	@echo "  VEC_DIM    = $(VEC_DIM)"
	@echo "  BATCH_SIZE = $(BATCH_SIZE)"
	@echo "  ANN_TREES  = $(ANN_TREES)"
	@echo "  LITSYNC    = $(LITSYNC)"
	@echo ""
	@echo "Targets:"
	@echo "  check    - Verify input directory and lit2vec installation"
	@echo "  run      - Run the full pipeline and produce $(ZIP_FILE)"
	@echo "  run-bg   - Run the pipeline in the background with nohup logging"
	@echo "  status   - Show whether the output zip exists and its size"
	@echo "  clean    - Remove the output directory (use with care)"

check:
	@echo "Checking input directory..."
	@test -d $(INPUT) || (echo "ERROR: Input directory does not exist: $(INPUT)" && exit 1)
	@echo "  OK: $(INPUT)"
	@echo "Checking lit2vec installation..."
	@python3 -m lit2vec --help >/dev/null 2>&1 || (echo "ERROR: lit2vec is not installed or not on PATH" && exit 1)
	@echo "  OK: lit2vec is available"
	@echo "Checking output parent directory..."
	@mkdir -p $(OUTPUT)
	@echo "  OK: $(OUTPUT)"

run: check
	@echo "Starting Lit2Vec pipeline..."
	@echo "  Input : $(INPUT)"
	@echo "  Output: $(OUTPUT)"
	@echo "  Zip   : $(ZIP_FILE)"
	$(LIT2VEC_RUN)
	@echo ""
	@echo "Pipeline complete. Output archive:"
	@ls -lh $(ZIP_FILE)

run-bg: check
	@mkdir -p $(LOG_DIR)
	@echo "Starting Lit2Vec pipeline in background..."
	@echo "  Input : $(INPUT)"
	@echo "  Output: $(OUTPUT)"
	@echo "  Log   : $(LOG_FILE)"
	nohup $(LIT2VEC_RUN) >$(LOG_FILE) 2>&1 &
	@echo "PID: $$!"
	@echo "Monitor with: tail -f $(LOG_FILE)"

status:
	@if [ -f $(ZIP_FILE) ]; then \
		echo "Output archive exists:"; \
		ls -lh $(ZIP_FILE); \
	else \
		echo "Output archive not found: $(ZIP_FILE)"; \
		echo "Pipeline may still be running or has not been started."; \
	fi
	@if [ -d $(OUTPUT) ]; then \
		echo ""; \
		echo "Output directory contents:"; \
		find $(OUTPUT) -maxdepth 2 -type d | sed 's|^$(OUTPUT)/||' | sort | head -30; \
	fi

clean:
	@echo "Removing output directory: $(OUTPUT)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf $(OUTPUT) && echo "Removed." || echo "Aborted."
