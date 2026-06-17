.PHONY: help build build-sandbox build-sandbox-fakeroot shell shell-writable shell-overlay exec run pip-install test overlay-create clean dev dev-overlay dev-overlay-fakeroot dev-noroot

# Project settings
PROJECT := lit2vec
DEF := Apptainer
SANDBOX := .appt-dev
SIF := $(PROJECT).sif
OVERLAY := overlay.img
OVERLAY_SIZE_MB := 8096

# Choose an image to use for exec/shell (prefer sandbox if present)
# This is evaluated at parse time; if sandbox exists, IMAGE := .appt-dev else lit2vec.sif
ifneq (,$(wildcard $(SANDBOX)))
IMAGE := $(SANDBOX)
else
IMAGE := $(SIF)
endif

help:
	@echo "Make targets for $(PROJECT) with Apptainer"
	@echo "  build                - Build immutable SIF from $(DEF) -> $(SIF)"
	@echo "  build-sandbox        - Build writable sandbox directory -> $(SANDBOX)"
	@echo "  build-sandbox-fakeroot - Build sandbox with --fakeroot (if needed)"
	@echo "  shell                - Enter container shell with source bound to /app (uses IMAGE=$(IMAGE), --nv)"
	@echo "  shell-writable       - Enter writable shell (requires sandbox)"
	@echo "  pip-install          - pip install -e /app inside the container"
	@echo "  run                  - Run a default command inside the container (edit CMD=...)"
	@echo "  exec CMD='...'       - Run arbitrary command inside the container"
	@echo "  test                 - Run pytest inside the container"
	@echo "  overlay-create       - Create a writable overlay image ($(OVERLAY_SIZE_MB) MB)"
	@echo "  shell-overlay        - Shell using SIF + writable overlay + bind /app"
	@echo "  dev-overlay          - Dev mode using immutable SIF + writable overlay (no sandbox)"
	@echo "  dev-overlay-fakeroot - Same as dev-overlay but uses --fakeroot (works on locked-down hosts)"
	@echo "  dev-noroot           - Dev with SIF + bind only, sets PYTHONPATH=/app (no installs, no rebuild)"
	@echo "  dev-tmpfs            - Dev shell with SIF + --writable-tmpfs (temporary writable, no artifacts)"
	@echo "  shell-sif            - Shell into the immutable SIF directly (bypass sandbox/autodetect)"
	@echo "  clean                - Remove build artifacts (SIF, sandbox, overlay)"

build:
	apptainer build $(SIF) $(DEF)

build-sandbox:
	apptainer build --sandbox $(SANDBOX) $(DEF)

build-sandbox-fakeroot:
	apptainer build --fakeroot --sandbox $(SANDBOX) $(DEF)

shell:
	apptainer shell --nv --bind "$(PWD)":/app $(IMAGE)

shell-writable:
	apptainer shell --nv --writable --bind "$(PWD)":/app $(SANDBOX)

dev:
	@if [ ! -d "$(SANDBOX)" ]; then \
		echo "[dev] Sandbox not found. Building $(SANDBOX) from $(DEF)..."; \
		apptainer build --sandbox $(SANDBOX) $(DEF); \
	fi
	@echo "[dev] Installing editable package inside sandbox..."
	apptainer exec --nv --writable --bind "$(PWD)":/app $(SANDBOX) pip install --root-user-action=ignore -e /app
	@echo "[dev] Launching writable shell with source bound to /app"
	apptainer shell --nv --writable --bind "$(PWD)":/app $(SANDBOX)

pip-install:
	apptainer exec --nv --bind "$(PWD)":/app $(IMAGE) pip install --root-user-action=ignore -e /app

# Default run command can be overridden e.g. make run CMD="python -m lit2vec"
CMD ?= python
run:
	$(MAKE) exec CMD='$(CMD)'

# Usage: make exec CMD="python -m lit2vec" (or any command)
exec:
	@if [ -z "$(CMD)" ]; then echo "Set CMD=..."; exit 1; fi
	apptainer exec --nv --bind "$(PWD)":/app $(IMAGE) sh -lc '$(CMD)'

# Testing (requires optional dependency [tests] if not present in image)
# You can ensure pytest is available with: make exec CMD="pip install -U pytest pytest-cov"
test:
	apptainer exec --nv --bind "$(PWD)":/app $(IMAGE) sh -lc 'pytest -q || (echo "Hint: install test deps with: pip install -U pytest pytest-cov" && exit 1)'

overlay-create:
	dd if=/dev/zero of=$(OVERLAY) bs=1M count=$(OVERLAY_SIZE_MB)
	mkfs.ext3 -F $(OVERLAY)

shell-overlay:
	@if [ ! -f "$(SIF)" ]; then \
		echo "[shell-overlay] $(SIF) not found, building from $(DEF)..."; \
		apptainer build $(SIF) $(DEF); \
	fi
	apptainer shell --nv --overlay $(OVERLAY):rw --bind "$(PWD)":/app $(SIF)

dev-overlay:
	@if [ ! -f "$(SIF)" ]; then \
		echo "[dev-overlay] $(SIF) not found, building from $(DEF)..."; \
		apptainer build $(SIF) $(DEF); \
	fi
	@if [ ! -f "$(OVERLAY)" ]; then \
		echo "[dev-overlay] Creating overlay $(OVERLAY) ($(OVERLAY_SIZE_MB) MB)..."; \
		dd if=/dev/zero of=$(OVERLAY) bs=1M count=$(OVERLAY_SIZE_MB); \
		mkfs.ext3 -F $(OVERLAY); \
	fi
	@echo "[dev-overlay] Installing editable package into overlay..."
	apptainer exec --nv --fakeroot --overlay $(OVERLAY):rw --bind "$(PWD)":/app $(SIF) pip install --root-user-action=ignore -e /app || echo "[dev-overlay] Install may fail without --fakeroot on some systems. Try 'make dev-overlay-fakeroot' if this fails."
	@echo "[dev-overlay] Launching shell with overlay + source bound to /app"
	apptainer shell --nv --fakeroot --overlay $(OVERLAY):rw --bind "$(PWD)":/app $(SIF)

dev-overlay-fakeroot:
	@if [ ! -f "$(SIF)" ]; then \
		echo "[dev-overlay-fakeroot] $(SIF) not found, building from $(DEF)..."; \
		apptainer build $(SIF) $(DEF); \
	fi
	@if [ ! -f "$(OVERLAY)" ]; then \
		echo "[dev-overlay-fakeroot] Creating overlay $(OVERLAY) ($(OVERLAY_SIZE_MB) MB)..."; \
		dd if=/dev/zero of=$(OVERLAY) bs=1M count=$(OVERLAY_SIZE_MB); \
		mkfs.ext3 -F $(OVERLAY); \
	fi
	@echo "[dev-overlay-fakeroot] Installing editable package into overlay..."
	apptainer exec --nv --fakeroot --overlay $(OVERLAY):rw --bind "$(PWD)":/app $(SIF) pip install --root-user-action=ignore -e /app
	@echo "[dev-overlay-fakeroot] Launching shell with overlay + source bound to /app"
	apptainer shell --nv --fakeroot --overlay $(OVERLAY):rw --bind "$(PWD)":/app $(SIF)

dev-noroot:
	@if [ ! -f "$(SIF)" ]; then \
		echo "[dev-noroot] $(SIF) not found, building from $(DEF)..."; \
		apptainer build $(SIF) $(DEF); \
	fi
	@echo "[dev-noroot] Launching shell with PYTHONPATH=/app (no install needed)"
	apptainer shell --nv --bind "$(PWD)":/app --env PYTHONPATH=/app $(SIF)

# Always shell directly into the SIF (bypass sandbox autodetect)
shell-sif:
	@if [ ! -f "$(SIF)" ]; then \
		echo "[shell-sif] $(SIF) not found, building from $(DEF)..."; \
		apptainer build $(SIF) $(DEF); \
	fi
	apptainer shell --nv --bind "$(PWD)":/app $(SIF)

# Temporary writable dev shell (no artifacts created, changes discarded on exit)
dev-tmpfs:
	@if [ ! -f "$(SIF)" ]; then \
		echo "[dev-tmpfs] $(SIF) not found, building from $(DEF)..."; \
		apptainer build $(SIF) $(DEF); \
	fi
	apptainer shell --nv --writable-tmpfs --bind "$(PWD)":/app $(SIF)

clean:
	rm -rf $(SANDBOX) $(SIF) $(OVERLAY)
