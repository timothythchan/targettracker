# EarningsLens / Moving Targets LM — convenience targets.
#
# Most users only need:
#
#     make install   # one-time: minimum deps to launch the app
#     make app       # launch the Gradio app on http://localhost:7860
#
# Docker users:
#
#     make docker-build
#     make docker-run

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip
HOST   ?= 127.0.0.1
PORT   ?= 7860

.PHONY: help install install-all app demo docker-build docker-run clean

help:
	@echo "Available targets:"
	@echo "  install         Install minimal deps to launch the Gradio app"
	@echo "  install-all     Install full research pipeline deps (requirements.txt + package)"
	@echo "  status          Print pipeline-stage status (which artifacts exist on disk)"
	@echo "  app             Launch the Gradio app on http://$(HOST):$(PORT)"
	@echo "  demo            Alias for 'app'"
	@echo "  pipeline        Run every pipeline stage in order"
	@echo "  cache           Build the Gradio demo cache (NB06 port)"
	@echo "  docker-build    Build the Docker image (tag: earningslens-app)"
	@echo "  docker-run      Run the Docker image, binding host port $(PORT)"
	@echo "  clean           Remove Python bytecode and cache directories"
	@echo
	@echo "All stages are also reachable via the unified CLI:"
	@echo "  python -m src --help"
	@echo "  python -m src status"
	@echo "  python -m src baseline --limit 20"

install:
	$(PIP) install -r requirements-app.txt

install-all:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

status:
	$(PYTHON) -m src status

app:
	$(PYTHON) -m src app --host $(HOST) --port $(PORT)

demo: app

pipeline:
	$(PYTHON) -m src pipeline

cache:
	$(PYTHON) -m src cache

docker-build:
	docker build -t earningslens-app .

docker-run:
	docker run --rm -p $(PORT):7860 earningslens-app

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
