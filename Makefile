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
	@echo "  app             Launch the Gradio app on http://$(HOST):$(PORT)"
	@echo "  demo            Alias for 'app'"
	@echo "  docker-build    Build the Docker image (tag: earningslens-app)"
	@echo "  docker-run      Run the Docker image, binding host port $(PORT)"
	@echo "  clean           Remove Python bytecode and cache directories"

install:
	$(PIP) install -r requirements-app.txt

install-all:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

app:
	$(PYTHON) app.py --host $(HOST) --port $(PORT)

demo: app

docker-build:
	docker build -t earningslens-app .

docker-run:
	docker run --rm -p $(PORT):7860 earningslens-app

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
