# EarningsLens / Moving Targets LM — convenience targets.
#
# Users only need:
#
#     python app.py
#
# Docker users:
#
#     make docker-build
#     make docker-run

PYTHON ?= python3
HOST   ?= 127.0.0.1
PORT   ?= 7860

.PHONY: help install app demo docker-build docker-run clean

help:
	@echo "Available targets:"
	@echo "  install         Install app dependencies (usually automatic on first launch)"
	@echo "  app             Launch the app on http://$(HOST):$(PORT)"
	@echo "  demo            Alias for 'app'"
	@echo "  docker-build    Build the Docker image (tag: earningslens-app)"
	@echo "  docker-run      Run the Docker image, binding host port $(PORT)"
	@echo "  clean           Remove Python bytecode and cache directories"

install:
	$(PYTHON) -m pip install -r requirements-app.txt

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
