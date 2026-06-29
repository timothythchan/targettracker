# EarningsLens / Moving Targets LM — runnable Gradio app image.
#
# Build:
#   docker build -t earningslens-app .
#
# Run (binds to host port 7860; visit http://localhost:7860):
#   docker run --rm -p 7860:7860 earningslens-app
#
# Mount a pre-computed demo cache from the host (optional):
#   docker run --rm -p 7860:7860 \
#       -v "$(pwd)/data/cache/demo:/app/data/cache/demo" \
#       earningslens-app
#
# Provide an LLM API key for live LangGraph runs (optional; the cached path
# does not need a key):
#   docker run --rm -p 7860:7860 -e OPENAI_API_KEY=sk-... earningslens-app
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

# Install only what the Gradio app needs at runtime. The full requirements.txt
# pulls heavy scientific deps (sentence-transformers, chromadb, statsmodels)
# that the cached demo path does not need. Users who want the live LangGraph
# pipeline inside the container can override the install layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt /app/requirements-app.txt
RUN pip install --no-cache-dir -r /app/requirements-app.txt \
    && python -m spacy download en_core_web_sm

COPY . /app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/ > /dev/null || exit 1

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
