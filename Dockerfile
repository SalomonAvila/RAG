# Imagen única para la interfaz (rag.server) y el indexador (cocoindex): cambia sólo el comando.
#   docker build -t rag-homelab .
#   docker run ... rag-homelab                                     # interfaz en :7860
#   docker run ... rag-homelab cocoindex update -L rag.indexer     # indexador en vivo

FROM ghcr.io/astral-sh/uv:0.13.0 AS uv

FROM python:3.14-slim

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app

WORKDIR /app

# Dependencias primero (capa cacheada mientras no cambie uv.lock)
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY rag ./rag

# Usuario sin privilegios. El uid/gid por defecto (1000) coincide con "homelab" en la torre,
# para que los volúmenes montados (.cocoindex_db, caché de fastembed) le pertenezcan.
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g ${APP_GID} app \
    && useradd -u ${APP_UID} -g ${APP_GID} -d /home/app -m app \
    && mkdir -p /home/app/.cache/fastembed /state \
    && chown -R app:app /home/app /state
USER app

ENV COCOINDEX_DB=/state/cocoindex_db \
    RAG_SOURCE_DIR=/data

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"RAG_PORT\",\"7860\")}/', timeout=4)"

CMD ["python", "-m", "rag.server"]
