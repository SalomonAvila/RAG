"""Configuración compartida entre el indexador y el módulo de consulta."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

# Estado de CocoIndex: debe definirse antes de que arranque el entorno de cocoindex.
os.environ.setdefault("COCOINDEX_DB", str(REPO_ROOT / ".cocoindex_db"))

SOURCE_DIR = Path(os.getenv("RAG_SOURCE_DIR", "~/rag-data")).expanduser()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6334")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_docs")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")

# Ventana de contexto del LLM. Ollama usa 4096 por defecto con < 24 GB de VRAM, lo que
# recorta en silencio el prompt (contexto + historial). 8192 cabe en 6 GB junto a un 7B.
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))
# Tiempo que Ollama mantiene los modelos cargados entre consultas (evita cargas en frío).
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
# Capas del embedder en GPU al consultar (0 = CPU). Con 6 GB de VRAM, si el embedder
# ocupa GPU, Ollama desaloja al LLM en cada pregunta y lo vuelve a cargar.
EMBED_NUM_GPU = int(os.getenv("EMBED_NUM_GPU", "0"))

# Umbral de similitud vectorial bajo el cual, sin coincidencias léxicas, no hay evidencia.
NOISE_FLOOR = float(os.getenv("NOISE_FLOOR", "0.64"))

# Reranker (cross-encoder vía fastembed, corre en CPU donde se ejecute la consulta).
# RERANK_MODEL vacío lo desactiva y se usa sólo la fusión RRF.
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3-int8")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "8"))
# Se conservan los fragmentos cuya probabilidad de relevancia sea >= este ratio de la mejor.
RERANK_KEEP_RATIO = float(os.getenv("RERANK_KEEP_RATIO", "0.5"))

# nomic-embed-text está entrenado con prefijos de tarea; usarlos mejora la búsqueda.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def chunk_header(rel_path: str, section: str | None = None) -> str:
    """Cabecera que se antepone a cada fragmento al embeberlo y al reordenarlo."""
    header = f"Archivo: {rel_path}\n"
    if section:
        header += f"Sección: {section}\n"
    return header


# Chunking (en bytes). ~1500 bytes ≈ 300-400 tokens: cabe de sobra en el contexto del embedder.
CHUNK_SIZE = 1500
MIN_CHUNK_SIZE = 400
CHUNK_OVERLAP = 200

# Archivos a indexar
INCLUDED_PATTERNS = [
    # Documentos
    "**/*.md", "**/*.mdx", "**/*.txt", "**/*.rst", "**/*.tex", "**/*.typ", "**/*.bib", "**/*.pdf", "**/*.docx",
    # Código
    "**/*.py", "**/*.ipynb", "**/*.js", "**/*.jsx", "**/*.ts", "**/*.tsx",
    "**/*.java", "**/*.kt", "**/*.go", "**/*.rs", "**/*.c", "**/*.h",
    "**/*.cpp", "**/*.hpp", "**/*.cs", "**/*.rb", "**/*.php", "**/*.swift",
    "**/*.dart", "**/*.scala", "**/*.sql", "**/*.sh", "**/*.r", "**/*.R",
    "**/*.html", "**/*.css", "**/*.scss", "**/*.vue", "**/*.svelte",
    # Configuración
    "**/*.toml", "**/*.yaml", "**/*.yml", "**/Dockerfile", "**/Makefile",
]
EXCLUDED_PATTERNS = [
    "**/.*",  # .git, .venv, .idea, etc.
    "**/node_modules", "**/__pycache__", "**/venv", "**/env",
    "**/target", "**/dist", "**/build", "**/out", "**/vendor",
    "**/*.min.js", "**/*.lock", "**/package-lock.json",
]

# Archivos más grandes que esto se ignoran (suelen ser datos generados, no contenido útil).
MAX_TEXT_FILE_BYTES = 1_000_000
MAX_PDF_FILE_BYTES = 50_000_000
