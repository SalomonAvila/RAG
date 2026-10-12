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

# Umbral mínimo de similitud para descartar fragmentos irrelevantes
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.58"))

# nomic-embed-text está entrenado con prefijos de tarea; usarlos mejora la búsqueda.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

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
