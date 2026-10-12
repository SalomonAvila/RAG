"""
Pipeline de indexación incremental con CocoIndex.

    Fuente (RAG_SOURCE_DIR) -> extraer texto (PDF/código/docs) -> chunking
    -> embeddings (Ollama: nomic-embed-text) -> Qdrant

Uso (desde la raíz del repo):
    uv run cocoindex update rag.indexer      # sincroniza una vez (sólo procesa cambios)
    uv run cocoindex update -L rag.indexer   # modo live: vigila la carpeta y re-indexa al vuelo
"""

from __future__ import annotations

import asyncio
import io
import pathlib
from typing import AsyncIterator

import cocoindex as coco
from cocoindex.connectors import localfs, qdrant
from cocoindex.ops.litellm import LiteLLMEmbedder
from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from cocoindex.resources.chunk import Chunk
from cocoindex.resources.file import FileLike, PatternFilePathMatcher
from cocoindex.resources.id import IdGenerator
from qdrant_client import QdrantClient

from rag import config

QDRANT_DB = coco.ContextKey[QdrantClient]("rag_qdrant")
EMBEDDER = coco.ContextKey[LiteLLMEmbedder]("rag_embedder", detect_change=True)

_splitter = RecursiveSplitter()


@coco.lifespan
async def coco_lifespan(builder: coco.EnvironmentBuilder) -> AsyncIterator[None]:
    builder.provide(QDRANT_DB, qdrant.create_client(config.QDRANT_URL, prefer_grpc=True))
    builder.provide(
        EMBEDDER,
        LiteLLMEmbedder(
            f"ollama/{config.EMBED_MODEL}",
            api_base=config.OLLAMA_URL,
            max_inflight_requests=16,  # Concurrencia acelerada aprovechando la VRAM
        ),
    )
    yield


def _extract_pdf_pages(data: bytes) -> list[tuple[int | None, str]]:
    import pymupdf  # Motor en C/C++ ultrarrápido

    doc = pymupdf.open(stream=data, filetype="pdf")
    pages: list[tuple[int | None, str]] = []
    for i, page in enumerate(doc, start=1):
        try:
            text = page.get_text() or ""
        except Exception:
            continue
        if text.strip():
            pages.append((i, text))
    return pages


def _extract_docx_text(data: bytes) -> str:
    import io
    import docx

    doc = docx.Document(io.BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)
    return "\n\n".join(paragraphs)


@coco.fn
async def process_chunk(
    chunk: Chunk,
    rel_path: str,
    source: str,
    language: str | None,
    page: int | None,
    id_gen: IdGenerator,
    target: qdrant.CollectionTarget,
) -> None:
    embedding = await coco.use_context(EMBEDDER).embed(config.DOC_PREFIX + chunk.text)
    target.declare_point(
        qdrant.PointStruct(
            id=await id_gen.next_id(chunk.text),
            vector=embedding.tolist(),
            payload={
                "path": rel_path,
                "source": source,
                "language": language,
                "page": page,
                "start_line": chunk.start.line,
                "end_line": chunk.end.line,
                "text": chunk.text,
            },
        )
    )


@coco.fn(memo=True)  # memo: si el archivo no cambió, no se vuelve a procesar
async def process_file(file: FileLike, target: qdrant.CollectionTarget) -> None:
    path = file.file_path.path
    rel_path = path.as_posix()
    # "source" = primera carpeta dentro de RAG_SOURCE_DIR (p.ej. el nombre del repo o materia)
    source = path.parts[0] if len(path.parts) > 1 else "(raíz)"
    suffix = path.suffix.lower()
    is_pdf = suffix == ".pdf"
    is_docx = suffix == ".docx"

    size = await file.size()
    if size > (config.MAX_PDF_FILE_BYTES if (is_pdf or is_docx) else config.MAX_TEXT_FILE_BYTES):
        return

    if is_pdf:
        pages = await asyncio.to_thread(_extract_pdf_pages, await file.read())
        language = None
    elif is_docx:
        text = await asyncio.to_thread(_extract_docx_text, await file.read())
        pages = [(None, text)]
        language = None
    else:
        text = await file.read_text()
        if "\x00" in text:  # binario disfrazado
            return
        pages = [(None, text)]
        language = detect_code_language(filename=path.name)

    id_gen = IdGenerator()
    for page, text in pages:
        chunks = _splitter.split(
            text,
            chunk_size=config.CHUNK_SIZE,
            min_chunk_size=config.MIN_CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            language=language or ("markdown" if path.suffix.lower() in (".md", ".mdx") else None),
        )
        chunks = [c for c in chunks if c.text.strip()]
        await coco.map(process_chunk, chunks, rel_path, source, language, page, id_gen, target)


@coco.fn
async def app_main(source_dir: pathlib.Path) -> None:
    if not source_dir.is_dir():
        raise SystemExit(
            f"La carpeta de fuentes no existe: {source_dir}\n"
            "Créala (o define RAG_SOURCE_DIR) y pon ahí tus repos/documentos."
        )

    collection = await qdrant.mount_collection_target(
        QDRANT_DB,
        collection_name=config.QDRANT_COLLECTION,
        schema=await qdrant.CollectionSchema.create(
            vectors=qdrant.QdrantVectorDef(schema=EMBEDDER, distance="cosine")
        ),
    )
    files = localfs.walk_dir(
        source_dir,
        recursive=True,
        path_matcher=PatternFilePathMatcher(
            included_patterns=config.INCLUDED_PATTERNS,
            excluded_patterns=config.EXCLUDED_PATTERNS,
        ),
        live=True,  # sólo vigila si se ejecuta con `cocoindex update -L`
    )
    await coco.mount_each(process_file, files.items(), collection)


app = coco.App(
    coco.AppConfig(name="RagHomelab", max_inflight_components=8),
    app_main,
    source_dir=config.SOURCE_DIR,
)
