"""
Servidor Web FastAPI para el RAG Local.
Ofrece la interfaz gráfica de tipo Aplicación (Libreta) y streaming en tiempo real (SSE).

Uso:
    uv run python -m rag.server
"""

from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator

import ollama
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient

from rag import config
from rag.query import build_messages, get_corpus, location_detail, search, stream_answer, warm_up_reranker


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Cargar el cross-encoder en segundo plano para que la primera pregunta no lo espere.
    threading.Thread(target=warm_up_reranker, daemon=True).start()
    yield


app = FastAPI(title="Tu Archivo · RAG Local", lifespan=lifespan)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

qdrant = QdrantClient(url=config.QDRANT_URL, prefer_grpc=True, check_compatibility=False)
llm = ollama.Client(host=config.OLLAMA_URL)


class QueryRequest(BaseModel):
    question: str
    source: str | None = None
    top_k: int = 7
    messages: list[dict[str, str]] | None = None


@app.get("/")
async def get_index():
    return FileResponse(STATIC_DIR / "index.html")


# Los endpoints que hablan con Qdrant/Ollama son síncronos (def): FastAPI los ejecuta en un
# threadpool y no bloquean el event loop mientras el LLM genera.
@app.get("/api/sources")
def api_sources():
    try:
        sources = get_corpus(qdrant).sources
    except Exception:
        sources = []
    return {"sources": ["(Todos)"] + sources}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/api/chat")
def api_chat(req: QueryRequest):
    history = req.messages or []
    points = search(qdrant, llm, req.question, top_k=req.top_k, source=req.source, history=history)

    def event_generator() -> Iterator[str]:
        # 1. Enviar primero la lista de fuentes consultadas para que el front las prepare
        yield _sse({
            "type": "sources",
            "sources": [
                {
                    "index": i,
                    "path": (pt.payload or {}).get("path", "?"),
                    "detail": location_detail(pt.payload or {}),
                    "score": f"{pt.score:.3f}",
                }
                for i, pt in enumerate(points, start=1)
            ],
        })

        if not points:
            yield _sse({"type": "token", "text": "No se encontraron documentos relevantes para esta consulta."})
            yield _sse({"type": "done"})
            return

        # 2. Transmitir token a token en tiempo real mientras el LLM los produce
        for token in stream_answer(llm, build_messages(req.question, points, history)):
            yield _sse({"type": "token", "text": token})

        yield _sse({"type": "done"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def main():
    uvicorn.run("rag.server:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
