"""
Servidor Web FastAPI para el RAG Local.
Ofrece la interfaz gráfica de tipo Aplicación (Libreta) y streaming en tiempo real (SSE).

Uso:
    uv run python -m rag.server
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncGenerator

import ollama
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient

from rag import config
from rag.query import SYSTEM_PROMPT, _location, search


def get_available_sources(qdrant: QdrantClient) -> list[str]:
    try:
        results, _ = qdrant.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=200,
            with_payload=["source"],
        )
        sources = sorted({p.payload["source"] for p in results if p.payload and "source" in p.payload})
        return ["(Todos)"] + sources
    except Exception:
        return ["(Todos)"]


app = FastAPI(title="Tu Archivo · RAG Local")

STATIC_DIR = Path(__file__).resolve().parent / "static"


class QueryRequest(BaseModel):
    question: str
    source: str | None = None
    top_k: int = 7


@app.get("/")
async def get_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/sources")
async def api_sources():
    qdrant = QdrantClient(url=config.QDRANT_URL, prefer_grpc=True, check_compatibility=False)
    sources = get_available_sources(qdrant)
    return {"sources": sources}


@app.post("/api/chat")
async def api_chat(req: QueryRequest):
    qdrant = QdrantClient(url=config.QDRANT_URL, prefer_grpc=True, check_compatibility=False)
    llm = ollama.Client(host=config.OLLAMA_URL)

    points = search(qdrant, llm, req.question, top_k=req.top_k, source=req.source)

    async def event_generator() -> AsyncGenerator[str, None]:
        # 1. Enviar primero la lista de fuentes consultadas para que el front las prepare
        sources_payload = []
        context_blocks = []
        for i, pt in enumerate(points, start=1):
            p = pt.payload or {}
            loc = _location(p)
            score = f"{pt.score:.3f}" if pt.score is not None else "1.000"
            sources_payload.append({
                "index": i,
                "path": p.get("path", "?"),
                "detail": f"pág. {p.get('page')}" if p.get("page") else f"L{p.get('start_line')}-{p.get('end_line')}",
                "score": score,
            })
            context_blocks.append(f"[{i}] {loc}\n{p.get('text', '')}")

        yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload})}\n\n"

        if not points:
            yield f"data: {json.dumps({'type': 'token', 'text': 'No se encontraron documentos relevantes para esta consulta.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        context_str = "\n\n---\n\n".join(context_blocks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Contexto:\n\n{context_str}\n\nPregunta: {req.question}"},
        ]

        # 2. Transmitir token a token en tiempo real mientras el LLM los produce
        for part in llm.chat(model=config.LLM_MODEL, messages=messages, stream=True):
            token_text = part.message.content
            yield f"data: {json.dumps({'type': 'token', 'text': token_text})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def main():
    uvicorn.run("rag.server:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
