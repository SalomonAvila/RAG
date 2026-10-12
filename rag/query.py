"""
Consulta al RAG: busca los fragmentos más relevantes en Qdrant y responde con Ollama.

Uso:
    uv run python -m rag.query "¿Cómo funciona la autenticación en mi-repo?"
    uv run python -m rag.query                      # modo interactivo
    uv run python -m rag.query -s mi-repo "..."     # filtrar por carpeta/fuente
    uv run python -m rag.query --solo-buscar "..."  # sólo muestra fragmentos, sin LLM
"""

from __future__ import annotations

import argparse
import sys

import ollama
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from rag import config

SYSTEM_PROMPT = """Eres un asistente que responde preguntas usando EXCLUSIVAMENTE el contexto \
proporcionado (fragmentos de los repositorios y documentos del usuario).
- Responde en el mismo idioma de la pregunta.
- Cita las fuentes con [n] según el número del fragmento.
- Si el contexto no contiene la respuesta, dilo claramente; no inventes."""


def search(
    qdrant: QdrantClient,
    llm: ollama.Client,
    question: str,
    *,
    top_k: int,
    source: str | None,
) -> list[qm.ScoredPoint]:
    vector = llm.embed(model=config.EMBED_MODEL, input=config.QUERY_PREFIX + question).embeddings[0]
    query_filter = (
        qm.Filter(must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=source))])
        if source
        else None
    )
    return qdrant.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    ).points


def _location(p: dict) -> str:
    loc = p.get("path", "?")
    if p.get("page"):
        loc += f" (pág. {p['page']})"
    elif p.get("start_line"):
        loc += f" (L{p['start_line']}-{p['end_line']})"
    return loc


def build_context(points: list[qm.ScoredPoint]) -> str:
    blocks = []
    for i, pt in enumerate(points, start=1):
        p = pt.payload or {}
        blocks.append(f"[{i}] {_location(p)}\n{p.get('text', '')}")
    return "\n\n---\n\n".join(blocks)


def answer(qdrant: QdrantClient, llm: ollama.Client, question: str, args: argparse.Namespace) -> None:
    points = search(qdrant, llm, question, top_k=args.k, source=args.source)
    if not points:
        print("No encontré fragmentos relevantes (¿ya corriste el indexador?).")
        return

    if args.solo_buscar:
        for i, pt in enumerate(points, start=1):
            p = pt.payload or {}
            print(f"[{i}] ({pt.score:.3f}) {_location(p)}\n{p.get('text', '')}\n{'-' * 60}")
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Contexto:\n\n{build_context(points)}\n\nPregunta: {question}"},
    ]
    for part in llm.chat(model=args.model, messages=messages, stream=True):
        print(part.message.content, end="", flush=True)

    print("\n\nFuentes:")
    for i, pt in enumerate(points, start=1):
        print(f"  [{i}] ({pt.score:.3f}) {_location(pt.payload or {})}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pregúntale a tus repos y documentos.")
    parser.add_argument("pregunta", nargs="*", help="Pregunta (si se omite, modo interactivo)")
    parser.add_argument("-k", type=int, default=6, help="Fragmentos a recuperar (default: 6)")
    parser.add_argument("-s", "--source", help="Filtrar por fuente (primera carpeta en RAG_SOURCE_DIR)")
    parser.add_argument("-m", "--model", default=config.LLM_MODEL, help="Modelo de Ollama para responder")
    parser.add_argument("--solo-buscar", action="store_true", help="Sólo mostrar fragmentos, sin LLM")
    args = parser.parse_args()

    qdrant = QdrantClient(url=config.QDRANT_URL, prefer_grpc=True)
    llm = ollama.Client(host=config.OLLAMA_URL)

    if args.pregunta:
        answer(qdrant, llm, " ".join(args.pregunta), args)
        return

    print(f"RAG local · modelo: {args.model} · colección: {config.QDRANT_COLLECTION} (Enter vacío para salir)")
    while True:
        try:
            q = input("\n❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        answer(qdrant, llm, q, args)


if __name__ == "__main__":
    sys.exit(main())
