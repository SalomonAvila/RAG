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

SYSTEM_PROMPT = """Eres un asistente de investigación de alto nivel y experto en ingeniería de software y computación.
Tu objetivo es responder las preguntas del usuario analizando y sintetizando en profundidad el contexto proporcionado (fragmentos de sus tesis, artículos y código).

Pautas de respuesta:
1. Comprensión conceptual: No te limites a copiar frases sueltas. Explica el "qué", el "por qué" y el "cómo", sintetizando la arquitectura, metodología, objetivos y hallazgos clave.
2. Estructura y claridad: Usa viñetas, secciones o párrafos bien articulados para facilitar la lectura.
3. Precisión estricta: Basa tus afirmaciones únicamente en el contexto proporcionado. Cita las fuentes relevantes usando [n] justo después de las ideas extraídas.
4. Si el contexto no contiene la información para responder con certeza, indícalo de forma clara y transparente sin inventar."""


def _extract_tokens(text: str) -> list[str]:
    import re
    import unicodedata

    # Separar palabras concatenadas (camelCase / PascalCase)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Reemplazar símbolos, rutas y extensiones por espacios
    text = re.sub(r"[\/\\\-_.]", " ", text)
    text = "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    )
    # Stopwords comunes
    stopwords = {
        "para", "como", "esta", "sobre", "entre", "donde", "cuando", "quien",
        "cual", "este", "estos", "estas", "unos", "unas", "tener", "hacer", "decir"
    }
    return [w for w in re.findall(r"\b[a-z0-9]+\b", text) if len(w) >= 3 and w not in stopwords]


_cached_corpus: list[tuple[qm.Record, list[str]]] | None = None
_cached_df: dict[str, int] = {}
_cached_paths: list[str] = []
_last_corpus_time: float = 0.0


def _get_corpus(qdrant: QdrantClient) -> tuple[list[tuple[qm.Record, list[str]]], dict[str, int], list[str]]:
    global _cached_corpus, _cached_df, _cached_paths, _last_corpus_time
    import time

    now = time.time()
    if _cached_corpus is not None and (now - _last_corpus_time) < 45.0:
        return _cached_corpus, _cached_df, _cached_paths

    # Recorrer colección completa para BM25 y catálogo de rutas
    all_points = []
    offset = None
    paths_set = set()

    while True:
        pts, offset = qdrant.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=250,
            offset=offset,
            with_payload=True,
        )
        for p in pts:
            all_points.append(p)
            if p.payload and "path" in p.payload:
                paths_set.add(p.payload["path"])
        if offset is None:
            break

    doc_corpus = []
    df: dict[str, int] = {}
    for p in all_points:
        payload = p.payload or {}
        p_path = payload.get("path", "")
        p_text = payload.get("text", "")
        # Se pondera la ruta para que menciones a carpetas, temas o autores pesen naturalmente
        content = f"{p_path} {p_path} {p_text}"
        toks = _extract_tokens(content)
        doc_corpus.append((p, toks))
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    _cached_corpus = doc_corpus
    _cached_df = df
    _cached_paths = sorted(paths_set)
    _last_corpus_time = now
    return _cached_corpus, _cached_df, _cached_paths


def search(
    qdrant: QdrantClient,
    llm: ollama.Client,
    question: str,
    *,
    top_k: int,
    source: str | None,
) -> list[qm.ScoredPoint]:
    """
    Búsqueda Híbrida Universal de 2 Fases:
    1. Análisis Semántico y Enrutamiento Universal: Identifica si la consulta apunta a archivos
       específicos del catálogo y complementa con sinónimos formales o académicos (sin reglas fijas).
    2. Fusión Equilibrada RRF (Reciprocal Rank Fusion): Combina búsqueda densa vectorial y
       búsqueda léxica BM25 (en ruta y contenido) calibrada para no sesgar ni descartar aciertos exactos.
    """
    import math

    doc_corpus, df, distinct_paths = _get_corpus(qdrant)

    # 1. Enrutamiento y expansión semántica universal con el LLM
    routed_paths: list[str] = []
    expanded_query = question

    if distinct_paths:
        try:
            catalog_str = "\n".join(f"- {p}" for p in distinct_paths)
            router_prompt = f"""Eres el analizador inteligente de consultas de un motor de búsqueda sobre documentos técnicos, diapositivas, artículos y código.
Catálogo de documentos disponibles:
{catalog_str}

Consulta del usuario: "{question}"

Instrucciones:
1. Determina qué archivo(s) del catálogo tienen relación directa con la consulta. Si la consulta es transversal, general o no menciona un tema específico, responde 'TODOS'.
2. Genera de 6 a 12 términos clave, sinónimos y componentes específicos con los que esta información suele estar redactada en los documentos fuente (por ejemplo: si pregunta por calificación o evaluación de cursos, incluye los componentes típicos como parciales, quices, talleres, proyectos, porcentajes, ponderación, notas; si pregunta por arquitectura de software, incluye módulos, servicios, contratos, diagramas; si pregunta por investigación académica o tesis, incluye objetivos, hipótesis, metodología, métricas).

Responde ÚNICAMENTE en este formato exacto:
DOCS: [archivos seleccionados separados por coma, o 'TODOS']
EXPANSION: [términos y componentes separados por espacio]"""

            resp = llm.chat(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": router_prompt}]
            )
            content = resp.message.content.strip()
            for line in content.splitlines():
                if line.startswith("DOCS:"):
                    docs_part = line.replace("DOCS:", "").strip()
                    if "TODOS" not in docs_part.upper():
                        for raw_name in docs_part.split(","):
                            clean_name = raw_name.strip().lstrip("- ")
                            for dp in distinct_paths:
                                if clean_name and (clean_name in dp or dp.endswith(clean_name)):
                                    routed_paths.append(dp)
                elif line.startswith("EXPANSION:"):
                    exp_part = line.replace("EXPANSION:", "").strip()
                    if exp_part:
                        expanded_query = f"{question} {exp_part}"
        except Exception:
            pass

    # 2. Búsqueda Vectorial Semántica
    search_vector = llm.embed(
        model=config.EMBED_MODEL,
        input=config.QUERY_PREFIX + expanded_query
    ).embeddings[0]

    base_filter = (
        qm.Filter(must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=source))])
        if source
        else None
    )

    candidate_depth = max(top_k * 5, 40)
    vector_results = qdrant.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=search_vector,
        query_filter=base_filter,
        limit=candidate_depth,
        with_payload=True,
    ).points

    # 3. Búsqueda Léxica BM25
    query_tokens = _extract_tokens(expanded_query)
    N = len(doc_corpus)
    avg_len = sum(len(toks) for _, toks in doc_corpus) / N if N > 0 else 1.0
    k1, b = 1.5, 0.75
    bm25_scores = []

    for p, toks in doc_corpus:
        payload = p.payload or {}
        if source and payload.get("source") != source:
            continue

        freqs = {}
        for t in toks:
            freqs[t] = freqs.get(t, 0) + 1
        score = 0.0
        for qt in query_tokens:
            if qt in freqs:
                tf = freqs[qt]
                n_t = df.get(qt, 0)
                idf = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1.0)
                num = tf * (k1 + 1.0)
                den = tf + k1 * (1.0 - b + b * (len(toks) / avg_len))
                score += idf * (num / den)

        if routed_paths and payload.get("path") in routed_paths:
            score *= 1.8

        if score > 0:
            bm25_scores.append((score, p))

    bm25_scores.sort(key=lambda x: x[0], reverse=True)
    bm25_results = [p for _, p in bm25_scores[:candidate_depth]]

    # 4. Fusión de Rangos Recíprocos Equilibrada (RRF calibrado con k=15)
    rrf_scores: dict[str, float] = {}
    points_by_id: dict[str, qm.ScoredPoint] = {}

    for rank, pt in enumerate(vector_results):
        boost = 1.6 if (routed_paths and pt.payload.get("path") in routed_paths) else 1.0
        rrf_scores[pt.id] = rrf_scores.get(pt.id, 0.0) + (boost / (15.0 + rank + 1.0))
        points_by_id[pt.id] = pt

    for rank, pt in enumerate(bm25_results):
        boost = 1.6 if (routed_paths and pt.payload.get("path") in routed_paths) else 1.0
        rrf_scores[pt.id] = rrf_scores.get(pt.id, 0.0) + (boost / (15.0 + rank + 1.0))
        points_by_id[pt.id] = pt

    sorted_ranks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    combined: list[qm.ScoredPoint] = []
    for pid, rrf_sc in sorted_ranks[:top_k]:
        orig = points_by_id[pid]
        combined.append(
            qm.ScoredPoint(
                id=orig.id,
                version=0,
                score=rrf_sc,
                payload=orig.payload,
                vector=None
            )
        )
    return combined



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
            score_val = f"{pt.score:.3f}" if getattr(pt, "score", None) is not None else "1.000"
            print(f"[{i}] ({score_val}) {_location(p)}\n{p.get('text', '')}\n{'-' * 60}")
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Contexto:\n\n{build_context(points)}\n\nPregunta: {question}"},
    ]
    for part in llm.chat(model=args.model, messages=messages, stream=True):
        print(part.message.content, end="", flush=True)

    print("\n\nFuentes:")
    for i, pt in enumerate(points, start=1):
        score_val = f"{pt.score:.3f}" if getattr(pt, "score", None) is not None else "1.000"
        print(f"  [{i}] ({score_val}) {_location(pt.payload or {})}")



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
