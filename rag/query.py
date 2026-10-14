"""
Consulta al RAG: busca los fragmentos más relevantes en Qdrant y responde con Ollama.

Uso:
    uv run python -m rag.query "¿Cómo funciona la autenticación en mi-repo?"
    uv run python -m rag.query                      # modo interactivo
    uv run python -m rag.query -s mi-repo "..."     # filtrar por carpeta/fuente
    uv run python -m rag.query --solo-buscar "..."  # sólo muestra fragmentos, sin LLM
    uv run python -m rag.query --sin-reranker "..." # desactiva el reranker (comparar)
"""

from __future__ import annotations

import argparse
import functools
import math
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath

import ollama
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from rag import config

SYSTEM_PROMPT = """Eres un asistente experto para responder preguntas basándote estrictamente en los documentos del usuario.

Pautas de respuesta:
1. Relevancia estricta y enfoque: Responde de forma precisa y directa ÚNICAMENTE a lo que el usuario preguntó. Si parte del contexto recuperado no tiene relación directa con la pregunta formulada, ignórala por completo y jamás agregues resúmenes o explicaciones de temas no solicitados.
2. Proporcionalidad: Si la pregunta es puntual, breve o específica, responde de forma concisa sin rodeos. Si la pregunta es amplia, metodológica o de arquitectura técnica, desarrolla la respuesta con la estructura y profundidad adecuadas.
3. Precisión estricta: Basa todas tus afirmaciones exclusivamente en los fragmentos relevantes del contexto. Cita las fuentes [n] justo después de cada dato o idea extraída.
4. Honestidad: Si el contexto no contiene información suficiente para responder con certeza a lo preguntado, indícalo brevemente y con total transparencia sin especular ni rellenar.
5. Secciones: Los fragmentos pueden empezar o terminar a mitad de una sección. Atribuye cada dato únicamente a la sección indicada en su línea "Sección:" y nunca mezcles contenido de secciones distintas bajo un mismo título.
6. Siglas: No expandas siglas ni inventes definiciones que no aparezcan en el contexto.
7. Visión general: Si el contexto incluye el "Índice del documento", úsalo para describir de qué trata el documento cuando la pregunta sea amplia (p.ej. "¿qué dice X?"), y complementa con los fragmentos."""


def _extract_tokens(text: str) -> list[str]:
    import unicodedata

    # Separar palabras concatenadas (camelCase / PascalCase)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Reemplazar símbolos, rutas y extensiones por espacios
    text = re.sub(r"[\/\\\-_.]", " ", text)
    text = "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    )
    # Stopwords comunes en español (artículos, preposiciones, pronombres, verbos auxiliares)
    stopwords = {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "del", "al",
        "a", "ante", "bajo", "cabe", "con", "contra", "de", "desde", "durante", "en",
        "entre", "hacia", "hasta", "mediante", "para", "por", "segun", "sin", "so",
        "sobre", "tras", "versus", "via", "y", "e", "ni", "o", "u", "pero", "sino",
        "aunque", "porque", "pues", "que", "si",
        "como", "cual", "cuales", "cuando", "cuanto", "cuanta", "cuantos", "cuantas",
        "donde", "adonde", "quien", "quienes",
        "este", "esta", "estos", "estas", "esto", "ese", "esa", "esos", "esas", "eso",
        "aquel", "aquella", "aquellos", "aquellas", "aquello",
        "mi", "mis", "tu", "tus", "su", "sus", "nuestro", "nuestra", "nuestros", "nuestras",
        "me", "te", "se", "nos", "les", "le",
        "ser", "es", "son", "era", "eran", "fue", "fueron", "sido", "siendo",
        "estar", "estan", "estaba", "estaban", "estado", "estando",
        "haber", "hay", "habia", "hubo", "haya", "he", "ha", "han", "hemos",
        "hacer", "hace", "hacen", "hacia", "hizo", "hecho",
        "tener", "tiene", "tienen", "tenia", "tuvo", "tenido",
        "decir", "dice", "dicen", "dijo", "dicho",
        "no", "ya", "muy", "mas", "menos", "bien", "mal", "asi", "tan", "tanto",
        "tanta", "tantos", "tantas", "todo", "toda", "todos", "todas", "otro", "otra",
        "otros", "otras", "mismo", "misma", "mismos", "mismas", "cada", "uno", "algo", "nada"
    }
    return [w for w in re.findall(r"\b[a-z0-9]+\b", text) if len(w) >= 3 and w not in stopwords]


# Mensajes de historial que se envían al LLM y caracteres máximos de cada uno.
HISTORY_MESSAGES = 6
HISTORY_MESSAGE_CHARS = 2000
# Si el catálogo de carpetas supera esto, el router sólo ve el primer nivel.
ROUTER_CATALOG_CHARS = 4000


def llm_options() -> dict:
    return {"num_ctx": config.LLM_NUM_CTX}


def embed_query(llm: ollama.Client, text: str) -> list[float]:
    return llm.embed(
        model=config.EMBED_MODEL,
        input=config.QUERY_PREFIX + text,
        options={"num_gpu": config.EMBED_NUM_GPU},
        keep_alive=config.OLLAMA_KEEP_ALIVE,
    ).embeddings[0]


@dataclass
class _Corpus:
    points: list[tuple[qm.Record, list[str]]]
    df: dict[str, int]
    avg_len: float
    folders: list[str]
    sources: list[str]
    stems: dict[str, frozenset[str]]  # ruta -> tokens del nombre del archivo
    loaded_at: float


_corpus: _Corpus | None = None


def _folder_catalog(paths: set[str], depth: int) -> list[str]:
    folders = set()
    for path in paths:
        parts = path.split("/")[:-1]
        if parts:
            folders.add("/".join(parts[:depth]))
    return sorted(folders)


def get_corpus(qdrant: QdrantClient) -> _Corpus:
    """Colección completa en memoria (para BM25 y catálogos), refrescada cada 45 s."""
    global _corpus
    now = time.time()
    if _corpus is not None and (now - _corpus.loaded_at) < 45.0:
        return _corpus

    points: list[tuple[qm.Record, list[str]]] = []
    df: dict[str, int] = {}
    paths: set[str] = set()
    sources: set[str] = set()
    offset = None
    while True:
        batch, offset = qdrant.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=500,
            offset=offset,
            with_payload=True,
        )
        for p in batch:
            payload = p.payload or {}
            path = payload.get("path", "")
            paths.add(path)
            if payload.get("source"):
                sources.add(payload["source"])
            # La ruta cuenta una sola vez: duplicarla hacía que en fragmentos cortos
            # (títulos de diapositivas) pesara más que el contenido.
            toks = _extract_tokens(f"{path} {payload.get('text', '')}")
            points.append((p, toks))
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        if offset is None:
            break

    folders = _folder_catalog(paths, depth=2)
    if sum(len(f) + 3 for f in folders) > ROUTER_CATALOG_CHARS:
        folders = _folder_catalog(paths, depth=1)

    _corpus = _Corpus(
        points=points,
        df=df,
        avg_len=(sum(len(t) for _, t in points) / len(points)) if points else 1.0,
        folders=folders,
        sources=sorted(sources),
        stems={path: frozenset(_extract_tokens(PurePosixPath(path).stem)) for path in paths},
        loaded_at=now,
    )
    return _corpus


@dataclass
class _Route:
    folders: list[str]
    expanded_query: str
    standalone_question: str


def _format_history(history: list[dict] | None) -> str:
    lines = []
    for m in (history or [])[-4:]:
        role = "Usuario" if m.get("role") == "user" else "Asistente"
        content = " ".join((m.get("content") or "").split())[:300]
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _route(llm: ollama.Client, question: str, folders: list[str], history: list[dict] | None) -> _Route:
    """Un solo llamado al LLM que (1) reescribe preguntas de seguimiento como consultas
    independientes, (2) elige carpetas relevantes y (3) expande la consulta con sinónimos."""
    route = _Route(folders=[], expanded_query=question, standalone_question=question)
    history_str = _format_history(history)
    catalog_str = "\n".join(f"- {f}" for f in folders) or "- (sin carpetas)"

    history_block = (
        f"Conversación previa:\n{history_str}\n\n" if history_str else ""
    )
    consulta_rule = (
        "3. CONSULTA: reescribe la consulta del usuario como una pregunta autocontenida, "
        "resolviendo referencias a la conversación previa (\"¿y la segunda?\", \"eso\", \"el mismo curso\"). "
        "Si ya es autocontenida, repítela igual.\n"
        if history_str
        else ""
    )
    consulta_fmt = "CONSULTA: [pregunta autocontenida]\n" if history_str else ""

    prompt = f"""Eres el analizador de consultas de un buscador sobre documentos personales (diapositivas, artículos, apuntes y código).
Carpetas disponibles:
{catalog_str}

{history_block}Consulta del usuario: "{question}"

Instrucciones:
1. DOCS: indica qué carpetas de la lista tienen relación directa con la consulta (copia los nombres exactos). Si es transversal responde 'TODOS'.
2. EXPANSION: genera de 6 a 12 términos clave y sinónimos con los que esta información suele estar redactada en los documentos (por ejemplo, para calificación: evaluación, parciales, quices, talleres, proyecto, porcentajes, ponderación; para arquitectura: módulos, servicios, contratos, diagramas).
{consulta_rule}
Responde ÚNICAMENTE en este formato:
{consulta_fmt}DOCS: [carpetas separadas por coma, o 'TODOS']
EXPANSION: [términos separados por espacio]"""

    try:
        resp = llm.chat(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={**llm_options(), "temperature": 0, "num_predict": 200},
            keep_alive=config.OLLAMA_KEEP_ALIVE,
        )
    except Exception as e:
        print(f"[router] falló, se busca sin expansión: {e}", file=sys.stderr)
        return route

    terms = ""
    for line in resp.message.content.splitlines():
        key, _, value = line.strip().partition(":")
        value = value.strip().strip("[]").strip()
        key = key.strip().upper()
        if key == "CONSULTA" and value:
            route.standalone_question = value
        elif key == "DOCS" and value and "TODOS" not in value.upper():
            for name in value.split(","):
                name = name.strip().strip("-'\" ").rstrip("/")
                route.folders += [f for f in folders if name and (f == name or f.endswith("/" + name))]
        elif key == "EXPANSION":
            terms = value

    route.expanded_query = f"{route.standalone_question} {terms}".strip()
    return route


def _in_folders(payload: dict, folders: list[str]) -> bool:
    path = payload.get("path", "")
    return any(path.startswith(f + "/") for f in folders)


def _bm25(
    corpus: _Corpus, query: str, source: str | None, path: str | None, folders: list[str], limit: int
) -> list[qm.Record]:
    query_tokens = _extract_tokens(query)
    N = len(corpus.points)
    k1, b = 1.5, 0.75
    scored = []
    for p, toks in corpus.points:
        payload = p.payload or {}
        if (source and payload.get("source") != source) or (path and payload.get("path") != path):
            continue
        freqs: dict[str, int] = {}
        for t in toks:
            freqs[t] = freqs.get(t, 0) + 1
        score = 0.0
        for qt in query_tokens:
            tf = freqs.get(qt)
            if tf:
                n_t = corpus.df.get(qt, 0)
                idf = math.log((N - n_t + 0.5) / (n_t + 0.5) + 1.0)
                score += idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * len(toks) / corpus.avg_len))
        if folders and _in_folders(payload, folders):
            score *= 1.8
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:limit]]


@functools.lru_cache(maxsize=1)
def _get_reranker():
    if not config.RERANK_MODEL:
        return None
    try:
        from pathlib import Path

        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(config.RERANK_MODEL, cache_dir=str(Path.home() / ".cache" / "fastembed"))
    except Exception as e:
        print(f"[reranker] no disponible ({e}); se usa sólo RRF.", file=sys.stderr)
        return None


def warm_up_reranker() -> None:
    _get_reranker()


def _rerank(question: str, candidates: list[qm.ScoredPoint], top_k: int) -> list[qm.ScoredPoint] | None:
    """Reordena con un cross-encoder y corta por relevancia relativa. None si no hay reranker."""
    model = _get_reranker()
    if model is None or not candidates:
        return None
    docs = [
        config.chunk_header((pt.payload or {}).get("path", ""), (pt.payload or {}).get("section"))
        + (pt.payload or {}).get("text", "")
        for pt in candidates
    ]
    logits = list(model.rerank(question, docs, batch_size=len(docs)))
    probs = [1.0 / (1.0 + math.exp(-x)) for x in logits]
    ranked = sorted(zip(probs, candidates), key=lambda x: x[0], reverse=True)
    best = ranked[0][0]
    return [
        qm.ScoredPoint(id=pt.id, version=0, score=prob, payload=pt.payload, vector=None)
        for prob, pt in ranked[:top_k]
        if prob >= best * config.RERANK_KEEP_RATIO
    ]


def _rrf_cutoff(sorted_ranks: list[tuple], points_by_id: dict, top_k: int) -> list[qm.ScoredPoint]:
    """Selección por RRF cuando no hay reranker: conserva fragmentos con >= 65% del mejor
    puntaje y se detiene ante una caída abrupta (> 25%)."""
    max_score = sorted_ranks[0][1]
    prev_score = max_score
    combined: list[qm.ScoredPoint] = []
    for i, (pid, rrf_sc) in enumerate(sorted_ranks[:top_k]):
        ratio = rrf_sc / max_score if max_score > 0 else 1.0
        gap = (prev_score - rrf_sc) / prev_score if prev_score > 0 else 0.0
        if i > 0 and (ratio < 0.65 or gap >= 0.25):
            break
        orig = points_by_id[pid]
        combined.append(qm.ScoredPoint(id=orig.id, version=0, score=rrf_sc, payload=orig.payload, vector=None))
        prev_score = rrf_sc
    return combined


# Máximo de caracteres del índice de un documento que se envía al LLM.
OUTLINE_CHARS = 3000
_OUTLINE_HEADING = re.compile(r"^(=+|#{1,6})\s+(.+?)\s*#*\s*$")


def _target_file(corpus: _Corpus, question: str) -> str | None:
    """Archivo nombrado explícitamente en la pregunta ("¿qué dice arch-description?").
    Exige que todos los tokens del nombre aparezcan y que el nombre sea distintivo."""
    q_tokens = set(_extract_tokens(question))
    matches = [
        path
        for path, stem in corpus.stems.items()
        if stem and sum(len(t) for t in stem) >= 6 and stem <= q_tokens
    ]
    if not matches:
        return None
    # Preferir el nombre más específico; si sigue ambiguo entre varios archivos, no elegir.
    best = max(len(corpus.stems[m]) for m in matches)
    matches = [m for m in matches if len(corpus.stems[m]) == best]
    return matches[0] if len(matches) == 1 else None


def _outline_point(corpus: _Corpus, path: str) -> qm.ScoredPoint | None:
    """Índice (títulos en orden) de un archivo, reconstruido desde sus fragmentos."""
    if PurePosixPath(path).suffix.lower() not in (".typ", ".md", ".mdx"):
        return None
    headings: dict[int, tuple[int, str]] = {}
    for p, _ in corpus.points:
        payload = p.payload or {}
        if payload.get("path") != path or not payload.get("start_line"):
            continue
        for offset, line in enumerate(payload.get("text", "").splitlines()):
            m = _OUTLINE_HEADING.match(line)
            if m:
                headings[payload["start_line"] + offset] = (len(m.group(1)), m.group(2))
    if not headings:
        return None
    ordered = [headings[k] for k in sorted(headings)]
    max_level = max(level for level, _ in ordered)
    while True:
        lines = [f"{'  ' * (level - 1)}- {title}" for level, title in ordered if level <= max_level]
        text = "\n".join(lines)
        if len(text) <= OUTLINE_CHARS or max_level == 1:
            break
        max_level -= 1
    return qm.ScoredPoint(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, path)),
        version=0,
        score=1.0,
        payload={"path": path, "outline": True, "text": f"Índice del documento:\n{text[:OUTLINE_CHARS]}"},
        vector=None,
    )


def search(
    qdrant: QdrantClient,
    llm: ollama.Client,
    question: str,
    *,
    top_k: int,
    source: str | None,
    history: list[dict] | None = None,
    use_reranker: bool = True,
) -> list[qm.ScoredPoint]:
    """
    Búsqueda híbrida:
    1. Router LLM: reescribe preguntas de seguimiento, elige carpetas y expande la consulta.
    2. Búsqueda densa (Qdrant) + BM25 en memoria, fusionadas con RRF (k=15).
    3. Reranker cross-encoder sobre los mejores candidatos (o corte por RRF si no hay reranker).
    """
    corpus = get_corpus(qdrant)
    # Si la pregunta nombra un archivo concreto, se busca sólo dentro de él y se añade su índice.
    target = _target_file(corpus, question)
    conditions = []
    if source:
        conditions.append(qm.FieldCondition(key="source", match=qm.MatchValue(value=source)))
    if target:
        conditions.append(qm.FieldCondition(key="path", match=qm.MatchValue(value=target)))
    base_filter = qm.Filter(must=conditions) if conditions else None

    def has_evidence(query: str) -> bool:
        # Guarda cero-match: sin términos presentes en el corpus y con similitud
        # vectorial bajo el piso de ruido, no hay nada que recuperar.
        if any(corpus.df.get(t, 0) > 0 for t in _extract_tokens(query)):
            return True
        res = qdrant.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=embed_query(llm, query),
            query_filter=base_filter,
            limit=1,
            with_payload=False,
        ).points
        return bool(res) and res[0].score >= config.NOISE_FLOOR

    # Con historial hay que reescribir la pregunta antes de juzgar si hay evidencia
    # ("¿y la segunda?" no tiene términos propios); sin historial, la guarda va primero
    # para ahorrarse el llamado al LLM.
    if history:
        route = _route(llm, question, corpus.folders, history)
        if not has_evidence(route.standalone_question):
            return []
    else:
        if not has_evidence(question):
            return []
        route = _route(llm, question, corpus.folders, history)

    candidate_depth = max(top_k * 5, 40)
    vector_results = qdrant.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=embed_query(llm, route.expanded_query),
        query_filter=base_filter,
        limit=candidate_depth,
        with_payload=True,
    ).points
    top_vector_score = vector_results[0].score if vector_results else 0.0

    bm25_results = _bm25(corpus, route.expanded_query, source, target, route.folders, candidate_depth)
    if not bm25_results and top_vector_score < config.NOISE_FLOOR:
        return []

    # Fusión RRF (k=15) con impulso a las carpetas elegidas por el router
    rrf_scores: dict = {}
    points_by_id: dict = {}
    for results in (vector_results, bm25_results):
        for rank, pt in enumerate(results):
            boost = 1.6 if (route.folders and _in_folders(pt.payload or {}, route.folders)) else 1.0
            rrf_scores[pt.id] = rrf_scores.get(pt.id, 0.0) + boost / (15.0 + rank + 1.0)
            points_by_id[pt.id] = pt

    sorted_ranks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    if not sorted_ranks:
        return []

    results = None
    if use_reranker:
        candidates = [points_by_id[pid] for pid, _ in sorted_ranks[: config.RERANK_CANDIDATES]]
        # Con la consulta expandida: la pregunta literal ("calificación") no se parece al
        # texto fuente ("Evaluación • 2 Parciales (20%)"); los términos del router cierran esa brecha.
        results = _rerank(route.expanded_query, candidates, top_k)
    if results is None:
        results = _rrf_cutoff(sorted_ranks, points_by_id, top_k)

    if target and (outline := _outline_point(corpus, target)):
        results = [outline] + results
    return results


def location_detail(p: dict) -> str:
    if p.get("outline"):
        return "índice"
    page, page_end = p.get("page"), p.get("page_end")
    if page:
        return f"pág. {page}-{page_end}" if page_end and page_end != page else f"pág. {page}"
    if p.get("start_line"):
        return f"L{p['start_line']}-{p['end_line']}"
    return ""


def _location(p: dict) -> str:
    detail = location_detail(p)
    return f"{p.get('path', '?')} ({detail})" if detail else p.get("path", "?")


def build_context(points: list[qm.ScoredPoint]) -> str:
    blocks = []
    for i, pt in enumerate(points, start=1):
        p = pt.payload or {}
        section = f"\nSección: {p['section']}" if p.get("section") else ""
        blocks.append(f"[{i}] {_location(p)}{section}\n{p.get('text', '')}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, points: list[qm.ScoredPoint], history: list[dict] | None = None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in (history or [])[-HISTORY_MESSAGES:]:
        role, content = m.get("role"), m.get("content") or ""
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:HISTORY_MESSAGE_CHARS]})
    messages.append({
        "role": "user",
        "content": f"Contexto de documentos recuperados:\n\n{build_context(points)}\n\nPregunta actual: {question}",
    })
    return messages


def stream_answer(llm: ollama.Client, messages: list[dict], model: str = config.LLM_MODEL):
    for part in llm.chat(
        model=model,
        messages=messages,
        stream=True,
        options=llm_options(),
        keep_alive=config.OLLAMA_KEEP_ALIVE,
    ):
        yield part.message.content


def answer(qdrant: QdrantClient, llm: ollama.Client, question: str, args: argparse.Namespace) -> None:
    points = search(qdrant, llm, question, top_k=args.k, source=args.source, use_reranker=not args.sin_reranker)
    if not points:
        print("No encontré fragmentos relevantes (¿ya corriste el indexador?).")
        return

    if args.solo_buscar:
        for i, pt in enumerate(points, start=1):
            p = pt.payload or {}
            print(f"[{i}] ({pt.score:.3f}) {_location(p)}\n{p.get('text', '')}\n{'-' * 60}")
        return

    for token in stream_answer(llm, build_messages(question, points), model=args.model):
        print(token, end="", flush=True)

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
    parser.add_argument("--sin-reranker", action="store_true", help="Desactivar el reranker (sólo RRF)")
    args = parser.parse_args()

    qdrant = QdrantClient(url=config.QDRANT_URL, prefer_grpc=True, check_compatibility=False)
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
