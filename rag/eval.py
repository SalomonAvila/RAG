"""
Evaluación de la recuperación: mide si los fragmentos esperados aparecen en los resultados.

Cada caso de eval/casos.json tiene una pregunta y uno o más fragmentos esperados
("ruta" = parte de la ruta del archivo; "pagina" opcional para PDFs). Un caso acierta si
algún resultado contiene algún esperado.

Uso:
    uv run python -m rag.eval                  # con reranker
    uv run python -m rag.eval --sin-reranker   # sólo RRF, para comparar
    uv run python -m rag.eval -v               # muestra lo recuperado en cada caso
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ollama
from qdrant_client import QdrantClient

from rag import config
from rag.query import _location, search

DEFAULT_CASES = config.REPO_ROOT / "eval" / "casos.json"


def _matches(payload: dict, expected: dict) -> bool:
    if expected["ruta"] not in payload.get("path", ""):
        return False
    page = expected.get("pagina")
    if page is None:
        return True
    start = payload.get("page")
    end = payload.get("page_end") or start
    return start is not None and start <= page <= end


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa la recuperación del RAG.")
    parser.add_argument("casos", nargs="?", type=Path, default=DEFAULT_CASES)
    parser.add_argument("-k", type=int, default=7, help="Fragmentos a recuperar (default: 7, como la interfaz)")
    parser.add_argument("--sin-reranker", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cases = json.loads(args.casos.read_text())
    qdrant = QdrantClient(url=config.QDRANT_URL, prefer_grpc=True, check_compatibility=False)
    llm = ollama.Client(host=config.OLLAMA_URL)

    hits, reciprocal_ranks, n_results, latencies = 0, [], [], []
    for case in cases:
        t0 = time.time()
        points = search(qdrant, llm, case["pregunta"], top_k=args.k, source=None, use_reranker=not args.sin_reranker)
        latencies.append(time.time() - t0)
        n_results.append(len(points))

        rank = next(
            (i for i, pt in enumerate(points, start=1) if any(_matches(pt.payload or {}, e) for e in case["esperado"])),
            None,
        )
        hits += rank is not None
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        mark = f"✅ #{rank}" if rank else "❌"
        print(f"{mark:6} {latencies[-1]:5.1f}s  {len(points)} frag.  {case['pregunta']}")
        if args.verbose or not rank:
            for i, pt in enumerate(points, start=1):
                print(f"         [{i}] ({pt.score:.3f}) {_location(pt.payload or {})}")

    n = len(cases)
    print(
        f"\nAciertos: {hits}/{n} ({hits / n:.0%})  ·  MRR: {sum(reciprocal_ranks) / n:.2f}  ·  "
        f"fragmentos promedio: {sum(n_results) / n:.1f}  ·  latencia promedio: {sum(latencies) / n:.1f}s"
        f"  ·  reranker: {'no' if args.sin_reranker else config.RERANK_MODEL or 'no'}"
    )


if __name__ == "__main__":
    main()
