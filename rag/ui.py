"""
Interfaz gráfica web minimalista y elegante para el RAG local con Gradio.
Enfoque limpio: pantalla completa, estilo ChatGPT / Claude, sin barras laterales complejas.

Uso:
    uv run python -m rag.ui
"""

from __future__ import annotations

import gradio as gr
import ollama
from qdrant_client import QdrantClient

from rag import config
from rag.query import SYSTEM_PROMPT, _location, search


def rag_chat_stream(message: str, history: list):
    query_text = message.get("text", "") if isinstance(message, dict) else str(message)
    if not query_text.strip():
        yield "Por favor ingresa una pregunta."
        return

    qdrant = QdrantClient(url=config.QDRANT_URL, prefer_grpc=True, check_compatibility=False)
    llm = ollama.Client(host=config.OLLAMA_URL)

    points = search(qdrant, llm, query_text, top_k=7, source=None)

    if not points:
        yield "No encontré información relevante en los documentos indexados."
        return

    context_blocks = []
    sources_summary = []
    for i, pt in enumerate(points, start=1):
        p = pt.payload or {}
        loc = _location(p)
        score = pt.score
        context_blocks.append(f"[{i}] {loc}\n{p.get('text', '')}")
        sources_summary.append(f"**[{i}]** `{loc}` *(similitud: {score:.3f})*")

    context_str = "\n\n---\n\n".join(context_blocks)
    sources_text = "\n\n<details><summary><b>📚 Fuentes consultadas (" + str(len(sources_summary)) + ")</b></summary>\n\n" + "\n\n".join(sources_summary) + "\n</details>"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Contexto:\n\n{context_str}\n\nPregunta: {query_text}"},
    ]

    accumulated_text = ""
    for part in llm.chat(model=config.LLM_MODEL, messages=messages, stream=True):
        accumulated_text += part.message.content
        yield accumulated_text

    # Al terminar, adjunta las fuentes en un desplegable colapsable elegante
    yield accumulated_text + "\n\n---\n" + sources_text


CSS = """
footer {visibility: hidden !important;}
.gradio-container {max-width: 900px !important; margin: 0 auto !important;}
#chat-header {text-align: center; margin-top: 1.5rem; margin-bottom: 0.5rem;}
#chat-header h1 {font-size: 2rem; font-weight: 700; color: #1e293b; margin-bottom: 0.2rem;}
#chat-header p {color: #64748b; font-size: 0.95rem;}
"""

def build_app() -> gr.Blocks:
    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    )

    with gr.Blocks(title="Homelab RAG") as demo:
        gr.HTML(
            """
            <div id="chat-header">
                <h1>📚 Asistente de Documentos</h1>
                <p>Consulta tus documentos de forma local</p>
            </div>
            """
        )

        gr.ChatInterface(
            fn=rag_chat_stream,
            fill_height=True,
        )

    return demo


def main():
    demo = build_app()
    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    )
    demo.launch(theme=custom_theme, css=CSS, server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
