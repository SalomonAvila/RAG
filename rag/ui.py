"""
Interfaz gráfica web para el RAG local con Gradio.

Uso:
    uv run python -m rag.ui
"""

from __future__ import annotations

import gradio as gr
import ollama
from qdrant_client import QdrantClient

from rag import config
from rag.query import SYSTEM_PROMPT, _location, search


def get_available_sources(qdrant: QdrantClient) -> list[str]:
    """Obtiene la lista de carpetas / repositorios indexados en Qdrant."""
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


def get_available_models(llm: ollama.Client) -> list[str]:
    """Lista los modelos LLM disponibles en Ollama."""
    try:
        models = [m.model for m in llm.list().models if not m.model.startswith("nomic-embed")]
        return models if models else [config.LLM_MODEL]
    except Exception:
        return [config.LLM_MODEL]


def rag_chat(
    message: str,
    history: list[dict],
    source_filter: str,
    selected_model: str,
    top_k: int,
):
    if not message.strip():
        yield "", ""
        return

    qdrant = QdrantClient(url=config.QDRANT_URL, prefer_grpc=True)
    llm = ollama.Client(host=config.OLLAMA_URL)

    source = None if source_filter == "(Todos)" else source_filter
    points = search(qdrant, llm, message, top_k=top_k, source=source)

    if not points:
        yield "No se encontraron fragmentos relevantes en la base de datos.", "Sin fuentes"
        return

    # Construir el contexto
    context_blocks = []
    sources_summary = []
    for i, pt in enumerate(points, start=1):
        p = pt.payload or {}
        loc = _location(p)
        score = pt.score
        context_blocks.append(f"[{i}] {loc}\n{p.get('text', '')}")
        sources_summary.append(f"**[{i}]** `{loc}` *(similitud: {score:.3f})*")

    context_str = "\n\n---\n\n".join(context_blocks)
    sources_panel = "\n\n".join(sources_summary)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Contexto:\n\n{context_str}\n\nPregunta: {message}"},
    ]

    # Generación en streaming para la UI
    accumulated_text = ""
    for part in llm.chat(model=selected_model, messages=messages, stream=True):
        accumulated_text += part.message.content
        yield accumulated_text, sources_panel


def build_app() -> gr.Blocks:
    qdrant = QdrantClient(url=config.QDRANT_URL, prefer_grpc=True)
    llm = ollama.Client(host=config.OLLAMA_URL)

    available_sources = get_available_sources(qdrant)
    available_models = get_available_models(llm)
    default_model = config.LLM_MODEL if config.LLM_MODEL in available_models else available_models[0]

    custom_theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
    )

    with gr.Blocks(theme=custom_theme, title="RAG Local Homelab") as demo:
        gr.Markdown(
            """
            # 🧠 RAG Local Homelab
            *Consulta inteligente sobre tus repositorios, apuntes y documentos con privacidad total.*
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversación",
                    height=520,
                    show_copy_button=True,
                    render_markdown=True,
                )
                with gr.Row():
                    msg = gr.Textbox(
                        label="Escribe tu pregunta",
                        placeholder="Ejemplo: ¿Cuáles son los objetivos del trabajo de grado?",
                        scale=9,
                        autofocus=True,
                    )
                    submit_btn = gr.Button("Enviar", variant="primary", scale=1)
                clear_btn = gr.ClearButton([msg, chatbot], value="Limpiar chat")

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Parámetros de Búsqueda")
                source_dropdown = gr.Dropdown(
                    label="Filtrar por Fuente / Materia",
                    choices=available_sources,
                    value=available_sources[0],
                )
                model_dropdown = gr.Dropdown(
                    label="Modelo LLM (Ollama)",
                    choices=available_models,
                    value=default_model,
                )
                top_k_slider = gr.Slider(
                    label="Fragmentos a recuperar (Top K)",
                    minimum=1,
                    maximum=15,
                    value=6,
                    step=1,
                )
                gr.Markdown("### 📚 Fuentes Consultadas")
                sources_box = gr.Markdown(
                    label="Fuentes del último mensaje",
                    value="*Las fuentes citadas aparecerán aquí tras cada respuesta.*",
                )

        def user_send(user_message, history):
            return "", history + [{"role": "user", "content": user_message}]

        def bot_reply(history, source_f, model_s, top_k):
            user_message = history[-1]["content"]
            history.append({"role": "assistant", "content": ""})
            for partial_text, sources_text in rag_chat(user_message, history, source_f, model_s, top_k):
                history[-1]["content"] = partial_text
                yield history, sources_text

        # Eventos
        msg.submit(
            user_send, [msg, chatbot], [msg, chatbot], queue=False
        ).then(
            bot_reply, [chatbot, source_dropdown, model_dropdown, top_k_slider], [chatbot, sources_box]
        )

        submit_btn.click(
            user_send, [msg, chatbot], [msg, chatbot], queue=False
        ).then(
            bot_reply, [chatbot, source_dropdown, model_dropdown, top_k_slider], [chatbot, sources_box]
        )

    return demo


def main():
    demo = build_app()
    # server_name="0.0.0.0" permite acceder desde cualquier equipo de tu red local
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
