#!/usr/bin/env bash
# Inicia la interfaz web en tu portátil abriendo un túnel SSH hacia el homelab.
# Esto canaliza las consultas a Qdrant y Ollama en la torre sin consumir GPU ni RAM local.
set -e

REMOTE_HOST="${RAG_REMOTE_HOST:-192.168.80.85}"
REMOTE_USER="${RAG_REMOTE_USER:-homelab}"
PORT_QDRANT=6334
PORT_OLLAMA=11434

echo "🔒 1. Estableciendo túnel SSH seguro hacia $REMOTE_USER@$REMOTE_HOST..."
# -L: mapea localhost:6334 -> torre:6334 (Qdrant) y localhost:11434 -> torre:11434 (Ollama)
# -N -f: corre el túnel en segundo plano
ssh -F /dev/null -o BatchMode=yes -f -N \
  -L "$PORT_QDRANT:localhost:$PORT_QDRANT" \
  -L "$PORT_OLLAMA:localhost:$PORT_OLLAMA" \
  "$REMOTE_USER@$REMOTE_HOST"

# Capturar el PID para cerrar el túnel al salir
TUNNEL_PID=$(pgrep -f "ssh.*-L $PORT_QDRANT:localhost:$PORT_QDRANT.*$REMOTE_HOST" | tail -1)
trap 'echo ""; echo "🔌 Cerrando túnel SSH..."; kill $TUNNEL_PID 2>/dev/null || true' EXIT

echo "🚀 2. Iniciando Interfaz Gráfica Local (Gradio)..."
echo "🌐 Conectado de forma transparente al motor de GPU en $REMOTE_HOST."
echo "👉 Abre en tu navegador: http://localhost:7860"
echo ""

uv run python -m rag.ui
