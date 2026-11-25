#!/usr/bin/env bash
# Inicia la interfaz web en tu equipo abriendo un túnel SSH hacia el servidor (RAG_REMOTE_HOST),
# donde corren Qdrant y Ollama, sin consumir GPU ni RAM local.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

if [ -f "$DIR/.env" ]; then
  set -a
  source "$DIR/.env" 2>/dev/null || . "$DIR/.env"
  set +a
fi

if [ "${RAG_AUTO_UPDATE:-0}" = "1" ] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "🔄 0. Actualizando el repositorio (RAG_AUTO_UPDATE=1)..."
  git pull --ff-only || echo "⚠️ No se pudo actualizar (¿sin internet o cambios locales?), continuando..."
fi

if [ -z "${RAG_REMOTE_HOST:-}" ]; then
  echo "❌ Define RAG_REMOTE_HOST en .env (copia .env.example como punto de partida)." >&2
  exit 1
fi
REMOTE_HOST="$RAG_REMOTE_HOST"
REMOTE_USER="${RAG_REMOTE_USER:-$USER}"
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

echo "🚀 2. Iniciando Aplicación Local (Tu Archivo)..."
echo "🌐 Conectado a Qdrant y Ollama en $REMOTE_HOST."
echo "👉 Abre en tu navegador: http://localhost:7860"
echo ""

uv run python -m rag.server
