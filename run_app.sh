#!/usr/bin/env bash
# Abre "Tu Archivo" como aplicación de escritorio (GTK/WebKitGTK), conectada por túnel SSH
# al servidor donde corren Qdrant y Ollama (RAG_REMOTE_HOST en .env).
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

# Mantener sincronizado el icono del sistema con el que esté en el repo
if [ -f "$DIR/rag/static/logo.png" ]; then
  cp "$DIR/rag/static/logo.png" "$HOME/.local/share/icons/tu-archivo.png" 2>/dev/null || true
  mkdir -p "$HOME/.local/share/icons/hicolor/512x512/apps/"
  cp "$DIR/rag/static/logo.png" "$HOME/.local/share/icons/hicolor/512x512/apps/tu-archivo.png" 2>/dev/null || true
fi

if [ -z "${RAG_REMOTE_HOST:-}" ]; then
  echo "❌ Define RAG_REMOTE_HOST en .env (copia .env.example como punto de partida)." >&2
  exit 1
fi
REMOTE_HOST="$RAG_REMOTE_HOST"
REMOTE_USER="${RAG_REMOTE_USER:-$USER}"
PORT_QDRANT=6334
PORT_OLLAMA=11434

echo "🔒 1. Estableciendo túnel SSH hacia $REMOTE_USER@$REMOTE_HOST..."
ssh -F /dev/null -o BatchMode=yes -f -N \
  -L "$PORT_QDRANT:localhost:$PORT_QDRANT" \
  -L "$PORT_OLLAMA:localhost:$PORT_OLLAMA" \
  "$REMOTE_USER@$REMOTE_HOST"

TUNNEL_PID=$(pgrep -f "ssh.*-L $PORT_QDRANT:localhost:$PORT_QDRANT.*$REMOTE_HOST" | tail -1)

# Asegurar que no haya servidores antiguos en el puerto 7860
fuser -k 7860/tcp 2>/dev/null || true

# Iniciar servidor backend en segundo plano
uv run python -m rag.server &
SERVER_PID=$!

cleanup() {
  echo ""
  echo "🔌 Cerrando aplicación y túnel..."
  kill $SERVER_PID 2>/dev/null || true
  fuser -k 7860/tcp 2>/dev/null || true
  kill $TUNNEL_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Esperar a que el backend esté listo
until curl -s http://localhost:7860/ >/dev/null 2>&1; do
  sleep 0.2
done

echo "🚀 Abriendo ventana de la aplicación..."
python3 -m rag.app_window
