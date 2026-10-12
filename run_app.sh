#!/usr/bin/env bash
# Inicia el motor RAG y abre "Tu Archivo" como aplicación nativa de escritorio en Fedora GNOME.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "🔄 0. Verificando actualizaciones en GitHub..."
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git pull origin main || echo "⚠️ Advertencia: No se pudo hacer git pull, continuando..."
fi

# Mantener sincronizado el icono del sistema con el que esté en el repo
if [ -f "$DIR/rag/static/logo.png" ]; then
  cp "$DIR/rag/static/logo.png" "$HOME/.local/share/icons/tu-archivo.png" 2>/dev/null || true
  mkdir -p "$HOME/.local/share/icons/hicolor/512x512/apps/"
  cp "$DIR/rag/static/logo.png" "$HOME/.local/share/icons/hicolor/512x512/apps/tu-archivo.png" 2>/dev/null || true
fi


REMOTE_HOST="${RAG_REMOTE_HOST:-192.168.80.85}"
REMOTE_USER="${RAG_REMOTE_USER:-homelab}"
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
