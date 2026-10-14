#!/usr/bin/env bash
# Script para sincronizar tus documentos locales con el servidor homelab e indexar automáticamente.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$DIR/.env" ]; then
  set -a
  source "$DIR/.env" 2>/dev/null || . "$DIR/.env"
  set +a
fi

SOURCE_DIR="${1:-$HOME/Documentos/documentosRag}"
REMOTE_HOST="${RAG_REMOTE_HOST:-192.168.80.85}"
REMOTE_USER="${RAG_REMOTE_USER:-homelab}"
REMOTE_DEST="${RAG_REMOTE_DEST:-~/rag-data/}"
REMOTE_REPO="${RAG_REMOTE_REPO:-~/Code/RAG}"

echo "🔄 1. Sincronizando '$SOURCE_DIR' hacia el homelab..."
rsync -avzP \
  -e "ssh -F /dev/null -o BatchMode=yes" \
  --exclude="*.mp4" \
  --exclude="*.mov" \
  --exclude="*.mkv" \
  --exclude="*.avi" \
  --delete \
  "$SOURCE_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DEST"

echo ""
echo "⚡ 2. Ejecutando indexación incremental en el servidor homelab..."
ssh -F /dev/null -o BatchMode=yes "$REMOTE_USER@$REMOTE_HOST" \
  "cd $REMOTE_REPO && git pull origin main >/dev/null 2>&1 || true; uv run cocoindex update rag.indexer"


echo ""
echo "✅ ¡Listo! Documentos sincronizados e indexados en Qdrant."
