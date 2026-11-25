#!/usr/bin/env bash
# Sincroniza una carpeta local de documentos con el servidor y lanza allí la indexación incremental.
# Uso: ./sync.sh [carpeta]   (por defecto, RAG_LOCAL_DIR de .env)
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$DIR/.env" ]; then
  set -a
  source "$DIR/.env" 2>/dev/null || . "$DIR/.env"
  set +a
fi

SOURCE_DIR="${1:-${RAG_LOCAL_DIR:-}}"
if [ -z "$SOURCE_DIR" ] || [ ! -d "$SOURCE_DIR" ]; then
  echo "❌ Indica la carpeta a sincronizar: ./sync.sh <carpeta>, o define RAG_LOCAL_DIR en .env." >&2
  exit 1
fi
if [ -z "${RAG_REMOTE_HOST:-}" ]; then
  echo "❌ Define RAG_REMOTE_HOST en .env (copia .env.example como punto de partida)." >&2
  exit 1
fi
REMOTE_HOST="$RAG_REMOTE_HOST"
REMOTE_USER="${RAG_REMOTE_USER:-$USER}"
# Rutas en el servidor (se expanden allá, por eso van entre comillas simples en .env)
REMOTE_DEST="${RAG_REMOTE_DEST:-~/rag-data/}"
REMOTE_REPO="${RAG_REMOTE_REPO:-~/RAG}"
REMOTE_UPDATE=""
[ "${RAG_AUTO_UPDATE:-0}" = "1" ] && REMOTE_UPDATE="git pull --ff-only >/dev/null 2>&1 || true; "

echo "🔄 1. Sincronizando '$SOURCE_DIR' hacia $REMOTE_HOST..."
rsync -avzP \
  -e "ssh -F /dev/null -o BatchMode=yes" \
  --exclude="*.mp4" \
  --exclude="*.mov" \
  --exclude="*.mkv" \
  --exclude="*.avi" \
  --delete \
  "$SOURCE_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DEST"

echo ""
if [ "${RAG_LIVE_INDEXER:-0}" = "1" ]; then
  # En la torre corre el contenedor rag-indexer en modo live (homelab/services/rag): ya vio
  # los cambios. Lanzar otro "cocoindex update" competiría con él por el estado (.cocoindex_db).
  echo "⚡ 2. El indexador en vivo de la torre detecta los cambios solo (RAG_LIVE_INDEXER=1)."
else
  echo "⚡ 2. Ejecutando la indexación incremental en el servidor..."
  ssh -F /dev/null -o BatchMode=yes "$REMOTE_USER@$REMOTE_HOST" \
    "cd $REMOTE_REPO && ${REMOTE_UPDATE}uv run cocoindex update rag.indexer"
fi


echo ""
echo "✅ ¡Listo! Documentos sincronizados e indexados en Qdrant."
