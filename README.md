# RAG Local en Homelab

Sistema RAG 100% privado y local corriendo en servidor torre homelab (`192.168.80.85` con GPU GTX 1660 Super) para consultar código fuente (repositorios) y documentos (apuntes, PDFs, Markdown).

## Componentes

- **Indexación incremental:** [CocoIndex](https://cocoindex.io/) (detecta cambios en archivos mediante fingerprints y sólo re-calcula embeddings de los deltas).
- **Base vectorial:** [Qdrant](https://qdrant.tech/) corriendo vía Docker con volumen persistente en disco y dashboard web en `http://<ip>:6333/dashboard`.
- **Embeddings:** `nomic-embed-text` vía Ollama (768 dimensiones).
- **Generación / LLM:** `qwen2.5:3b` vía Ollama (rápido y adaptado a 6GB VRAM).

---

## Estructura del Proyecto

```text
.
├── docker-compose.yml     # Contenedor Qdrant con almacenamiento persistente
├── pyproject.toml         # Dependencias (cocoindex, qdrant-client, ollama, pypdf...)
├── .env.example           # Variables de entorno opcionales
├── .gitignore             # Ignora carpetas de datos, bases de datos y .env
└── rag/
    ├── __init__.py
    ├── config.py          # Parámetros centralizados (rutas, modelos, chunking)
    ├── indexer.py         # Pipeline incremental de CocoIndex
    └── query.py           # CLI de consulta interactiva / por comando
```

---

## Puesta en Marcha en el Servidor Homelab

### 1. Clonar o sincronizar repositorio
```bash
cd ~/Code/RAG
git pull origin main
```

### 2. Levantar la base de datos vectorial (Qdrant)
```bash
docker compose up -d
```
Puedes abrir la interfaz web de Qdrant en cualquier navegador de tu red local:
`http://192.168.80.85:6333/dashboard`

### 3. Sincronizar dependencias de Python
Con `uv` instalado:
```bash
uv sync
```

### 4. Poner tus archivos o repositorios
Por defecto, el sistema busca documentos en `~/rag-data` (puedes crear carpetas con nombres de materias o clonar tus repos ahí dentro):
```bash
mkdir -p ~/rag-data/mis-repos
mkdir -p ~/rag-data/clases
```

### 5. Ejecutar la indexación
- **Indexación puntual (catch-up):**
  ```bash
  uv run cocoindex update rag.indexer
  ```
  *CocoIndex procesará los archivos nuevos o modificados; si un archivo no cambió, no se recalcula nada (0.0s).*

- **Modo en vivo (Live watch):**
  ```bash
  uv run cocoindex update -L rag.indexer
  ```
  *Queda observando la carpeta y re-indexa automáticamente cualquier guardado de archivo.*

### 6. Consultar el RAG

- **Consulta directa por argumento:**
  ```bash
  uv run python -m rag.query "¿Cuándo es el examen parcial?"
  ```

- **Filtrar por repositorio o materia:**
  ```bash
  uv run python -m rag.query -s mis-repos "¿Cómo funciona la autenticación?"
  ```

- **Modo interactivo (chat en terminal):**
  ```bash
  uv run python -m rag.query
  ```

- **Sólo buscar fragmentos sin llamar al LLM:**
  ```bash
  uv run python -m rag.query --solo-buscar "algoritmo de ordenamiento"
  ```
