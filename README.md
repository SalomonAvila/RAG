# RAG Local con CocoIndex, Qdrant y Ollama

Sistema RAG 100% privado y local para consultar repositorios de código y documentos personales (apuntes, Markdown, PDFs), diseñado para correr en servidores domésticos (homelab) o estaciones locales de trabajo con aceleración por GPU.

## Características

- **Indexación incremental eficiente:** Integra [CocoIndex](https://cocoindex.io/) para calcular deltas de cambios y evitar reprocesar documentos no modificados.
- **Base de datos vectorial:** [Qdrant](https://qdrant.tech/) sobre Docker con volumen persistente en disco y panel de control web.
- **Embeddings locales:** `nomic-embed-text` vía Ollama (modelo ligero y multilingüe).
- **Generación fundamentada:** Modelos locales en Ollama (por defecto `qwen2.5:3b`, configurable a cualquier modelo instalado).
- **Citas automáticas:** Respuestas en streaming referenciando archivo, sección o número de línea.

---

## Estructura del Proyecto

```text
.
├── docker-compose.yml     # Servicio Qdrant con almacenamiento persistente
├── pyproject.toml         # Definición del entorno y dependencias del proyecto
├── .env.example           # Plantilla de variables de configuración
├── .gitignore             # Exclusión de almacenamiento local, cachés y variables
└── rag/
    ├── __init__.py
    ├── config.py          # Configuración y patrones de filtrado
    ├── indexer.py         # Pipeline de indexación con CocoIndex
    └── query.py           # CLI de consulta interactiva / por argumentos
```

---

## Requisitos Previos

1. **Docker y Docker Compose** instalados y activos.
2. **Ollama** corriendo con los modelos descargados:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen2.5:3b
   ```
3. Gestor de paquetes **[uv](https://github.com/astral-sh/uv)** (o Python 3.11+ con `venv`).

---

## Puesta en Marcha

### 1. Clonar el repositorio
```bash
git clone https://github.com/SalomonAvila/RAG.git
cd RAG
```

### 2. Configuración opcional
Si deseas cambiar rutas o modelos, copia la plantilla `.env.example`:
```bash
cp .env.example .env
```
Por defecto, el indexador busca archivos en `~/rag-data`.

### 3. Iniciar la base vectorial (Qdrant)
```bash
docker compose up -d
```
El panel de administración web quedará accesible en:
`http://localhost:6333/dashboard` *(o `http://<IP-DE-TU-SERVIDOR>:6333/dashboard` si accedes por red local)*.

### 4. Instalar dependencias
```bash
uv sync
```

### 5. Colocar documentos a indexar
Coloca tus carpetas de apuntes o clona repositorios dentro de la ruta configurada (por defecto `~/rag-data`):
```bash
mkdir -p ~/rag-data/mis-repos
mkdir -p ~/rag-data/documentos
```

### 6. Ejecutar la indexación

- **Sincronización puntual (one-shot):**
  ```bash
  uv run cocoindex update rag.indexer
  ```
  *CocoIndex detectará qué archivos son nuevos o han cambiado y procesará únicamente esos.*

- **Modo en tiempo real (Live Watch):**
  ```bash
  uv run cocoindex update -L rag.indexer
  ```
  *Permanece en ejecución vigilando el directorio y sincronizando automáticamente al guardar cambios.*

---

## Consultar el RAG

- **Pregunta directa desde la terminal:**
  ```bash
  uv run python -m rag.query "¿Cómo funciona el sistema de autenticación?"
  ```

- **Filtrar por subcarpeta o repositorio específico:**
  ```bash
  uv run python -m rag.query -s mis-repos "¿Dónde se define la base de datos?"
  ```

- **Modo interactivo (conversacional continuo):**
  ```bash
  uv run python -m rag.query
  ```

- **Inspeccionar fragmentos recuperados sin invocar al LLM:**
  ```bash
  uv run python -m rag.query --solo-buscar "algoritmo de búsqueda"
  ```
