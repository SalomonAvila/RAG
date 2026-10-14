# RAG Local y Privado con CocoIndex, Qdrant y Ollama

Sistema RAG (Retrieval-Augmented Generation) 100% privado, local y modular para consultar repositorios de código, notas y documentos personales (Word, Markdown, PDFs, Typst), optimizado para estaciones de trabajo o servidores domésticos (*homelab*) con aceleración por GPU.

---

## Características Principales

- **Búsqueda Híbrida Inteligente (Dense Vector + BM25 con RRF):**
  Combina la búsqueda vectorial semántica densa (`nomic-embed-text`) con búsqueda léxica BM25 (calibrada para palabras clave exactas, código, nombres propios y rutas de archivo) mediante *Reciprocal Rank Fusion* (RRF, $k=15$).
- **Enrutamiento y Expansión Semántica Universal:**
  El LLM analiza dinámicamente la consulta frente al catálogo de documentos disponibles, enrutando a los archivos relevantes y enriqueciendo la búsqueda con términos específicos, componentes y sinónimos formales o académicos (por ejemplo, si se consulta por "calificación", mapea automáticamente a "parciales, quices, talleres, proyectos, notas, ponderación").
- **Filtro de Ruido Cero-Match (Zero-Match Guard):**
  Si una consulta trata sobre una entidad o tema que no existe en tus documentos (presencia léxica nula) y su similitud semántica pura no supera el piso de ruido de fondo ($< 0.64$), el sistema responde de inmediato sin inventar respuestas ni traer fragmentos no relacionados.
- **Poda Dinámica de Fuentes (Dynamic Relevance Cutoff):**
  En lugar de forzar un número fijo arbitrario de fuentes, el motor selecciona automáticamente solo los fragmentos con alta relevancia relativa ($\ge 65\%$ del mejor puntaje) y descarta caídas abruptas ($< 25\%$).
- **Indexación Incremental con [CocoIndex](https://cocoindex.io/):**
  Detección de cambios a nivel de archivo mediante *fingerprints* criptográficos. Actualiza exclusivamente los fragmentos modificados en la base vectorial (tiempo de re-indexación si no hay cambios: **0.0s**).
- **Soporte Multiformato de Alto Rendimiento:**
  - **PDFs:** Procesamiento acelerado en C/C++ con **PyMuPDF (`fitz`)**, extrayendo cientos de páginas en fracciones de segundo.
  - **Word (`.docx`):** Extracción completa de párrafos y tablas estructuradas.
  - **Documentos académicos y texto:** Markdown (`.md`, `.mdx`), Typst (`.typ`), BibTeX (`.bib`), texto plano (`.txt`, `.rst`).
  - **Código fuente:** Soporte para más de 25 lenguajes (`.py`, `.ts`, `.go`, `.rs`, `.c`, `.cpp`, `.sh`, `.sql`, etc.).
- **Interfaz Gráfica Moderna ("Tu Archivo"):**
  - Backend en **FastAPI** con streaming en tiempo real (Server-Sent Events / SSE).
  - Renderizado completo de Markdown con botón para copiar bloques de código.
  - Persistencia de múltiples conversaciones en el navegador (*LocalStorage*).
  - Detalle visual de fuentes consultadas con página o líneas exactas y puntaje de relevancia.
  - Aplicación de escritorio nativa para Linux (Fedora/GNOME) con integración de ventana e icono del sistema vía WebKit2GTK.
- **Modelos Locales en Ollama:**
  - Generación: **`qwen2.5:7b`** (configurable a cualquier modelo local).
  - Embeddings: **`nomic-embed-text`** con prefijos optimizados (`search_query:` / `search_document:`).

---

## Estructura del Proyecto

```text
.
├── docker-compose.yml     # Servicio Qdrant con almacenamiento persistente
├── pyproject.toml         # Dependencias (cocoindex, qdrant-client, ollama, pymupdf, fastapi...)
├── sync.sh                # Script de sincronización rsync --delete + indexación remota
├── run_app.sh             # Lanzador de la app de escritorio nativa (Fedora/GNOME) + túnel SSH
├── run_ui_local.sh        # Lanzador de la aplicación web para navegador + túnel SSH
├── .env.example           # Plantilla de variables de configuración
├── .gitignore             # Exclusión de almacenamiento local, cachés y variables
└── rag/
    ├── __init__.py
    ├── config.py          # Configuración de modelos, umbrales, extensiones y rutas
    ├── indexer.py         # Pipeline de indexación incremental con CocoIndex
    ├── query.py           # Motor de búsqueda híbrida (RRF, BM25, enrutamiento, guardas) y CLI
    ├── server.py          # Backend FastAPI con streaming SSE y API de chat
    ├── app_window.py      # Ventana nativa GTK3/WebKit2GTK para escritorio
    └── static/            # Frontend SPA (HTML5, CSS, JS, logo)
        ├── index.html     # Interfaz web con historial, markdown y streaming
        ├── logo.png       # Logo de la aplicación
        └── logo.svg       # Vector original del logo
```

---

## Requisitos Previos

1. **Docker y Docker Compose** instalados y activos.
2. **Ollama** corriendo con los modelos descargados:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen2.5:7b
   ```
3. Gestor de paquetes **[uv](https://github.com/astral-sh/uv)** (o Python 3.11+).

---

## Puesta en Marcha en el Servidor (Homelab)

### 1. Clonar el repositorio
```bash
git clone https://github.com/SalomonAvila/RAG.git
cd RAG
```

### 2. Iniciar la base vectorial (Qdrant)
```bash
docker compose up -d
```
El panel web de administración queda accesible en: `http://localhost:6333/dashboard` *(o usando la IP de tu red local)*.

### 3. Instalar dependencias
```bash
uv sync
```

### 4. Directorio de documentos
Por defecto, el indexador busca archivos en `~/rag-data`:
```bash
mkdir -p ~/rag-data/documentos
mkdir -p ~/rag-data/mis-repos
```

### 5. Ejecutar la indexación

- **Sincronización puntual (one-shot):**
  ```bash
  uv run cocoindex update rag.indexer
  ```
- **Modo en tiempo real (Live Watch):**
  ```bash
  uv run cocoindex update -L rag.indexer
  ```
  *Permanece en ejecución vigilando el directorio y sincronizando automáticamente al guardar cambios.*

---

## Sincronización Remota Automática (`sync.sh`)

Si editas o agregas archivos desde tu portátil y el servidor homelab está en tu red local, puedes sincronizar y re-indexar en un solo comando:

```bash
./sync.sh
```

El script:
1. Transmite tus archivos locales hacia el servidor mediante `rsync -avzP --delete` (eliminando en el servidor los archivos que hayas borrado localmente y omitiendo videos pesados).
2. Ejecuta automáticamente la indexación incremental de CocoIndex en el servidor vía SSH, actualizando Qdrant en segundos.

---

## Formas de Consultar el RAG

### Opción 1: Aplicación de Escritorio Nativa (Recomendado en Fedora / GNOME)

Ejecuta el asistente como una aplicación de escritorio nativa e independiente:

```bash
./run_app.sh
```

Este script:
- Sincroniza automáticamente las últimas actualizaciones desde GitHub.
- Configura el icono de "Tu Archivo" en tu entorno de escritorio (`~/.local/share/icons`).
- Abre un túnel SSH cifrado hacia el servidor homelab (`Qdrant :6334` y `Ollama :11434`).
- Inicia el backend FastAPI y abre la ventana nativa mediante WebKit2GTK sin consumir GPU local.
- Cierra limpiamente todos los procesos y túneles al salir.

---

### Opción 2: Interfaz Web en el Navegador

Si prefieres usarlo directamente desde el navegador web:

```bash
# Desde tu portátil (con túnel hacia el homelab):
./run_ui_local.sh

# O ejecutándolo directamente dentro del servidor:
uv run python -m rag.server
```

Abre en tu navegador:
`http://localhost:7860`

---

### Opción 3: Línea de Comandos (CLI)

- **Pregunta directa desde la terminal:**
  ```bash
  uv run python -m rag.query "¿De qué trata el proyecto y cuáles son sus objetivos?"
  ```

- **Filtrar por subcarpeta o materia específica:**
  ```bash
  uv run python -m rag.query -s "Septimo Semestre" "¿Cómo es la calificación en aprendizaje de máquina?"
  ```

- **Modo interactivo continuo (chat de terminal):**
  ```bash
  uv run python -m rag.query
  ```

- **Inspeccionar fragmentos recuperados sin invocar al LLM:**
  ```bash
  uv run python -m rag.query --solo-buscar "algoritmo de búsqueda"
  ```
