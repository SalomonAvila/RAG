# RAG Local y Privado con CocoIndex, Qdrant y Ollama

Sistema RAG (Retrieval-Augmented Generation) 100% privado, local y modular para consultar repositorios de código, notas y documentos personales (Word, Markdown, PDFs, Typst), optimizado para estaciones de trabajo o servidores domésticos (*homelab*) con aceleración por GPU.

---

## Características Principales

- **Búsqueda Híbrida Inteligente (Dense Vector + BM25 con RRF):**
  Combina la búsqueda vectorial semántica densa (`nomic-embed-text`) con búsqueda léxica BM25 (calibrada para palabras clave exactas, código, nombres propios y rutas de archivo) mediante *Reciprocal Rank Fusion* (RRF, $k=15$).
- **Enrutamiento, Expansión Semántica y Preguntas de Seguimiento:**
  En un solo llamado, el LLM reescribe las preguntas de seguimiento como consultas autocontenidas ("¿y la segunda?" → la pregunta completa), elige las carpetas relevantes del catálogo y enriquece la búsqueda con términos específicos, componentes y sinónimos formales o académicos (por ejemplo, si se consulta por "calificación", mapea automáticamente a "parciales, quices, talleres, proyectos, notas, ponderación").
- **Filtro de Ruido Cero-Match (Zero-Match Guard):**
  Si una consulta trata sobre una entidad o tema que no existe en tus documentos (presencia léxica nula) y su similitud semántica pura no supera el piso de ruido de fondo ($< 0.64$), el sistema responde de inmediato sin inventar respuestas ni traer fragmentos no relacionados.
- **Modo Documento:**
  Si la pregunta nombra un archivo (p. ej. "¿qué dice arch-description?"), la búsqueda se restringe a ese archivo y se envía al LLM su índice de títulos, para que pueda dar una visión general y no solo dos fragmentos sueltos.
- **Fragmentos con Sección:**
  En Typst y Markdown cada fragmento guarda la ruta de títulos vigente (`Decisiones > ADR-04: Kafka`), que se usa en el embedding y en el contexto del LLM. Los títulos sueltos al final de un fragmento se eliminan para que el modelo no atribuya contenido a la sección equivocada.
- **Reranker Multilingüe (Cross-Encoder):**
  Los mejores candidatos de RRF se reordenan con **`BAAI/bge-reranker-v2-m3`** (int8, vía `fastembed`, en CPU, ~2.5 s para 8 candidatos) usando la consulta expandida por el router. Se conservan solo los fragmentos con probabilidad de relevancia $\ge 50\%$ de la mejor. Sin reranker, se usa un corte por RRF ($\ge 65\%$ del mejor puntaje, sin caídas $> 25\%$).
- **Indexación Incremental con [CocoIndex](https://cocoindex.io/):**
  Detección de cambios a nivel de archivo mediante *fingerprints* criptográficos. Actualiza exclusivamente los fragmentos modificados en la base vectorial (tiempo de re-indexación si no hay cambios: **0.0s**).
- **Soporte Multiformato de Alto Rendimiento:**
  - **PDFs:** Procesamiento acelerado en C/C++ con **PyMuPDF (`fitz`)**. Las páginas consecutivas se agrupan hasta ~1500 bytes, para que las diapositivas con poco texto no generen fragmentos casi vacíos.
  - Cada fragmento se embebe con su ruta como cabecera (`Archivo: Materia/clase.pdf`), lo que da contexto a fragmentos ambiguos.
  - **Word (`.docx`):** Extracción completa de párrafos y tablas estructuradas.
  - **Documentos académicos y texto:** Markdown (`.md`, `.mdx`), Typst (`.typ`), BibTeX (`.bib`), texto plano (`.txt`, `.rst`).
  - **Código fuente:** Soporte para más de 25 lenguajes (`.py`, `.ts`, `.go`, `.rs`, `.c`, `.cpp`, `.sh`, `.sql`, etc.). De los notebooks (`.ipynb`) se indexan solo las celdas, sin salidas ni metadatos.
- **Interfaz Gráfica ("Tu Archivo")**, diseñada con los principios de interfaz de Apple:
  - Backend en **FastAPI** con streaming en tiempo real (Server-Sent Events / SSE); las respuestas se pueden detener.
  - Tipografía del sistema con interletrado según el tamaño, modo claro/oscuro automático y materiales translúcidos (barra superior y campo de pregunta flotantes).
  - Animaciones con resortes críticamente amortiguados, respuesta inmediata al presionar y soporte para movimiento reducido, transparencia reducida y alto contraste.
  - Markdown saneado con DOMPurify y botón para copiar bloques de código. Sin dependencias externas: `marked` y DOMPurify se sirven desde `rag/static/vendor/`, así que la interfaz no hace peticiones a internet.
  - Conversaciones guardadas en el navegador (*LocalStorage*), con deshacer al eliminar.
  - Fuentes desplegables con archivo, carpeta y página o líneas exactas.
  - Aplicación de escritorio nativa para Linux (Fedora/GNOME) vía WebKit2GTK, que sigue el modo claro/oscuro del sistema.
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
├── eval/
│   └── casos.json         # Casos de evaluación: pregunta → archivo/página esperados
└── rag/
    ├── __init__.py
    ├── config.py          # Configuración de modelos, umbrales, extensiones y rutas
    ├── indexer.py         # Pipeline de indexación incremental con CocoIndex
    ├── query.py           # Motor de búsqueda híbrida (RRF, BM25, enrutamiento, guardas) y CLI
    ├── eval.py            # Evaluación de la recuperación (aciertos, MRR, latencia)
    ├── server.py          # Backend FastAPI con streaming SSE y API de chat
    ├── app_window.py      # Ventana nativa GTK3/WebKit2GTK para escritorio
    └── static/            # Frontend SPA (HTML5, CSS, JS, logo)
        ├── index.html     # Interfaz web con historial, markdown y streaming
        ├── vendor/        # marked y DOMPurify servidos localmente
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

## Configuración y Variables de Entorno (`.env`)

Copia la plantilla `.env.example` a `.env` para personalizar la conexión a tu servidor homelab y las opciones del motor:

```bash
cp .env.example .env
```

Variables principales:
```env
# Conexión al servidor Homelab (para túnel SSH y rsync)
RAG_REMOTE_HOST=192.168.80.85
RAG_REMOTE_USER=homelab

# Parámetros del motor RAG
RAG_SOURCE_DIR=~/rag-data
QDRANT_URL=http://localhost:6334
OLLAMA_URL=http://localhost:11434
LLM_MODEL=qwen2.5:7b
EMBED_MODEL=nomic-embed-text

# Opcionales (valores por defecto)
LLM_NUM_CTX=8192            # ventana de contexto del LLM (Ollama usa 4096 si no se indica)
OLLAMA_KEEP_ALIVE=30m       # tiempo que los modelos quedan cargados entre consultas
EMBED_NUM_GPU=0             # 0 = embeddings de consulta en CPU (deja la VRAM al LLM)
NOISE_FLOOR=0.64            # similitud mínima cuando no hay coincidencias léxicas
RERANK_MODEL=BAAI/bge-reranker-v2-m3-int8   # vacío = sin reranker
RERANK_CANDIDATES=8
RERANK_KEEP_RATIO=0.5
```

Todos los scripts (`run_app.sh`, `run_ui_local.sh`, `sync.sh`) leen automáticamente estas variables desde tu archivo `.env`.

---

## Ajustes de Ollama para GPUs de 6 GB

Con una GPU de 6 GB (p. ej. GTX 1660 Super), `qwen2.5:7b` y `nomic-embed-text` no caben juntos con un contexto de 8k. Si ambos van a GPU, Ollama los alterna en VRAM en cada consulta (decenas de segundos por pregunta). Por eso las consultas embeben en CPU (`EMBED_NUM_GPU=0`) y conviene comprimir la caché KV del LLM en el servidor:

```bash
sudo systemctl edit ollama
```
```ini
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
```
```bash
sudo systemctl restart ollama
ollama ps   # tras una consulta: qwen2.5:7b debe aparecer con PROCESSOR = 100% GPU y CONTEXT = 8192
```

---

## Evaluación de la Recuperación

`eval/casos.json` contiene preguntas con el archivo (y la página, en PDFs) donde está la respuesta. Antes de cambiar umbrales o modelos, mide:

```bash
uv run python -m rag.eval                  # con reranker
uv run python -m rag.eval --sin-reranker   # solo RRF, para comparar
uv run python -m rag.eval -v               # muestra lo recuperado en cada caso
```

Reporta aciertos, MRR, fragmentos promedio y latencia. Conviene tener 20–30 casos reales de distintas materias y repositorios.

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

- **Comparar sin reranker:**
  ```bash
  uv run python -m rag.query --solo-buscar --sin-reranker "algoritmo de búsqueda"
  ```
