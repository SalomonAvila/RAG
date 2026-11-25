# RAG Local y Privado con CocoIndex, Qdrant y Ollama

Sistema RAG (Retrieval-Augmented Generation) 100% privado, local y modular para consultar repositorios de código, notas y documentos personales (Word, Markdown, PDFs, Typst), optimizado para estaciones de trabajo o servidores domésticos (*homelab*) con aceleración por GPU.

---

## Características Principales

- **Búsqueda Híbrida Inteligente (Dense Vector + BM25 con RRF):**
  Combina la búsqueda vectorial semántica densa (`nomic-embed-text`) con búsqueda léxica BM25 (calibrada para palabras clave exactas, código, nombres propios y rutas de archivo) mediante *Reciprocal Rank Fusion* (RRF, $k=15$).
- **Enrutamiento, Expansión Semántica y Preguntas de Seguimiento:**
  En un solo llamado, el LLM reescribe las preguntas de seguimiento como consultas autocontenidas ("¿y la segunda?" → la pregunta completa), elige las carpetas relevantes del catálogo y enriquece la búsqueda con términos específicos, componentes y sinónimos formales o académicos (por ejemplo, si se consulta por "calificación", busca también "evaluación, exámenes, porcentajes, ponderación").
- **Filtro de Ruido Cero-Match (Zero-Match Guard):**
  Si una consulta trata sobre una entidad o tema que no existe en tus documentos (presencia léxica nula) y su similitud semántica pura no supera el piso de ruido de fondo ($< 0.64$), el sistema responde de inmediato sin inventar respuestas ni traer fragmentos no relacionados.
- **Modo Documento:**
  Si la pregunta nombra un archivo (p. ej. "¿qué dice guia-instalacion?"), la búsqueda se restringe a ese archivo y se envía al LLM su índice de títulos, para que pueda dar una visión general y no solo dos fragmentos sueltos.
- **Fragmentos con Sección:**
  En Typst y Markdown cada fragmento guarda la ruta de títulos vigente (`Diseño > Decisiones > Base de datos`), que se usa en el embedding y en el contexto del LLM. Los títulos sueltos al final de un fragmento se eliminan para que el modelo no atribuya contenido a la sección equivocada.
- **Reranker Multilingüe (Cross-Encoder):**
  Los mejores candidatos de RRF se reordenan con **`BAAI/bge-reranker-v2-m3`** (int8, vía `fastembed`, en CPU, ~2.5 s para 8 candidatos) usando la consulta expandida por el router. Se conservan los fragmentos con probabilidad de relevancia $\ge 25\%$ de la mejor, y siempre al menos 3, para que un fragmento muy genérico (como una portada) no deje fuera al resto. Sin reranker, se usa un corte por RRF ($\ge 65\%$ del mejor puntaje, sin caídas $> 25\%$).
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
  - "Buscar en": árbol de carpetas y subcarpetas (con número de archivos) para limitar la búsqueda a cualquier nivel; la selección se recuerda y se muestra junto al campo de pregunta.
  - Aplicación de escritorio para Linux (GTK3 + WebKit2GTK), que sigue el modo claro/oscuro del sistema.
- **Modelos Locales en Ollama:**
  - Generación: **`qwen2.5:7b`** (configurable a cualquier modelo local).
  - Embeddings: **`nomic-embed-text`** con prefijos optimizados (`search_query:` / `search_document:`).

---

## Cómo se Reparte

El sistema está pensado para dos equipos, aunque todo puede correr en uno solo:

- **Servidor** (el equipo con GPU): Qdrant, Ollama, tus documentos en `RAG_SOURCE_DIR` y el indexador.
- **Cliente** (tu portátil): la interfaz y el motor de consulta. `run_app.sh` y `run_ui_local.sh` abren un túnel SSH hacia el servidor (`RAG_REMOTE_HOST`), así que Qdrant y Ollama se ven como si fueran locales.

Para usarlo en **un solo equipo**, sigue la puesta en marcha del servidor y abre la interfaz con `uv run python -m rag.server` (sin túnel).

---

## Estructura del Proyecto

```text
.
├── docker-compose.yml     # Servicio Qdrant con almacenamiento persistente
├── pyproject.toml         # Dependencias (cocoindex, qdrant-client, ollama, pymupdf, fastapi...)
├── sync.sh                # Envía una carpeta local al servidor (rsync --delete) e indexa allá
├── run_app.sh             # App de escritorio (GTK/WebKitGTK) + túnel SSH al servidor
├── run_ui_local.sh        # Interfaz web en el navegador + túnel SSH al servidor
├── .env.example           # Plantilla de variables de configuración
├── .gitignore             # Exclusión de almacenamiento local, cachés y variables
├── eval/
│   ├── casos.ejemplo.json # Plantilla de casos de evaluación (pregunta → archivo/página)
│   └── casos.local.json   # Tus casos reales (ignorado por git)
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
4. *(Opcional, solo para la app de escritorio)* GTK3 y WebKit2GTK 4.1 para el Python del sistema:
   ```bash
   sudo dnf install python3-gobject webkit2gtk4.1          # Fedora
   sudo apt install python3-gi gir1.2-webkit2-4.1          # Debian / Ubuntu
   ```

---

## Puesta en Marcha en el Servidor

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
Por defecto, el indexador busca archivos en `~/rag-data` (configurable con `RAG_SOURCE_DIR`). Cada carpeta de primer nivel funciona como una "fuente" y todas sus subcarpetas se pueden usar como filtro en la interfaz:
```bash
mkdir -p ~/rag-data/Apuntes ~/rag-data/mis-repos
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

Copia la plantilla `.env.example` a `.env` (ignorado por git) y ajústala a tu equipo:

```bash
cp .env.example .env
```

Variables principales:
```env
# Conexión al servidor (túnel SSH y rsync)
RAG_REMOTE_HOST=192.168.1.100      # IP o nombre del servidor
RAG_REMOTE_USER=usuario            # tu usuario SSH allá (por defecto, el local)
RAG_LOCAL_DIR=${HOME}/Documentos/rag   # carpeta que sync.sh envía al servidor
RAG_REMOTE_DEST='~/rag-data/'      # destino en el servidor (comillas simples: "~" se expande allá)
RAG_REMOTE_REPO='~/RAG'            # dónde está clonado este repo en el servidor
RAG_AUTO_UPDATE=0                  # 1 = "git pull --ff-only" automático al arrancar los scripts

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
RERANK_KEEP_RATIO=0.25
RERANK_MIN_KEEP=3
RERANK_MIN_SCORE=0.05       # relevancia mínima del mejor fragmento; debajo, "no hay evidencia"
ROUTE_PENALTY=0.25          # penalización a fragmentos fuera de las carpetas elegidas por el router
```

Todos los scripts (`run_app.sh`, `run_ui_local.sh`, `sync.sh`) leen automáticamente estas variables desde tu archivo `.env`.

---

## Ajustes de Ollama para GPUs de 6 GB

Con una GPU de 6 GB, `qwen2.5:7b` y `nomic-embed-text` no caben juntos con un contexto de 8k. Si ambos van a GPU, Ollama los alterna en VRAM en cada consulta (decenas de segundos por pregunta). Por eso las consultas embeben en CPU (`EMBED_NUM_GPU=0`) y conviene comprimir la caché KV del LLM en el servidor:

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

Los casos de evaluación son preguntas sobre **tus** documentos junto con el archivo (y la página, en PDFs) donde está la respuesta. Crea los tuyos a partir de la plantilla; `eval/casos.local.json` no se sube a git:

```bash
cp eval/casos.ejemplo.json eval/casos.local.json   # y edítalo con preguntas reales
```

Antes de cambiar umbrales o modelos, mide:

```bash
uv run python -m rag.eval                  # con reranker
uv run python -m rag.eval --sin-reranker   # solo RRF, para comparar
uv run python -m rag.eval -v               # muestra lo recuperado en cada caso
```

Reporta aciertos, MRR, fragmentos promedio y latencia. Conviene tener 20–30 casos reales de distintos temas y repositorios.

---

## Sincronización Remota Automática (`sync.sh`)

Si editas tus documentos en el portátil, puedes enviarlos al servidor y re-indexar en un solo comando:

```bash
./sync.sh                 # usa RAG_LOCAL_DIR de .env
./sync.sh ~/otra/carpeta  # o una carpeta concreta
```

El script:
1. Transmite tus archivos locales hacia el servidor mediante `rsync -avzP --delete` (eliminando en el servidor los archivos que hayas borrado localmente y omitiendo videos pesados).
2. Ejecuta automáticamente la indexación incremental de CocoIndex en el servidor vía SSH, actualizando Qdrant en segundos.

---

## Formas de Consultar el RAG

### Opción 1: Aplicación de Escritorio (Linux)

Ejecuta el asistente como una aplicación de escritorio nativa e independiente:

```bash
./run_app.sh
```

Este script:
- Si `RAG_AUTO_UPDATE=1`, actualiza el repositorio con `git pull --ff-only`.
- Configura el icono de "Tu Archivo" en tu entorno de escritorio (`~/.local/share/icons`).
- Abre un túnel SSH cifrado hacia el servidor (`Qdrant :6334` y `Ollama :11434`).
- Inicia el backend FastAPI y abre la ventana nativa mediante WebKit2GTK sin consumir GPU local.
- Cierra limpiamente todos los procesos y túneles al salir.

---

### Opción 2: Interfaz Web en el Navegador

Si prefieres usarlo directamente desde el navegador web:

```bash
# Desde tu portátil (con túnel hacia el servidor):
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

- **Buscar sólo en una carpeta, de cualquier nivel:**
  ```bash
  uv run python -m rag.query -s "Apuntes" "¿Cómo se evalúa el curso de cálculo?"
  uv run python -m rag.query -s "Apuntes/Calculo" "¿Cómo se evalúa el curso?"
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
