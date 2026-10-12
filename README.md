# RAG Local con CocoIndex, Qdrant y Ollama

Sistema RAG 100% privado y local para consultar repositorios de código y documentos personales (apuntes, Markdown, PDFs), optimizado para estaciones de trabajo o servidores domésticos (homelab) con aceleración por GPU.

## Características Principales

- **Indexación incremental con [CocoIndex](https://cocoindex.io/):** Detecta cambios a nivel de archivo mediante fingerprints criptográficos y actualiza exclusivamente los fragmentos modificados en la base vectorial (tiempo de re-indexación cuando no hay cambios: **0.0s**).
- **Extracción de PDFs acelerada en C/C++:** Integración con **PyMuPDF (`fitz`)**, procesando documentos extensos de cientos de páginas en fracciones de segundo.
- **Base de datos vectorial:** [Qdrant](https://qdrant.tech/) sobre Docker con volumen persistente en disco y panel de control web.
- **Embeddings locales:** `nomic-embed-text` vía Ollama (modelo ligero con concurrencia optimizada para GPU).
- **Generación fundamentada:** Modelos locales en Ollama (por defecto `qwen2.5:3b`, configurable a cualquier modelo instalado).
- **Citas automáticas:** Respuestas en tiempo real (streaming) referenciando el archivo, sección o número de página exacto.

---

## Estructura del Proyecto

```text
.
├── docker-compose.yml     # Servicio Qdrant con almacenamiento persistente
├── pyproject.toml         # Dependencias (cocoindex, qdrant-client, ollama, pymupdf...)
├── sync.sh                # Script de sincronización automática por red + indexación
├── .env.example           # Plantilla de variables de configuración
├── .gitignore             # Exclusión de almacenamiento local, cachés y variables
└── rag/
    ├── __init__.py
    ├── config.py          # Configuración y patrones de filtrado
    ├── indexer.py         # Pipeline de indexación con CocoIndex (concurrencia optimizada)
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
El panel web queda accesible en: `http://localhost:6333/dashboard` *(o usando la IP de tu red local)*.

### 3. Instalar dependencias
```bash
uv sync
```

### 4. Directorio de documentos
Por defecto, el indexador busca archivos en `~/rag-data`:
```bash
mkdir -p ~/rag-data/mis-repos
mkdir -p ~/rag-data/documentos
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

Si trabajas desde tu portátil y el servidor está en tu red local, puedes sincronizar tus archivos locales e indexarlos en un solo paso:

```bash
# Ejemplo: sincroniza tu carpeta local hacia el servidor y lanza la indexación
./sync.sh
```

El script utiliza `rsync` (transmite solo archivos nuevos/modificados y excluye videos pesados) y ejecuta la indexación incremental vía SSH en el servidor.

---

## Consultar el RAG

### Opción 1: Interfaz Gráfica Local con Túnel SSH (Recomendado)
Puedes ejecutar la interfaz web en tu propio computador de trabajo conectada de forma transparente al servidor por SSH. De esta forma, **la interfaz corre en tu máquina, pero todos los vectores y cómputos de GPU (Ollama) ocurren en el servidor sin consumir tus recursos locales**:

```bash
# En tu computador local:
./run_ui_local.sh
```
El script abre automáticamente un túnel SSH cifrado hacia el servidor (`Qdrant :6334` y `Ollama :11434`), inicia la aplicación web y la cierra limpiamente al salir. Solo abre en tu navegador:
`http://localhost:7860`

---

### Opción 2: Ejecutar la Interfaz Directamente en el Servidor
Si prefieres correr la interfaz dentro del servidor homelab:

```bash
# En el servidor:
uv run python -m rag.ui
```
Y accede desde cualquier dispositivo en tu red local: `http://<IP-DE-TU-SERVIDOR>:7860`.

---

### Opción 3: Línea de Comandos (CLI)

- **Pregunta directa desde la terminal:**
  ```bash
  uv run python -m rag.query "¿De qué trata el proyecto y cuáles son sus objetivos?"
  ```

- **Filtrar por subcarpeta o materia específica:**
  ```bash
  uv run python -m rag.query -s TESIS "¿Cómo se define la arquitectura de microservicios?"
  ```

- **Modo interactivo continuo (chat de terminal):**
  ```bash
  uv run python -m rag.query
  ```

- **Inspeccionar fragmentos recuperados sin invocar al LLM:**
  ```bash
  uv run python -m rag.query --solo-buscar "algoritmo de búsqueda"
  ```
