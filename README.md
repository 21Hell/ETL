
# ETL — Pipeline de ingestión y RAG UI

Resumen
-------
Este repositorio implementa un pipeline ETL para documentos (PDF/EPUB), generación de chunks, cálculo de embeddings y carga en Qdrant. Además incluye una pequeña aplicación RAG (FastAPI) con una UI web (3 columnas) para búsqueda semántica, visualización de fragmentos y generación de respuestas con LLMs (Ollama, Deepseek, OpenAI).

Contenido clave
---------------
- `src/etl/` — módulos ETL (chunking, utilidades)
- `src/tools/ingest_books.py` — script para ingestar libros (soporta `--file`, `--dry-run`)
- `src/tools/rag_chat.py` — FastAPI app: `/api/search`, `/api/chat`, `/api/point/{id}`, `/api/llm_status`, `/ui`
- `static/` — frontend (HTML/JS) con 3 columnas: Search | Related vectors | RAG output
- `data/raw/` — documentos fuente
- `data/chunks/` — resultados chunk/parquet
- `data/qdrant/` — volumen persistente para Qdrant (si se usa local)
#!/usr/bin/env bash

# ETL — Pipeline de ingestión y RAG UI

Resumen
-------
Este repositorio implementa un pipeline ETL para documentos (PDF/EPUB), generación de chunks, cálculo de embeddings y carga en Qdrant. Además incluye una pequeña aplicación RAG (FastAPI) con una UI web (3 columnas) para búsqueda semántica, visualización de fragmentos y generación de respuestas con LLMs (Ollama, Deepseek, OpenAI).

Contenido clave
---------------
- `src/etl/` — módulos ETL (chunking, utilidades)
- `src/tools/ingest_books.py` — script para ingestar libros (soporta `--file`, `--dry-run`)
- `src/tools/rag_chat.py` — FastAPI app: `/api/search`, `/api/chat`, `/api/point/{id}`, `/api/llm_status`, `/ui`
- `static/` — frontend (HTML/JS) con 3 columnas: Search | Related vectors | RAG output
- `data/raw/` — documentos fuente
- `data/chunks/` — resultados chunk/parquet
- `data/qdrant/` — volumen persistente para Qdrant (si se usa local)
- `podman-compose.yml` — orquestación (si está presente)

Requisitos (local)
------------------
- Python 3.10+ recomendado
- Podman (para ejecutar Qdrant y/o la pila)
- Opcional: `ollama` instalado si piensas usar Ollama local
- Acceso a internet para descargar modelos (sentence-transformers) la primera vez

Instalación local (rápida)
--------------------------
1. Crear y activar entorno virtual (desde la raíz del repo):
```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
```

2. Instalar dependencias mínimas (si no existe `requirements.txt`):
```bash
.venv/bin/pip install fastapi uvicorn qdrant-client==1.11.0 sentence-transformers \
		pdfminer.six ebooklib requests deepseek
```

Levantar Qdrant con Podman (local, persistente)
----------------------------------------------
```bash
# crear carpeta de almacenamiento
mkdir -p data/qdrant

# descargar imagen (si no la tienes)
podman pull docker.io/qdrant/qdrant:v1.11.0

# arrancar Qdrant
podman run -d --name qdrant-local \
	-p 6333:6333 -p 6334:6334 \
	-v $(pwd)/data/qdrant:/qdrant/storage:Z \
	docker.io/qdrant/qdrant:v1.11.0
```

Comprobar Qdrant:
```bash
curl http://127.0.0.1:6333/collections | jq .
```

(Opcional) Levantar la pila completa con Podman Compose
-------------------------------------------------------
Si tienes `podman compose` y el archivo `podman-compose.yml`:
```bash
# exportar UID/GID para evitar problemas de permisos
export AIRFLOW_UID=$(id -u)
export AIRFLOW_GID=$(id -g)

# inicializar y levantar
podman compose -f podman-compose.yml up -d
```

ETL — Ingestión de libros
-------------------------
- Dry-run (solo chunking — no calcula embeddings ni sube a Qdrant):
```bash
.venv/bin/python -m src.tools.ingest_books --dry-run
```

- Ingestión completa (calcula embeddings y sube a Qdrant):
```bash
.venv/bin/python -m src.tools.ingest_books --collection books_collection
```

- Ingestar un único archivo (incremental):
```bash
.venv/bin/python -m src.tools.ingest_books --collection books_collection --file data/raw/NOMBRE.epub
```

Notas:
- El script genera IDs deterministas (UUIDv5) por `source::chunk_index`. Re-ejecutar el ETL no duplica puntos; actualiza los existentes.
- Por defecto se usa `sentence-transformers` (modelo `all-MiniLM-L6-v2`) para embeddings.

RAG app (FastAPI) — uso local
-----------------------------
1. Arrancar servidor (desarrollo):
```bash
.venv/bin/python -m uvicorn src.tools.rag_chat:app --host 127.0.0.1 --port 5051 --reload
```
2. UI en navegador:
- Abrir: http://127.0.0.1:5051/ui

Endpoints
- POST `/api/search` — cuerpo: `{ "text": "...", "top_k": N }`
	- Devuelve vectores encontrados + metadata y `rerank_score`.
- POST `/api/chat` — cuerpo: `{ "text": "...", "top_k": N }`
	- Devuelve: `{ "answer": "...", "contexts": [...], "raw_results": [...] }`
	- `answer` será generado por LLM (si está configurado) o será un ensamblado de contextos como fallback.
- GET `/api/point/{id}` — devuelve payload/text almacenado para ese punto.
- GET `/api/llm_status` — estado del LLM detectado (Ollama/Deepseek/OpenAI).

Interfaz (UI)
-------------
La UI tiene 3 columnas:
- Izquierda: búsqueda y lista de resultados.
- Centro: detalles / payload del resultado seleccionado (incluye `payload.text` y `payload.page` cuando estén disponibles).
- Derecha: sección "Respuesta" (respuesta LLM), "Contextos" (fragmentos recuperados) y "Raw Results" (debug).

LLMs: Ollama / Deepseek / OpenAI
--------------------------------
La app intenta preferir (en orden):
1. Ollama local (HTTP/CLI) si está disponible.
2. Deepseek (cliente remoto) si `USE_DEEPSEEK=1` y `DEEPSEEK_API_KEY` está configurado.
3. OpenAI si `OPENAI_API_KEY` está en el entorno.

Variables de entorno típicas:
```bash
# Ollama no suele requerir clave (si está corriendo local)
export USE_DEEPSEEK=1
export DEEPSEEK_API_KEY="..."
export OPENAI_API_KEY="..."
```

Depuración rápida
-----------------
- Qdrant collections:
```bash
curl http://127.0.0.1:6333/collections | jq .
```
- Ver logs de uvicorn:
```bash
tail -f logs/rag_uvicorn.log
```
- Si `GET /api/point/{id}` falla, revisa:
	- que el id exista en la colección (`qdrant_client.get_point`)
	- que `payload.text` se haya guardado en la ingest (ingest_books.py guarda `payload.text`)

---

Versión: 2025-11-02

Licencia
--------
Este proyecto se distribuye bajo la licencia GNU General Public License v2 (GPLv2).
Consulta el archivo `LICENSE` en la raíz del repositorio para el texto completo de la licencia.