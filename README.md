# Ubatik RAG Histologia — Grupo 3

**Qdrant + ImageBIND/CLIP + LangGraph/ADK**

Sistema multimodal de Recuperacion Aumentada por Generacion (RAG) para la identificacion de imagenes histologicas. Permite al usuario subir una imagen del manual de histologia y que el agente la reconozca, o describir una imagen con texto para que el sistema sepa de cual se esta hablando.

## Objetivos del Proyecto

1. **Reconocimiento de imagenes**: Al pasarle una imagen del manual, el agente identifica cual es y la asocia con su descripcion correcta.
2. **Busqueda texto -> imagen**: Al describir una imagen con lenguaje natural, el agente recupera la imagen correspondiente del manual.
3. **Busqueda hibrida**: Combina busqueda semantica de texto con busqueda visual de imagenes usando CLIP.

## Stack Tecnologico

| Componente | Tecnologia |
|---|---|
| Vector Store | Qdrant (Cloud o local) |
| Embeddings | CLIP ViT-B/32 (texto + imagen) |
| Orquestacion | LangGraph (StateGraph) |
| LLM | Gemini 2.0 Flash / Groq Llama 4 Scout |
| Backend | FastAPI + uvicorn |
| Frontend | Vanilla JS + CSS (glassmorphism dark theme) |
| Observabilidad | LangSmith (opcional) |

## Arquitectura

```
Usuario → [Frontend JS] → [FastAPI] → [LangGraph Agent]
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                    [Qdrant Texto]  [Qdrant Imagenes]  [Gemini/Groq]
```

### Flujo del Agente (LangGraph)

```
START
  → inicializar (reescribe, detecta modo)
  → procesar_imagen (CLIP embed + vision LLM analisis)
  → clasificar (extrae entidades, valida dominio)
  → generar_consulta (reformula queries de busqueda)
  → buscar (hybrid search en Qdrant)
  → filtrar_contexto (deduplica, ordena por score)
  → generar_respuesta (LLM genera respuesta final)
  → finalizar (extrae imagenes referenciadas, guarda memoria)
→ END
```

## Requisitos

- Python >= 3.10
- uv (gestor de paquetes)
- API Keys: Google Gemini, Groq, Qdrant Cloud, HuggingFace
- Poppler (para pdf2image): `sudo apt install poppler-utils`
- Tesseract OCR (opcional): `sudo apt install tesseract-ocr tesseract-ocr-spa`

## Versiones de Arquitectura

El repositorio contiene **dos versiones separadas** del sistema. No están diseñadas para ejecutarse simultáneamente en el mismo servidor; debes elegir cuál inicializar:

1. **Versión Estable (Qdrant Puro)**: Ubicada en la raíz (`src/` y `server.py`). Utiliza Qdrant como base de datos vectorial principal y el modelo genérico CLIP de OpenAI. Es ideal para estabilidad y bajo consumo de recursos, pero **su rendimiento para recuperar imágenes a partir de descripciones médicas complejas en lenguaje natural es básico**.
2. **Versión Avanzada "Laboratorio" (v4.2)**: Ubicada aislada en la carpeta `/histo-test-main`. Utiliza **Neo4j** como base de conocimiento vectorial principal y Qdrant solo para memoria conversacional. Incorpora modelos médicos especializados como UNI (requiere `HF_TOKEN`) y PLIP. **Esta versión tiene un rendimiento inmensamente superior al buscar imágenes usando lenguaje natural médico**, ya que sus modelos fueron entrenados específicamente con miles de pares de textos e imágenes de patología histológica.

## Instalación y Ejecución

Primero, instala las dependencias iniciales y configura tus variables de entorno en la raíz del proyecto:

```bash
# Clonar el repositorio
git clone <repo-url>
cd proyecto-ubatik-Qdrant-ImageBIND-CLIP-Langgraph-ADK-

# Instalar uv si no lo tenes
curl -LsSf https://astral.sh/uv/install.sh | sh

# Configurar variables de entorno
cp .env.example .env
# IMPORTANTE: Edita .env con tus API keys (Gemini, Groq, Qdrant, y HF_TOKEN para la version avanzada)

# Instalar dependencias
uv sync
```

### Opción A: Ejecutar la Versión Estable (Raíz / Qdrant)

Si deseas utilizar la arquitectura principal, ejecuta los comandos desde la raíz del proyecto:

```bash
# (Opcional) Indexar PDFs del manual
mkdir -p data/pdf
# Copia tus PDFs a data/pdf/ y luego:
uv run python -m src.ingestion.pipeline

# Iniciar el servidor (levanta FastAPI y sirve el Frontend)
npm run dev
# o directamente: uv run uvicorn server:app --host 0.0.0.0 --port 10010 --reload
```
👉 **Interactuar**: Abre [http://localhost:10010](http://localhost:10010) en tu navegador.

### Opción B: Ejecutar la Versión Avanzada (Neo4j / UNI)

Si deseas utilizar las capacidades gráficas y los modelos médicos de alto rendimiento, debes operar dentro de la carpeta de pruebas:

```bash
cd histo-test-main

# 1. Iniciar el backend RAG con Neo4j
python ne4j-histo.py

# 2. En otra ventana de terminal, sirve el frontend (cliente estático)
# Si estás usando npm:
npm run dev
# Alternativa simple con Python:
python -m http.server 10005 -d client/
```
👉 **Interactuar**: Abre [http://localhost:10005](http://localhost:10005) (o el puerto que asigne tu servidor) en tu navegador.

## Endpoints API

| Metodo | Ruta | Descripcion |
|---|---|---|
| GET | `/` | Frontend (SPA) |
| GET | `/api/status` | Estado del asistente |
| GET | `/api/temario` | Temario de histologia |
| POST | `/api/chat` | Chat RAG multimodal |
| POST | `/api/imagen/limpiar` | Limpiar imagen activa |
| GET | `/imagenes_extraidas/{file}` | Servir imagenes del manual |

### POST /api/chat

Request:
```json
{
  "query": "Que tejido es este?",
  "image_base64": "...",
  "image_filename": "microscopio.png"
}
```

Response:
```json
{
  "respuesta": "Basado en el analisis...",
  "estructura_identificada": "Tejido cartilaginoso hialino",
  "imagenes_recuperadas": [{"etiqueta": "Imagen 11.1", "base64": "...", "nombre_archivo": "..."}],
  "mostrar_imagenes": true,
  "trayectoria": [...]
}
```

## Estructura del Proyecto

```
/
├── server.py                    # Entry point FastAPI
├── pyproject.toml               # Dependencias Python
├── package.json                 # Scripts npm
├── .env.example                 # Template de API keys
├── src/
│   ├── config.py                # Configuracion central
│   ├── models/
│   │   ├── embeddings.py        # CLIP embedder (CLIPModel + SentenceTransformer)
│   │   └── llm.py              # LLM providers (Gemini + Groq)
│   ├── ingestion/
│   │   ├── extractor.py         # PDF extractor (PyMuPDF + OCR)
│   │   ├── chunker.py           # Text chunker (500 chars, 200 overlap)
│   │   ├── indexer.py           # Qdrant indexer (CRUD + busqueda)
│   │   └── pipeline.py          # Pipeline de ingestion completo
│   ├── agents/
│   │   ├── state.py             # AgentState (TypedDict)
│   │   ├── graph.py             # LangGraph StateGraph builder
│   │   └── nodes/               # Nodos del grafo
│   │       ├── initialize.py    # Reescritura + deteccion de modo
│   │       ├── process_image.py # CLIP embedding + vision LLM
│   │       ├── classify.py      # Clasificacion semantica + entidades
│   │       ├── generate_query.py
│   │       ├── search.py        # Busqueda hibrida Qdrant
│   │       ├── filter_context.py
│   │       ├── generate_response.py
│   │       └── finalize.py      # Extraccion de imagenes + memoria
│   ├── search/
│   │   ├── hybrid_search.py     # Funciones de busqueda auxiliares
│   │   └── semantic_classifier.py # Clasificador de dominio
│   ├── memory/
│   │   └── conversation.py      # Memoria conversacional (Qdrant local)
│   └── api/
│       └── routes.py            # Endpoints FastAPI
├── client/
│   ├── index.html               # SPA frontend
│   ├── app.js                   # Logica del chat
│   └── style.css                # Estilos dark theme
├── docs/                        # Documentacion detallada
├── tests/                       # Tests
└── data/
    └── pdf/                     # PDFs del manual (gitignored)
```

## Licencia

MIT — Grupo 3, UBATIC 2026-27
