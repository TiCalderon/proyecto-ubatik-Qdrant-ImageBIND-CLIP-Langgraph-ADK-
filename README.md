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

## Instalacion

```bash
# Clonar el repositorio
git clone <repo-url>
cd proyecto-ubatik-Qdrant-ImageBIND-CLIP-Langgraph-ADK-

# Instalar uv si no lo tenes
curl -LsSf https://astral.sh/uv/install.sh | sh

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys

# Instalar dependencias
uv sync

# (Opcional) Indexar PDFs del manual
mkdir -p data/pdf
# Copia tus PDFs a data/pdf/
uv run python -m src.ingestion.pipeline

# Iniciar el servidor
npm run dev
# o directamente: uv run uvicorn server:app --host 0.0.0.0 --port 10010 --reload
```

Abrir http://localhost:10010 en el navegador.

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
