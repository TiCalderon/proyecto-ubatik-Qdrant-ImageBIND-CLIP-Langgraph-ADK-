# Guia de Instalacion y Setup

## Requisitos del Sistema

- Python >= 3.10
- uv (gestor de paquetes Python)
- Poppler (para pdf2image): `sudo apt install poppler-utils`
- Tesseract OCR (opcional): `sudo apt install tesseract-ocr tesseract-ocr-spa`

## Paso 1: API Keys

Obtene las siguientes API keys:

| Servicio | URL | Variable |
|---|---|---|
| Google AI Studio | https://aistudio.google.com | `GOOGLE_API_KEY` |
| Groq Cloud | https://console.groq.com | `GROQ_API_KEY` |
| HuggingFace | https://huggingface.co/settings/tokens | `HF_TOKEN` |
| Qdrant Cloud | https://cloud.qdrant.io | `QDRANT_URL`, `QDRANT_API_KEY` |
| LangSmith | https://smith.langchain.com (opcional) | `LANGSMITH_API_KEY` |

## Paso 2: Clonar y Configurar

```bash
git clone <repo-url>
cd proyecto-ubatik-Qdrant-ImageBIND-CLIP-Langgraph-ADK-

# Copiar template de variables
cp .env.example .env

# Editar .env con tus API keys
nano .env
```

## Paso 3: Instalar Dependencias

```bash
# Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sincronizar dependencias
uv sync
```

## Paso 4: Preparar los PDFs

Coloca los PDFs del manual de histologia en `data/pdf/`:

```bash
mkdir -p data/pdf
# cp /ruta/a/tus/pdfs/*.pdf data/pdf/
```

## Paso 5: Indexar en Qdrant

```bash
uv run python -m src.ingestion.pipeline
```

Esto extrae texto e imagenes de los PDFs, los chunkea (500 chars), genera embeddings CLIP y los indexa en Qdrant.

## Paso 6: Ejecutar el Servidor

```bash
# Desarrollo (con hot-reload)
npm run dev

# Produccion
npm start
```

Abre http://localhost:10010 en tu navegador.

## Verificacion

```bash
# Verificar estado
curl http://localhost:10010/api/status
# {"ready":true,"chunks_indexed":150,"images_indexed":58,"device":"cuda"}

# Probar chat
curl -X POST http://localhost:10010/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Que tipos de cartilago existen?"}'
```

## Variables de Entorno Completas

```bash
GOOGLE_API_KEY=tu-api-key-de-gemini
GROQ_API_KEY=tu-api-key-de-groq
HF_TOKEN=tu-token-huggingface-read
QDRANT_URL=https://tu-cluster.qdrant.cloud
QDRANT_API_KEY=tu-qdrant-api-key
LANGSMITH_API_KEY=lsv2_pt_...    # opcional
LANGSMITH_PROJECT=ubatik-rag-histologia-g3
PORT=10010
HOST=0.0.0.0
```
