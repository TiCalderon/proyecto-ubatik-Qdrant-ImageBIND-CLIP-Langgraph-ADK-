# Sprint 1 — Chunking e Ingestion Multimodal

**Grupo 3: Qdrant + ImageBIND/CLIP + LangGraph/ADK**

## Objetivos

- Extraer texto e imagenes del manual de histologia
- Generar embeddings multimodales con CLIP
- Determinar tamanio optimo de chunk para contexto semantico
- Extraccion inicial de ontologia (Organo -> Tejido -> Celula)

## Implementacion

### Extraccion de PDF (`src/ingestion/extractor.py`)

Usamos PyMuPDF (fitz) para extraccion de texto e imagenes con precision de pixel:

- Texto: `page.get_text("text")` pagina por pagina
- Imagenes: `page.get_images(full=True)` con filtro de tamano minimo (150x150px)
- Preprocesamiento: contraste (1.2x) + brillo (1.1x)
- Magnificacion: escala a 868px si es menor (LANCZOS)
- Fallback: renderizado de pagina completa a 150 DPI con pdf2image
- OCR: pytesseract para texto embebido en imagenes
- Caption: extraccion por proximidad espacial debajo del bbox de la imagen
- Etiquetas: regex para "Imagen X.X", "Figura X.X", "Lamina X"

### Chunking (`src/ingestion/chunker.py`)

- Tamanio de chunk: 500 caracteres
- Overlap: 200 caracteres
- Sin solapamiento semantico adicional (chunks secuenciales)
- Cada chunk mantiene referencia a fuente y pagina

### Embeddings (`src/models/embeddings.py`)

- Modelo: `openai/clip-vit-base-patch32`
- Dimension: 512
- Texto: `CLIPModel.get_text_features()` normalizado
- Imagen: `CLIPModel.get_image_features()` normalizado
- Alternativa: SentenceTransformer `all-MiniLM-L6-v2` para busquedas rapidas

### Evaluacion de Chunk Size

Realizamos pruebas con 3 tamanios de chunk:
- 250 chars (con 200 overlap)
- 500 chars (con 200 overlap) ← seleccionado
- Pagina completa

El chunk de 500 chars con 200 overlap resulto el mejor balance entre contexto semantico y precision de recuperacion.

### Ontologia Inicial

Anclas semanticas predefinidas (10 categorias):
1. Tejido cartilaginoso (hialino, elastico, fibroso)
2. Tejido oseo (compacto, esponjoso, osteonas)
3. Tejido muscular (estriado esqueletico, cardiaco, liso)
4. Tejido nervioso (neuronas piramidales, estrelladas, piriformes)
5. Neuroglia (astrocitos, oligodendrocitos, microglia, ependimarias)
6. Celulas epiteliales
7. Glandulas (exocrinas, endocrinas)
8. Tejido conjuntivo (laxo, denso, adiposo)
9. Vasos sanguineos (arterias, venas, capilares)
10. Organos linfoides (ganglio, bazo, timo)

## Resultados

- Texto extraido: ~50 paginas procesadas
- Imagenes extraidas: ~58 por PDF
- Chunks generados: ~150 por PDF de 58 paginas
- Tiempo de ingestion: ~3 minutos con GPU
