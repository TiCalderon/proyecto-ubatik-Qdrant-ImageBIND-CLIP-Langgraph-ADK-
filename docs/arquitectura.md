# Arquitectura del Sistema

## Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (SPA)                            │
│  client/index.html + app.js + style.css                     │
│  ┌─────────┐  ┌──────────┐  ┌──────────────┐               │
│  │ Chat UI │  │ Upload   │  │ Gallery      │               │
│  └────┬────┘  └────┬─────┘  └──────┬───────┘               │
└───────┼────────────┼───────────────┼────────────────────────┘
        │            │               │
        ▼            ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Server (server.py)                 │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────────┐     │
│  │ GET /     │  │ POST /chat   │  │ GET /imagenes/*  │     │
│  └─────┬─────┘  └──────┬───────┘  └────────┬─────────┘     │
└────────┼───────────────┼───────────────────┼───────────────┘
         │               │                   │
         ▼               ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│           AsistenteHistologia (LangGraph Agent)              │
│                                                              │
│  START → inicializar → procesar_imagen → clasificar         │
│              │                                     │         │
│              ▼                                     ▼         │
│        generar_consulta ←──────────────────────────┘         │
│              │                                               │
│              ▼                                               │
│          buscar → filtrar_contexto → generar_respuesta       │
│                                                  │          │
│                                                  ▼          │
│                                             finalizar → END  │
└──────────────┬──────────────────────────────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌─────────┐ ┌───────┐ ┌──────────┐
│ Qdrant  │ │Gemini │ │Groq      │
│ Texto   │ │2.0    │ │Llama 4   │
├─────────┤ │Flash  │ │Scout     │
│ Qdrant  │ └───────┘ └──────────┘
│ Imagenes│
└─────────┘
```

## Flujo de Datos Detallado

### Modo Texto (sin imagen)
1. Query → `inicializar`: reescribe anáforas, detecta si pide imágenes
2. → `clasificar`: extrae entidades, genera text embedding CLIP, valida dominio
3. → `generar_consulta`: LLM reformula query para búsqueda
4. → `buscar`: hybrid search en Qdrant (solo texto)
5. → `filtrar_contexto`: deduplica y ordena
6. → `generar_respuesta`: LLM responde con contexto del manual
7. → `finalizar`: guarda en memoria

### Modo Multimodal (con imagen)
1. Query + imagen → `inicializar`: detecta imagen
2. → `procesar_imagen`: embedding CLIP + análisis visual con Gemini Vision
3. → `clasificar`: extrae entidades, valida dominio (threshold reducido)
4. → `generar_consulta`: consultas texto + visual
5. → `buscar`: búsqueda en ambas colecciones Qdrant
6. → `filtrar_contexto`: merge + sort de resultados texto + imagen
7. → `generar_respuesta`: respuesta multimodal con estructura identificada
8. → `finalizar`: extrae imágenes referenciadas, guarda memoria

### Modo Solicitud Imágenes
1. Query → `inicializar`: detecta "mostrame imagen de..."
2. → `clasificar`: extrae entidades, text embedding
3. → `buscar`: text→image search en Qdrant con threshold reducido (0.50)
4. → `generar_respuesta`: LLM referencia etiquetas reales del manual
5. → `finalizar`: extrae imágenes por regex + búsqueda semántica de etiquetas

## Modelo de Datos Qdrant

### Colección: `histologia_g3_chunks` (texto)
- Vector: 512-dim (CLIP text embedding)
- Payload: `tipo, texto, fuente, pagina, chunk_idx, id_chunk`
- Distancia: Cosine

### Colección: `histologia_g3_imagenes` (imágenes)
- Vector: 512-dim (CLIP image embedding)
- Payload: `tipo, path, nombre_archivo, fuente_pdf, pagina, ocr_text, texto_pagina, caption, etiqueta, width, height`
- Distancia: Cosine

### Colección: `memoria_histo_g3` (conversacional, Qdrant local)
- Vector: 512-dim (CLIP text embedding)
- Payload: `resumen, turno_fin, tiene_imagen, structure`
- Distancia: Cosine
