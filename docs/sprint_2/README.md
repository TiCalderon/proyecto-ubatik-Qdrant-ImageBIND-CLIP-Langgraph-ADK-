# Sprint 2 — Persistencia en Qdrant

**Grupo 3: Configuracion de colecciones Qdrant con vectores densos (Coseno)**

## Objetivos

- Configurar colecciones en Qdrant con vectores densos
- Probar recuperacion con similitud de coseno
- Comparar con otras bases de datos vectoriales

## Implementacion

### Colecciones Qdrant (`src/ingestion/indexer.py`)

**Coleccion de Texto:** `histologia_g3_chunks`
- Dimension: 512 (CLIP text embedding)
- Distancia: Cosine
- Payload: `tipo, texto, fuente, pagina, chunk_idx, id_chunk`

**Coleccion de Imagenes:** `histologia_g3_imagenes`
- Dimension: 512 (CLIP image embedding)
- Distancia: Cosine
- Payload: `tipo, path, nombre_archivo, fuente_pdf, pagina, ocr_text, texto_pagina, caption, etiqueta`

### Indices y Busqueda

- Busqueda de texto: `client.search()` con `score_threshold=0.60`
- Busqueda de imagenes: `client.search()` con `score_threshold=0.70`
- Busqueda texto->imagen: `client.search()` con `score_threshold=0.50`
- Busqueda hibrida: combinacion de resultados texto + imagen con pesos configurables

### Umbrales de Similitud

| Modo | Texto | Imagen |
|---|---|---|
| Solo texto | 0.45 | 0.95 |
| Solicitud imagenes | 0.45 | 0.50 |
| Multimodal | 0.60 | 0.70 |

### Qdrant Cloud vs Local

El sistema soporta ambos modos:
- **Cloud**: configurando `QDRANT_URL` y `QDRANT_API_KEY` en `.env`
- **Local**: usando `QDRANT_URL=http://localhost:6333` y corriendo Qdrant localmente

### Comparativa Qdrant vs Otras BD Vectoriales

| Caracteristica | Qdrant | Neo4j | Milvus |
|---|---|---|---|
| Tipo | Vectorial pura | Grafos + vectores | Vectorial pura |
| Distancia | Cosine/Euclid/Dot | Cosine/Euclid | Multiple |
| Filtrado | Payload filters | Cypher queries | Scalar filters |
| Escalabilidad | Cloud + On-prem | AuraDB | Zilliz Cloud |
| Indices | HNSW | Vector indexes | Multiple ANN |
| Complejidad | Baja | Alta | Media |

**Conclusion:** Qdrant es optimo para el caso de uso del Grupo 3 porque ofrece busqueda vectorial pura con payload filtering, sin la complejidad de modelado de grafos que requiere Neo4j (Grupo 1).
