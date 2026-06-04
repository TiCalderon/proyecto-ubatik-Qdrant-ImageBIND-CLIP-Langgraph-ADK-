# API Reference

Base URL: `http://localhost:10010`

## GET /api/status

Estado del asistente.

**Response:**
```json
{
  "ready": true,
  "temario_count": 10,
  "chunks_indexed": 150,
  "images_indexed": 58,
  "active_image": false,
  "device": "cuda"
}
```

## GET /api/temario

Lista de anclas semanticas del temario de histologia.

**Response:**
```json
{
  "anclas": ["tejido cartilaginoso...", "tejido oseo..."]
}
```

## POST /api/chat

Envia una consulta al agente RAG multimodal.

**Request Body:**
```json
{
  "query": "Que tejido es este?",
  "image_base64": "...",
  "image_filename": "microscopio.png"
}
```

**Response Body:**
```json
{
  "respuesta": "Basado en el analisis...",
  "estructura_identificada": "Tejido cartilaginoso hialino",
  "imagenes_recuperadas": [{"etiqueta": "Imagen 11.1", "nombre_archivo": "histo_pag011.png", "score": 0.92}],
  "imagenes_base64": ["..."],
  "trayectoria": [{"nodo": "inicializar", "modo": "multimodal"}],
  "imagen_activa": true,
  "mostrar_imagenes": true,
  "error": ""
}
```

## POST /api/imagen/limpiar

Limpia la imagen activa en memoria conversacional.

**Response:**
```json
{"ok": true}
```
