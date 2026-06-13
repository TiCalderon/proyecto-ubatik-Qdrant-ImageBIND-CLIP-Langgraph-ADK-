import os
import re
import base64
import logging
from src.agents.state import AgentState
from src.models.embeddings import MultimodalEmbedder
from src.ingestion.indexer import QdrantIndexer
from src.memory.conversation import ConversationMemory
from src.config import Config

logger = logging.getLogger(__name__)


async def nodo_finalizar(
    state: AgentState,
    embedder: MultimodalEmbedder,
    memory: ConversationMemory,
    indexer: QdrantIndexer,
) -> AgentState:
    state["trayectoria"].append({"nodo": "finalizar", "accion": "inicio"})

    respuesta = state.get("respuesta", "")
    query_vec = state.get("texto_embedding", [])
    max_imgs = Config.MAX_IMAGES_POR_RESPUESTA

    # ── Búsqueda por similitud coseno query ↔ caption directamente en Qdrant ──
    caption_results = []
    if query_vec:
        caption_results = indexer.caption_search(
            query_vec=query_vec,
            top_k=max_imgs,
            threshold=Config.IMAGE_CAPTION_SIMILARITY_THRESHOLD,
        )
        logger.info(f"caption_search devolvió {len(caption_results)} imágenes")

    # ── Construir lista final con base64 ──
    imagenes_recuperadas = []
    seen_nombres = set()
    for score, payload in caption_results:
        nombre = payload.get("nombre_archivo", "")
        if nombre in seen_nombres:
            continue
        seen_nombres.add(nombre)
        path = payload.get("path", "")
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                b64 = ""
        else:
            b64 = ""
        imagenes_recuperadas.append({
            "etiqueta": payload.get("etiqueta", ""),
            "nombre_archivo": nombre,
            "path": path,
            "base64": b64,
            "caption": payload.get("caption", "") or payload.get("texto_pagina", "")[:300],
            "pagina": payload.get("pagina", 0),
            "score": round(score, 4),
        })

    state["imagenes_detectadas"] = _extraer_referencias_imagenes(respuesta)
    state["imagenes_recuperadas"] = imagenes_recuperadas
    state["mostrar_imagenes"] = len(imagenes_recuperadas) > 0

    memory.add_interaction(
        query=state["query_original"],
        response=respuesta,
        image_path=state.get("imagen_path", ""),
        image_analysis=state.get("imagen_analisis", ""),
        structure=state.get("estructura_identificada", ""),
    )

    state["trayectoria"].append({
        "nodo": "finalizar",
        "caption_results": len(caption_results),
        "imagenes_recuperadas": len(imagenes_recuperadas),
    })
    return state


def _extraer_referencias_imagenes(texto: str) -> list[str]:
    patterns = [
        r"(?:Imagen|Figura|Fig\.|Lámina|Lamina)\s*(\d+\.\d+)",
        r"(?:Imagen|Figura|Fig\.|Lámina|Lamina)\s*(\d+)",
    ]
    encontradas = set()
    for pat in patterns:
        for match in re.finditer(pat, texto, re.IGNORECASE):
            encontradas.add(match.group(0).strip())
    return list(encontradas)[:10]
