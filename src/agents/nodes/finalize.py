import os
import re
import base64
import logging
import numpy as np
from src.agents.state import AgentState
from src.models.embeddings import MultimodalEmbedder
from src.memory.conversation import ConversationMemory

logger = logging.getLogger(__name__)


async def nodo_finalizar(state: AgentState, embedder: MultimodalEmbedder, memory: ConversationMemory) -> AgentState:
    state["trayectoria"].append({"nodo": "finalizar", "accion": "inicio"})

    respuesta = state.get("respuesta", "")
    imagenes_detectadas = _extraer_referencias_imagenes(respuesta)

    imagenes_recuperadas = []
    for label in imagenes_detectadas:
        for img_dict in state.get("contexto_filtrado", {}).get("imagenes", []):
            if img_dict.get("etiqueta", "").lower() == label.lower():
                path = img_dict.get("path", "")
                nombre = img_dict.get("nombre_archivo", "")
                if path and os.path.exists(path):
                    try:
                        with open(path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                    except Exception:
                        b64 = ""
                else:
                    b64 = ""
                imagenes_recuperadas.append({
                    "etiqueta": label,
                    "nombre_archivo": nombre,
                    "path": path,
                    "base64": b64,
                    "caption": img_dict.get("caption", ""),
                    "pagina": img_dict.get("pagina", 0),
                    "score": img_dict.get("score", 0),
                })
                break

    if not imagenes_recuperadas and state.get("modo") in ("solicitud_imagenes", "multimodal"):
        sem_matches = _busqueda_semantica_etiquetas(respuesta, state, embedder)
        for sm in sem_matches:
            if sm["etiqueta"] not in [ir["etiqueta"] for ir in imagenes_recuperadas]:
                imagenes_recuperadas.append(sm)

    state["imagenes_detectadas"] = imagenes_detectadas
    state["imagenes_recuperadas"] = imagenes_recuperadas

    if imagenes_recuperadas:
        state["mostrar_imagenes"] = True
    else:
        state["mostrar_imagenes"] = state.get("solicita_imagenes", False)

    memory.add_interaction(
        query=state["query_original"],
        response=respuesta,
        image_path=state.get("imagen_path", ""),
        image_analysis=state.get("imagen_analisis", ""),
        structure=state.get("estructura_identificada", ""),
    )

    state["trayectoria"].append({
        "nodo": "finalizar",
        "imagenes_detectadas": len(imagenes_detectadas),
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


def _busqueda_semantica_etiquetas(respuesta: str, state: AgentState, embedder: MultimodalEmbedder) -> list[dict]:
    matches = []
    try:
        if not state.get("contexto_filtrado", {}).get("imagenes"):
            return matches
        query_vec = np.array(state.get("texto_embedding", []))
        if len(query_vec) == 0:
            query_vec = embedder.embed_text(respuesta[:500])
        for img in state["contexto_filtrado"]["imagenes"][:10]:
            etiqueta = img.get("etiqueta", "")
            caption = img.get("caption", "")
            path = img.get("path", "")
            if etiqueta or caption:
                etiqueta_vec = embedder.embed_text(etiqueta + " " + caption[:200])
                sim = float(np.dot(query_vec, etiqueta_vec))
                if sim > 0.55:
                    nombre = img.get("nombre_archivo", "")
                    b64 = ""
                    if path and os.path.exists(path):
                        try:
                            with open(path, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode("utf-8")
                        except Exception:
                            pass
                    matches.append({
                        "etiqueta": etiqueta,
                        "nombre_archivo": nombre,
                        "path": path,
                        "base64": b64,
                        "caption": caption,
                        "score": sim,
                    })
        matches.sort(key=lambda x: x["score"], reverse=True)
    except Exception as e:
        logger.warning(f"Error en busqueda semantica de etiquetas: {e}")
    return matches[:5]
