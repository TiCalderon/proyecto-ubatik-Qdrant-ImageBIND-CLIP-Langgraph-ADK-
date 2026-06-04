import os
import logging
from src.agents.state import AgentState
from src.models.embeddings import ClipEmbedder
from src.models.llm import LLMProvider

logger = logging.getLogger(__name__)

SYSTEM_ANALIZAR_IMAGEN = """Eres un patologo experto. Analiza esta imagen histologica y responde en 3 secciones:

1. DESCRIPCION OBJETIVA: Describe que ves (color, forma, patrones celulares, organizacion tisular).
2. CARACTERISTICAS DISCRIMINATIVAS: Senala caracteristicas clave que permitan diferenciar este tejido de otros.
3. DIAGNOSTICO DIFERENCIAL: Propon 3 posibles tipos de tejido, ordenados por probabilidad.

Se conciso y tecnico. Responde en español."""


async def nodo_procesar_imagen(state: AgentState, embedder: ClipEmbedder) -> AgentState:
    state["trayectoria"].append({"nodo": "procesar_imagen", "accion": "inicio"})

    if not state["tiene_imagen"]:
        state["trayectoria"].append({"nodo": "procesar_imagen", "estado": "sin_imagen"})
        return state

    image_path = state["imagen_path"]
    image_base64 = state["imagen_base64"]

    if image_base64 and not image_path:
        import tempfile
        import base64
        fd, image_path = tempfile.mkstemp(suffix=".png", dir="/tmp")
        os.close(fd)
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(image_base64))
        state["imagen_path"] = image_path

    if image_path and os.path.exists(image_path):
        try:
            emb = embedder.embed_image(image_path)
            state["imagen_embedding"] = emb.tolist()
            state["trayectoria"].append({"nodo": "procesar_imagen", "embedding": "ok"})

            b64 = state["imagen_base64"] or LLMProvider.image_to_base64(image_path)
            analisis = await LLMProvider.invoke_vision(
                system_prompt=SYSTEM_ANALIZAR_IMAGEN,
                user_text="Analiza esta imagen histologica:",
                image_base64=b64,
            )
            state["imagen_analisis"] = analisis
            state["trayectoria"].append({"nodo": "procesar_imagen", "analisis": "ok"})
        except Exception as e:
            logger.error(f"Error procesando imagen: {e}")
            state["error"] = f"Error procesando imagen: {e}"

    return state
