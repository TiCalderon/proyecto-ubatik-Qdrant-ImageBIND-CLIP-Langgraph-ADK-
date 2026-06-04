import logging
from src.agents.state import AgentState
from src.ingestion.indexer import QdrantIndexer
from src.config import Config

logger = logging.getLogger(__name__)


async def nodo_buscar(state: AgentState, indexer: QdrantIndexer) -> AgentState:
    state["trayectoria"].append({"nodo": "buscar", "accion": "inicio"})

    if not state["en_temario"]:
        state["trayectoria"].append({"nodo": "buscar", "estado": "omitido"})
        state["resultados_busqueda"] = {"texto": [], "imagenes": []}
        return state

    text_vec = state["texto_embedding"]
    image_vec = state.get("imagen_embedding") if state.get("imagen_embedding") else None

    modo = state["modo"]
    text_threshold = Config.SIMILARITY_THRESHOLD_TEXTO
    image_threshold = Config.SIMILARITY_THRESHOLD_IMAGEN

    if modo == "solicitud_imagenes":
        text_threshold = 0.45
        image_threshold = 0.50
    elif modo == "texto":
        text_threshold = 0.45
        image_threshold = 0.95
    elif modo == "multimodal":
        text_threshold = 0.60
        image_threshold = 0.70

    if text_vec:
        resultados = indexer.hybrid_search(
            text_vec=text_vec,
            image_vec=image_vec,
            text_top_k=Config.TOP_K_TEXTO,
            image_top_k=Config.TOP_K_IMAGEN,
            text_threshold=text_threshold,
            image_threshold=image_threshold,
            include_images=(modo != "texto"),
        )
    else:
        resultados = {"texto": [], "imagenes": []}

    state["resultados_busqueda"] = resultados
    n_text = len(resultados.get("texto", []))
    n_img = len(resultados.get("imagenes", []))
    state["trayectoria"].append({
        "nodo": "buscar",
        "resultados": f"texto={n_text}, imagenes={n_img}",
        "thresholds": f"texto={text_threshold}, imagen={image_threshold}",
    })
    return state
