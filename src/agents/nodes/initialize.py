import logging
from src.agents.state import AgentState
from src.models.embeddings import MultimodalEmbedder
from src.models.llm import LLMProvider
from src.memory.conversation import ConversationMemory
from src.config import Config

logger = logging.getLogger(__name__)

SYSTEM_INICIALIZAR = """Eres un asistente de histologia. Reescribe la consulta del usuario para resolver referencias anaforicas (como "eso", "esa imagen", "cuentame mas", etc.) usando el historial de conversacion.

Historial: {historial}
Consulta del usuario: {query}

Devuelve SOLO la consulta reescrita, sin explicaciones."""

SYSTEM_DETECTAR_IMAGEN = """Determina si el usuario esta solicitando EXPLICITAMENTE ver imagenes del manual de histologia en su mensaje.

Historial: {historial}
Mensaje: {mensaje}

Responde SOLO con "SI" o "NO"."""


async def nodo_inicializar(state: AgentState, embedder: MultimodalEmbedder, memory: ConversationMemory) -> AgentState:
    query = state["query"]
    historial = memory.get_history_text()

    state["query_original"] = query
    state["trayectoria"].append({"nodo": "inicializar", "accion": "inicio"})

    if historial:
        prompt = SYSTEM_INICIALIZAR.format(historial=historial, query=query)
        try:
            rewritten = await LLMProvider.invoke_text(
                system_prompt="Reescribe consultas para resolver anaforas.",
                user_prompt=prompt,
            )
            if rewritten and len(rewritten) > 5:
                state["query_reescrita"] = rewritten.strip()
                state["trayectoria"].append({"nodo": "inicializar", "reescritura": rewritten.strip()})
        except Exception as e:
            logger.warning(f"Error reescribiendo query: {e}")

    detect_prompt = SYSTEM_DETECTAR_IMAGEN.format(historial=historial, mensaje=state["query_reescrita"])
    try:
        result = await LLMProvider.invoke_text(
            system_prompt="Detecta si el usuario pide imagenes.",
            user_prompt=detect_prompt,
        )
        state["solicita_imagenes"] = "SI" in result.upper()
    except Exception:
        state["solicita_imagenes"] = False

    keywords_imagen = ["mostrar imagen", "muestrame imagen", "ver imagen", "foto de",
                       "imagen de", "micrografia", "microfotografia", "lamina"]
    if any(kw in query.lower() for kw in keywords_imagen):
        state["solicita_imagenes"] = True

    if state["tiene_imagen"]:
        state["modo"] = "multimodal"
    elif memory.has_active_image():
        state["modo"] = "multimodal"
        state["tiene_imagen"] = True
        state["imagen_path"] = memory.get_active_image_path()
    elif state["solicita_imagenes"]:
        state["modo"] = "solicitud_imagenes"
    else:
        state["modo"] = "texto"

    state["trayectoria"].append({"nodo": "inicializar", "modo": state["modo"]})
    return state
