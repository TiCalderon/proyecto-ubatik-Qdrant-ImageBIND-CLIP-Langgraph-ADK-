import logging
from src.agents.state import AgentState
from src.models.llm import LLMProvider
from src.memory.conversation import ConversationMemory

logger = logging.getLogger(__name__)

SYSTEM_TEXTO = """Eres un asistente de histologia. Responde usando UNICAMENTE el contenido del manual que se te proporciona como contexto. Si la informacion no esta en el contexto, dilo claramente.

Reglas:
- Responde en español, en prosa fluida y natural.
- Cita las fuentes con [Manual: {fuente}].
- No inventes informacion que no este en el contexto.
- Si te piden explicar un concepto, usa la informacion del manual.
- Si hay imagenes referenciadas (ej: "Imagen 15.1"), menciona la etiqueta exacta.

Historial: {historial}
Contexto del manual:
{contexto}"""

SYSTEM_MULTIMODAL = """Eres un patologo experto. El usuario ha subido una imagen histologica para identificacion.

Prioridad maxima: Reporta la ESTRUCTURA IDENTIFICADA basandote en el analisis comparativo y el contexto.

Contexto del manual:
{contexto}

Analisis de la imagen subida:
{analisis_imagen}

Analisis comparativo con imagenes del manual:
{analisis_comparativo}

Historial: {historial}

Responde en español, con estas secciones:
1. ESTRUCTURA IDENTIFICADA: (obligatorio, max prioridad)
2. DESCRIPCION: segun el manual
3. COMPARACION: con las imagenes de referencia
4. FUENTES: cita las paginas del manual"""

SYSTEM_SOLICITUD_IMAGENES = """Eres un asistente de histologia. El usuario quiere ver imagenes del manual.

Reglas:
- Responde en español, prosa natural.
- Menciona las imagenes del manual que sean relevantes usando su etiqueta exacta del formato "Imagen X.X" o "Figura X.X".
- Describe brevemente cada imagen referenciada.

Contexto del manual:
{contexto}

Historial: {historial}"""


async def nodo_generar_respuesta(state: AgentState, memory: ConversationMemory) -> AgentState:
    state["trayectoria"].append({"nodo": "generar_respuesta", "accion": "inicio"})

    modo = state["modo"]
    historial = memory.get_history_text()
    contexto_textos = "\n\n".join([
        f"[{c['fuente']} p.{c['pagina']} score={c['score']:.2f}] {c['texto']}"
        for c in state["contexto_filtrado"]["texto"]
    ])

    if modo == "texto":
        prompt = SYSTEM_TEXTO.format(
            fuente="manual_histologia",
            historial=historial,
            contexto=contexto_textos or "No se encontro contexto relevante.",
        )
        user_msg = state["query_reescrita"]

    elif modo == "solicitud_imagenes":
        prompt = SYSTEM_SOLICITUD_IMAGENES.format(
            historial=historial,
            contexto=contexto_textos or "No se encontro contexto.",
        )
        user_msg = state["query_reescrita"]

    elif modo == "multimodal":
        prompt = SYSTEM_MULTIMODAL.format(
            contexto=contexto_textos or "No se encontro contexto.",
            analisis_imagen=state.get("imagen_analisis", "No disponible"),
            analisis_comparativo=state.get("analisis_comparativo", "No disponible"),
            historial=historial,
        )
        user_msg = state["query_reescrita"]
    else:
        prompt = SYSTEM_TEXTO.format(
            fuente="manual_histologia",
            historial=historial,
            contexto=contexto_textos or "No disponible",
        )
        user_msg = state["query_reescrita"]

    try:
        respuesta = await LLMProvider.invoke_text(
            system_prompt=prompt,
            user_prompt=user_msg,
        )
        state["respuesta"] = respuesta
        state["trayectoria"].append({"nodo": "generar_respuesta", "longitud": len(respuesta)})
    except Exception as e:
        logger.error(f"Error generando respuesta: {e}")
        state["respuesta"] = f"Error al generar la respuesta: {e}"
        state["error"] = str(e)

    return state
