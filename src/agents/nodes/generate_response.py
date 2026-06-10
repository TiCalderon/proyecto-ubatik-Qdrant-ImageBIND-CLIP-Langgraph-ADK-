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

SYSTEM_MULTIMODAL = """Eres un asistente de histologia. Responde SOLO con el contenido del manual o la imagen visible en el chat.

REGLAS FUNDAMENTALES:
1. PRIORIDAD ABSOLUTA: La DESCRIPCION TEXTUAL DEL MANUAL de las imágenes recuperadas es la fuente de verdad. Si el texto del manual dice 'Tejido nervioso corteza cerebelosa', ESO es lo correcto, sin importar tu propia interpretación visual de la imagen.
   ATENCION: La imagen marcada como '⭐ MEJOR MATCH VISUAL' es el resultado de una comparación matemática directa imagen-vs-referencia y tiene MÁXIMA PRIORIDAD. Usa la descripción del manual CORRESPONDIENTE A ESTA ESTRUCTURA para tu diagnóstico.
2. Cita: [Manual: archivo] | [Imagen: archivo]
3. Para cada 'IMAGEN DE REFERENCIA' recuperada, indica el nombre y la descripción textual del manual.
4. NO hagas diagnósticos propios basados en tu interpretación visual. Usa SIEMPRE el texto del manual asociado a la imagen.
5. No des diagnósticos clínicos salvo que estén explícitos en el manual.

Contexto del manual (Texto e Imágenes recuperadas):
{contexto}

Analisis preliminar de la imagen subida:
{analisis_imagen}

Historial: {historial}

Responde en español, con estas secciones:
1. ESTRUCTURA IDENTIFICADA: (obligatorio, basado en el MEJOR MATCH VISUAL)
2. DESCRIPCION: según el manual para esa estructura
3. COMPARACION: con las otras imágenes de referencia
4. FUENTES: cita las páginas del manual"""

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
        for c in state.get("contexto_filtrado", {}).get("texto", [])
    ])

    contexto_imagenes = []
    for i, c in enumerate(state.get("contexto_filtrado", {}).get("imagenes", [])):
        marcador = " ⭐ MEJOR MATCH VISUAL" if i == 0 else ""
        bloque = f"[{c['etiqueta']} p.{c['pagina']} score={c['score']:.2f}{marcador}]\n"
        bloque += f"Caption: {c['caption']}"
        contexto_imagenes.append(bloque)
    contexto_imagenes_str = "\n\n".join(contexto_imagenes)

    contexto_completo = contexto_textos
    if contexto_imagenes_str:
        contexto_completo += "\n\n--- IMAGENES RECUPERADAS (SIMILITUD VISUAL) ---\n\n" + contexto_imagenes_str

    if modo == "texto":
        prompt = SYSTEM_TEXTO.format(
            fuente="manual_histologia",
            historial=historial,
            contexto=contexto_completo or "No se encontro contexto relevante.",
        )
        user_msg = state["query_reescrita"]

    elif modo == "solicitud_imagenes":
        prompt = SYSTEM_SOLICITUD_IMAGENES.format(
            historial=historial,
            contexto=contexto_completo or "No se encontro contexto.",
        )
        user_msg = state["query_reescrita"]

    elif modo == "multimodal":
        prompt = SYSTEM_MULTIMODAL.format(
            contexto=contexto_completo or "No se encontro contexto.",
            analisis_imagen=state.get("imagen_analisis", "No disponible"),
            historial=historial,
        )
        user_msg = state["query_reescrita"]
    else:
        prompt = SYSTEM_TEXTO.format(
            fuente="manual_histologia",
            historial=historial,
            contexto=contexto_completo or "No disponible",
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
