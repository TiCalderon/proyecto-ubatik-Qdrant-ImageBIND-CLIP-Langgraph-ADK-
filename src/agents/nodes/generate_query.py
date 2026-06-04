import logging
from src.agents.state import AgentState
from src.models.llm import LLMProvider

logger = logging.getLogger(__name__)

SYSTEM_GENERAR_CONSULTA = """Genera 2 consultas de busqueda optimizadas para recuperar informacion de un manual de histologia.

Contexto: {contexto}
Entidades detectadas: {entidades}
Modo: {modo}

Responde en JSON:
{{
    "consulta_texto": "consulta para busqueda semantica de texto",
    "consulta_visual": "descripcion visual para busqueda de imagenes (solo si hay imagen)"
}}

La consulta_texto debe ser concisa y enfocada en terminos histologicos del dominio."""


async def nodo_generar_consulta(state: AgentState, embedder=None) -> AgentState:
    state["trayectoria"].append({"nodo": "generar_consulta", "accion": "inicio"})

    if not state["en_temario"]:
        state["trayectoria"].append({"nodo": "generar_consulta", "estado": "fuera_temario"})
        return state

    contexto = state["query_reescrita"][:500]
    if state["imagen_analisis"]:
        contexto += "\nAnalisis de imagen: " + state["imagen_analisis"][:400]

    try:
        prompt = SYSTEM_GENERAR_CONSULTA.format(
            contexto=contexto,
            entidades=str(state["entidades"]),
            modo=state["modo"],
        )
        resp = await LLMProvider.invoke_text(
            system_prompt="Genera consultas de busqueda optimizadas.",
            user_prompt=prompt,
        )
        import json
        resp_clean = resp.strip()
        if "```json" in resp_clean:
            resp_clean = resp_clean.split("```json")[1].split("```")[0]
        elif "```" in resp_clean:
            resp_clean = resp_clean.split("```")[1].split("```")[0]
        consultas = json.loads(resp_clean)
        state["trayectoria"].append({"nodo": "generar_consulta", "consultas": consultas})
    except Exception as e:
        logger.warning(f"Error generando consultas: {e}")
        state["trayectoria"].append({"nodo": "generar_consulta", "error": str(e)})

    return state
