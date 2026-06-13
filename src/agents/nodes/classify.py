import logging
from src.agents.state import AgentState
from src.models.embeddings import MultimodalEmbedder
from src.models.llm import LLMProvider
from src.search.semantic_classifier import ClasificadorSemantico

logger = logging.getLogger(__name__)

SYSTEM_ENTIDADES = """Extrae del siguiente texto las entidades histologicas presentes.
Responde en JSON con este formato exacto:
{{
    "tejidos": ["tejido1", "tejido2"],
    "estructuras": ["estructura1"],
    "tinciones": ["tincion1"]
}}

Maximo 3 elementos por categoria. Si no hay, lista vacia.
Texto: {texto}"""


async def nodo_clasificar(state: AgentState, embedder: MultimodalEmbedder, clasificador: ClasificadorSemantico) -> AgentState:
    state["trayectoria"].append({"nodo": "clasificar", "accion": "inicio"})

    query_text = state["query_reescrita"]
    if state["imagen_analisis"]:
        query_text += " " + state["imagen_analisis"][:500]

    try:
        emb = embedder.embed_text(query_text)
        state["texto_embedding"] = emb.tolist()
    except Exception as e:
        logger.error(f"Error generando texto embedding: {e}")

    try:
        prompt = SYSTEM_ENTIDADES.format(texto=query_text[:1000])
        resp = await LLMProvider.invoke_text(
            system_prompt="Extrae entidades histologicas en JSON.",
            user_prompt=prompt,
        )
        import json
        resp_clean = resp.strip()
        if "```json" in resp_clean:
            resp_clean = resp_clean.split("```json")[1].split("```")[0]
        elif "```" in resp_clean:
            resp_clean = resp_clean.split("```")[1].split("```")[0]
        entidades = json.loads(resp_clean)
        state["entidades"] = entidades
        state["trayectoria"].append({"nodo": "clasificar", "entidades": entidades})
    except Exception as e:
        logger.warning(f"Error extrayendo entidades: {e}")
        state["entidades"] = {"tejidos": [], "estructuras": [], "tinciones": []}

    domain_valid = True
    if state.get("tiene_imagen"):
        logger.debug("Clasificacion semantica: Omitida porque se subio una imagen (siempre en temario).")
    else:
        domain_valid = clasificador.clasificar(query_text, state["texto_embedding"])
        
    state["en_temario"] = domain_valid
    state["trayectoria"].append({"nodo": "clasificar", "en_temario": domain_valid})

    return state
