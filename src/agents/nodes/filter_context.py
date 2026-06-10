import logging
from src.agents.state import AgentState

logger = logging.getLogger(__name__)


async def nodo_filtrar_contexto(state: AgentState) -> AgentState:
    state["trayectoria"].append({"nodo": "filtrar_contexto", "accion": "inicio"})

    resultados = state["resultados_busqueda"]
    contexto = {"texto": [], "imagenes": []}

    seen_text = set()
    for score, payload in resultados.get("texto", []):
        txt = payload.get("texto", "")
        key = txt[:100]
        if key not in seen_text:
            seen_text.add(key)
            contexto["texto"].append({
                "score": score,
                "texto": txt,
                "fuente": payload.get("fuente", ""),
                "pagina": payload.get("pagina", 0),
            })

    seen_img = set()
    for score, payload in resultados.get("imagenes", []):
        path = payload.get("path", "")
        if path and path not in seen_img:
            seen_img.add(path)
            contexto["imagenes"].append({
                "score": score,
                "path": path,
                "nombre_archivo": payload.get("nombre_archivo", ""),
                "etiqueta": payload.get("etiqueta", ""),
                "caption": payload.get("caption", ""),
                "pagina": payload.get("pagina", 0),
            })

    contexto["texto"].sort(key=lambda x: x["score"], reverse=True)
    contexto["imagenes"].sort(key=lambda x: x["score"], reverse=True)

    max_results = 10
    contexto["texto"] = contexto["texto"][:max_results]
    contexto["imagenes"] = contexto["imagenes"][:10]

    state["contexto_filtrado"] = contexto
    state["trayectoria"].append({
        "nodo": "filtrar_contexto",
        "resultado": f"texto={len(contexto['texto'])}, imagenes={len(contexto['imagenes'])}",
    })
    return state
