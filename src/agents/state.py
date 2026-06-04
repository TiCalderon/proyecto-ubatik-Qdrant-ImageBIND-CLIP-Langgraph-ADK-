from typing import TypedDict, Optional, Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    query: str
    query_original: str
    query_reescrita: str
    tiene_imagen: bool
    imagen_path: str
    imagen_base64: str
    imagen_embedding: list[float]
    imagen_analisis: str
    imagen_activa: bool
    solicita_imagenes: bool
    mostrar_imagenes: bool
    modo: str
    en_temario: bool
    entidades: dict
    terminos_histologicos: list[str]
    texto_embedding: list[float]
    resultados_busqueda: dict
    contexto_filtrado: dict
    analisis_comparativo: str
    estructura_identificada: str
    respuesta: str
    imagenes_detectadas: list[str]
    imagenes_recuperadas: list[dict]
    trayectoria: list[dict]
    error: str
    messages: Annotated[Sequence[BaseMessage], add_messages]


def initial_state(query: str, image_path: str = "", image_base64: str = "") -> AgentState:
    return AgentState(
        query=query,
        query_original=query,
        query_reescrita=query,
        tiene_imagen=bool(image_path or image_base64),
        imagen_path=image_path,
        imagen_base64=image_base64,
        imagen_embedding=[],
        imagen_analisis="",
        imagen_activa=False,
        solicita_imagenes=False,
        mostrar_imagenes=False,
        modo="texto",
        en_temario=True,
        entidades={"tejidos": [], "estructuras": [], "tinciones": []},
        terminos_histologicos=[],
        texto_embedding=[],
        resultados_busqueda={"texto": [], "imagenes": []},
        contexto_filtrado={"texto": [], "imagenes": []},
        analisis_comparativo="",
        estructura_identificada="",
        respuesta="",
        imagenes_detectadas=[],
        imagenes_recuperadas=[],
        trayectoria=[],
        error="",
        messages=[],
    )
