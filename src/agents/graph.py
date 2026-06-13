import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.agents.state import AgentState, initial_state
from src.agents.nodes.initialize import nodo_inicializar
from src.agents.nodes.process_image import nodo_procesar_imagen
from src.agents.nodes.classify import nodo_clasificar
from src.agents.nodes.generate_query import nodo_generar_consulta
from src.agents.nodes.search import nodo_buscar
from src.agents.nodes.filter_context import nodo_filtrar_contexto
from src.agents.nodes.generate_response import nodo_generar_respuesta
from src.agents.nodes.finalize import nodo_finalizar
from src.models.embeddings import MultimodalEmbedder
from src.ingestion.indexer import QdrantIndexer
from src.search.semantic_classifier import ClasificadorSemantico
from src.memory.conversation import ConversationMemory

logger = logging.getLogger(__name__)


class AsistenteHistologia:
    def __init__(self):
        self.embedder = MultimodalEmbedder()
        self.indexer = QdrantIndexer()
        self.clasificador = ClasificadorSemantico(self.embedder)
        self.memory = ConversationMemory()
        self.graph = None
        self.checkpointer = MemorySaver()
        self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("inicializar", self._wrap(nodo_inicializar, self.embedder, self.memory))
        builder.add_node("procesar_imagen", self._wrap(nodo_procesar_imagen, self.embedder))
        builder.add_node("clasificar", self._wrap(nodo_clasificar, self.embedder, self.clasificador))
        builder.add_node("generar_consulta", self._wrap(nodo_generar_consulta, self.embedder))
        builder.add_node("buscar", self._wrap(nodo_buscar, self.indexer))
        builder.add_node("filtrar_contexto", self._wrap(nodo_filtrar_contexto))
        builder.add_node("generar_respuesta", self._wrap(nodo_generar_respuesta, self.memory))
        builder.add_node("finalizar", self._wrap(nodo_finalizar, self.embedder, self.memory, self.indexer))

        builder.add_edge(START, "inicializar")

        builder.add_conditional_edges(
            "inicializar",
            self._route_after_init,
            {
                "procesar_imagen": "procesar_imagen",
                "clasificar": "clasificar",
            },
        )

        builder.add_edge("procesar_imagen", "clasificar")

        builder.add_conditional_edges(
            "clasificar",
            self._route_after_classify,
            {
                "generar_consulta": "generar_consulta",
                "finalizar": "finalizar",
            },
        )

        builder.add_edge("generar_consulta", "buscar")
        builder.add_edge("buscar", "filtrar_contexto")

        builder.add_conditional_edges(
            "filtrar_contexto",
            self._route_after_filter,
            {
                "generar_respuesta": "generar_respuesta",
            },
        )

        builder.add_edge("generar_respuesta", "finalizar")
        builder.add_edge("finalizar", END)

        self.graph = builder.compile(checkpointer=self.checkpointer)

    def _wrap(self, fn, *args):
        import functools

        @functools.wraps(fn)
        async def wrapper(state: AgentState):
            return await fn(state, *args)
        return wrapper

    def _route_after_init(self, state: AgentState) -> str:
        if state["tiene_imagen"]:
            return "procesar_imagen"
        return "clasificar"

    def _route_after_classify(self, state: AgentState) -> str:
        if state["en_temario"]:
            return "generar_consulta"
        return "finalizar"

    def _route_after_filter(self, state: AgentState) -> str:
        return "generar_respuesta"

    async def chat(self, query: str, image_base64: str = "", image_filename: str = "") -> AgentState:
        state = initial_state(query=query, image_base64=image_base64)

        try:
            config = {"configurable": {"thread_id": "session_default"}}
            result = await self.graph.ainvoke(state, config)
            return result
        except Exception as e:
            logger.error(f"Error en el grafo: {e}")
            state["respuesta"] = f"Lo siento, ocurrio un error: {e}"
            state["error"] = str(e)
            return state

    def get_status(self) -> dict:
        counts = self.indexer.count_points()
        return {
            "ready": True,
            "temario_count": len(self.clasificador.anchors),
            "chunks_indexed": counts.get("texto", 0),
            "images_indexed": counts.get("imagenes", 0),
            "active_image": self.memory.has_active_image(),
            "device": self.embedder.device,
        }
