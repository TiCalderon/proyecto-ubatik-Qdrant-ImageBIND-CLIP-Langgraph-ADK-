import logging
from src.agents.state import AgentState
from src.ingestion.indexer import QdrantIndexer
from src.models.embeddings import ClipEmbedder
from src.config import Config

logger = logging.getLogger(__name__)


def buscar_similares_imagen(
    embedder: ClipEmbedder,
    indexer: QdrantIndexer,
    image_path: str,
    top_k: int = None,
    threshold: float = None,
) -> list[dict]:
    top_k = top_k or Config.TOP_K_IMAGEN
    threshold = threshold or Config.SIMILARITY_THRESHOLD_IMAGEN
    vec = embedder.embed_image(image_path)
    return indexer.image_search(vec.tolist(), top_k=top_k, threshold=threshold)


def buscar_imagenes_por_texto(
    embedder: ClipEmbedder,
    indexer: QdrantIndexer,
    texto: str,
    top_k: int = None,
    threshold: float = None,
) -> list[dict]:
    top_k = top_k or Config.TOP_K_IMAGEN
    threshold = threshold or 0.50
    vec = embedder.embed_text(texto)
    return indexer.image_search_by_text(vec.tolist(), top_k=top_k, threshold=threshold)


def fusionar_resultados(resultados_texto: list, resultados_imagenes: list, peso_texto: float = 0.7, peso_imagen: float = 0.3) -> dict:
    return {
        "texto": resultados_texto,
        "imagenes": resultados_imagenes,
    }
