import logging
import numpy as np
from src.models.embeddings import MultimodalEmbedder
from src.config import Config

logger = logging.getLogger(__name__)


class ClasificadorSemantico:
    def __init__(self, embedder: MultimodalEmbedder = None):
        self.embedder = embedder or MultimodalEmbedder()
        self.anchors = Config.TEMARIO_ANCHORS
        self._anchor_embeddings = None

    @property
    def anchor_embeddings(self):
        if self._anchor_embeddings is None:
            self._anchor_embeddings = []
            for anchor in self.anchors:
                emb = self.embedder.embed_text(anchor)
                self._anchor_embeddings.append(emb)
        return self._anchor_embeddings

    def clasificar(self, query: str, query_embedding: list[float] = None) -> bool:
        if query_embedding is None:
            query_embedding = self.embedder.embed_text(query)
        query_vec = np.array(query_embedding)

        max_sim = 0.0
        for anchor_emb in self.anchor_embeddings:
            sim = float(np.dot(query_vec, anchor_emb))
            if sim > max_sim:
                max_sim = sim

        threshold = Config.CLASIFICADOR_SEMANTICO_THRESHOLD
        es_valido = max_sim >= threshold

        logger.debug(f"Clasificacion semantica: max_sim={max_sim:.4f}, threshold={threshold}, valido={es_valido}")
        return es_valido
