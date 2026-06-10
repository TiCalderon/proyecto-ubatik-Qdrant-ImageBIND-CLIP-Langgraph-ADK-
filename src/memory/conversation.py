import os
import json
import uuid
import logging
import numpy as np
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from src.models.embeddings import MultimodalEmbedder
from src.config import Config

logger = logging.getLogger(__name__)


class ConversationMemory:
    def __init__(self, embedder: MultimodalEmbedder = None, qdrant_path: str = None):
        self.embedder = embedder or MultimodalEmbedder()
        self.qdrant_path = qdrant_path or Config.DIRECTORIO_QDRANT_MEMORIA
        self.collection = "memoria_histo_g3"
        self._client = None
        self.history: list[dict] = []
        self.active_image_path: str = ""
        self.active_image_analysis: str = ""
        self.turn_count: int = 0
        self._ensure_client()

    @property
    def client(self):
        if self._client is None:
            os.makedirs(self.qdrant_path, exist_ok=True)
            self._client = QdrantClient(path=self.qdrant_path)
            self._ensure_collection()
        return self._client

    def _ensure_client(self):
        pass

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=Config.DIM_TEXTO,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def add_interaction(self, query: str, response: str, image_path: str = "", image_analysis: str = "", structure: str = ""):
        self.turn_count += 1
        interaction = {
            "turno": self.turn_count,
            "query": query,
            "response": response[:500],
            "image_path": image_path,
            "structure": structure,
        }
        self.history.append(interaction)

        if image_path:
            self.active_image_path = image_path
            self.active_image_analysis = image_analysis

        if self.turn_count % Config.MEMORIA_RESUMEN_CADA == 0:
            self._store_summary(interaction)

        if len(self.history) > Config.MAX_MEMORIA_INTERACCIONES:
            self.history = self.history[-Config.MAX_MEMORIA_INTERACCIONES:]

    def _store_summary(self, interaction: dict):
        try:
            text = f"Pregunta: {interaction['query']}\nRespuesta: {interaction['response']}"
            vec = self.embedder.embed_text(text, use_minilm=False)
            self.client.upsert(
                collection_name=self.collection,
                points=[qmodels.PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vec.tolist(),
                    payload={
                        "resumen": text[:1000],
                        "turno_fin": self.turn_count,
                        "tiene_imagen": bool(interaction.get("image_path")),
                        "structure": interaction.get("structure", ""),
                    },
                )],
            )
        except Exception as e:
            logger.warning(f"Error guardando resumen: {e}")

    def get_history_text(self, last_n: int = 3) -> str:
        if not self.history:
            return ""
        recent = self.history[-last_n:]
        lines = []
        for h in recent:
            lines.append(f"Usuario: {h['query']}")
            lines.append(f"Asistente: {h['response'][:300]}")
        return "\n".join(lines)

    def get_context(self, query: str = "") -> str:
        parts = []
        if self.active_image_analysis and self.turn_count <= self.history[-1]["turno"] + 3 if self.history else True:
            parts.append(f"[Imagen activa analisis previo]: {self.active_image_analysis[:400]}")
        return "\n".join(parts)

    def has_active_image(self) -> bool:
        return bool(self.active_image_path and os.path.exists(self.active_image_path))

    def get_active_image_path(self) -> str:
        return self.active_image_path

    def clear_image(self):
        self.active_image_path = ""
        self.active_image_analysis = ""

    def clear_all(self):
        self.history = []
        self.active_image_path = ""
        self.active_image_analysis = ""
        self.turn_count = 0
        try:
            self.client.delete_collection(self.collection)
            self._ensure_collection()
        except Exception:
            pass
