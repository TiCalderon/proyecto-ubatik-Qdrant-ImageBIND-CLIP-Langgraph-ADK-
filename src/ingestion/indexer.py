import os
import uuid
import logging
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from src.config import Config
from src.models.embeddings import MultimodalEmbedder

logger = logging.getLogger(__name__)


class QdrantIndexer:
    def __init__(self, url: str = None, api_key: str = None, embedder: MultimodalEmbedder = None):
        self.url = url or Config.QDRANT_URL
        self.api_key = api_key or Config.QDRANT_API_KEY
        self.embedder = embedder or MultimodalEmbedder()
        self.client = QdrantClient(path=Config.DIRECTORIO_QDRANT_MEMORIA)
        self.col_texto = Config.QDRANT_COLLECTION_TEXTO
        self.col_imagenes = Config.QDRANT_COLLECTION_IMAGENES

    def _create_collection_if_not_exists(self, collection_name: str, vectors_config):
        try:
            self.client.get_collection(collection_name)
            logger.info(f"Coleccion {collection_name} ya existe")
        except Exception:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )
            logger.info(f"Coleccion {collection_name} creada")

    def ensure_collections(self):
        self._create_collection_if_not_exists(
            self.col_texto, 
            qmodels.VectorParams(size=Config.DIM_TEXTO, distance=qmodels.Distance.COSINE)
        )
        self._create_collection_if_not_exists(
            self.col_imagenes, 
            {
                "uni": qmodels.VectorParams(size=Config.DIM_UNI, distance=qmodels.Distance.COSINE),
                "plip": qmodels.VectorParams(size=Config.DIM_PLIP, distance=qmodels.Distance.COSINE)
            }
        )

    def index_chunks(self, chunks: list[dict]) -> int:
        self.ensure_collections()
        points = []
        for i, chunk in enumerate(chunks):
            try:
                vec = self.embedder.embed_text(chunk["texto"])
                points.append(qmodels.PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vec.tolist(),
                    payload={
                        "tipo": "texto",
                        "texto": chunk["texto"],
                        "fuente": chunk["fuente"],
                        "pagina": chunk["pagina"],
                        "chunk_idx": chunk["chunk_idx"],
                        "id_chunk": chunk["id"],
                    },
                ))
            except Exception as e:
                logger.warning(f"Error indexando chunk {chunk.get('id')}: {e}")
            if len(points) >= 50:
                self.client.upsert(collection_name=self.col_texto, points=points)
                points = []
        if points:
            self.client.upsert(collection_name=self.col_texto, points=points)
        return len(chunks)

    def index_images(self, images: list[dict]) -> int:
        self.ensure_collections()
        points = []
        for i, img in enumerate(images):
            try:
                if not os.path.exists(img["path"]):
                    logger.warning(f"Imagen no encontrada: {img['path']}")
                    continue
                vecs = self.embedder.embed_image(img["path"], preprocess=False)
                points.append(qmodels.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={"uni": vecs["uni"].tolist(), "plip": vecs["plip"].tolist()},
                    payload={
                        "tipo": "imagen",
                        "path": img["path"],
                        "nombre_archivo": img["nombre_archivo"],
                        "fuente_pdf": img["fuente_pdf"],
                        "pagina": img["pagina"],
                        "ocr_text": img["ocr_text"],
                        "texto_pagina": img["texto_pagina"],
                        "caption": img["caption"],
                        "etiqueta": img["etiqueta"],
                        "width": img["width"],
                        "height": img["height"],
                    },
                ))
            except Exception as e:
                logger.warning(f"Error indexando imagen pag {img.get('pagina')}: {e}")
            if len(points) >= 50:
                self.client.upsert(collection_name=self.col_imagenes, points=points)
                points = []
        if points:
            self.client.upsert(collection_name=self.col_imagenes, points=points)
        return len(images)

    def text_search(self, query_vec: list[float], top_k: int = None, threshold: float = None) -> list:
        top_k = top_k or Config.TOP_K_TEXTO
        threshold = threshold or Config.SIMILARITY_THRESHOLD_TEXTO
        response = self.client.query_points(
            collection_name=self.col_texto,
            query=query_vec,
            limit=top_k,
            score_threshold=threshold,
        )
        return [(r.score, r.payload) for r in response.points]

    def image_search(self, query_vec: list[float], top_k: int = None, threshold: float = None, using: str = "uni") -> list:
        top_k = top_k or Config.TOP_K_IMAGEN
        threshold = threshold or Config.SIMILARITY_THRESHOLD_IMAGEN
        response = self.client.query_points(
            collection_name=self.col_imagenes,
            query=query_vec,
            using=using,
            limit=top_k,
            score_threshold=threshold,
        )
        return [(r.score, r.payload) for r in response.points]

    def hybrid_search(
        self,
        text_vec: list[float],
        image_vec: list[float] = None,
        text_top_k: int = None,
        image_top_k: int = None,
        text_threshold: float = None,
        image_threshold: float = None,
        include_images: bool = True,
    ) -> dict:
        resultados = {"texto": [], "imagenes": []}
        if text_vec:
            resultados["texto"] = self.text_search(
                text_vec, top_k=text_top_k, threshold=text_threshold
            )
        if image_vec and include_images:
            resultados["imagenes"] = self.image_search(
                image_vec, top_k=image_top_k, threshold=image_threshold, using="uni"
            )
        elif text_vec and include_images:
            # If no image_vec provided but we want images, we search by text using PLIP
            # We lower the threshold to 0.20 for PLIP zero-shot text-to-image
            resultados["imagenes"] = self.image_search(
                text_vec, top_k=image_top_k, threshold=0.20, using="plip"
            )
            
            # Recuperación determinística: traer imágenes de la misma página que los mejores chunks de texto
            if resultados["texto"]:
                page_filters = set()
                for score, payload in resultados["texto"][:10]: # top 10 text chunks
                    if "fuente" in payload and "pagina" in payload:
                        page_filters.add((payload["fuente"], payload["pagina"]))
                
                for f_pdf, pag in page_filters:
                    try:
                        scroll_res, _ = self.client.scroll(
                            collection_name=self.col_imagenes,
                            scroll_filter=qmodels.Filter(
                                must=[
                                    qmodels.FieldCondition(key="fuente_pdf", match=qmodels.MatchValue(value=f_pdf)),
                                    qmodels.FieldCondition(key="pagina", match=qmodels.MatchValue(value=pag)),
                                ]
                            ),
                            limit=5
                        )
                        for point in scroll_res:
                            # Avoid duplicates
                            if not any(img_payload.get("nombre_archivo") == point.payload.get("nombre_archivo") for _, img_payload in resultados["imagenes"]):
                                # Add with high pseudo-score because it's exactly on the referenced page
                                resultados["imagenes"].append((0.99, point.payload)) 
                    except Exception as e:
                        logger.warning(f"Error en recuperación determinística de imágenes: {e}")
                
                # Sort images by score
                resultados["imagenes"].sort(key=lambda x: x[0], reverse=True)
                # Keep top K
                image_top_k = image_top_k or Config.TOP_K_IMAGEN
                resultados["imagenes"] = resultados["imagenes"][:image_top_k]
                
        return resultados

    def count_points(self) -> dict:
        try:
            tc = self.client.count(collection_name=self.col_texto).count
        except Exception:
            tc = 0
        try:
            ic = self.client.count(collection_name=self.col_imagenes).count
        except Exception:
            ic = 0
        return {"texto": tc, "imagenes": ic}

    def delete_all(self):
        try:
            self.client.delete_collection(self.col_texto)
            self.client.delete_collection(self.col_imagenes)
        except Exception:
            pass
