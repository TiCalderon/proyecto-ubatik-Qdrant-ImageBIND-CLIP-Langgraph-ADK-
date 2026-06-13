import os
import base64
import logging
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from src.agents.graph import AsistenteHistologia

logger = logging.getLogger(__name__)

asistente: AsistenteHistologia = None


class ChatRequest(BaseModel):
    query: str
    image_base64: Optional[str] = ""
    image_filename: Optional[str] = ""


class ChatResponse(BaseModel):
    respuesta: str
    estructura_identificada: str = ""
    imagenes_recuperadas: list[dict] = []
    imagenes_base64: list[str] = []
    trayectoria: list[dict] = []
    imagen_activa: bool = False
    mostrar_imagenes: bool = False
    error: str = ""


class ImageClearResponse(BaseModel):
    ok: bool = True


class StatusResponse(BaseModel):
    ready: bool
    temario_count: int = 0
    chunks_indexed: int = 0
    images_indexed: int = 0
    active_image: bool = False
    device: str = "cpu"


def setup_routes(app: FastAPI):
    @app.get("/api/status", response_model=StatusResponse)
    async def api_status():
        if asistente is None:
            return StatusResponse(ready=False)
        status = asistente.get_status()
        return StatusResponse(**status)

    @app.get("/api/temario")
    async def api_temario():
        return JSONResponse(content={"anclas": asistente.clasificador.anchors if asistente else []})

    @app.post("/api/chat", response_model=ChatResponse)
    async def api_chat(req: ChatRequest):
        if asistente is None:
            raise HTTPException(status_code=503, detail="Asistente no inicializado")
        if not req.query or not req.query.strip():
            raise HTTPException(status_code=400, detail="Query vacia")

        result = await asistente.chat(
            query=req.query.strip(),
            image_base64=req.image_base64 or "",
            image_filename=req.image_filename or "",
        )

        imagenes_b64 = []
        for ir in result.get("imagenes_recuperadas", []):
            if ir.get("base64"):
                imagenes_b64.append(ir["base64"])

        return ChatResponse(
            respuesta=result.get("respuesta", ""),
            estructura_identificada=result.get("estructura_identificada", ""),
            imagenes_recuperadas=result.get("imagenes_recuperadas", []),
            imagenes_base64=imagenes_b64,
            trayectoria=result.get("trayectoria", []),
            imagen_activa=asistente.memory.has_active_image(),
            mostrar_imagenes=result.get("mostrar_imagenes", False),
            error=result.get("error", ""),
        )

    @app.post("/api/imagen/limpiar", response_model=ImageClearResponse)
    async def api_clear_image():
        if asistente:
            asistente.memory.clear_image()
        return ImageClearResponse(ok=True)

    # Estado compartido de reindexación
    _reindex_state = {"running": False, "resultado": None, "error": None}

    @app.post("/api/reindex")
    async def api_reindex(background_tasks: __import__("fastapi").BackgroundTasks):
        from fastapi import BackgroundTasks
        if _reindex_state["running"]:
            return JSONResponse(content={"ok": False, "message": "Ya hay una reindexación en curso."}, status_code=409)

        def _run_reindex():
            from src.ingestion.pipeline import ingest_pdfs
            _reindex_state["running"] = True
            _reindex_state["resultado"] = None
            _reindex_state["error"] = None
            try:
                resultado = ingest_pdfs(reindex=True, indexer=asistente.indexer if asistente else None)
                _reindex_state["resultado"] = resultado
            except Exception as e:
                logger.error(f"Error en reindexación: {e}")
                _reindex_state["error"] = str(e)
            finally:
                _reindex_state["running"] = False

        background_tasks.add_task(_run_reindex)
        return JSONResponse(content={"ok": True, "message": "Reindexación iniciada en segundo plano."})

    @app.get("/api/reindex/status")
    async def api_reindex_status():
        return JSONResponse(content={
            "running": _reindex_state["running"],
            "resultado": _reindex_state["resultado"],
            "error": _reindex_state["error"],
        })
