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
