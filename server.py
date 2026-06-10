#!/usr/bin/env python3
import os
import sys
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.agents.graph import AsistenteHistologia
from src.api.routes import setup_routes
from src.config import Config

class ColorFormatter(logging.Formatter):
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    def format(self, record):
        if record.levelno >= logging.ERROR:
            # Pinta todo el mensaje de error en rojo
            return f"{self.RED}{super().format(record)}{self.RESET}"
        elif record.levelno == logging.WARNING:
            return f"{self.YELLOW}{super().format(record)}{self.RESET}"
        return super().format(record)

_handler = logging.StreamHandler()
_handler.setFormatter(ColorFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
logger = logging.getLogger("server")

asistente = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global asistente
    import src.api.routes as routes_mod
    logger.info("Inicializando Asistente de Histologia...")
    try:
        asistente = AsistenteHistologia()
        routes_mod.asistente = asistente
        asistente.indexer.ensure_collections()
        
        from src.ingestion.pipeline import ingest_pdfs
        logger.info("Verificando PDFs en la carpeta data/pdf para auto-indexacion...")
        ingest_pdfs(reindex=False, indexer=asistente.indexer)
        
        status = asistente.get_status()
        logger.info(f"Asistente listo: {status}")
    except Exception as e:
        logger.error(f"Error inicializando: {e}")
        raise
    yield
    logger.info("Apagando...")
    routes_mod.asistente = None
    asistente = None


app = FastAPI(
    title="Ubatik RAG Histologia — Grupo 3",
    description="RAG Multimodal de Histologia con Qdrant + CLIP + LangGraph",
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_routes(app)

client_dir = os.path.join(os.path.dirname(__file__), "client")
if os.path.isdir(client_dir):
    app.mount("/client", StaticFiles(directory=client_dir), name="client")

imagenes_dir = os.path.join(os.path.dirname(__file__), "data", "imagenes_extraidas")
if os.path.isdir(imagenes_dir):
    app.mount("/imagenes_extraidas", StaticFiles(directory=imagenes_dir), name="imagenes_extraidas")

import starlette.responses as _responses_module


@app.get("/")
async def root():
    index_path = os.path.join(os.path.dirname(__file__), "client", "index.html")
    if os.path.isfile(index_path):
        return _responses_module.FileResponse(index_path)
    return {"message": "Ubatik RAG Histologia API v5.0.0 — Grupo 3"}


@app.get("/app.js")
async def app_js():
    path = os.path.join(os.path.dirname(__file__), "client", "app.js")
    return _responses_module.FileResponse(path, media_type="application/javascript")


@app.get("/style.css")
async def style_css():
    path = os.path.join(os.path.dirname(__file__), "client", "style.css")
    return _responses_module.FileResponse(path, media_type="text/css")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=True,
        log_level="info",
    )
