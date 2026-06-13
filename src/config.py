import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION = "histologia_g3_multimodal"
    QDRANT_COLLECTION_TEXTO = "histologia_g3_chunks"
    QDRANT_COLLECTION_IMAGENES = "histologia_g3_imagenes"
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "ubatik-rag-histologia-g3")
    PORT = int(os.getenv("PORT", "10010"))
    HOST = os.getenv("HOST", "0.0.0.0")

    LLM_TEMPERATURE = 0.4
    LLM_MAX_TOKENS = 2048
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 200
    DIM_TEXTO = 512
    DIM_UNI = 1024
    DIM_PLIP = 512
    SIMILARITY_THRESHOLD_TEXTO = 0.65
    SIMILARITY_THRESHOLD_IMAGEN = 0.75
    # Umbral de similitud coseno (query embedding vs caption embedding) para seleccionar imágenes
    IMAGE_CAPTION_SIMILARITY_THRESHOLD = 0.1
    # Cantidad fija de imágenes a devolver al usuario por respuesta
    MAX_IMAGES_POR_RESPUESTA = 3
    CLASIFICADOR_SEMANTICO_THRESHOLD = 0.45
    CLASIFICADOR_IMAGEN_THRESHOLD = 0.27
    TOP_K_TEXTO = 10
    TOP_K_IMAGEN = 10
    MAX_RESULTADOS = 15
    MAX_MEMORIA_INTERACCIONES = 10
    MEMORIA_RESUMEN_CADA = 5
    DIRECTORIO_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    DIRECTORIO_PDFS = os.path.join(DIRECTORIO_DATA, "pdf")
    DIRECTORIO_IMAGENES_EXTRAIDAS = os.path.join(DIRECTORIO_DATA, "imagenes_extraidas")
    DIRECTORIO_IMAGENES_CHAT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "imagenes_chat")
    DIRECTORIO_QDRANT_MEMORIA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qdrant_memoria")

    TEMARIO_ANCHORS = [
        "tejido cartilaginoso hilaino elastico fibroso condrocitos condroblastos",
        "tejido oseo compacto esponjoso osteonas osteocitos osteoblastos osteoclastos",
        "tejido muscular estriado esqueletico cardiaco liso sarcomero",
        "tejido nervioso neuronas piramidales estrelladas piriformes neuroglia",
        "neuroglia astrocitos oligodendrocitos microglia celulas ependimarias",
        "celulas epiteliales epitelio simple estratificado pseudoestratificado",
        "glandulas exocrinas endocrinas acinos conductos excretores",
        "tejido conjuntivo conectivo laxo denso adiposo areolar",
        "vasos sanguineos arterias venas capilares endotelio",
        "organos linfoides ganglio bazo timo amigdala tejido linfoide",
    ]
