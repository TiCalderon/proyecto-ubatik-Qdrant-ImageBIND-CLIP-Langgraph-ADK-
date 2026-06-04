import os
import logging
from src.ingestion.extractor import PDFExtractor
from src.ingestion.chunker import TextChunker
from src.ingestion.indexer import QdrantIndexer
from src.config import Config

logger = logging.getLogger(__name__)


def ingest_pdfs(pdf_dir: str = None, reindex: bool = False):
    pdf_dir = pdf_dir or Config.DIRECTORIO_PDFS
    if not os.path.isdir(pdf_dir):
        logger.error(f"Directorio no encontrado: {pdf_dir}")
        return

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning(f"No hay PDFs en {pdf_dir}")
        return
    logger.info(f"PDFs encontrados: {pdf_files}")

    indexer = QdrantIndexer()
    if reindex:
        logger.info("Reindexando: borrando colecciones existentes...")
        indexer.delete_all()
    indexer.ensure_collections()

    total_chunks = 0
    total_images = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        logger.info(f"Procesando: {pdf_file}")

        extractor = PDFExtractor(pdf_path)
        extractor.open()

        pages = extractor.extract_text_by_page()
        chunker = TextChunker()
        chunks = chunker.chunk_pages(pages)
        n_chunks = indexer.index_chunks(chunks)
        total_chunks += n_chunks
        logger.info(f"  Chunks indexados: {n_chunks}")

        images = extractor.extraer_imagenes()
        if images:
            n_images = indexer.index_images(images)
            total_images += n_images
            logger.info(f"  Imagenes indexadas: {n_images}")

        extractor.close()

    counts = indexer.count_points()
    logger.info(f"Ingestion completada: {total_chunks} chunks, {total_images} imagenes")
    logger.info(f"Total en Qdrant: {counts}")
    return {"chunks": total_chunks, "imagenes": total_images, "counts": counts}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_pdfs(reindex=True)
