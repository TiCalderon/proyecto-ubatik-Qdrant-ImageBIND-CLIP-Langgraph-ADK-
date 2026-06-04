import re
from typing import List, Dict
from src.config import Config


class TextChunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

    def chunk_text(self, text: str, source: str = "", page_num: int = 0) -> list[dict]:
        if not text or not text.strip():
            return []
        clean = re.sub(r'\s+', ' ', text).strip()
        chunks = []
        start = 0
        chunk_idx = 0
        while start < len(clean):
            end = min(start + self.chunk_size, len(clean))
            chunk_text = clean[start:end]
            if len(chunk_text.strip()) > 20:
                chunks.append({
                    "id": f"{source}_p{page_num}_c{chunk_idx}",
                    "texto": chunk_text,
                    "fuente": source,
                    "pagina": page_num,
                    "chunk_idx": chunk_idx,
                })
            chunk_idx += 1
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def chunk_pages(self, pages: list[dict]) -> list[dict]:
        all_chunks = []
        for page in pages:
            chunks = self.chunk_text(
                text=page["texto"],
                source=page["fuente"],
                page_num=page["numero"],
            )
            all_chunks.extend(chunks)
        return all_chunks
