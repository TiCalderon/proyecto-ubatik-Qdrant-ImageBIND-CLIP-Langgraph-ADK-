import os
import re
import logging
from typing import Optional
from PIL import Image, ImageEnhance
import fitz
import pytesseract
from pdf2image import convert_from_path
from src.config import Config

logger = logging.getLogger(__name__)

MIN_WIDTH = 150
MIN_HEIGHT = 150
TARGET_SIZE = 868


class PDFExtractor:
    def __init__(self, pdf_path: str, output_dir: str = None):
        self.pdf_path = pdf_path
        self.pdf_filename = os.path.basename(pdf_path)
        self.output_dir = output_dir or Config.DIRECTORIO_IMAGENES_EXTRAIDAS
        os.makedirs(self.output_dir, exist_ok=True)
        self.doc = None

    def open(self):
        self.doc = fitz.open(self.pdf_path)
        return self

    def close(self):
        if self.doc:
            self.doc.close()
            self.doc = None

    def extract_full_text(self) -> str:
        if not self.doc:
            self.open()
        full_text = []
        for page in self.doc:
            text = page.get_text("text")
            full_text.append(text)
        return "\n\n".join(full_text)

    def extract_text_by_page(self) -> list[dict]:
        if not self.doc:
            self.open()
        pages = []
        for i, page in enumerate(self.doc):
            text = page.get_text("text")
            pages.append({
                "numero": i + 1,
                "texto": text,
                "fuente": self.pdf_filename,
            })
        return pages

    def extraer_imagenes(
        self,
        min_width: int = MIN_WIDTH,
        min_height: int = MIN_HEIGHT,
        target_size: int = TARGET_SIZE,
    ) -> list[dict]:
        if not self.doc:
            self.open()
        imagenes = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            pix = page.get_pixmap(dpi=150)
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', self.pdf_filename.replace(".pdf", ""))
            filename = f"{safe_name}_pag{page_num+1:03d}.png"
            output_path = os.path.join(self.output_dir, filename)
            pix.save(output_path)
            
            page_text = page.get_text("text")
            
            # Since we render the whole page, OCR is not strictly needed because we have page_text,
            # but we can keep it empty to save processing time.
            ocr_text = ""
            caption = self._extraer_caption(page_text, {}, page)
            label = self._extraer_etiqueta(page_text)
            
            imagenes.append({
                "path": output_path,
                "nombre_archivo": filename,
                "fuente_pdf": self.pdf_filename,
                "pagina": page_num + 1,
                "ocr_text": ocr_text,
                "texto_pagina": page_text.strip()[:2000],
                "caption": caption,
                "etiqueta": label,
                "width": pix.width,
                "height": pix.height,
            })
        return imagenes

    @staticmethod
    def _preprocess_image(img: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
        return img

    @staticmethod
    def _magnify_image(img: Image.Image, target: int) -> Image.Image:
        w, h = img.size
        if w >= h:
            ratio = target / w
        else:
            ratio = target / h
        new_w, new_h = int(w * ratio), int(h * ratio)
        return img.resize((new_w, new_h), Image.LANCZOS)

    @staticmethod
    def _extraer_ocr(image_bytes: bytes) -> str:
        try:
            img = Image.open(__import__("io").BytesIO(image_bytes))
            return pytesseract.image_to_string(img, lang="spa")
        except Exception:
            return ""

    @staticmethod
    def _extraer_caption(page_text: str, img_info: dict, page) -> str:
        text_lower = page_text.lower()
        keywords = ["figura", "fig.", "imagen", "lamina", "lámina", "fotografia", "fotografía"]
        best = ""
        for kw in keywords:
            idx = text_lower.find(kw)
            if idx >= 0:
                best = page_text[idx:idx+300].strip()
                break
        return best

    @staticmethod
    def _extraer_etiqueta(page_text: str) -> str:
        patterns = [
            r"(?:Imagen|Figura|Fig\.|Lámina|Lamina)\s*([\d]+\.[\d]+)",
            r"(?:Imagen|Figura|Fig\.|Lámina|Lamina)\s*([\d]+)",
        ]
        for pat in patterns:
            match = re.search(pat, page_text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        return ""
