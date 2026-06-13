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
        import io
        imagenes = []
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', self.pdf_filename.replace(".pdf", ""))

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            page_text = page.get_text("text")
            image_list = page.get_images(full=True)

            # Recopilar candidatos: imágenes individuales a color
            candidatos = []
            seen_xrefs = set()
            for img_info in image_list:
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                
                # CRÍTICO: Verificar que la imagen realmente se dibuja en esta página
                if not page.get_image_rects(xref):
                    continue

                try:
                    base_image = self.doc.extract_image(xref)
                except Exception:
                    continue
                if not base_image:
                    continue
                w, h = base_image["width"], base_image["height"]
                if w < min_width or h < min_height:
                    continue
                
                try:
                    img = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                    # Calcular área para elegir la más grande si hay varias
                    candidatos.append({
                        "xref": xref,
                        "bytes": base_image["image"],
                        "img": img,
                        "width": w,
                        "height": h,
                        "area": w * h,
                    })
                except Exception as e:
                    logger.debug(f"Pag {page_num+1} xref={xref}: error cargando imagen: {e}")
                    continue

            if candidatos:
                # Elegir la imagen más grande dibujada en la página
                candidatos.sort(key=lambda c: c["area"], reverse=True)
                elegida = candidatos[0]
                img = elegida["img"]
                img = self._preprocess_image(img)
                filename = f"{safe_name}_pag{page_num+1:03d}.png"
                output_path = os.path.join(self.output_dir, filename)
                img.save(output_path, "PNG")
                ocr_text = ""
                caption = self._extraer_caption(page_text, elegida, page)
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
                    "width": elegida["width"],
                    "height": elegida["height"],
                })
                logger.info(f"Pag {page_num+1}: imagen individual extraída ({elegida['width']}x{elegida['height']})")
            else:
                # Fallback: renderizar la página completa (no se encontró imagen de contenido)
                pix = page.get_pixmap(dpi=150)
                filename = f"{safe_name}_pag{page_num+1:03d}_render.png"
                output_path = os.path.join(self.output_dir, filename)
                pix.save(output_path)
                caption = self._extraer_caption(page_text, {}, page)
                label = self._extraer_etiqueta(page_text)
                imagenes.append({
                    "path": output_path,
                    "nombre_archivo": filename,
                    "fuente_pdf": self.pdf_filename,
                    "pagina": page_num + 1,
                    "ocr_text": "",
                    "texto_pagina": page_text.strip()[:2000],
                    "caption": caption,
                    "etiqueta": label,
                    "width": pix.width,
                    "height": pix.height,
                })
                logger.info(f"Pag {page_num+1}: sin imagen de contenido, se usó render de página completa.")
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
