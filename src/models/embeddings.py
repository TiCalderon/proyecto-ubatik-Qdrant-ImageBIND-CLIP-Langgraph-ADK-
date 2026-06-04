import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from src.config import Config


class ClipEmbedder:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._clip_model = None
        self._clip_processor = None
        self._text_model = None

    @property
    def clip_model(self):
        if self._clip_model is None:
            model_id = "openai/clip-vit-base-patch32"
            self._clip_model = CLIPModel.from_pretrained(
                model_id, token=Config.HF_TOKEN or True
            ).to(self.device).eval()
            self._clip_processor = CLIPProcessor.from_pretrained(model_id)
        return self._clip_model

    @property
    def clip_processor(self):
        if self._clip_processor is None:
            _ = self.clip_model
        return self._clip_processor

    @property
    def text_model(self):
        if self._text_model is None:
            self._text_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device=self.device,
            )
        return self._text_model

    def embed_text(self, text: str, use_minilm: bool = False) -> np.ndarray:
        if use_minilm:
            return self.text_model.encode(text, normalize_embeddings=True)
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self.clip_model.get_text_features(**inputs)
        emb = emb.cpu().numpy().flatten()
        return emb / np.linalg.norm(emb)

    def embed_texts(self, texts: list[str], use_minilm: bool = False) -> list[np.ndarray]:
        if use_minilm:
            return self.text_model.encode(texts, normalize_embeddings=True)
        results = []
        for text in texts:
            results.append(self.embed_text(text, use_minilm=False))
        return results

    def embed_image(self, image_path: str, preprocess: bool = True) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        if preprocess:
            img = self._preprocess_image(img)
        inputs = self.clip_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self.clip_model.get_image_features(**inputs)
        emb = emb.cpu().numpy().flatten()
        return emb / np.linalg.norm(emb)

    def embed_image_pil(self, pil_image: Image.Image, preprocess: bool = True) -> np.ndarray:
        img = pil_image.convert("RGB")
        if preprocess:
            img = self._preprocess_image(img)
        inputs = self.clip_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = self.clip_model.get_image_features(**inputs)
        emb = emb.cpu().numpy().flatten()
        return emb / np.linalg.norm(emb)

    @staticmethod
    def _preprocess_image(img: Image.Image) -> Image.Image:
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
        return img

    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b))

    def compute_similarities(self, query_vec: np.ndarray, candidates: list[np.ndarray]) -> list[float]:
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        return [float(np.dot(query_vec, c / (np.linalg.norm(c) + 1e-8))) for c in candidates]
