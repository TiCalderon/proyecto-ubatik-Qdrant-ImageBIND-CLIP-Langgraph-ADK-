import os
import numpy as np
import torch
from PIL import Image
import timm
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import login
from src.config import Config


class MultimodalEmbedder:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._uni_model = None
        self._uni_transform = None
        self._plip_model = None
        self._plip_processor = None
        
        if Config.HF_TOKEN:
            try:
                login(token=Config.HF_TOKEN)
            except Exception as e:
                print(f"Error login HF: {e}")

    @property
    def plip_model(self):
        if self._plip_model is None:
            self._plip_model = CLIPModel.from_pretrained("vinid/plip").to(self.device).eval()
            self._plip_processor = CLIPProcessor.from_pretrained("vinid/plip")
        return self._plip_model

    @property
    def plip_processor(self):
        if self._plip_processor is None:
            _ = self.plip_model
        return self._plip_processor

    @property
    def uni_model(self):
        if self._uni_model is None:
            self._uni_model = timm.create_model(
                "hf_hub:MahmoodLab/UNI", 
                pretrained=True, 
                init_values=1e-5, 
                dynamic_img_size=True
            ).to(self.device).eval()
        return self._uni_model

    @property
    def uni_transform(self):
        if self._uni_transform is None:
            from torchvision import transforms
            self._uni_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        return self._uni_transform

    @staticmethod
    def _preprocess_image(img: Image.Image) -> Image.Image:
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
        return img

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.plip_processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_out = self.plip_model.text_model(**inputs)
            emb = self.plip_model.text_projection(text_out.pooler_output)
        emb = emb.cpu().numpy().flatten()
        return emb / np.linalg.norm(emb)

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        inputs = self.plip_processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_out = self.plip_model.text_model(**inputs)
            embs = self.plip_model.text_projection(text_out.pooler_output)
        embs = embs.cpu().numpy()
        return [e / np.linalg.norm(e) for e in embs]

    def embed_image(self, image_path: str, preprocess: bool = True) -> dict:
        img = Image.open(image_path).convert("RGB")
        if preprocess:
            img = self._preprocess_image(img)
        
        # PLIP
        plip_inputs = self.plip_processor(images=img, return_tensors="pt")
        plip_inputs = {k: v.to(self.device) for k, v in plip_inputs.items()}
        with torch.no_grad():
            vision_out = self.plip_model.vision_model(**plip_inputs)
            plip_emb = self.plip_model.visual_projection(vision_out.pooler_output)
        plip_emb = plip_emb.cpu().numpy().flatten()
        plip_emb = plip_emb / np.linalg.norm(plip_emb)

        # UNI
        uni_tensor = self.uni_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            uni_emb = self.uni_model(uni_tensor)
        uni_emb = uni_emb.cpu().numpy().flatten()
        
        return {
            "uni": uni_emb,
            "plip": plip_emb
        }

    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b))

    def compute_similarities(self, query_vec: np.ndarray, candidates: list[np.ndarray]) -> list[float]:
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        return [float(np.dot(query_vec, c / (np.linalg.norm(c) + 1e-8))) for c in candidates]
