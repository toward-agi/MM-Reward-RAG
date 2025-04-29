from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from typing import Union, List
import numpy as np
from pathlib import Path
from .base import MultiModalEmbedder
from config.config import settings

class CLIPEmbedder(MultiModalEmbedder):
    def __init__(self):
        self.config = settings.get_embedder_config("multimodal")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = self.config["models"]["clip"]["name"]
        print(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def embed_images(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]
            
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features.cpu().numpy()
    
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, max_length=77)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
        # Normalize features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features.cpu().numpy()
    
    def compute_similarity(self, image_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
        return np.dot(image_features, text_features.T)


def test_clip_embedder():
    embedder = CLIPEmbedder()
    image = Image.open("test/images/hk.png")
    images = [image] * 3
    text = "test text"
    image_features = embedder.embed_images(images)
    text_features = embedder.embed_text(text)
    similarity = embedder.compute_similarity(image_features, text_features)
    print(similarity)
    print(image_features.shape)
    print(text_features.shape)
    print(type(text_features))

if __name__ == "__main__":
    test_clip_embedder()
