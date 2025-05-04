from transformers import AutoModel, CLIPImageProcessor, CLIPTokenizer
import torch
import sys
sys.path.append("/data/track/mm-RewardRAG")
from PIL import Image 
from typing import Union, List
import numpy as np
from pathlib import Path
from .base import MultiModalEmbedder
from config.config import settings

class EVAEmbedder(MultiModalEmbedder):
    def __init__(self):
        self.config = settings.get_embedder_config("multimodal")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = self.config["models"]["eva"]["name"]  # Assuming you have this in your config
        print(f"Loading EVA-CLIP model: {model_name}")
        
        # Load model with float16 precision to save memory
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device).eval()
        
        # Load processor and tokenizer
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        
    def embed_images(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        if isinstance(images, Image.Image):
            images = [images]
            
        # Process images
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs.pixel_values.to(self.device)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(pixel_values)
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
        return image_features.cpu().numpy()
    
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize texts
        input_ids = self.tokenizer(texts, return_tensors="pt", padding=True, max_length=77, truncation=True).input_ids.to(self.device)
        # 这里是就截取前77个就好了；
        # print(f"input_ids: {input_ids.shape}")
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model.encode_text(input_ids)
            # Normalize features
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
        return text_features.cpu().numpy()
    
    def compute_similarity(self, image_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between image and text features"""
        return np.dot(image_features, text_features.T)

def test_eva_embedder():
    embedder = EVAEmbedder()
    image = Image.open("test/images/hk.png")
    images = [image] * 3
    texts = ["a diagram", "a dog", "a cat"]
    
    # Get embeddings
    image_features = embedder.embed_images(images)
    text_features = embedder.embed_text(texts)
    
    # Compute similarity
    similarity = embedder.compute_similarity(image_features, text_features)
    
    print(f"Similarity scores:\n{similarity}")
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    
    # Print softmax probabilities (like in your original code)
    label_probs = torch.softmax(torch.tensor(similarity * 100.0), dim=-1)
    print(f"Label probabilities: {label_probs}")

if __name__ == "__main__":
    test_eva_embedder()