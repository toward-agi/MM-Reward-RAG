from openai import OpenAI
import numpy as np
from typing import List, Union
from config.config import settings
from .base import TextEmbedder

class OpenAITextEmbedder(TextEmbedder):
    def __init__(self):
        self.config = settings.get_embedder_config("text")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = self.config["name"]
        
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embeddings.append(response.data[0].embedding)
            
        return np.array(embeddings)
    
    def compute_similarity(self, query_embedding: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        # Normalize embeddings
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        
        return np.dot(query_embedding, text_embeddings.T)

class SentenceTransformerEmbedder(TextEmbedder):
    def __init__(self):
        self.config = settings.get_embedder_config("text")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.config["name"])
        
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True)
    
    def compute_similarity(self, query_embedding: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        return np.dot(query_embedding, text_embeddings.T)
