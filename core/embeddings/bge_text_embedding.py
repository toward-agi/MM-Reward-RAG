from .base import TextEmbedder
from sentence_transformers import SentenceTransformer
from typing import Union, List
import numpy as np

class BGETextEmbedder(TextEmbedder):
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model = SentenceTransformer(model_name)
    
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True)
    
    def compute_similarity(self, query_embedding: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        pass

def test_bge_text_embedder():
    embedder = BGETextEmbedder()
    text = "test text"
    texts = [text] * 3
    text_features = embedder.embed_text(texts)
    print(text_features)
    print(text_features.shape)
    print(type(text_features))

if __name__ == "__main__":
    test_bge_text_embedder()