from .base import TextEmbedder
from sentence_transformers import SentenceTransformer
from typing import Union, List
import numpy as np
import torch
import torch.nn.functional as F

class E5TextEmbedder(TextEmbedder):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large", device: str = None):
        # Set device (cuda if available, otherwise cpu)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Load model and move to appropriate device
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
    
    def _add_prefix(self, texts: Union[str, List[str]], prefix: str) -> List[str]:
        """
        Add required prefix to input texts
        """
        if isinstance(texts, str):
            texts = [texts]
            
        return [f"{prefix} {text}" for text in texts]
    
    def embed_query(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embed query texts with the 'query:' prefix
        """
        prefixed_texts = self._add_prefix(texts, "query:")
        return self.model.encode(prefixed_texts, normalize_embeddings=True)
    
    def embed_corpus(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Embed corpus texts with the 'passage:' prefix
        """
        prefixed_texts = self._add_prefix(texts, "passage:")
        return self.model.encode(prefixed_texts, normalize_embeddings=True)
    
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        General embedding method that treats inputs as queries
        Maintained for compatibility with base class
        """
        return self.embed_query(texts)
    
    def compute_similarity(self, query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute dot product similarity between query embedding and corpus embeddings.
        Returns scores scaled by 100 as in the reference code.
        """
        # Convert numpy arrays to PyTorch tensors
        query_embedding = torch.from_numpy(query_embedding)
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
        
        # Ensure embeddings are normalized
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.unsqueeze(0)
            
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        corpus_embeddings = F.normalize(corpus_embeddings, p=2, dim=1)
        
        # Compute similarity scores
        scores = (query_embedding @ corpus_embeddings.T) * 100
        return scores.numpy()

def test_e5_text_embedder():
    embedder = E5TextEmbedder()
    
    print(f"Using device: {embedder.device}")
    
    # Test with individual text as query
    query_text = "how much protein should a female eat"
    query_embedding = embedder.embed_query(query_text)
    print(f"Single query shape: {query_embedding.shape}")
    
    # Test with multiple texts as corpus
    corpus_texts = [
        "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
        "1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精"
    ]
    
    corpus_embeddings = embedder.embed_corpus(corpus_texts)
    print(f"Corpus texts shape: {corpus_embeddings.shape}")
    
    # Test similarity computation
    similarity_scores = embedder.compute_similarity(query_embedding, corpus_embeddings)
    print(f"Similarity scores: \n{similarity_scores}")
    
    # Demonstrate SentenceTransformer's direct encoding
    print("\nDemonstrating SentenceTransformer direct encoding:")
    input_texts = [
        'query: how much protein should a female eat',
        'query: 南瓜的家常做法',
        "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day.",
        "passage: 1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精"
    ]
    direct_embeddings = embedder.model.encode(input_texts, normalize_embeddings=True)
    print(f"Direct encoding shape: {direct_embeddings.shape}")

if __name__ == "__main__":
    test_e5_text_embedder()