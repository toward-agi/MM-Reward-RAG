import faiss
import numpy as np
from typing import List, Tuple, Optional
import pickle
from pathlib import Path
from config.config import settings

class VectorStore:
    def __init__(self, dimension: int = settings.embedding_dimension, index_path: Optional[Path] = None):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product similarity
        self.metadata = []
        self.index_path = index_path or settings.vector_store_path
        
    def add_vectors(self, vectors: np.ndarray, metadata: List[dict]):
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
            
        # Normalize vectors
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(metadata)
        
    def search(self, query_vector: np.ndarray, k: int = settings.top_k) -> Tuple[np.ndarray, np.ndarray]:
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
            
        # Normalize query vector
        faiss.normalize_L2(query_vector)
        scores, indices = self.index.search(query_vector, k)
        
        return scores[0], [self.metadata[idx] for idx in indices[0]]
    
    def save(self, path: Optional[Path] = None):
        path = path or self.index_path
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save the metadata
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
            
    def load(self, path: Optional[Path] = None):
        path = path or self.index_path
        
        # Load the index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load the metadata
        with open(path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
            
    @property
    def size(self) -> int:
        return self.index.ntotal
