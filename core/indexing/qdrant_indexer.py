from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from typing import List, Dict, Tuple, Any
import json
from pathlib import Path

from .base import BaseIndexer
from config.config import settings

class QdrantIndexer(BaseIndexer):
    """Qdrant-based vector indexer."""
    
    def __init__(self, dimension: int, collection_name: str = None):
        """Initialize Qdrant indexer.
        
        Args:
            dimension: Dimension of vectors to index
            collection_name: Name of Qdrant collection
        """
        self.dimension = dimension
        self.config = settings.indexers.qdrant
        self.collection_name = collection_name or self.config.collection_name
        
        self.client = QdrantClient(url=self.config.url)
        self._initialize_collection()
        
    def _initialize_collection(self):
        """Initialize Qdrant collection."""
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance[self.config.distance]
                )
            )
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add vectors and metadata to Qdrant collection."""
        points = []
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            points.append(models.PointStruct(
                id=i + len(self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1
                )[0]),
                vector=vector.tolist(),
                payload=meta
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Search for nearest neighbors of query vector."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        
        scores = []
        metadata_list = []
        
        for hit in results:
            scores.append(hit.score)
            metadata_list.append(hit.payload)
        
        return scores, metadata_list
    
    def save(self, path: str) -> None:
        """Save collection info to disk (Qdrant data is persisted in Qdrant server)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        info = {
            "dimension": self.dimension,
            "collection_name": self.collection_name
        }
        
        with open(path / "qdrant_info.json", "w") as f:
            json.dump(info, f)
    
    def load(self, path: str) -> None:
        """Load collection info from disk and reconnect to Qdrant."""
        path = Path(path)
        
        with open(path / "qdrant_info.json", "r") as f:
            info = json.load(f)
            
        self.dimension = info["dimension"]
        self.collection_name = info["collection_name"]
        self._initialize_collection()
