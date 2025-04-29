from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
import numpy as np
from typing import List, Dict, Tuple, Any
import json
from pathlib import Path

from .base import BaseIndexer
from config.config import settings

class MilvusIndexer(BaseIndexer):
    """Milvus-based vector indexer."""
    
    def __init__(self, dimension: int, collection_name: str = None):
        """Initialize Milvus indexer.
        
        Args:
            dimension: Dimension of vectors to index
            collection_name: Name of Milvus collection
        """
        self.dimension = dimension
        self.config = settings.indexers.milvus
        self.collection_name = collection_name or self.config.collection_name
        
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=self.config.host,
            port=self.config.port
        )
        
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize Milvus collection with schema."""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description=f"Collection for {self.dimension}-dimensional vectors"
        )
        
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using="default"
        )
        
        # Create index if it doesn't exist
        if not self.collection.has_index():
            index_params = {
                "metric_type": self.config.metric_type,
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(
                field_name="vector",
                index_params=index_params
            )
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add vectors and metadata to Milvus collection."""
        # Convert metadata to strings
        metadata_strings = [json.dumps(m) for m in metadata]
        
        entities = [
            vectors.tolist(),  # vectors
            metadata_strings   # metadata
        ]
        
        self.collection.insert(entities)
        self.collection.flush()
    
    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Search for nearest neighbors of query vector."""
        self.collection.load()
        
        search_params = {
            "metric_type": self.config.metric_type,
            "params": {"nprobe": 16}
        }
        
        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["metadata"]
        )
        
        scores = []
        metadata_list = []
        
        for hits in results:
            for hit in hits:
                scores.append(hit.score)
                metadata_list.append(json.loads(hit.entity.get('metadata')))
        
        return scores, metadata_list
    
    def save(self, path: str) -> None:
        """Save collection info to disk (Milvus data is persisted in Milvus server)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        info = {
            "dimension": self.dimension,
            "collection_name": self.collection_name
        }
        
        with open(path / "milvus_info.json", "w") as f:
            json.dump(info, f)
    
    def load(self, path: str) -> None:
        """Load collection info from disk and reconnect to Milvus."""
        path = Path(path)
        
        with open(path / "milvus_info.json", "r") as f:
            info = json.load(f)
            
        self.dimension = info["dimension"]
        self.collection_name = info["collection_name"]
        self._initialize_collection()
