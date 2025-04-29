from typing import Optional
from config.config import settings
from .base import BaseIndexer
from .faiss_indexer import FaissIndexer
from .milvus_indexer import MilvusIndexer
from .qdrant_indexer import QdrantIndexer

def get_indexer(dimension: int, collection_name: Optional[str] = None) -> BaseIndexer:
    """Get vector indexer based on configuration.
    
    Args:
        dimension: Dimension of vectors to index
        collection_name: Optional name for collection (for Milvus/Qdrant)
        
    Returns:
        Configured vector indexer
    """
    indexer_type = settings.indexers.type.lower()
    
    if indexer_type == "faiss":
        return FaissIndexer(dimension)
    elif indexer_type == "milvus":
        return MilvusIndexer(dimension, collection_name)
    elif indexer_type == "qdrant":
        return QdrantIndexer(dimension, collection_name)
    else:
        raise ValueError(f"Unsupported indexer type: {indexer_type}")
