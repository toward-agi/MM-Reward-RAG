from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Tuple, Any

class BaseIndexer(ABC):
    """Base class for vector indexers."""
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add vectors and their metadata to the index.
        
        Args:
            vectors: Array of vectors to add
            metadata: List of metadata dictionaries for each vector
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Search for nearest neighbors of the query vector.
        
        Args:
            query_vector: Query vector to search for
            top_k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (scores, metadata) where scores is a list of similarity scores
            and metadata is a list of metadata dictionaries for the nearest neighbors
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the index to disk.
        
        Args:
            path: Path to save the index to
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the index from disk.
        
        Args:
            path: Path to load the index from
        """
        pass
