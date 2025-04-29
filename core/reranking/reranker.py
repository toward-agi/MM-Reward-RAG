"""
Base class for rerankers that compute relevance scores between queries and documents.
"""
from abc import ABC, abstractmethod
from typing import List, Union, Any, Dict, Tuple


class Reranker(ABC):
    """
    Abstract base class for rerankers.
    
    A reranker computes relevance scores between a query and a list of documents,
    then returns documents ranked by their relevance to the query.
    """
    
    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the reranker with relevant parameters.
        """
        pass
    
    @abstractmethod
    def compute_relevance(self, query: str, docs: List[Any]) -> List[Tuple[Any, float]]:
        """
        Compute relevance scores between a query and a list of documents.
        
        Args:
            query: The query string
            docs: List of documents (can be strings, URLs, images, etc.)
            
        Returns:
            List of tuples (document, score) ranked by relevance score in descending order
        """
        pass
    
    def rank(self, query: str, docs: List[Any]) -> List[Any]:
        """
        Rank documents by their relevance to the query.
        
        Args:
            query: The query string
            docs: List of documents to rank
            
        Returns:
            List of documents ranked by relevance
        """
        scored_docs = self.compute_relevance(query, docs)
        # Return only the documents in ranked order
        return [doc for doc, _ in scored_docs]