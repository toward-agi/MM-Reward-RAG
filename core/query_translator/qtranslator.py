"""
Base class for rerankers that compute relevance scores between queries and documents.
"""
from abc import ABC, abstractmethod
from typing import List, Union, Any, Dict, Tuple


class Translator(ABC):
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
    def translate(self, query: Dict) -> Any:
        """
        Compute relevance scores between a query and a list of documents.
        
        Args:
            query: The query string
            docs: List of documents (can be strings, URLs, images, etc.)
            
        Returns:
            List of tuples (document, score) ranked by relevance score in descending order
        """
        pass
