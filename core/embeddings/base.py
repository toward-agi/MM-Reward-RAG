from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List
from PIL import Image

class MultiModalEmbedder(ABC):
    @abstractmethod
    def embed_images(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        pass
        
    @abstractmethod
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        pass
        
    @abstractmethod
    def compute_similarity(self, image_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
        pass

class TextEmbedder(ABC):
    @abstractmethod
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        pass
        
    @abstractmethod
    def compute_similarity(self, query_embedding: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        pass
