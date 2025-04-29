"""
# !pip install transformers einops timm pillow
from transformers import AutoModel

# Initialize the model
model = AutoModel.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)

# Corpus
sentences = [
    'غروب جميل على الشاطئ', # Arabic
    '海滩上美丽的日落', # Chinese
    'Un beau coucher de soleil sur la plage', # French
    'Ein wunderschöner Sonnenuntergang am Strand', # German
    'Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία', # Greek
    'समुद्र तट पर एक खूबसूरत सूर्यास्त', # Hindi
    'Un bellissimo tramonto sulla spiaggia', # Italian
    '浜辺に沈む美しい夕日', # Japanese
    '해변 위로 아름다운 일몰', # Korean
]

# Public image URLs or PIL Images
image_urls = ['https://i.ibb.co/nQNGqL0/beach1.jpg', 'https://i.ibb.co/r5w8hG8/beach2.jpg']

# Choose a matryoshka dimension, set to None to get the full 1024-dim vectors
truncate_dim = 512

# Encode text and images
text_embeddings = model.encode_text(sentences, truncate_dim=truncate_dim)
image_embeddings = model.encode_image(
    image_urls, truncate_dim=truncate_dim
)  # also accepts PIL.Image.Image, local filenames, dataURI

# Encode query text
query = 'beautiful sunset over the beach' # English
query_embeddings = model.encode_text(
    query, task='retrieval.query', truncate_dim=truncate_dim
)

# Text to Image
print('En -> Img: ' + str(query_embeddings @ image_embeddings[0].T))
# Image to Image
print('Img -> Img: ' + str(image_embeddings[0] @ image_embeddings[1].T))
# Text to Text
print('En -> Ar: ' + str(query_embeddings @ text_embeddings[0].T))
print('En -> Zh: ' + str(query_embeddings @ text_embeddings[1].T))
print('En -> Fr: ' + str(query_embeddings @ text_embeddings[2].T))
print('En -> De: ' + str(query_embeddings @ text_embeddings[3].T))
print('En -> Gr: ' + str(query_embeddings @ text_embeddings[4].T))
print('En -> Hi: ' + str(query_embeddings @ text_embeddings[5].T))
print('En -> It: ' + str(query_embeddings @ text_embeddings[6].T))
print('En -> Jp: ' + str(query_embeddings @ text_embeddings[7].T))
print('En -> Ko: ' + str(query_embeddings @ text_embeddings[8].T))

"""

import torch
from typing import Union, List, Optional
import numpy as np
from PIL import Image
from transformers import AutoModel
from core.embeddings.base import MultiModalEmbedder, TextEmbedder

class JinaClipV2Embedder(MultiModalEmbedder, TextEmbedder):
    def __init__(self, batch_size = 1024, truncate_dim: Optional[int] = None):
        """
        Initialize the Jina CLIP V2 embedder.
        
        Args:
            truncate_dim: Optional dimension to truncate embeddings to. If None, uses full 1024-dim vectors.
        """
        self.model = AutoModel.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True).cuda()
        self.truncate_dim = truncate_dim
        self.batch_size = batch_size
        
    def embed_images(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """
        Generate embeddings for one or more images.
        
        Args:
            images: Single image or list of images (PIL.Image.Image objects, URLs, or local file paths)
            
        Returns:
            numpy.ndarray: Image embeddings
        """
        if isinstance(images, (str, Image.Image)):
            images = [images]
        
        with torch.no_grad():
            embeddings = self.model.encode_image(images,truncate_dim=self.truncate_dim, batch_size=self.batch_size,show_progress_bar=False) # TODO: modift config
        return embeddings
    
    def embed_text(self, texts: Union[str, List[str]], is_query: bool = False) -> np.ndarray:
        """
        Generate embeddings for one or more text strings.
        
        Args:
            texts: Single text string or list of strings
            is_query: Whether the text is a query (affects encoding strategy)
            
        Returns:
            numpy.ndarray: Text embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            
        task = 'retrieval.query' if is_query else None
        embeddings = self.model.encode_text(texts, task=task, truncate_dim=self.truncate_dim, show_progress_bar=False, batch_size=self.batch_size)
        return embeddings
    
    def compute_similarity(self, 
                         features_a: np.ndarray, 
                         features_b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of features.
        
        Args:
            features_a: First set of feature vectors
            features_b: Second set of feature vectors
            
        Returns:
            numpy.ndarray: Similarity scores matrix
        """
        return np.dot(features_a, features_b.T)
    
    def search_images(self, 
                     query_text: str, 
                     image_list: List[Union[str, Image.Image]]) -> List[float]:
        """
        Search for images using a text query.
        
        Args:
            query_text: Text query to search with
            image_list: List of images to search through
            
        Returns:
            List[float]: Similarity scores for each image
        """
        query_embedding = self.embed_text(query_text, is_query=True)
        image_embeddings = self.embed_images(image_list)
        similarities = self.compute_similarity(query_embedding, image_embeddings)
        return similarities.flatten().tolist()
    
    def search_texts(self, 
                    query_text: str, 
                    text_list: List[str]) -> List[float]:
        """
        Search for texts using a text query.
        
        Args:
            query_text: Text query to search with
            text_list: List of texts to search through
            
        Returns:
            List[float]: Similarity scores for each text
        """
        query_embedding = self.embed_text(query_text, is_query=True)
        text_embeddings = self.embed_text(text_list)
        similarities = self.compute_similarity(query_embedding, text_embeddings)
        return similarities.flatten().tolist()


def test():
    images = [Image.open("test/images/dog.png"), Image.open("test/images/cat.png"), Image.open("test/images/hk.png")]
    texts = ["a dog", "a cat", "a city"]
    query = "a dog"
    embedder = JinaClipV2Embedder()
    image_embeddings = embedder.embed_images(images)
    print(image_embeddings.shape)
    print(type(image_embeddings))
    text_embeddings = embedder.embed_text(texts)
    print(text_embeddings.shape)
    query_embedding = embedder.embed_text(query, is_query=True)
    print(query_embedding.shape)
    image_similarities = embedder.search_images(query, images)
    print(image_similarities)
    text_similarities = embedder.search_texts(query, texts)
    print(text_similarities)

if __name__ == "__main__":
    test()
