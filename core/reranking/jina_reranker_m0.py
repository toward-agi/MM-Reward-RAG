"""
Implementation of Jina AI's multimodal reranker based on the Reranker base class.
"""
from typing import List, Union, Any, Tuple, Optional
import torch
from PIL import Image
from transformers import AutoModel
from .reranker import Reranker


class JinaReranker(Reranker):
    """
    Reranker implementation using Jina AI's multimodal reranker models.
    
    This reranker supports text and image documents for reranking.
    """
    
    def __init__(
        self, 
        model_name: str = "jinaai/jina-reranker-m0", 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_flash_attention: bool = True,
        max_length: int = 2048,
        **kwargs
    ):
        """
        Initialize the Jina reranker with a pre-trained model.
        
        Args:
            model_name: Name of the pretrained model to use
            device: Device to load the model on ('cuda' or 'cpu')
            use_flash_attention: Whether to use flash attention for faster processing
            max_length: Maximum sequence length for inputs
            **kwargs: Additional arguments to pass to the model loading function
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": "auto",
            "trust_remote_code": True,
            **kwargs
        }
        
        # Add flash attention if requested and on CUDA
        if use_flash_attention and device == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load the model
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model.to(device)
        self.model.eval()
    
    def compute_relevance(self, query: Any, docs: List[Any], query_type: str, doc_type: str) -> List[int]:
        """
        Compute relevance scores between a query and a list of documents and return document indices
        ranked by relevance.
        
        Args:
            query: The query string or image
            docs: List of documents (can be URLs, PIL Images, or paths to images)
            doc_type: Type of document ("text" or "image")
            
        Returns:
            List of document indices ranked by relevance score in descending order
        """
        # Construct query-document pairs for scoring
        pairs = [[query, doc] for doc in docs]
        
        # Compute scores
        with torch.no_grad():
            scores = self.model.compute_score(
                pairs, 
                max_length=self.max_length, 
                query_type=query_type,
                doc_type=doc_type
            )
        
        # Create list of (document_index, score) tuples
        scored_docs = [(i, float(score)) for i, score in enumerate(scores)]
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return only the document indices in ranked order
        return [doc_index for doc_index, _ in scored_docs]


def test():
    reranker = JinaReranker()
    # Example query and documents
    query = "slm markdown"
    documents = [
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png",
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png",
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/wired-preview.png",
        "https://jina.ai/blog-banner/using-deepseek-r1-reasoning-model-in-deepsearch.webp"
    ]

    # Get ranked document indices
    ranked_indices = reranker.compute_relevance(query, documents, doc_type="image")
    print("Ranked document indices:")
    for idx in ranked_indices:
        print(idx)


if __name__ == "__main__":
    test()