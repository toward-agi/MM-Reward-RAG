"""
Implementation of Qwen2-VL based multimodal reranker.
Only supports query-image pairs and uses True/False probabilities for relevance scoring.
"""
from typing import List, Union, Any, Tuple, Optional
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from .reranker import Reranker
import requests
from io import BytesIO


class MonoQwenVLReranker(Reranker):
    """
    Reranker implementation using Qwen2-VL multimodal model.
    
    This reranker evaluates the relevance of images to text queries using
    Qwen2-VL's instruction-following capabilities and True/False probabilities.
    """
    
    def __init__(
        self, 
        model_name: str = "lightonai/MonoQwen2-VL-v0.1", 
        processor_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device_map: str = "auto",
        use_flash_attention: bool = False, # this model does not support flash attention
        dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        """
        Initialize the Qwen-VL reranker with a pre-trained model.
        
        Args:
            model_name: Name of the pretrained model to use
            processor_name: Name of the processor to use
            device_map: Device mapping strategy ('auto', 'cuda', 'cpu', etc.)
            use_flash_attention: Whether to use flash attention for faster processing
            dtype: Torch data type for the model (None, torch.float16, torch.bfloat16)
            **kwargs: Additional arguments to pass to the model loading function
        """
        self.model_name = model_name
        self.processor_name = processor_name
        
        # Prepare model loading arguments
        model_kwargs = {
            "device_map": device_map,
            **kwargs
        }
        
        # Add flash attention if requested
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Add dtype if specified
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(processor_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        
        # Cache token IDs for "True" and "False"
        self.true_token_id = self.processor.tokenizer.convert_tokens_to_ids("True")
        self.false_token_id = self.processor.tokenizer.convert_tokens_to_ids("False")
    
    def compute_relevance(self, query: str, docs: List[Image.Image], doc_type: str = "image", **kwargs) -> List[int]:
        """
        Compute relevance scores between a query and a list of documents and return document indices
        ranked by relevance.
        
        Args:
            query: The query string
            docs: List of documents (PIL Image objects)
            doc_type: Type of document (currently only supports "image")
            
        Returns:
            List of document indices ranked by relevance score in descending order
        """
        if doc_type != "image":
            raise ValueError("MonoQwenVLReranker currently only supports image documents")
        images = []
        for doc in docs:
            if isinstance(doc, str):
                try:
                    # Try to load from URL
                    if doc.startswith(('http://', 'https://')):
                        response = requests.get(doc)
                        doc_image = Image.open(BytesIO(response.content)).convert('RGB')
                    # Try to load from file path
                    else:
                        doc_image = Image.open(doc).convert('RGB')
                except Exception as e:
                    print(f"Error loading image {doc}: {e}")
                    continue
            else:
                doc_image = doc
            images.append(doc_image)
        
        # Process each document and compute relevance scores
        scores = []
        for i, image in enumerate(images):
            # Compute True and False probabilities
            true_prob, false_prob = self._compute_relevance_probabilities(query, image)
            
            # Calculate a relevance score that considers both probabilities
            # We use true_prob - false_prob as a score to capture the confidence difference
            # This gives better discrimination than just using true_prob
            relevance_score = true_prob - false_prob
            
            scores.append((i, relevance_score))
        
        # Sort by relevance score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return only the document indices in ranked order
        return [doc_index for doc_index, _ in scores]
    
    def _compute_relevance_probabilities(self, query: str, image: Image.Image) -> Tuple[float, float]:
        """
        Compute True and False probabilities for a single query-image pair.
        
        Args:
            query: The query string
            image: PIL image object
            
        Returns:
            Tuple of (true_probability, false_probability)
        """
        try:
            # Construct the prompt
            prompt = (
                "Assert the relevance of the previous image document to the following query, "
                "answer True or False. The query is: {query}"
            ).format(query=query)
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Apply chat template and tokenize
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.model.device)
            
            # Run inference to obtain logits
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_for_last_token = outputs.logits[:, -1, :]
            
            # Calculate relevance scores (probabilities of "True" and "False")
            relevance_scores = torch.softmax(
                logits_for_last_token[:, [self.true_token_id, self.false_token_id]], 
                dim=-1
            )
            
            # Extract true and false probabilities
            true_prob = relevance_scores[0, 0].item()
            false_prob = relevance_scores[0, 1].item()
            
            return true_prob, false_prob
            
        except Exception as e:
            print(f"Error computing relevance: {e}")
            return 0.0, 1.0  # Return lowest relevance on error
    
    def get_relevance_details(self, query: str, docs: List[Image.Image]) -> List[Tuple[int, float, float]]:
        """
        Get detailed relevance information including both True and False probabilities.
        Useful for debugging and analysis.
        
        Args:
            query: The query string
            docs: List of PIL Image objects
            
        Returns:
            List of tuples (doc_index, true_probability, false_probability) sorted by relevance
        """
        detailed_scores = []
        
        for i, image in enumerate(docs):
            # Get True and False probabilities
            true_prob, false_prob = self._compute_relevance_probabilities(query, image)
            detailed_scores.append((i, true_prob, false_prob))
        
        # Sort by true_prob - false_prob (same as in compute_relevance)
        detailed_scores.sort(key=lambda x: x[1] - x[2], reverse=True)
        
        return detailed_scores


def test():
    # Initialize reranker
    reranker = MonoQwenVLReranker()
    
    # Example query and image documents
    query = "slm markdown"
    image_urls = [
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/handelsblatt-preview.png",
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/paper-11.png",
        "https://raw.githubusercontent.com/jina-ai/multimodal-reranker-test/main/wired-preview.png",
        "https://jina.ai/blog-banner/using-deepseek-r1-reasoning-model-in-deepsearch.webp"
    ]
    
    # Load images
    images = []
    for url in image_urls:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            images.append(image)
        except Exception as e:
            print(f"Error loading image from {url}: {e}")

    # Get ranked document indices
    ranked_indices = reranker.compute_relevance(query, images, doc_type="image")
    print("Ranked document indices:")
    for rank, idx in enumerate(ranked_indices):
        print(f"Rank {rank+1}: Document index {idx} - {image_urls[idx]}")
    
    # Get detailed relevance information
    print("\nDetailed relevance scores:")
    detailed_scores = reranker.get_relevance_details(query, images)
    for rank, (idx, true_prob, false_prob) in enumerate(detailed_scores):
        print(f"Rank {rank+1}: Document index {idx}")
        print(f"  True probability: {true_prob:.4f}")
        print(f"  False probability: {false_prob:.4f}")
        print(f"  Relevance score (true-false): {true_prob-false_prob:.4f}")


if __name__ == "__main__":
    # Uncomment to run test
    test()