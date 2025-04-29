"""
only supports:
query2text, query2image;
Implementation of a reranker using MrLight's DSE Qwen2 model for embedding-based document retrieval.
"""
from typing import List, Union, Any, Optional, Tuple
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoConfig
from transformers.utils.hub import cached_file
from torch.nn.functional import cosine_similarity
from .reranker import Reranker
from qwen_vl_utils import process_vision_info

class DseReranker(Reranker):
    """
    Reranker implementation using MrLight's DSE Qwen2 embedding model.
    
    This reranker uses embedding similarity between queries and documents (text or images)
    to rank documents by relevance.
    """
    
    def __init__(
        self, 
        model_name: str = "MrLight/dse-qwen2-2b-mrl-v1", 
        device: str = "cuda:0",
        embedding_dim: int = 1536,
        use_flash_attention: bool = True,
        **kwargs
    ):
        """
        Initialize the DSE reranker with the pre-trained model.
        
        Args:
            model_name: Name of the pretrained model to use
            device: Device to load the model on
            embedding_dim: Dimension of the embedding vectors
            use_flash_attention: Whether to use flash attention for faster processing
            **kwargs: Additional arguments to pass to the model loading function
        """
        self.model_name = model_name
        self.device = device
        self.embedding_dim = embedding_dim
        
        # Define minimum and maximum pixels for image processing
        min_pixels = 1 * 28 * 28
        max_pixels = 2560 * 28 * 28
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
        
        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "cuda",
            **kwargs
        }
        
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained("/data/track/mm-RewardRAG/models/dse-qwen2-2b-mrl-v1", **model_kwargs)
        
        # Set padding configuration
        self.processor.tokenizer.padding_side = "left"
        self.model.padding_side = "left"
        
        # Create a dummy image for text-only inputs
        self.dummy_image = Image.new('RGB', (28, 28))
    
    def _get_embedding(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Extract normalized embeddings from the model's last hidden state.
        
        Args:
            last_hidden_state: The last hidden state from the model
            
        Returns:
            Normalized embedding vectors
        """
        reps = last_hidden_state[:, -1]
        reps = torch.nn.functional.normalize(reps[:, :self.embedding_dim], p=2, dim=-1)
        return reps
    
    def _get_query_embeddings(self, queries: List[str]) -> torch.Tensor:
        """
        Generate embeddings for a list of queries.
        
        Args:
            queries: List of query strings
            
        Returns:
            Tensor of query embeddings
        """
        query_messages = []
        for query in queries:
            message = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': self.dummy_image, 'resized_height': 1, 'resized_width': 1},
                        {'type': 'text', 'text': f'Query: {query}'},
                    ]
                }
            ]
            query_messages.append(message)
        
        # Apply chat template
        query_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
            for msg in query_messages
        ]
        
        # Process vision info using the imported function
        query_image_inputs, query_video_inputs = process_vision_info(query_messages)
        
        # Process inputs
        query_inputs = self.processor(
            text=query_texts, 
            images=query_image_inputs,
            videos=query_video_inputs,
            padding=True, 
            max_length=2048,
            return_tensors='pt'
        ).to(self.device)
        
        # Prepare inputs for generation
        cache_position = torch.arange(0, len(query_texts)).to(self.device)
        query_inputs = self.model.prepare_inputs_for_generation(
            **query_inputs, 
            cache_position=cache_position, 
            use_cache=False
        )
        
        # Generate embeddings
        with torch.no_grad():
            output = self.model(**query_inputs, return_dict=True, output_hidden_states=True)
        
        return self._get_embedding(output.hidden_states[-1])
    
    def _get_text_doc_embeddings(self, documents: List[str]) -> torch.Tensor:
        """
        Generate embeddings for a list of text documents.
        
        Args:
            documents: List of document strings
            
        Returns:
            Tensor of document embeddings
        """
        doc_messages = []
        for doc in documents:
            message = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': self.dummy_image, 'resized_height': 1, 'resized_width': 1},
                        {'type': 'text', 'text': f'Document: {doc}'}
                    ]
                }
            ]
            doc_messages.append(message)
        
        # Apply chat template
        doc_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
            for msg in doc_messages
        ]
        
        # Process vision info using the imported function
        doc_image_inputs, doc_video_inputs = process_vision_info(doc_messages)
        
        # Process inputs
        doc_inputs = self.processor(
            text=doc_texts, 
            images=doc_image_inputs,
            videos=doc_video_inputs,
            padding=True, 
            max_length=2048,
            return_tensors='pt'
        ).to(self.device)
        
        # Prepare inputs for generation
        cache_position = torch.arange(0, len(doc_texts)).to(self.device)
        doc_inputs = self.model.prepare_inputs_for_generation(
            **doc_inputs, 
            cache_position=cache_position, 
            use_cache=False
        )
        
        # Generate embeddings
        with torch.no_grad():
            output = self.model(**doc_inputs, return_dict=True, output_hidden_states=True)
        
        return self._get_embedding(output.hidden_states[-1])
    
    def _get_image_doc_embeddings(self, documents: List[Union[str, Image.Image]]) -> torch.Tensor:
        """
        Generate embeddings for a list of image documents.
        
        Args:
            documents: List of image paths or PIL Image objects
            
        Returns:
            Tensor of document embeddings
        """
        doc_images = []
        for doc in documents:
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
                    doc_image = self.dummy_image
            else:
                doc_image = doc
            
            doc_images.append(doc_image)
        
        doc_messages = []
        for doc in doc_images:
            message = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': doc},
                        {'type': 'text', 'text': 'What is shown in this image?'}
                    ]
                }
            ]
            doc_messages.append(message)
        
        # Apply chat template
        doc_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
            for msg in doc_messages
        ]
        
        # Process vision info using the imported function
        doc_image_inputs, doc_video_inputs = process_vision_info(doc_messages)
        
        # Process inputs
        doc_inputs = self.processor(
            text=doc_texts, 
            images=doc_image_inputs,
            videos=doc_video_inputs,
            padding=True, 
            max_length=2048,
            return_tensors='pt'
        ).to(self.device)
        
        # Prepare inputs for generation
        cache_position = torch.arange(0, len(doc_texts)).to(self.device)
        doc_inputs = self.model.prepare_inputs_for_generation(
            **doc_inputs, 
            cache_position=cache_position, 
            use_cache=False
        )
        
        # Generate embeddings
        with torch.no_grad():
            output = self.model(**doc_inputs, return_dict=True, output_hidden_states=True)
        
        return self._get_embedding(output.hidden_states[-1])
    
    def compute_relevance(self, query: str, docs: List[Union[str, Image.Image], ], doc_type: str = "text", **kwargs) -> List[int]:
        """
        Compute relevance scores between a query and a list of documents and return document indices
        ranked by relevance.
        
        Args:
            query: The query string
            docs: List of documents (text strings or images)
            doc_type: Type of document ("text" or "image")
            
        Returns:
            List of document indices ranked by relevance score in descending order
        """
        # Get query embeddings
        query_embeddings = self._get_query_embeddings([query])
        
        # Get document embeddings based on doc_type
        if doc_type == "text":
            doc_embeddings = self._get_text_doc_embeddings(docs)
        elif doc_type == "image":
            doc_embeddings = self._get_image_doc_embeddings(docs)
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")
        
        # Compute similarities
        query_embedding = query_embeddings[0].unsqueeze(0)
        similarities = cosine_similarity(query_embedding, doc_embeddings)
        print(f"Similarities: {similarities}")
        
        # Create list of (document_index, similarity) tuples
        scored_docs = [(i, float(similarity)) for i, similarity in enumerate(similarities)]
        
        # Sort by similarity score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return only the document indices in ranked order
        return [doc_index for doc_index, _ in scored_docs]


def test():
    # Initialize reranker
    reranker = DseReranker()
    
    # Example queries
    queries = [
        "Where can we see Llama?",
        "What is the LLaMA AI model?"
    ]
    
    # Example image documents
    image_urls = [
        "https://huggingface.co/Tevatron/dse-phi3-docmatix-v2/resolve/main/animal-llama.png",
        "https://huggingface.co/Tevatron/dse-phi3-docmatix-v2/resolve/main/meta-llama.png"
    ]
    
    # Example text documents
    text_docs = [
        "The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.",
        "Llama (acronym for Large Language Model Meta AI, and formerly stylized as LLaMA) is a family of autoregressive large language models (LLMs) released by Meta AI starting in February 2023."
    ]
    
    print("Testing image document reranking:")
    for query in queries:
        ranked_indices = reranker.compute_relevance(query, image_urls, doc_type="image")
        print(f"\nQuery: {query}")
        print(f"Ranked indices: {ranked_indices}")
        for idx in ranked_indices:
            print(f"  {idx}: {image_urls[idx]}")
    
    print("\nTesting text document reranking:")
    for query in queries:
        ranked_indices = reranker.compute_relevance(query, text_docs, doc_type="text")
        print(f"\nQuery: {query}")
        print(f"Ranked indices: {ranked_indices}")
        for idx in ranked_indices:
            print(f"  {idx}: {text_docs[idx][:50]}...")


if __name__ == "__main__":
    test()
