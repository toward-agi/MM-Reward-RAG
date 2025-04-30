from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import base64
from io import BytesIO
from openai import OpenAI
from config.config import settings
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)

class BaseGenerator(ABC):
    def __init__(self, generator_name: str):
        """Initialize the generator with configuration."""
        self.config = settings.get_generator_config(generator_name)
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config.get("url"),
        )
        self.model = self.config["model"]
        self.prompts = self.config["prompts"]
        self.generation_config = self.config["generation_config"]
        self.batch_size = self.config.get("batch_size", 1)
        
    def _encode_image(self, image: Image.Image) -> str:
        """Encode image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    @abstractmethod
    def _prepare_messages(self, **kwargs) -> Any:
        """Prepare messages for the different API call."""
        pass

    @abstractmethod
    def _prepare_messages_for_batch(self, **kwargs) -> Any:
        """Prepare messages for the different API call."""
        pass
    
    def _generate_single(self, messages: List[Dict[str, Any]], request_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """Generate a single response with optional request ID for tracking."""

        # try:
        #     response = self.client.chat.completions.create(
        #         model=self.model,
        #         messages=messages,
        #         **self.generation_config
        #     )
        #     return response.choices[0].message.content, request_id
        # except Exception as e:
        #     logger.error(f"Error generating response for request {request_id}: {str(e)}")
        #     raise
        max_retries = 5
        retry_count = 0
        backoff_factor = 1  # seconds
        
        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **self.generation_config
                )
                return response.choices[0].message.content, request_id
                
            except Exception as e:
                retry_count += 1
                wait_time = backoff_factor * (2 ** (retry_count - 1))  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                
                if retry_count < max_retries:
                    logger.warning(f"Attempt {retry_count}/{max_retries} failed for request {request_id}: {str(e)}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} attempts failed for request {request_id}: {str(e)}")
                    raise
        
        # This should not be reached due to the raise in the else clause above, but added for completeness
        raise Exception(f"Failed after {max_retries} attempts for request {request_id}")
    
    def generate_batch(self, message_list: List[Tuple[List[Dict[str, Any]], Optional[str]]]) -> List[Tuple[str, Optional[str]]]:
        """Generate responses for a batch of messages using ThreadPoolExecutor with context manager."""
        results = []
        future_to_idx = {}
        
        # with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
        # with ProcessPoolExecutor(max_workers=self.batch_size) as executor:
            # Submit all tasks to the executor
            for idx, (messages, request_id) in enumerate(message_list):
                future = executor.submit(self._generate_single, messages, request_id)
                future_to_idx[future] = idx
            
            # Process futures as they complete with progress bar
            with tqdm(total=len(future_to_idx), desc="Generating responses") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        request_id = message_list[idx][1]
                        logger.error(f"Error in batch generation for request {request_id}: {str(e)}")
                        results.append((str(e), request_id))
                    pbar.update(1)
        
        # Sort results to maintain original order
        if all(r[1] is not None for r in results):
            results.sort(key=lambda x: next(i for i, (_, rid) in enumerate(message_list) if rid == x[1]))
            
        return results
    
    @abstractmethod
    def generate(self, **kwargs) -> Any:
        """Generate content based on input."""
        pass