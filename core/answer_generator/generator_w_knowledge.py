"""
Implementation of a reranker using vLLM served through an OpenAI-compatible API.
Extends the BaseGenerator class for consistent API usage.
"""

#TODO: 这里要确定一下
# ----------------------------
import sys
sys.path.append("/data/track/mm-RewardRAG")

import copy
from abc import abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union
from PIL import Image
import json
import logging
from config.config import settings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from generators.base import BaseGenerator

logger = logging.getLogger(__name__)

single_text_block = [
    {
        "type": "text",
        "text": " Document: {text_document} \n\n"
    }
]

single_image_block = [
    {
        "type": "text",
        "text": "Image:"
    },
    {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,{settings.image_base64}"
        }
    },
    {
        "type": "text",
        "text": "Caption: {caption} \n\n"
    }
]

end_block = [
    {
        "type": "text",
        "text": "Answer:"
    }
]

# 然后，根据输入的image和text，按照这个去选择block，加到messages里面就好了；

prompt_templates = {
    "ti":[
        # query first:
        {
            "type": "text",
            "text": """
            Given a Visual Question Answering (VQA) question and supporting contents, please generate the answer to the question.
            """
        },
        {
            "type": "text",
            "text": "Here is the VQA question:"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,{settings.image_base64}"
            }
        },
        {
            "type": "text",
            "text": "Question: {query} \n\n"
        },

    ],
    "t":[
        # query first:
        {
            "type": "text",
            "text": """
            Given a question and supporting contents, please generate the answer to the question.
            """
        },
        {
            "type": "text",
            "text": "Here is the question: {query} \n\n"
        },
        {
            "type": "text",
            "text": "Supporting contents:\n\n"
        }
    ]
}

# 如果有image的话，先添加image然后再添加text；


class VLLMAnswerGenerator_w_Knowledge(BaseGenerator):
    """
    Reranker implementation using vLLM served through an OpenAI-compatible API.
    
    This reranker leverages ThreadPoolExecutor for concurrent evaluation of 
    document relevance based on the BaseGenerator infrastructure.
    """
    
    def __init__(
        self,
        generator_name: str = "vllm_answer_generator"
    ):
        """
        Initialize the vLLM reranker with configuration from settings.
        
        Args:
            generator_name: Name of the generator configuration to use
        """
        BaseGenerator.__init__(self, generator_name)
        self.prompts = prompt_templates
        
    def _prepare_messages(self, query: Dict) -> List[Dict[str, Any]]: # type: ignore
        """
        Prepare messages for a single query-document pair.
        
        Args:
            query: The query dictionary containing type and text/image
            document: The document dictionary containing type and text/image
            
        Returns:
            List of message dictionaries for the API call
        """
        if query["type"] == "ti":
            prompt = copy.deepcopy(self.prompts["ti"]) 
            prompt[2]["image_url"]["url"] = f"data:image/jpeg;base64,{self._encode_image(query['image'])}" #这里就是把这里的符号换成是转换之后的base64 string就可以；
            prompt[3]["text"] = prompt[3]["text"].format(query=query["text"]) # 这里是把query的question放进去；
            for document in query["text_documents"]: # only have text documents
                prompt.append(single_text_block[0].copy())
                prompt[-1]["text"] = prompt[-1]["text"].format(text_document=document)
            prompt.append(end_block[0].copy())
            
        elif query["type"] == "t": # 这里就是需要把supported image或者text放进去；然后去请求就可以了；
            # prompt = self.prompts["t->t"]
            prompt = copy.deepcopy(self.prompts["t"])
            prompt[1]["text"] = prompt[1]["text"].format(query=query["text"]) # 这里是把query的question放进去；
            if query["image_documents"] != []:
                for image_document in query["image_documents"]:
                    prompt.extend(copy.deepcopy(single_image_block))
                    prompt[-2]["image_url"]["url"] = f"data:image/jpeg;base64,{self._encode_image(image_document["image"])}"
                    prompt[-1]["text"] = prompt[-1]["text"].format(caption=image_document["caption"])
            if query["text_documents"] != []:
                for text_document in query["text_documents"]:
                    prompt.append(single_text_block[0].copy())
                    prompt[-1]["text"] = prompt[-1]["text"].format(text_document=text_document)
            prompt.append(end_block[0].copy())
            
        else:
            raise ValueError("Unsupported query-document type combination")
        
        return [{"role": "user", "content": prompt}]
    
    def _prepare_messages_for_batch(self, queries: List[Dict]) -> List[Tuple[List[Dict[str, Any]], str]]:
        """
        Prepare messages for a batch of documents.
        
        Args:
            query: The query string
            documents: List of document texts
            
        Returns:
            List of tuples (messages, document_index) for batch processing
        """
        message_list = []
        
        for idx, query in enumerate(queries):
            messages = self._prepare_messages(query)
            # Use index as the request_id for tracking
            # print(f"messages: {messages}")
            message_list.append((messages, query["id"]))
            
        return message_list
    
    def _extract_answer(self, response_text: str) -> float:
        """
        Extract the relevance score from the LLM response by converting the label to a float.
        
        Args:
            response_text: The response text from the LLM
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        try:
            # Try to extract JSON from the response
            response_text = response_text.strip()
            
            # If response is wrapped in backticks or has text before/after, extract just the JSON part
            if "```json" in response_text and "```" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].strip()
            else:
                # Look for JSON object at the end of the response
                import re
                json_match = re.search(r'\{.*"label"\s*:\s*"([^"]+)".*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return 0.0  # No JSON found
            
            # Parse JSON and extract label
            import json
            data = json.loads(json_str)
            #TODO: 修改一下后面这部分就好了；
            label = data.get("label", "")
            
            # Convert label to score
            if label == "useful":
                score = 0.9
            elif label == "neutral":
                score = 0.5
            elif label == "useless":
                score = 0.1
            else:
                # Default to 0.0 if label is not recognized
                score = 0.0
                
            # Clamp score between 0 and 1
            return max(0.0, min(1.0, score))
        
        except Exception as e:
            print(f"Error extracting score from response: {e}")
            # Return a default low score on error
            return 0.0
            
    def generate_answer(self, queries: List[Dict]) -> List[int]:
        
        message_list = self._prepare_messages_for_batch(queries)
        
        # Process in batches using ThreadPoolExecutor
        results = self.generate_batch(message_list)

        return [{"data_id": result[1],"answer": result[0]} for result in results]
    
    def generate(self, **kwargs) -> List[int]:
        pass

def test():
    
    # answer_generator = VLLMAnswerGenerator_wo_Knowledge()
    answer_generator = VLLMAnswerGenerator_w_GT()

    query = [{
        "type": "t",
        "text": "machine learning techniques for time series forecasting",
        "id": "string"
    }]

    answers = answer_generator.generate_answer(query)
    for answer in answers:
        print(f"Answer: {answer}")


if __name__ == "__main__":
    # Uncomment to run test
    test()
