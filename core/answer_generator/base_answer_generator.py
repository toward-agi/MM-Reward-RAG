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

prompt_templates = {
    "ti->t":[
        # query first:
        {
            "type": "text",
            "text": """
            Given a Visual Question Answering (VQA) question and a knowledge snippet, please generate the answer to the question.
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
        # then the document:
        {
            "type": "text",
            "text": "Here is the knowledge snippet: {document} \n\n"
        },
        {
            "type": "text",
            "text": "Please output the answer to the question."
        }
    ],
    "t->t":[
        # query first:
        {
            "type": "text",
            "text": """
            Given a question and a knowledge snippet, please generate the answer to the question.
            """
        },
        {
            "type": "text",
            "text": "Here is the question: {query} \n\n"
        },
        # then the document:
        {
            "type": "text",
            "text": "Here is the knowledge snippet: {document} \n\n"
        },
        {
            "type": "text",
            "text": "Please output the answer to the question."
        }

    ],
    "t->ti":[
        # query first:
        {
            "type": "text",
            "text": """
            Given a question and an image with caption, please generate the answer to the question.
            """
        },
        {
            "type": "text",
            "text": "Here is the question: {query} \n\n"
        },
        # then the document:
        {
            "type": "text",
            "text": "Here is the image with caption:"
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
        },
        {
            "type": "text",
            "text": "Please output the answer to the question."
        }
    ]
}


class VLLMAnswerGenerator(BaseGenerator):
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
        
    def _prepare_messages(self, query: Dict, document: Dict) -> List[Dict[str, Any]]: # type: ignore
        """
        Prepare messages for a single query-document pair.
        
        Args:
            query: The query dictionary containing type and text/image
            document: The document dictionary containing type and text/image
            
        Returns:
            List of message dictionaries for the API call
        """
        # TODO: 这里修改一下，修改成一个function来处理
        # TODO: 这里都不用管，直接修改下prompt就行
        if query["type"] == "ti" and document["type"] == "t":
            prompt = copy.deepcopy(self.prompts["ti->t"]) # 这里应该使用deepcopy来处理一下，避免直接修改原来的prompt
            # 这里应该使用deepcopy来处理一下，避免直接修改原来的prompt
            # print(f"prompt: {prompt}")
            prompt[2]["image_url"]["url"] = f"data:image/jpeg;base64,{self._encode_image(query['image'])}" #这里就是把这里的符号换成是转换之后的base64 string就可以；
            prompt[3]["text"] = prompt[3]["text"].format(query=query["text"]) # 这里是把query的question放进去；
            prompt[4]["text"] = prompt[4]["text"].format(document=document["text"]) # 这里是把document的text放进去；
            
        elif query["type"] == "t" and document["type"] == "t":
            # prompt = self.prompts["t->t"]
            prompt = copy.deepcopy(self.prompts["t->t"]) # 这里应该使用deepcopy来处理一下，避免直接修改原来的prompt
            prompt[1]["text"] = prompt[1]["text"].format(query=query["text"]) # 这里是把query的question放进去；
            prompt[2]["text"] = prompt[2]["text"].format(document=document["text"]) # 这里是把document的text放进去；
            
        elif query["type"] == "t" and document["type"] == "ti":
            # prompt = self.prompts["t->ti"]
            prompt = copy.deepcopy(self.prompts["t->ti"]) # 这里应该使用deepcopy来处理一下，避免直接修改原来的prompt
            prompt[1]["text"] = prompt[1]["text"].format(query=query["text"]) # 这里是把query的question放进去；
            prompt[3]["image_url"]["url"] = f"data:image/jpeg;base64,{self._encode_image(document['image'])}" #这里就是把这里的符号换成是转换之后的base64 string就可以；
            prompt[4]["text"] = prompt[4]["text"].format(caption=document["text"]) # 这里是把document的caption放进去；

        else:
            raise ValueError("Unsupported query-document type combination")
        
        return [{"role": "user", "content": prompt}]
    
    def _prepare_messages_for_batch(self, query: Dict, documents: List[Dict]) -> List[Tuple[List[Dict[str, Any]], str]]:
        """
        Prepare messages for a batch of documents.
        
        Args:
            query: The query string
            documents: List of document texts
            
        Returns:
            List of tuples (messages, document_index) for batch processing
        """
        message_list = []
        
        for idx, document in enumerate(documents):
            messages = self._prepare_messages(query, document)
            # Use index as the request_id for tracking
            # print(f"messages: {messages}")
            message_list.append((messages, str(idx)))
            
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
            
    def generate_answer(self, query: Dict, docs: List[Dict],**kwargs) -> List[int]:
        
        message_list = self._prepare_messages_for_batch(query, docs)
        
        # Process in batches using ThreadPoolExecutor
        results = self.generate_batch(message_list)

        return [result[0] for result in results]
    
    def generate(self, query: str, documents: List[str], doc_type: str = "text") -> List[int]:
        
        return self.generate_answer(query, documents)

def test():
    
    answer_generator = VLLMAnswerGenerator()
    
    # Example query and documents
    # query = "machine learning techniques for time series forecasting"
    # documents = [
    #     "Machine learning models like LSTM and transformer architectures have shown great success in time series forecasting tasks.",
    #     "The stock market showed significant volatility yesterday, with tech stocks experiencing the largest decline.",
    #     "Time series forecasting is a critical task in many domains, from weather prediction to financial market analysis.",
    #     "Deep learning approaches, particularly recurrent neural networks, have become popular for sequence prediction tasks.",
    #     "Reinforcement learning has applications in robotics and game playing.",
    #     "Natural language processing enables computers to understand human language."
    # ]

    query = {
        "type": "t",
        "text": "machine learning techniques for time series forecasting"
    }
    documents = [
        {
            "type": "t",
            "text": "Machine learning models like LSTM and transformer architectures have shown great success in time series forecasting tasks."
        },
        {
            "type": "t",
            "text": "The stock market showed significant volatility yesterday, with tech stocks experiencing the largest decline."
        },
        {
            "type": "t",
            "text": "Time series forecasting is a critical task in many domains, from weather prediction to financial market analysis."
        },
        {
            "type": "t",
            "text": "Deep learning approaches, particularly recurrent neural networks, have become popular for sequence prediction tasks."
        },
        {
            "type": "t",
            "text": "Reinforcement learning has applications in robotics and game playing."
        },
        {
            "type": "t",
            "text": "Natural language processing enables computers to understand human language."
        }
    ]


    # Get ranked document indices
    answers = answer_generator.generate_answer(query, documents)
    for answer in answers:
        print(f"Answer: {answer}")


if __name__ == "__main__":
    # Uncomment to run test
    test()
