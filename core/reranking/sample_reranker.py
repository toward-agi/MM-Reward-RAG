"""
Implementation of a reranker using vLLM served through an OpenAI-compatible API.
Extends the BaseGenerator class for consistent API usage.
"""

#TODO: 这个函数要做的事情就是，对一个document sample出来多个response，然后选择label不一样的，用正确的那个来给模型做fine-tuning；


# ----------------------------
import sys
sys.path.append("/data/track/mm-RewardRAG")

import copy
from abc import abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union
from PIL import Image
import json
import logging
from .reranker import Reranker
from config.config import settings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from generators.sample_base import SampleBaseGenerator

logger = logging.getLogger(__name__)

prompt_templates = {
    "ti->t":[
        # query first:
        {
            "type": "text",
            "text": """
            Given a Visual Question Answering (VQA) question and a knowledge snippet, please determine whether this knowledge snippet is useful for answering the VQA question. Please output one of the following labels: useful, neutral, useless. First perform Chain of Thought (CoT) reasoning and then output the label. Please output strict JSON that can be directly parsed by Python, in the following format: {"CoT": your analysis here, "label": the label obtained after analysis}.
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
            "text": "Please output the Chain of Thought reasoning and the label in strict JSON format."
        }
    ],
    "t->t":[
        # query first:
        {
            "type": "text",
            "text": """
            Given a question and a knowledge snippet, please determine whether this knowledge snippet is useful for answering the question. Please output one of the following labels: useful, neutral, useless. First perform Chain of Thought (CoT) reasoning and then output the label. Please output strict JSON that can be directly parsed by Python, in the following format: {"CoT": your analysis here, "label": the label obtained after analysis}.
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
            "text": "Please output the Chain of Thought reasoning and the label in strict JSON format."
        }

    ],
    "t->ti":[
        # query first:
        {
            "type": "text",
            "text": """
            Given a question and an image with caption, please determine whether this image is useful for answering the question. Please output one of the following labels: useful, neutral, useless. First perform Chain of Thought reasoning and then output the label. Please output strict JSON that can be directly parsed by Python, in the following format: {"CoT": your analysis here, "label": the label obtained after analysis}.
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
            "text": "Please output the Chain of Thought reasoning and the label in strict JSON format."
        }
    ]
}


class VLLMSampleReranker(Reranker, SampleBaseGenerator):
    """
    Reranker implementation using vLLM served through an OpenAI-compatible API.
    
    This reranker leverages ThreadPoolExecutor for concurrent evaluation of 
    document relevance based on the BaseGenerator infrastructure.
    """
    
    def __init__(
        self,
        generator_name: str = "vllm_sample_reranker"
    ):
        """
        Initialize the vLLM reranker with configuration from settings.
        
        Args:
            generator_name: Name of the generator configuration to use
        """
        SampleBaseGenerator.__init__(self, generator_name)
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
    
    def _extract_label(self, response_text: str) -> str:
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
                    return "not"
            
            # Parse JSON and extract label
            import json
            data = json.loads(json_str)
            label = data.get("label", "")

            if label in ["useful", "neutral", "useless"]:
                return label
            else:
                return "not"
        
        except Exception as e:
            print(f"Error extracting score from response: {e}")
            # Return a default low score on error
            return "not"
            
    def compute_relevance(self, query: Dict, docs: List[Dict],**kwargs) -> List[int]:
        
        message_list = self._prepare_messages_for_batch(query, docs)
        
        # Process in batches using ThreadPoolExecutor
        results = self.generate_batch(message_list)
        
        # Extract scores and document indices
        docs = []
        for responses, doc_idx in results:
            # print(len(responses))
            # 这里修改完之后都是responses，是一个list，从这个list里面抽取完label，然后只留下不一样的label的选项就可以了；
            labels_to_responses = {"useful": [], "neutral": [], "useless": [], "not": []}
            for response in responses:
                label = self._extract_label(response)
                labels_to_responses[label].append(response)
            # labels_to_responses = {self._extract_label(response): response for response in responses}
            docs.append((labels_to_responses, doc_idx))
        
        return docs
    
    def generate(self, query: str, documents: List[str], doc_type: str = "text") -> List[int]:
        """
        Generate ranked document indices based on relevance to query.
        Implementation of the abstract method from BaseGenerator.
        
        Args:
            query: The query string
            documents: List of document texts
            doc_type: Type of document (currently only supports "text")
            
        Returns:
            List of document indices ranked by relevance score
        """
        return self.compute_relevance(query, documents, doc_type)


def test():
    """Test function for the VLLMReranker"""
    # Configure with your vLLM generator configuration
    reranker = VLLMSampleReranker()
    
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


    results = reranker.compute_relevance(query, documents)
    from pprint import pprint as pp
    pp(results)


if __name__ == "__main__":
    # Uncomment to run test
    test()
