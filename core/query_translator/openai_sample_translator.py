# -*- coding: utf-8 -*-
# ----------------------------
import sys
sys.path.append("/data/track/mm-RewardRAG")

import copy
from abc import abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union
from PIL import Image
import json
import logging
from .qtranslator import Translator
from config.config import settings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from generators.sample_base import SampleBaseGenerator
from copy import deepcopy

logger = logging.getLogger(__name__)

prompt_templates = {
    "ti":[
        # query first:
        {
            "type": "text",
            "text": """
            Given a Visual Question Answering (VQA) question, please generate some possible search engine queries and key words that can be used to retrieve knowledge from an external knowledge base that can answer the question without referring the input image.
            Please output strict JSON that can be directly parsed by Python, in the following format: {"Chain_of_Thought": your analysis here, "queries": search engine queries here, "key_words_string": key words for bm25 here.}.
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
            "text": "Question: {query}\n\n"
        },
        {
            "type": "text",
            "text": "Please output the Chain of Thought reasoning and the queries in strict JSON format. Note that the queries must contain the entity name."
        }

    ],
    "t":[
        # query first:
        {
            "type": "text",
            "text": """
            Given a question, please generate some possible search engine queries and key words that can be used to retrieve knowledge from an external knowledge base that can answer the question.
            Please output strict JSON that can be directly parsed by Python, in the following format: {"Chain_of_Thought": your analysis here, "queries": search engine queries here, "key_words_string": key words for bm25 here.}.
            """
        },
        {
            "type": "text",
            "text": "Here is the question: {query} \n\n"
        },
        {
            "type": "text",
            "text": "Please output the Chain of Thought reasoning and the queries in strict JSON format."
        }

    ]
}

class VLLMSampleTranslator(Translator, SampleBaseGenerator):

    def __init__(
        self,
        generator_name: str = "vllm_sample_translator"
    ):

        SampleBaseGenerator.__init__(self, generator_name)
        self.prompts = prompt_templates
        
    def _prepare_messages(self, query: Dict) -> List[Dict[str, Any]]: # type: ignore

        if query["type"] == "ti":
            prompt = deepcopy(self.prompts["ti"])
            prompt[2]["image_url"]["url"] = f"data:image/jpeg;base64,{self._encode_image(query['image'])}"
            prompt[3]["text"] = prompt[3]["text"].format(query=query["text"])
            
        elif query["type"] == "t":
            prompt = deepcopy(self.prompts["t"])
            prompt[1]["text"] = prompt[1]["text"].format(query=query["text"])
        else:
            raise ValueError("Unsupported query-document type combination")
        
        return [{"role": "user", "content": prompt}]
    
    def _prepare_messages_for_batch(self, queries: Dict) -> List[Tuple[List[Dict[str, Any]], str]]:
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
            message_list.append((messages, str(idx)))
            
        return message_list
    
    def _extract_dict(self, response_text: str) -> float:
        #TODO: 这里需要将response转换成dict；
        #TODO: 这里肯定是要抽取出来的；
        #TODO: 感觉这里不需要先；
        pass
            
      
        
    
    def translate(self, queries: List[Dict], **kwargs) -> List[int]:
        
        message_list = self._prepare_messages_for_batch(queries)
        
        # Process in batches using ThreadPoolExecutor
        results = self.generate_batch(message_list)
        
        valid_results = []
        for responses, doc_idx in results:
            # query_dict = self._extract_dict(response)
            # TODO: 这里就是将response转换成dict就好了
            # Convert string index back to integer
            # valid_results.append(query_dict)
            valid_results.append((responses, doc_idx))
        
        return valid_results
    
    def generate(self, queries: List[Dict]) -> List[int]:

        return self.translate(queries)


def test():
    """Test function for the VLLMTranslator class."""
    # Configure with your vLLM generator configuration
    translator = VLLMSampleTranslator(generator_name="vllm_sample_translator")
    
    query_1 = {
        "type": "t",
        "text": "Do people in Beskiden decorate for Halloween?"
    }

    query_2 = {
        "type": "ti",
        "text": "What kind of clusters does this plant grow in?",
        "image": Image.open("/data/mingyan/encyclopedic/iNaturalist2021/train/08469_Plantae_Tracheophyta_Magnoliopsida_Lamiales_Lamiaceae_Monarda_didyma/68813c21-d93e-4ee5-a6a1-3cdfb9d53b6b.jpg").convert("RGB").resize((512, 512)),
    } 

    query_3 = {
        "type": "ti",
        "text": "What country does this building belong to?",
        "image": Image.open("/data/dataset/mm_RewardRAG/InfoSeek/oven/00/oven_00115764.jpg").convert("RGB").resize((512, 512)),
    }

    query_4 = {
        "type": "t",
        "text": "What symbol is on the top of the crown in the Real Oviedo logo?"
    }

    queries = [query_1, query_2, query_3, query_4]
    # queries = [query_1]

    structured_results = translator.translate(queries)
    for idx in structured_results:
        print(idx)


if __name__ == "__main__":
    # Uncomment to run test
    test()
