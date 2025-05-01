"""
Implementation of a reranker using vLLM served through an OpenAI-compatible API.
Extends the BaseGenerator class for consistent API usage.
"""

# TODO: 这里主要的区别就是，这里就是单条的query和document，因为是要进行大量的并发处理的，然后最后就是，返回，question id, document, label就好了；
# TODO: 之后可以整一个majority vote的，就还是这么请求，只不过返回之后选择最大的那个label，换成sample generator就好了；

# TODO: 这里再做下两部分：

# TODO: prompt里面模型输出的label需要再确认一下，然后


# ----------------------------
import sys

sys.path.append("/data/track/mm-RewardRAG")

import copy
from abc import abstractmethod
from typing import Optional, Dict, Any, List, Tuple, Union
from PIL import Image
import json
import logging

# from .reranker import Reranker
from config.config import settings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
# from generators.base import BaseGenerator
from generators.sample_base import SampleBaseGenerator
from collections import Counter

logger = logging.getLogger(__name__)

prompt_templates = {
    "ti->t": [
        # query first:
        {
            "type": "text",
            "text": """
            Given a Visual Question Answering (VQA) question and a knowledge snippet, please determine whether this knowledge snippet is useful for answering the VQA question. Please output one of the following labels: useful, neutral, useless. First perform Chain of Thought (CoT) reasoning and then output the label. Please output strict JSON that can be directly parsed by Python, in the following format: {"CoT": your analysis here, "label": the label obtained after analysis}.
            """,
        },
        {"type": "text", "text": "Here is the VQA question:"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,{settings.image_base64}"},
        },
        {"type": "text", "text": "Question: {query} \n\n"},
        # then the document:
        {"type": "text", "text": "Here is the knowledge snippet: {document} \n\n"},
        {
            "type": "text",
            "text": "Please output the Chain of Thought reasoning and the label in strict JSON format.",
        },
    ],
    "t->t": [
        # query first:
        {
            "type": "text",
            "text": """
            Given a question and a knowledge snippet, please determine whether this knowledge snippet is useful for answering the question. Please output one of the following labels: useful, neutral, useless. First perform Chain of Thought (CoT) reasoning and then output the label. Please output strict JSON that can be directly parsed by Python, in the following format: {"CoT": your analysis here, "label": the label obtained after analysis}.
            """,
        },
        {"type": "text", "text": "Here is the question: {query} \n\n"},
        # then the document:
        {"type": "text", "text": "Here is the knowledge snippet: {document} \n\n"},
        {
            "type": "text",
            "text": "Please output the Chain of Thought reasoning and the label in strict JSON format.",
        },
    ],
    "t->ti": [
        # query first:
        {
            "type": "text",
            "text": """
            Given a question and an image with caption, please determine whether this image is useful for answering the question. Please output one of the following labels: useful, neutral, useless. First perform Chain of Thought reasoning and then output the label. Please output strict JSON that can be directly parsed by Python, in the following format: {"CoT": your analysis here, "label": the label obtained after analysis}.
            """,
        },
        {"type": "text", "text": "Here is the question: {query} \n\n"},
        # then the document:
        {"type": "text", "text": "Here is the image with caption:"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,{settings.image_base64}"},
        },
        {"type": "text", "text": "Caption: {caption} \n\n"},
        {
            "type": "text",
            "text": "Please output the Chain of Thought reasoning and the label in strict JSON format.",
        },
    ],
}


# class InferenceReranker(BaseGenerator):
class InferenceReranker_MajorityVote(SampleBaseGenerator): #这里换成sample_base_generator就好了；
    """
    Reranker implementation using vLLM served through an OpenAI-compatible API.

    This reranker leverages ThreadPoolExecutor for concurrent evaluation of
    document relevance based on the BaseGenerator infrastructure.
    """

    def __init__(self, generator_name: str = "vllm_reranker_sample"):
        """
        Initialize the vLLM reranker with configuration from settings.

        Args:
            generator_name: Name of the generator configuration to use
        """
        SampleBaseGenerator.__init__(self, generator_name)
        self.prompts = prompt_templates

    def _prepare_messages(self, query: Dict, document: Dict) -> List[Dict[str, Any]]:  # type: ignore
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
            prompt = copy.deepcopy(
                self.prompts["ti->t"]
            )  # 这里应该使用deepcopy来处理一下，避免直接修改原来的prompt
            # 这里应该使用deepcopy来处理一下，避免直接修改原来的prompt
            # print(f"prompt: {prompt}")
            prompt[2]["image_url"][
                "url"
            ] = f"data:image/jpeg;base64,{self._encode_image(query['image'])}"  # 这里就是把这里的符号换成是转换之后的base64 string就可以；
            prompt[3]["text"] = prompt[3]["text"].format(
                query=query["text"]
            )  # 这里是把query的question放进去；
            prompt[4]["text"] = prompt[4]["text"].format(
                document=document["text"]
            )  # 这里是把document的text放进去；

        elif query["type"] == "t" and document["type"] == "t":
            # prompt = self.prompts["t->t"]
            prompt = copy.deepcopy(
                self.prompts["t->t"]
            )  # 这里应该使用deepcopy来处理一下，避免直接修改原来的prompt
            prompt[1]["text"] = prompt[1]["text"].format(
                query=query["text"]
            )  # 这里是把query的question放进去；
            prompt[2]["text"] = prompt[2]["text"].format(
                document=document["text"]
            )  # 这里是把document的text放进去；

        elif query["type"] == "t" and document["type"] == "ti":
            # prompt = self.prompts["t->ti"]
            prompt = copy.deepcopy(
                self.prompts["t->ti"]
            )  # 这里应该使用deepcopy来处理一下，避免直接修改原来的prompt
            prompt[1]["text"] = prompt[1]["text"].format(
                query=query["text"]
            )  # 这里是把query的question放进去；
            prompt[3]["image_url"][
                "url"
            ] = f"data:image/jpeg;base64,{self._encode_image(document['image'])}"  # 这里就是把这里的符号换成是转换之后的base64 string就可以；
            prompt[4]["text"] = prompt[4]["text"].format(
                caption=document["text"]
            )  # 这里是把document的caption放进去；

        else:
            raise ValueError("Unsupported query-document type combination")

        return [{"role": "user", "content": prompt}]

    def _prepare_messages_for_batch(
        self, query_doc_dicts: List[Dict]
    ) -> List[Tuple[List[Dict[str, Any]], str]]:
        """
        Prepare messages for a batch of documents.

        Args:
            query: The query string
            documents: List of document texts

        Returns:
            List of tuples (messages, document_index) for batch processing
        """
        message_list = []

        for index, item in enumerate(query_doc_dicts):
            messages = self._prepare_messages(item["question"], item["document"])
            # Use index as the request_id for tracking
            # print(f"messages: {messages}")
            message_list.append(
                (messages, str(index))
            )  # 这里的这个id不重要，只要保证返回的是和输入的顺序一致就好了，然后其他的可以等返回label之后再去匹配就好了；

        return message_list

    def _extract_label(self, response_text: str) -> str:
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

                json_match = re.search(
                    r'\{.*"label"\s*:\s*"([^"]+)".*\}', response_text, re.DOTALL
                )
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return 0.0  # No JSON found

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

    def majority_vote(self, strings):
        if not strings:
            return "not"

        # 使用Counter计数并找出最常见的元素
        counter = Counter(strings)
        return counter.most_common(1)[0][0]

    def generate(self, query_doc_dicts) -> List[int]:
        messages = self._prepare_messages_for_batch(query_doc_dicts)
        results = self.generate_batch(messages)
        labels = [
            self.majority_vote([self._extract_label(item) for item in result[0]])
            for result in results
        ]
        # majority vote的话，就是这里的results[0]变成了一个list而不是string；
        return [
            {
                "question_id": query_doc_dicts[i]["question_id"],
                # "question": query_doc_dicts[i]["question"],
                "document": query_doc_dicts[i]["document"],
                "response": results[i][0],
                "label": labels[i],
            }
            for i in range(len(query_doc_dicts))
        ]

    def organize_by_question_id(self, data_list):
        """
        Organizes documents by question_id and their corresponding labels.

        Args:
            data_list: List of dictionaries containing question_id, document, and label

        Returns:
            Dict with question_ids as keys and categorized documents as values
        """
        result = {}

        for item in data_list:
            question_id = item["question_id"]
            document = item["document"]
            label = item["label"]

            # Initialize the question_id entry if it doesn't exist
            if question_id not in result:
                result[question_id] = {
                    "useful": [],
                    "neutral": [],
                    "useless": [],
                    "not": [],
                }

            # Add the document to the appropriate category based on label
            if label == "useful":
                result[question_id]["useful"].append(document)
            elif label == "neutral":
                result[question_id]["neutral"].append(document)
            elif label == "useless":
                result[question_id]["useless"].append(document)
            else:  # Assuming anything else falls into "not" category
                result[question_id]["not"].append(document)

        # 得到这个结果之后，然后保存成pickle文件
        # 感觉这个是对inference友好啊，就是测试跑结果的时候，那reranker的指标怎么搞呢？难道一个结果让我跑两遍？
        # 那可不行，也不是不能测，这样子可能还更好一点，把entity_id或者doc_id带上就好了，这样应该比之前的更快；
        # 然后根据这个转换成ranking就好了；

        return result


def test():
    """Test function for the VLLMReranker"""
    # Configure with your vLLM generator configuration
    reranker = InferenceReranker_MajorityVote(generator_name="vllm_reranker_sample")

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
        "text": "machine learning techniques for time series forecasting",
    }
    documents = [
        {
            "type": "t",
            "text": "Machine learning models like LSTM and transformer architectures have shown great success in time series forecasting tasks.",
        },
        {
            "type": "t",
            "text": "The stock market showed significant volatility yesterday, with tech stocks experiencing the largest decline.",
        },
        {
            "type": "t",
            "text": "Time series forecasting is a critical task in many domains, from weather prediction to financial market analysis.",
        },
        {
            "type": "t",
            "text": "Deep learning approaches, particularly recurrent neural networks, have become popular for sequence prediction tasks.",
        },
        {
            "type": "t",
            "text": "Reinforcement learning has applications in robotics and game playing.",
        },
        {
            "type": "t",
            "text": "Natural language processing enables computers to understand human language.",
        },
    ]

    query_doc_dicts = [
        {"question_id": i, "question": query, "document": doc}
        for i, doc in enumerate(documents)
    ]

    results = reranker.generate(query_doc_dicts)
    from pprint import pprint

    pprint(results)


if __name__ == "__main__":
    # Uncomment to run test
    test()
