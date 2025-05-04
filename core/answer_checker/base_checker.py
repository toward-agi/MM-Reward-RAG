# 这个的用处就是，检查一下gt和pred到底一不一致，一致的话，就直接用gt代替pred，免得evaluation函数出错。
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


class AnswerChecker(BaseGenerator):
    """
    Reranker implementation using vLLM served through an OpenAI-compatible API.

    This reranker leverages ThreadPoolExecutor for concurrent evaluation of
    document relevance based on the BaseGenerator infrastructure.
    """

    def __init__(self, generator_name: str = "vllm_answer_generator"):
        """
        Initialize the vLLM reranker with configuration from settings.

        Args:
            generator_name: Name of the generator configuration to use
        """
        BaseGenerator.__init__(self, generator_name)
        self.prompts = """
        Given a question, a ground truth answer, and a prediction answer, please evaluate the prediction answer.
        If the prediction answer is correct, please return "True".
        If the prediction answer is wrong, please return "False".
        The question is: {{ {question} }}
        The ground truth answer is: {{ {ground_truth_answer} }}
        The prediction answer is: {{ {prediction_answer} }}
        Please answer with "True" or "False".
        """

    def _prepare_messages(self, question: str, ground_truth: str, prediction: str) -> List[Dict[str, Any]]:  # type: ignore
        prompt = copy.deepcopy(self.prompts)
        prompt = prompt.format(
            question=question,
            ground_truth_answer=ground_truth,
            prediction_answer=prediction,
        )

        return [{"role": "user", "content": prompt}]

    def _prepare_messages_for_batch(
        self, triples: List[Dict]
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

        for triple in triples:
            messages = self._prepare_messages(
                question=triple["question"],
                ground_truth=triple["ground_truth"],
                prediction=triple["prediction"],
            )

            message_list.append((messages, triple["data_id"]))

        return message_list

    def _extract_answer(self, response_text: str) -> str:

        try:
            response_text = response_text.lower()
            if "true" in response_text and "false" in response_text:
                raise ValueError("Response contains both 'True' and 'False'.")
            if "true" in response_text:
                return "True"
            elif "false" in response_text:
                return "False"
            else:
                raise ValueError("Response does not contain 'True' or 'False'.")

        except Exception as e:
            print(f"Error extracting score from response: {e}")
            return "False"

    def generate_answer(self, triples: List[Dict]) -> List[int]:

        message_list = self._prepare_messages_for_batch(triples)

        # Process in batches using ThreadPoolExecutor
        results = self.generate_batch(message_list)
        results = {
                result[1]: self._extract_answer(result[0])
                for result in results
            } 
    
        return results
    

    def generate(
        self, triples: List[Dict]
    ) -> List[int]:

        return self.generate_answer(triples)

def test():

    answer_checker = AnswerChecker(generator_name="vllm_answer_generator")

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

    triples = [
        {
            "question": "1 + 1 = ?",
            "ground_truth": "2",
            "prediction": "2.00",
            "data_id": "1",
        }
    ]

    # Get ranked document indices
    results = answer_checker.generate_answer(triples)
    from pprint import pprint
    pprint(results)


if __name__ == "__main__":
    # Uncomment to run test
    test()
