from PIL import Image
from typing import Dict, Optional, Tuple, List, Any
from .base import BaseGenerator
import json

class LLMKB(BaseGenerator):
    def __init__(self):
        super().__init__("llm_kb")

    def _prepare_messages(self, **kwargs) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        assert "question" in kwargs and kwargs["question"] is not None
        assert "id" in kwargs
        prompts = self.prompts
        messages = [
            {
                "role": "system",
                "content": prompts["system"]
            }
        ]
        
        user_content = [
            {
                "type": "text",
                "text": prompts["user"].replace("{question}", kwargs["question"])
            }
        ]
        
        messages.append({
            "role": "user",
            "content": user_content
        })

        # print(messages)
        
        return messages, kwargs["id"]

    def _prepare_messages_for_batch(self, **kwargs) -> List[Tuple[List[Dict[str, Any]], Optional[str]]]:
        """
        input format:
        [
            (question, id),
            (question, id),
            ...
        ]
        """
        assert "question_id_list" in kwargs and kwargs["question_id_list"] is not None
        question_id_list = kwargs["question_id_list"]
        messages_list = []
        for question, id in question_id_list:
            messages, id = self._prepare_messages(question=question, id=id)
            messages_list.append((messages, id))
        return messages_list

    def generate(self, **kwargs) -> Tuple[str, Optional[str]]:
        """Generate decomposed queries for multimodal search."""
        question = kwargs["question"]
        id = kwargs.get("id")
        messages, _ = self._prepare_messages(question=question, id=id)
        response, _ = self._generate_single(messages)
        return response, id
    
    def generate_knowledge_batch(self, question_id_list: List[Tuple[str, int]]):
        """
        Input format:
        [
            (question, id),
            ...
        ]
        Output format:
        [
            (query, id),
            ...
        ]
        """
        messages = self._prepare_messages_for_batch(question_id_list=question_id_list)
        return self.generate_batch(message_list=messages)


def test():
    query_translator = LLMKB()
    question = "珠穆朗玛峰多高？"
    print(query_translator.generate(question=question, id=0))
    question_id_list = [
        ("What breed is this dog?", 0),
        ("What is the name of this cat?", 1)
    ]
    messages_list = query_translator._prepare_messages_for_batch(question_id_list=question_id_list)
    response = query_translator.generate_batch(message_list=messages_list)
    print(response)

if __name__ == "__main__":
    test()


