from PIL import Image
from typing import Dict, Optional, Tuple, List, Any
from .base import BaseGenerator
import json

class QueryTranslator(BaseGenerator):
    def __init__(self):
        super().__init__("query_translator")

    def _prepare_messages(self, **kwargs) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        assert "question" in kwargs and kwargs["question"] is not None
        assert "id" in kwargs
        prompts = self.prompts
        image = kwargs.get("image")
        
        messages = [
            {
                "role": "system",
                "content": prompts["system"]
            }
        ]
        
        if image is not None:
            user_content = [
                {
                    "type": "text",
                    "text": prompts["user"].replace("{question}", kwargs["question"])
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self._encode_image(image)}"
                    }
                }
            ]
        else:
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
            (question, image, id),
            (question, image, id),
            ...
        ]
        """
        assert "question_img_id_list" in kwargs and kwargs["question_img_id_list"] is not None
        question_img_id_list = kwargs["question_img_id_list"]
        messages_list = []
        for question, image, id in question_img_id_list:
            messages, id = self._prepare_messages(question=question, image=image, id=id)
            messages_list.append((messages, id))
        return messages_list

    def generate(self, **kwargs) -> Tuple[str, Optional[str]]:
        """Generate decomposed queries for multimodal search."""
        question = kwargs["question"]
        image = kwargs.get("image")
        id = kwargs.get("id")
        messages, _ = self._prepare_messages(question=question, image=image, id=id)
        response, _ = self._generate_single(messages)
        return response, id
    
    def generate_queries_batch(self, question_img_id_list: List[Tuple[str, Image.Image, int]]):
        """
        Input format:
        [
            (question, image, id),
            ...
        ]
        Output format:
        [
            (query, id),
            ...
        ]
        """
        messages = self._prepare_messages_for_batch(question_img_id_list=question_img_id_list)
        return self.generate_batch(message_list=messages)


def test():
    query_translator = QueryTranslator()
    question = "这个动物生活在哪个国家？"
    image = Image.open("test/images/dog.png").convert("RGB").resize((512, 512))
    print(query_translator.generate(question=question, image=image, id=0))
    question_img_id_list = [(question, image, i) for i in range(2)]
    messages_list = query_translator._prepare_messages_for_batch(question_img_id_list=question_img_id_list)
    response = query_translator.generate_batch(message_list=messages_list)
    print(response)

if __name__ == "__main__":
    test()


