from re import A
import re
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any

from transformers.models.deprecated import retribert
from .base import BaseGenerator


class ResponseGenerator(BaseGenerator):
    def __init__(self):
        super().__init__("response_generator")
        
    def _prepare_messages(self, **kwargs) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        input format:
        (
            question,
            mode,
            image,
            retrieved_image, # TODO: multiple images
            retrieved_description,
            retrieved_knowledge,
            id
        )
        """
        assert "question" in kwargs and kwargs["question"] is not None
        assert "mode" in kwargs and kwargs["mode"] in ["question_image", "question_image_description", "question_description"]
        assert "id" in kwargs
        assert "image" in kwargs
        assert "retrieved_image" in kwargs
        assert "retrieved_description" in kwargs
        
        prompts = self.prompts
        mode = kwargs["mode"]
        image = kwargs.get("image", None)
        use_knowledge = self.config.get("use_knowledge", True)
        
        if use_knowledge:
            assert "retrieved_knowledge" in kwargs and kwargs["retrieved_knowledge"] is not None, "Knowledge is required when use_knowledge is True"
            knowledge = kwargs["retrieved_knowledge"]
            user_content = prompts[mode]["user"] + prompts[mode]['knowledge_suffix']
            user_content = user_content.replace("{knowledge}", knowledge)
        else:
            user_content = prompts[mode]["user"]
            knowledge = None
        
        if mode == "question_image":
            assert "retrieved_image" in kwargs and kwargs["retrieved_image"] is not None, "Retrieved image is required for mode 'question_image'"
            retrieved_image = kwargs["retrieved_image"]
            user_content = user_content.replace("{question}", kwargs["question"])

            # TODO: the num of retrieved images?
            messages = [
                {
                    "role": "system",
                    "content": prompts[mode]["system"]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_content
                        },
                    ]
                }
            ]

            if image is not None:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self._encode_image(image)}"
                    }
                })
            
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self._encode_image(retrieved_image)}"
                }
            })

        elif mode == "question_image_description":
            assert "retrieved_image" in kwargs and kwargs["retrieved_image"] is not None, "Retrieved image is required for mode 'question_image_description'"
            retrieved_image = kwargs["retrieved_image"]

            assert "retrieved_description" in kwargs and kwargs["retrieved_description"] is not None, "Retrieved description is required for mode 'question_image_description'"
            description = kwargs["retrieved_description"]

            user_content = prompts[mode]["user"].replace("{question}", kwargs["question"])
            user_content = user_content.replace("{description}", description)
            messages = [
                {
                    "role": "system",
                    "content": prompts[mode]["system"]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_content
                        }
                    ]
                }
            ]
            if image is not None:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self._encode_image(image)}"
                    }
                })

            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self._encode_image(retrieved_image)}"
                }
            })
        elif mode == "question_description": # the retrieved image is not required, but the input image is required
            assert "retrieved_description" in kwargs and kwargs["retrieved_description"] is not None, "Retrieved description is required for mode 'question_description'"
            description = kwargs["retrieved_description"]

            user_content = prompts[mode]["user"].replace("{question}", kwargs["question"])
            user_content = user_content.replace("{description}", description)
            messages = [
                {
                    "role": "system",
                    "content": prompts[mode]["system"]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_content
                        }
                    ]
                }
            ]
            if image is not None:
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self._encode_image(image)}"
                    }
                })
        else:
            raise ValueError(f"Invalid mode: {mode}, expected one of ['question_image', 'question_image_description', 'question_description']")
        
        return messages, kwargs["id"]

    def _prepare_messages_for_batch(self, **kwargs) -> List[Tuple[List[Dict[str, Any]], Optional[str]]]:
        """
        input format:
        [
            (question, mode, image, retrieved_image, retrieved_description, retrieved_knowledge, id),
            (question, mode, image, retrieved_image, retrieved_description, retrieved_knowledge, id),
            ...
        ]
        """
        assert "input_list" in kwargs and kwargs["input_list"] is not None
        input_list = kwargs["input_list"]
        messages_list = []
        for question, mode, image, retrieved_image,retrieved_description,retrieved_knowledge, id in input_list:
            messages, id = self._prepare_messages(
                question=question,
                mode=mode,
                image=image,
                retrieved_image=retrieved_image,
                retrieved_description=retrieved_description,
                retrieved_knowledge=retrieved_knowledge,
                id=id
            )
            messages_list.append((messages, id))
        return messages_list

    def generate(self, **kwargs) -> Tuple[str, Optional[str]]:
        """Generate response based on the question and retrieved information."""
        messages, id = self._prepare_messages(**kwargs)
        print(messages[0])
        print(messages[1]['content'][0])
        exit()
        response, _ = self._generate_single(messages)
        return response, id


def test():
    response_generator = ResponseGenerator()

    # TODO: should be update
    # input image + retrieved image
    # Test question_image mode
    question = "What breed is this dog?"
    image = Image.open("test/images/hk.png")
    print("Testing question_image mode with knowledge (required if use_knowledge is true):")
    input_item = (question, "question_image", image, image, "A golden retriever sitting on the grass", "Golden retrievers are known for their friendly temperament and golden coat.", 0)
    print(response_generator.generate(
        question=input_item[0],
        mode=input_item[1],
        image=input_item[2],
        retrieved_image=input_item[3],
        retrieved_description=input_item[4],
        retrieved_knowledge=input_item[5],
        id=input_item[6]
    ))
    
    # Test question_image_description mode
    # print("\nTesting question_image_description mode with knowledge (required if use_knowledge is true):")
    # print(response_generator.generate(
    #     question=question,
    #     mode="question_image_description",
    #     image=image,
    #     description="A golden retriever sitting on the grass",
    #     knowledge="Golden retrievers are known for their friendly temperament and golden coat.",
    #     id=1
    # ))
    
    # Test question_description mode
    # print("\nTesting question_description mode with knowledge (required if use_knowledge is true):")
    # print(response_generator.generate(
    #     question=question,
    #     mode="question_description",
    #     description="A golden retriever sitting on the grass",
    #     knowledge="Golden retrievers are known for their friendly temperament and golden coat.",
    #     id=2
    # ))
    messages_for_batch = [
        (question, "question_image", image, image, "A golden retriever sitting on the grass", "Golden retrievers are known for their friendly temperament and golden coat.", 0),
        (question, "question_image", image, image, "A golden retriever sitting on the grass", "Golden retrievers are known for their friendly temperament and golden coat.", 1),
    ]
    print("\nTesting question_image mode with knowledge (required if use_knowledge is true):")
    print(response_generator.generate_batch(message_list=response_generator._prepare_messages_for_batch(input_list=messages_for_batch)))

if __name__ == "__main__":
    test()
