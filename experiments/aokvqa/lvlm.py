
# import sys
# sys.path.append("/data/track/mm-RewardRAG")
from generators.base import BaseGenerator

from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

import json
from tqdm import tqdm

import resource
# 提高限制（在Linux/Unix系统中）
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

class BaselineLVLMGenerator(BaseGenerator):
    def __init__(self):
        super().__init__("vanilla_lvlm_generator")

    def _prepare_messages(self, **kwargs) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        assert "image" in kwargs and kwargs["image"] is not None
        assert "id" in kwargs and kwargs["id"] is not None
        assert "user_prompt" in kwargs and kwargs["user_prompt"] is not None

        prompts = self.prompts
        messages = [
            # {
            #     "role": "system",
            #     "content": prompts["system"]
            # }, for baseline, system prompt is not necessary
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": kwargs["user_prompt"]
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self._encode_image(kwargs['image'])}"
                        }
                    }
                ]
            }
        ]
        

        # print(messages)
        
        return messages, kwargs["id"]
    
    def generate(self, **kwargs) -> Tuple[str, Optional[str]]:
        image = kwargs["image"]
        id = kwargs["id"]
        user_prompt = kwargs["user_prompt"]
        messages,id = self._prepare_messages(image=image,id=id,user_prompt=user_prompt)
        return self._generate_single(messages,id)

    def _prepare_messages_for_batch(self, **kwargs) -> List[Tuple[List[Dict[str, Any]], Optional[str]]]:
        assert "image_id_prompts_list" in kwargs and kwargs["image_id_prompts_list"] is not None
        image_id_prompts_list = kwargs["image_id_prompts_list"]
        messages_list = []
        for image, id, user_prompt in tqdm(image_id_prompts_list):
            messages, id = self._prepare_messages(image=image, id=id, user_prompt=user_prompt)
            messages_list.append((messages, id))
        return messages_list

    def generate_description_batch(self, image_id_list: List[Tuple[Image.Image, int, str]]) -> List[Tuple[str, int]]:
        """
        Input format:
        [
            (image: PIL.Image, id, user_prompt),
            ...
        ]
        Output format:
        [
            (description, id),
            ...
        ]
        """
        messages = self._prepare_messages_for_batch(image_id_prompts_list=image_id_list)
        print("prepare done")
        return self.generate_batch(message_list=messages)

def test():
    lvlm_generator = BaselineLVLMGenerator()
    # image = Image.open("test/images/hk.png")
    # print(description_generator.generate(image=image, id=0))
    # image_list = [(image, i) for i in range(2)]
    # messages_list = description_generator._prepare_messages_for_batch(image_id_list=image_list)
    # # print(messages_list[0][0][0])
    # print(len(messages_list))
    # descriptions = description_generator.generate_batch(message_list=messages_list)
    # print(descriptions)
    from datasets import load_dataset

    aokvqa = load_dataset("HuggingFaceM4/A-OKVQA")

    # input_data_file = "/data/track/mm-RewardRAG/eval_datasets/encyclopedia/test.json"
    """
    {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x569>,
    'question_id': '22jbM6gDxdaMaunuzgrsBB',
    'question': "What is in the motorcyclist's mouth?",
    'choices': ['toothpick', 'food', 'popsicle stick', 'cigarette'],
    'correct_choice_idx': 3,
    'direct_answers': "['cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette']",
    'difficult_direct_answer': False,
    'rationales': ["He's smoking while riding.",
    'The motorcyclist has a lit cigarette in his mouth while he rides on the street.',
    'The man is smoking.']}
    """

    output_file_dir= "/data/track/mm-RewardRAG/results/aokvqa/vanilla_lvlm"
    import os
    os.makedirs(output_file_dir, exist_ok=True)

    output_file_path = f"{output_file_dir}/qwen-2-vl.json"

    
    # valid_data = json.load(open(input_data_file))
    valid_data = aokvqa["validation"]

    buffer_size = 1000000
    for i in tqdm(range(0, len(valid_data), buffer_size)):
        subset = valid_data[i:i + buffer_size]
        keys = subset.keys()
        values = list(zip(*subset.values()))
        buffer = [dict(zip(keys, value_tuple)) for value_tuple in values]
        image_id_prompts_list = []
        for data in tqdm(buffer):
        # for k in range(len(valid_data)):
            # image_path = f"/data/mingyan/encyclopedic/{data['image_path']}"
            # data = valid_data[k]
            try:
                image = data['image'].convert("RGB")

                image_id_prompts_list.append((image, data["question_id"], data["question"]))
            except Exception as e:
                print(f"Error processing {data['question_id']}:{str(e)}")
                # print(f"Error opening {image_path}: {str(e)}")
        print(f"Generating descriptions for {len(image_id_prompts_list)} images...")
        descriptions = lvlm_generator.generate_description_batch(image_id_list=image_id_prompts_list)
        with open(output_file_path, "a") as f:
            for description, id in descriptions:
                f.write(json.dumps({"data_id": id, "response": description}) + "\n")

def test_image():
    input_data_file = "/data/dataset/mm_RewardRAG/InfoSeek/valid.json"
    data = json.load(open(input_data_file))
    image_id_prompts_list = []
    for item in tqdm(data):
        image_path = item["image_path"]
        image = Image.open(image_path).convert("RGB")
        image_id_prompts_list.append((image, item["data_id"], item["question"]))
    
if __name__ == "__main__":
    test()
    # test_image()
