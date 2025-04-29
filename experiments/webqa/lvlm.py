
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import json
from tqdm import tqdm

import resource

import sys
sys.path.append("/data/track/mm-RewardRAG/")
from generators.base import BaseGenerator
# 提高限制（在Linux/Unix系统中）
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

class BaselineLVLMGenerator(BaseGenerator):
    def __init__(self):
        super().__init__("vanilla_lvlm_generator")
        self.prompts = """
        You are a helpful assistant. Please answer the question.
        Question: {Q}
        Answer: 
        """

    def _prepare_messages(self, **kwargs) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        assert "image_id_prompts" in kwargs and kwargs["image_id_prompts"] is not None
        image_id_prompts = kwargs["image_id_prompts"]
        prompts = self.prompts
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompts.format(Q=image_id_prompts["Q"])
                    },
                ]
            }
        ]
        return messages, image_id_prompts["Guid"]
    
    def generate(self, **kwargs) -> Tuple[str, Optional[str]]:
        image = kwargs["image"]
        id = kwargs["id"]
        user_prompt = kwargs["user_prompt"]
        messages, id = self._prepare_messages(image=image,id=id,user_prompt=user_prompt)
        return self._generate_single(messages,id)

    def _prepare_messages_for_batch(self, **kwargs) -> List[Tuple[List[Dict[str, Any]], Optional[str]]]:
        assert "image_id_prompts_list" in kwargs and kwargs["image_id_prompts_list"] is not None
        image_id_prompts_list = kwargs["image_id_prompts_list"]
        messages_list = []
        for image_id_prompts in tqdm(image_id_prompts_list):
            messages, id = self._prepare_messages(image_id_prompts=image_id_prompts)
            messages_list.append((messages, id))
        return messages_list
    
    def generate_description_batch(self, image_id_list: List[Dict]) -> List[Tuple[str, int]]:
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
    input_data_file = "/data/track/mm-RewardRAG/eval_datasets/webqa/WebQA_data_first_release/WebQA_train_val.json"
    """
    {
        "data_id": {
            "Q": "",
            "A": [""],
            "topic": "",
            "split": "",
            "Qcate": "",
            "Guid": "",
            "img_posFacts": [
                {
                    "image_id": "",
                    "title": "",
                    "caption: "",
                    "url": "",
                    "imgUrl": ""
                },
            ],
            "img_negFacts": [
                {
                    "image_id": "",
                    "title": "",
                    "caption: "",
                    "url": "",
                    "imgUrl": ""
                },
            ],
            "txt_negFacts": [
                {
                    "title": "",
                    "fact": "",
                    "url": "",
                    "snippet_id": ""
                },
            ],
            "txt_posFacts": [],
        }
    },
    """

    output_file_dir= "results/webqa/vanilla_lvlm"
    import os
    os.makedirs(output_file_dir, exist_ok=True)

    output_file_path = f"{output_file_dir}/qwen-2-vl.json"

    
    valid_data = json.load(open(input_data_file))
    buffer_size = 1000000
    valid_data_items = list(valid_data.items())
    for i in tqdm(range(0, len(valid_data_items), buffer_size)):
        buffer = dict(valid_data_items[i:i + buffer_size])
        image_id_prompts_list = []
        for id, data in tqdm(buffer.items()):
            split = data.get("split", "")
            if split != "val": continue
            Guid = data.get("Guid", "")
            Qcate = data.get("Qcate", "")
            Q = data.get("Q", "")
            A = data.get("A", [])
            Output_conf = data.get("Output_conf", [])
            image_id_prompts_list.append({
                "Guid": Guid,
                "Qcate": Qcate,
                "Q": Q,
                "A": A,
                "Output_conf": Output_conf,
            })
            
        print(f"Generating descriptions for {len(image_id_prompts_list)} questions...")
        descriptions = lvlm_generator.generate_description_batch(image_id_list=image_id_prompts_list)
        formatted_descriptions = []
        for description, id in descriptions:
            # Find matching original data
            original_data = next(item for item in image_id_prompts_list if item["Guid"] == id)
            # Ensure data is properly formatted for JSON serialization by cleaning quotes
            formatted_descriptions.append({
                "Guid": original_data["Guid"],
                "Qcate": original_data["Qcate"],
                "Q": original_data["Q"].strip('"'),  # Remove outer quotes
                "A": [a.strip('"') for a in original_data["A"]],  # Clean quotes from answers
                "Output_conf": original_data["Output_conf"],
                "Output": [description]
            })
        with open(output_file_path, "w") as f:
            json.dump(formatted_descriptions, f, indent=4)

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
