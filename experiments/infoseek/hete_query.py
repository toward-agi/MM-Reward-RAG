from generators.base import BaseGenerator
from generators.translator import QueryTranslator
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple

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

import json
from tqdm import tqdm
from PIL import Image

# TODO: the output_dir should be changed to the correct path
output_dir = "results/infoseek/hete_query"
retrieved_results_file = f"{output_dir}/retrieved_results.json"
import os
os.makedirs(output_dir, exist_ok=True)

# 这里是加载用来测试的数据集
valid_dataset = json.load(open("/data/dataset/mm_RewardRAG/InfoSeek/valid.json", "r"))

# 先做检索
if not os.path.exists(retrieved_results_file):

    translator = QueryTranslator()
    
    bs = 512

    retrieved_results = []

    for i in tqdm(range(0, len(valid_dataset), bs)):
        batch = valid_dataset[i:i+bs]
        data_ids = [item["data_id"] for item in batch]
        questions = [item["question"] for item in batch]
        image_paths = [item["image_path"] for item in batch]
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        results = translator.generate_queries_batch([(question, image, data_id) for question, image, data_id in zip(questions, images, data_ids)])
        for data_id, result in zip(data_ids, results):
            retrieved_results.append({
                "data_id": data_id,
                "retrieved": result
            })


    with open(retrieved_results_file, "w") as f:
        json.dump(retrieved_results, f, indent=2)
else:
    id_to_queries = json.load(open(retrieved_results_file, "r"))

generator = BaselineLVLMGenerator()

image_id_prompts_list = []
for data, retrieved_item in tqdm(zip(valid_dataset, id_to_retrieved)):
    knowledge_context = retrieved_item["retrieved"][0]
    input_question = data["question"]
    input_question = "External knowledge: " + knowledge_context + " Question: " + input_question
    image = Image.open(data["image_path"]).convert("RGB")
    image_id_prompts_list.append((image, data["data_id"], input_question))

print(image_id_prompts_list[0])

output_file_path = f"{output_dir}/qwen-2-vl.json"
buffer_size = 1000000
for i in tqdm(range(0, len(image_id_prompts_list), buffer_size)):
    buffer = image_id_prompts_list[i:i + buffer_size]
    descriptions = generator.generate_description_batch(buffer)
    with open(output_file_path, "a") as f:
        for description, id in descriptions:
            f.write(json.dumps({
                "data_id": id,
                "description": description
            }) + "\n")

