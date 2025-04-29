from generators.base import BaseGenerator

from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import pandas as pd

class BaselineLVLMGenerator(BaseGenerator):
    def __init__(self):
        super().__init__("baseline_lvlm_generator")

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
        for image, id, user_prompt in image_id_prompts_list:
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
    input_data_file = "/data/dataset/mm_RewardRAG/encyclopedic/test.csv"
    image_root = "/data/mingyan/encyclopedic"
    input_data = pd.read_csv(input_data_file)
    print(input_data.iloc[0])
    print(len(input_data))
    """
    {
        "data_id": "infoseek_val_00000000",
        "image_id": "oven_04990048",
        "question": "What is the objective of this object?",
        "answer": [
            "energy transformation"
        ],
        "answer_eval": [
            "energy conversion",
            "energy transformation",
            "conversion"
        ],
        "data_split": "val_unseen_question",
        "entity_id": "Q178185",
        "image_path": "/wikipedia_images_full/Q178/Q178185.jpg"
    },
    """

    output_file_path = "results/baseline_lvlm/encyclopedic/llama-11-vision.json"
    import os
    os.makedirs("results/baseline_lvlm/encyclopedic", exist_ok=True)

    import json
    from tqdm import tqdm
    buffer_size = 1000
    for i in tqdm(range(0, len(input_data), buffer_size)):
        # 提取当前缓冲区的数据
        buffer = input_data.iloc[i:i + buffer_size]  # 用 DataFrame 的切片
        image_id_prompts_list = []
        
        # 遍历缓冲区中的每一行
        for _, row in buffer.iterrows():
            image_path = f"{image_root}/{row['image_path']}"  # 拼接图像路径
            try:
                image = Image.open(image_path)  # 打开图像
                image_id_prompts_list.append((image, row["question_id"], row["question"]))
            except Exception as e:
                print(f"Error opening {image_path}: {str(e)}")
        
        # 使用 LVLM 生成描述
        descriptions = lvlm_generator.generate_description_batch(image_id_list=image_id_prompts_list)
        
        # 写入输出文件
        with open(output_file_path, "a") as f:
            for description, id in descriptions:
                f.write(json.dumps({"data_id": id, "response": description}) + "\n")
if __name__ == "__main__":
    test()
