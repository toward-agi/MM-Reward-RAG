
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import json
import os
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
        External knowledge: {knowledge_context}
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
                        "text": prompts.format(Q=image_id_prompts["Q"], knowledge_context=image_id_prompts["knowledge_context"])
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

from core.indexing.faiss_indexer import FaissIndexer
from core.embeddings.jina_clip_v2 import JinaClipV2Embedder

valid_dataset = json.load(open("/data/track/mm-RewardRAG/eval_datasets/webqa/WebQA_data_first_release/WebQA_val.json", "r"))
output_dir = "results/webqa/text_kb"
retrieved_results_file = f"{output_dir}/retrieved_results.json"
os.makedirs(output_dir, exist_ok=True)
if not os.path.exists(retrieved_results_file):
    embedder = JinaClipV2Embedder() #将question或者image转换为向量
    indexer = FaissIndexer(mode="jina-clip-v2") #建的索引
    
    # 加载index
    # 不同benchmark加载的index不一样，都在textual_index_gpu/visual_index_gpu下  
    indexer.load("textual_index_gpu/webqa")
    bs = 512
    id_to_retrieved = {}
    valid_dataset_list = list(valid_dataset.items())
    for i in tqdm(range(0, len(valid_dataset_list), bs)):
        batch = valid_dataset_list[i:i+bs]
        data_ids = [item[0] for item in batch]
        questions = [item[1]["Q"] for item in batch]
        query_vectors = embedder.embed_text(questions)
        results = indexer.search_batch(query_vectors, top_k=5) #检索,这里top-k就是返回的检索结果的数量
        for data_id, result in zip(data_ids, results):
            id_to_retrieved[data_id] = {
                "data_id": data_id,
                "retrieved": result
            }
    with open(retrieved_results_file, "w") as f:
        json.dump(id_to_retrieved, f, indent=2)
else:
    id_to_retrieved = json.load(open(retrieved_results_file, "r"))

def create_knowledge_context(retrieved_result):
    return " ".join([item['fact'] for item in retrieved_result['retrieved'][2][:3]])

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

    output_file_dir= "results/webqa/text_kb"
    import os
    os.makedirs(output_file_dir, exist_ok=True)

    output_file_path = f"{output_file_dir}/qwen-2-vl-full.json"

    
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
            knowledge_context = create_knowledge_context(id_to_retrieved[id])
            Output_conf = data.get("Output_conf", [])
            image_id_prompts_list.append({
                "Guid": Guid,
                "Qcate": Qcate,
                "Q": Q,
                "A": A,
                "Output_conf": Output_conf,
                "knowledge_context": knowledge_context
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
