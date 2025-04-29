from generators.base import BaseGenerator
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
    
from core.indexing.faiss_indexer import FaissIndexer
from core.embeddings.jina_clip_v2 import JinaClipV2Embedder
import json
from tqdm import tqdm
from PIL import Image

output_dir = "results/okvqa/text_kb_whole"
retrieved_results_file = f"{output_dir}/retrieved_results.json"
import os
os.makedirs(output_dir, exist_ok=True)

# 这里是加载用来测试的数据集
# valid_dataset = json.load(open("/data/track/mm-RewardRAG/eval_datasets/aokvqa/test.json", "r"))
from datasets import load_dataset

okvqa = load_dataset("lmms-lab/OK-VQA")
valid_dataset = okvqa["val2014"]

# 先做检索
if not os.path.exists(retrieved_results_file):
    embedder = JinaClipV2Embedder() #将question或者image转换为向量
    indexer = FaissIndexer(mode="jina-clip-v2") #建的索引
    
    # 加载index
    # 不同benchmark加载的index不一样，都在textual_index_gpu/visual_index_gpu下  
    indexer.load("textual_index_gpu/encyclopedia/whole")
    bs = 512
    id_to_retrieved = []
    for i in tqdm(range(0, len(valid_dataset), bs)):
        batch = valid_dataset[i:i+bs]
        # data_ids = [item["question_id"] for item in batch]
        # questions = [item["question"] for item in batch]
        data_ids = batch["question_id"]
        questions = batch["question"]
        query_vectors = embedder.embed_text(questions)
        print(query_vectors.shape)
        results = indexer.search_batch(query_vectors, top_k=3) #检索,这里top-k就是返回的检索结果的数量
        for data_id, result in zip(data_ids, results):
            id_to_retrieved.append({
                "data_id": data_id,
                "retrieved": result
            })

    with open(retrieved_results_file, "w") as f:
        json.dump(id_to_retrieved, f, indent=2)
else:
    id_to_retrieved = json.load(open(retrieved_results_file, "r"))

# 这里就是把top-3的检索结果拼接起来，需要注意下有些benchmark的kb很长，可能会超过模型的max_length
def create_knowledge_context(retrieved_result):
    return " ".join([' '.join(item["section_texts"]) for item in retrieved_result[2][:3]])

generator = BaselineLVLMGenerator()

image_id_prompts_list = []
for data, retrieved_item in tqdm(zip(valid_dataset, id_to_retrieved)):
    knowledge_context = create_knowledge_context(retrieved_item["retrieved"])
    input_question = data["question"]
    input_question = "External knowledge: " + knowledge_context + " Question: " + input_question
    # image = Image.open("/data/mingyan/encyclopedic/" + data["image_path"]).convert("RGB")
    image = data['image'].convert("RGB")
    image_id_prompts_list.append((image, data["question_id"], input_question))

print(image_id_prompts_list[0])

output_file_dir= "results/okvqa/text_kb_whole"
import os
os.makedirs(output_file_dir, exist_ok=True)

output_file_path = f"{output_file_dir}/qwen-2-vl.json"
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

