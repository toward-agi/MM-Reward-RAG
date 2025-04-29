
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
        Attached images are External knowledge:
        """

    def _prepare_messages(self, **kwargs) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        assert "image_id_prompts" in kwargs and kwargs["image_id_prompts"] is not None
        image_id_prompts = kwargs["image_id_prompts"]
        image_dir = '/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/final_dataset_images'
        prompts = self.prompts
        content = [
            {
                "type": "text",
                "text": prompts.format(Q=image_id_prompts["question"])
            },
        ]
        for image_path in image_id_prompts["knowledge_context"]:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(Image.open(os.path.join(image_dir, image_path)).convert('RGB'))}"
                }
            })
        content.append({
            "type": "text",
            "text": "Answer: "
        })
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        return messages, image_id_prompts["qid"]
    
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

def read_jsonl(filename):
    with open(filename, 'r') as f:
        data = [json.loads(l.strip()) for l in f.readlines()]
    return data

valid_dataset = {}
text_docs = read_jsonl("/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/MMQA_texts.jsonl")
text_docs = {doc["id"]: doc for doc in text_docs}
image_docs = read_jsonl("/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/MMQA_images.jsonl")
image_docs = {doc["id"]: doc for doc in image_docs}
with open("/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/MMQA_dev_textQ_ImageQ.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        valid_dataset[item['qid']] = item
output_dir = "results/mmqa/image_kb"
retrieved_results_file = f"{output_dir}/retrieved_results_hard_neg.json"
os.makedirs(output_dir, exist_ok=True)
if not os.path.exists(retrieved_results_file):
    embedder = JinaClipV2Embedder() #将question或者image转换为向量
    
    # 加载index
    # 不同benchmark加载的index不一样，都在textual_index_gpu/visual_index_gpu下  
    #indexer.load("textual_index_gpu/webqa")
    bs = 512
    id_to_retrieved = {}
    valid_dataset_list = list(valid_dataset.items())
    for i in tqdm(range(0, len(valid_dataset_list), bs)):
        batch = valid_dataset_list[i:i+bs]
        data_ids = [item[0] for item in batch]
        questions = [item[1]["question"] for item in batch]
        image_doc_ids = [item[1]['metadata']['image_doc_ids'] for item in batch]
        image_doc_vectors = []
        image_dir = '/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/final_dataset_images'
        for id_list in image_doc_ids:
            images = []
            for id in id_list:
                images.append(Image.open(os.path.join(image_dir, image_docs[id]["path"])).convert("RGB"))
            
            image_doc_vectors.append(embedder.embed_images(images))
        # Get embeddings for questions
        query_vectors = embedder.embed_text(questions)
        # Process each question separately
        for j, (data_id, query_vec, doc_vectors) in enumerate(zip(data_ids, query_vectors, image_doc_vectors)):
            # Get embeddings for this question's facts
            if len(doc_vectors) == 0:
                id_to_retrieved[data_id] = {
                    "data_id": data_id,
                    "retrieved": []
                }
                continue
            # Calculate scores between query and facts
            scores = (query_vec @ doc_vectors.T).tolist()
            # Get top 5 fact indices
            top_k_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:5]
            
            id_to_retrieved[data_id] = {
                "data_id": data_id,
                "retrieved": [{
                    "image": image_docs[image_doc_ids[j][idx]]["path"], 
                    "score": scores[idx],
                } for idx in top_k_indices]
            }
    with open(retrieved_results_file, "w") as f:
        json.dump(id_to_retrieved, f, indent=2)
else:
    id_to_retrieved = json.load(open(retrieved_results_file, "r"))

def create_knowledge_context(retrieved_result):
    return [item['image'] for item in retrieved_result['retrieved'][:1]]

def test():
    lvlm_generator = BaselineLVLMGenerator()
    valid_dataset = {}
    with open("/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/MMQA_dev_textQ_ImageQ.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            valid_dataset[item['qid']] = item 
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

    output_file_dir= "results/mmqa/text_kb"
    import os
    os.makedirs(output_file_dir, exist_ok=True)

    output_file_path = f"{output_file_dir}/qwen-2-vl-hard_neg.json"

    
    buffer_size = 1000000
    valid_data_items = list(valid_dataset.items())
    for i in tqdm(range(0, len(valid_data_items), buffer_size)):
        buffer = dict(valid_data_items[i:i + buffer_size])
        image_id_prompts_list = []
        for id, data in tqdm(buffer.items()):
            qid = data.get("qid", "")
            question = data.get("question", "")
            answers = data.get("answers", [])
            metadata = data.get("metadata", [])
            supporting_context = data.get("supporting_context", [])
            knowledge_context = create_knowledge_context(id_to_retrieved[id])
            image_id_prompts_list.append({
                "qid": qid,
                "question": question,
                "answers": answers,
                "metadata": metadata,
                "supporting_context": supporting_context,
                "knowledge_context": knowledge_context
            })
            
        print(f"Generating descriptions for {len(image_id_prompts_list)} questions...")
        descriptions = lvlm_generator.generate_description_batch(image_id_list=image_id_prompts_list)
        formatted_descriptions = []
        for description, id in descriptions:
            # Find matching original data
            original_data = next(item for item in image_id_prompts_list if item["qid"] == id)
            # Ensure data is properly formatted for JSON serialization by cleaning quotes
            formatted_descriptions.append({
                "qid": original_data["qid"],
                "question": original_data["question"],
                "answers": original_data["answers"],
                "metadata": original_data["metadata"],
                "supporting_context": original_data["supporting_context"],
                "knowledge_context": original_data["knowledge_context"],
                "prediction": description
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
