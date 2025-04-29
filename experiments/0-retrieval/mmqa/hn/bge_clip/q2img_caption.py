import os
import sys
import json
import numpy as np
from tqdm import tqdm

import sys
sys.path.append("/data/track/mm-RewardRAG/")

# from core.embeddings.jina_clip_v2 import JinaClipV2Embedder
from core.embeddings.bge_text_embedding import BGETextEmbedder

def read_jsonl(filename):
    with open(filename, 'r') as f:
        data = [json.loads(l.strip()) for l in f.readlines()]
    return data

# Load datasets
valid_dataset = {}
text_docs = read_jsonl("/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/MMQA_texts.jsonl")
text_docs = {doc["id"]: doc for doc in text_docs}
image_docs = read_jsonl("/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/MMQA_images.jsonl")
image_docs = {doc["id"]: doc for doc in image_docs}

# Load only questions with images
with open("/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/new_MMQA_dev_textQ_ImageQ.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        if "ImageQ" in item['metadata']['type']:
            valid_dataset[item['qid']] = item

print(f"Image questions in valid_dataset: {len(valid_dataset)}")

output_dir = "/data/track/mm-RewardRAG/experiments/0-retrieval/results/mmqa/hn/bge_clip/q2img_caption"
retrieved_results_file = f"{output_dir}/retrieved_results_hard_neg.json"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(retrieved_results_file):
    # embedder = JinaClipV2Embedder()
    embedder = BGETextEmbedder()
    
    bs = 64  # Batch size for text processing
    id_to_retrieved = {}
    valid_dataset_list = list(valid_dataset.items())
    
    for i in tqdm(range(0, len(valid_dataset_list), bs), desc="Processing batches"):
        batch = valid_dataset_list[i:i+bs]
        data_ids = [item[0] for item in batch]
        questions = [item[1]["question"] for item in batch]
        image_doc_ids = [item[1]['metadata']['image_doc_ids'] for item in batch]
        
        # Get embeddings for questions
        query_vectors = embedder.embed_text(questions)
        
        for j, (data_id, query_vec, id_list) in tqdm(enumerate(zip(data_ids, query_vectors, image_doc_ids)), 
                                                 desc="Processing questions", leave=False):
            if len(id_list) == 0:
                id_to_retrieved[data_id] = {
                    "data_id": data_id,
                    "retrieved": []
                }
                continue
            
            # Create caption texts for image docs
            captions = []
            for img_id in id_list:
                # Use title as caption since MMQA doesn't have separate caption field
                caption = image_docs[img_id]["title"]
                captions.append(caption)
            
            # Get caption embeddings
            caption_vectors = embedder.embed_text(captions)
            
            # Calculate scores
            scores = (query_vec @ caption_vectors.T).tolist()
            
            # Get all indices sorted by score
            sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            
            # Create retrieved results
            id_to_retrieved[data_id] = {
                "data_id": data_id,
                "retrieved": [{
                    "title": image_docs[id_list[idx]]["title"],
                    "image_path": image_docs[id_list[idx]]["path"],
                    "url": image_docs[id_list[idx]]["url"],
                    "score": scores[idx],
                    "doc_id": id_list[idx]
                } for idx in sorted_indices]
            }
    
    with open(retrieved_results_file, "w") as f:
        json.dump(id_to_retrieved, f, indent=2)
else:
    id_to_retrieved = json.load(open(retrieved_results_file, "r"))

# Calculate recall@k
def calculate_recall_at_k(id_to_retrieved, valid_dataset, k_values=[1, 3, 5, 10, 20]):
    recalls = {k: [] for k in k_values}
    max_k = max(k_values)
    
    for data_id, retrieved_data in tqdm(id_to_retrieved.items(), desc="Calculating recall"):
        if data_id not in valid_dataset:
            continue
        
        # Get ground truth image document IDs
        gt_doc_ids = [item["doc_id"] for item in valid_dataset[data_id]["supporting_context"] if item["doc_part"] == "image"]
        gt_doc_ids = set(gt_doc_ids)
        
        if not gt_doc_ids:
            continue
        
        # Get retrieved IDs up to max_k
        retrieved = retrieved_data["retrieved"][:max_k]
        retrieved_doc_ids = [item["doc_id"] for item in retrieved]
        
        # Calculate recall for each k
        for k in k_values:
            ids_at_k = set(retrieved_doc_ids[:min(k, len(retrieved_doc_ids))])
            correct_retrievals = len(gt_doc_ids.intersection(ids_at_k))
            recall = correct_retrievals / len(gt_doc_ids) if gt_doc_ids else 0
            recalls[k].append(recall)
    
    avg_recalls = {f"recall@{k}": np.mean(recalls[k]) for k in k_values}
    return avg_recalls

# Calculate and save results
k_values = [1, 3, 5, 10, 20]
recall_results = calculate_recall_at_k(id_to_retrieved, valid_dataset, k_values)
print("MMQA Image Caption Retrieval Recall@k结果:")
for k, recall in recall_results.items():
    print(f"{k}: {recall:.4f}")

with open(f"{output_dir}/recall_results.json", "w") as f:
    json.dump(recall_results, f, indent=2)