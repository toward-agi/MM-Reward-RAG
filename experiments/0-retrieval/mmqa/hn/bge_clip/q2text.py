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

valid_dataset = {}
text_docs = read_jsonl("/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/MMQA_texts.jsonl")
text_docs = {doc["id"]: doc for doc in text_docs}
with open("/data/track/mm-RewardRAG/eval_datasets/multimodal_qa/multimodalqa/dataset/new_MMQA_dev_textQ_ImageQ.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        if "TextQ" in item['metadata']['type']:
            valid_dataset[item['qid']] = item

print(f"valid_dataset size: {len(valid_dataset)}")
# 这里是不是要想办法对valid_dataset过滤一下，只保留有TextQ的question

output_dir = "/data/track/mm-RewardRAG/experiments/0-retrieval/results/mmqa/hn/bge_clip/q2text-fact-only"
retrieved_results_file = f"{output_dir}/retrieved_results_hard_neg.json"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(retrieved_results_file):
    # embedder = JinaClipV2Embedder()
    embedder = BGETextEmbedder()
    
    bs = 512
    id_to_retrieved = {}
    valid_dataset_list = list(valid_dataset.items())
    for i in tqdm(range(0, len(valid_dataset_list), bs)):
        batch = valid_dataset_list[i:i+bs]
        data_ids = [item[0] for item in batch]
        questions = [item[1]["question"] for item in batch]
        text_doc_ids = [item[1]['metadata']['text_doc_ids'] for item in batch]
        
        text_docs_text = []
        text_doc_vectors = []
        
        for id_list in tqdm(text_doc_ids):
            texts = []
            for id in id_list:
                texts.append(text_docs[id]["text"])
            text_docs_text.append(texts)
            text_doc_vectors.append(embedder.embed_text(texts))
            
        # Get embeddings for questions
        query_vectors = embedder.embed_text(questions)
        
        # Process each question separately
        for j, (data_id, query_vec, doc_vectors) in tqdm(enumerate(zip(data_ids, query_vectors, text_doc_vectors))):
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
            top_k_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:len(doc_vectors)]
            
            id_to_retrieved[data_id] = {
                "data_id": data_id,
                "retrieved": [{
                    "text": text_docs_text[j][idx], 
                    "score": scores[idx],
                    "doc_id": text_doc_ids[j][idx]  # Store document ID for later use
                } for idx in top_k_indices]
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
            
        # Get ground truth document IDs
        # gt_doc_ids = set(valid_dataset[data_id]['metadata']['supporting_context']['text'])
        gt_doc_ids = [item["doc_id"] for item in valid_dataset[data_id]["supporting_context"] if item["doc_part"] == "text"]
        gt_doc_ids = set(gt_doc_ids)
        
        if not gt_doc_ids:
            continue
            
        # Get all retrieved ids up to max_k
        retrieved = retrieved_data["retrieved"][:max_k]
        retrieved_doc_ids = [fact["doc_id"] for fact in retrieved]
        
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
print("MMQA Text Retrieval Recall@k结果:")
for k, recall in recall_results.items():
    print(f"{k}: {recall:.4f}")

with open(f"{output_dir}/recall_results.json", "w") as f:
    json.dump(recall_results, f, indent=2)