import os
import sys
import json
import numpy as np
from tqdm import tqdm
import bm25s
import Stemmer  # You'll need to install PyStemmer

sys.path.append("/data/track/mm-RewardRAG/")

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

output_dir = "/data/track/mm-RewardRAG/experiments/0-retrieval/results/mmqa/hn/bm25s/q2img_caption"
retrieved_results_file = f"{output_dir}/retrieved_results_hard_neg.json"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(retrieved_results_file):
    # Create stemmer for English
    stemmer = Stemmer.Stemmer("english")
    
    id_to_retrieved = {}
    valid_dataset_list = list(valid_dataset.items())
    
    for data_id, data in tqdm(valid_dataset_list, desc="Processing questions"):
        question = data["question"]
        image_doc_ids = data['metadata']['image_doc_ids']
        
        if len(image_doc_ids) == 0:
            id_to_retrieved[data_id] = {
                "data_id": data_id,
                "retrieved": []
            }
            continue
        
        # Get all image captions (titles) for this question
        captions = []
        for img_id in image_doc_ids:
            captions.append(image_docs[img_id]["title"])
        
        # Tokenize the corpus
        corpus_tokens = bm25s.tokenize(captions, stopwords="en", stemmer=stemmer)
        
        # Create the BM25 model and index the corpus
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        
        # Tokenize query
        query_tokens = bm25s.tokenize([question], stemmer=stemmer)
        
        # Get scores - retrieve all documents
        results, scores = retriever.retrieve(query_tokens, k=len(captions))
        
        # Create a list of retrieved images with scores
        retrieved_images = []
        for i in range(results.shape[1]):
            doc_idx = int(results[0, i])  # Convert to Python int
            score = float(scores[0, i])   # Convert to Python float
            img_id = image_doc_ids[doc_idx]
            
            retrieved_images.append({
                "title": image_docs[img_id]["title"],
                "image_path": image_docs[img_id]["path"],
                "url": image_docs[img_id]["url"],
                "score": score,
                "doc_id": img_id
            })
        
        id_to_retrieved[data_id] = {
            "data_id": data_id,
            "retrieved": retrieved_images
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
print("MMQA Image Caption Retrieval (BM25S) Recall@k结果:")
for k, recall in recall_results.items():
    print(f"{k}: {recall:.4f}")

with open(f"{output_dir}/recall_results.json", "w") as f:
    json.dump(recall_results, f, indent=2)
