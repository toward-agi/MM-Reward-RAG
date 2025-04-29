import os
import sys
import json
import numpy as np
from tqdm import tqdm
import bm25s
import Stemmer  # You'll need to install PyStemmer
sys.path.append("/data/track/mm-RewardRAG/")

# Load dataset
valid_dataset = json.load(open("/data/track/mm-RewardRAG/eval_datasets/webqa/WebQA_data_first_release/WebQA_val.json", "r"))
output_dir = "/data/track/mm-RewardRAG/experiments/0-retrieval/results/webqa/hn/bm25s/q2img_caption"
retrieved_results_file = f"{output_dir}/retrieved_results_hard_neg.json"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(retrieved_results_file):
    id_to_retrieved = {}
    valid_dataset_list = list(valid_dataset.items())
    
    # Create stemmer for English
    stemmer = Stemmer.Stemmer("english")
    
    for data_id, data in tqdm(valid_dataset_list, desc="Processing questions"):
        question = data["Q"]
        pos_facts = data["img_posFacts"]
        neg_facts = data["img_negFacts"]
        facts = pos_facts + neg_facts
        
        if len(facts) == 0:
            id_to_retrieved[data_id] = {
                "data_id": data_id,
                "retrieved": []
            }
            continue
        
        # Extract caption texts for corpus
        fact_texts = [fact["title"] + " " + fact["caption"] for fact in facts]
        
        # Tokenize the corpus
        corpus_tokens = bm25s.tokenize(fact_texts, stopwords="en", stemmer=stemmer)
        
        # Create the BM25 model and index the corpus
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        
        # Tokenize query
        query_tokens = bm25s.tokenize([question], stemmer=stemmer)
        
        # Get scores - retrieve all facts
        results, scores = retriever.retrieve(query_tokens, k=len(facts))
        
        # Create a list of retrieved facts with scores
        retrieved_facts = []
        for i in range(results.shape[1]):
            doc_idx = int(results[0, i])  # Convert numpy int to Python int
            score = float(scores[0, i])   # Convert numpy float to Python float
            
            retrieved_facts.append({
                "score": score,
                "title": facts[doc_idx]["title"],
                "caption": facts[doc_idx]["caption"],
                "url": facts[doc_idx]["url"],
                "imgUrl": facts[doc_idx]["imgUrl"],
                "image_id": facts[doc_idx]["image_id"]
            })
        
        id_to_retrieved[data_id] = {
            "data_id": data_id,
            "retrieved": retrieved_facts
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
            
        gt_pos_facts = valid_dataset[data_id]["img_posFacts"]
        if not gt_pos_facts:
            continue
            
        gt_image_ids = set(fact["image_id"] for fact in gt_pos_facts)
        
        # Get all retrieved ids up to max_k
        retrieved = retrieved_data["retrieved"][:max_k]
        retrieved_image_ids = [fact["image_id"] for fact in retrieved]
        
        # Calculate recall for each k
        for k in k_values:
            ids_at_k = set(retrieved_image_ids[:min(k, len(retrieved_image_ids))])
            correct_retrievals = len(gt_image_ids.intersection(ids_at_k))
            recall = correct_retrievals / len(gt_image_ids) if gt_image_ids else 0
            recalls[k].append(recall)
    
    avg_recalls = {f"recall@{k}": np.mean(recalls[k]) for k in k_values}
    return avg_recalls

# Calculate and save results
k_values = [1, 3, 5, 10, 20]
recall_results = calculate_recall_at_k(id_to_retrieved, valid_dataset, k_values)
print("BM25S Image Caption Recall@k结果:")
for k, recall in recall_results.items():
    print(f"{k}: {recall:.4f}")

with open(f"{output_dir}/recall_results.json", "w") as f:
    json.dump(recall_results, f, indent=2)