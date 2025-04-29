import os
import sys
import json
from tqdm import tqdm
sys.path.append("/data/track/mm-RewardRAG/")


valid_dataset = json.load(open("/data/track/mm-RewardRAG/eval_datasets/webqa/WebQA_data_first_release/WebQA_val.json", "r"))
output_dir = "/data/track/mm-RewardRAG/experiments/0-retrieval/results/webqa/hn/bge/q2text-fact-only"
retrieved_results_file = f"{output_dir}/retrieved_results_hard_neg.json"
os.makedirs(output_dir, exist_ok=True)
if not os.path.exists(retrieved_results_file):
    # from core.embeddings.jina_clip_v2 import JinaClipV2Embedder
    from core.embeddings.bge_text_embedding import BGETextEmbedder
    # embedder = JinaClipV2Embedder() #将question或者image转换为向量
    embedder = BGETextEmbedder()
    bs = 512
    id_to_retrieved = {}
    valid_dataset_list = list(valid_dataset.items())
    for i in tqdm(range(0, len(valid_dataset_list), bs)):
        batch = valid_dataset_list[i:i+bs]
        data_ids = [item[0] for item in batch]
        questions = [item[1]["Q"] for item in batch]
        pos_text_facts = [item[1]["txt_posFacts"] for item in batch]
        neg_text_facts = [item[1]["txt_negFacts"] for item in batch]

        # Get embeddings for questions
        query_vectors = embedder.embed_text(questions)
        
        # Process each question separately
        for j, (data_id, query_vec, pos_facts, neg_facts) in tqdm(enumerate(zip(data_ids, query_vectors, pos_text_facts, neg_text_facts))):
            # Get embeddings for this question's facts
            facts = pos_facts + neg_facts
            if len(facts) == 0:
                id_to_retrieved[data_id] = {
                    "data_id": data_id,
                    "retrieved": []
                }
                continue
            # fact_vectors = embedder.embed_text([fact["title"] + " " + fact["fact"] for fact in facts])
            fact_vectors = embedder.embed_text([fact["fact"] for fact in facts])
            # Calculate scores between query and facts
            scores = (query_vec @ fact_vectors.T).tolist()
            
            # Get top 5 fact indices
            
            top_k_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:len(facts)]
            # 这里就是希望把全部的hard-neg都得到，然后再去慢慢算结果去看；
            
            id_to_retrieved[data_id] = {
                "data_id": data_id,
                "retrieved": [{
                    "fact": facts[idx]["fact"], 
                    "score": scores[idx],
                    "title": facts[idx]["title"],
                    "url": facts[idx]["url"],
                    "snippet_id": facts[idx]["snippet_id"]
                } for idx in top_k_indices]
            }
    with open(retrieved_results_file, "w") as f:
        json.dump(id_to_retrieved, f, indent=2)
else:
    id_to_retrieved = json.load(open(retrieved_results_file, "r"))

# 继续之前的代码，计算recall@k
import numpy as np

def calculate_recall_at_k(id_to_retrieved, valid_dataset, k_values=[1, 3, 5]):
    recalls = {k: [] for k in k_values}
    
    for data_id, retrieved_data in id_to_retrieved.items():
        if data_id not in valid_dataset:
            continue
            
        gt_pos_facts = valid_dataset[data_id]["txt_posFacts"]
        if not gt_pos_facts:
            continue
            
        gt_snippet_ids = set(fact["snippet_id"] for fact in gt_pos_facts)
        
        for k in k_values:
            top_k_retrieved = retrieved_data["retrieved"][:k]
            retrieved_snippet_ids = set(fact["snippet_id"] for fact in top_k_retrieved)
            
            correct_retrievals = len(gt_snippet_ids.intersection(retrieved_snippet_ids))
            recall = correct_retrievals / len(gt_snippet_ids) if gt_snippet_ids else 0
            recalls[k].append(recall)
    
    avg_recalls = {f"recall@{k}": np.mean(recalls[k]) for k in k_values}
    return avg_recalls

# 计算并保存结果
k_values = [1, 3, 5, 10, 20]
recall_results = calculate_recall_at_k(id_to_retrieved, valid_dataset, k_values)
print("Recall@k结果:")
for k, recall in recall_results.items():
    print(f"{k}: {recall:.4f}")

with open(f"{output_dir}/recall_results.json", "w") as f:
    json.dump(recall_results, f, indent=2)
