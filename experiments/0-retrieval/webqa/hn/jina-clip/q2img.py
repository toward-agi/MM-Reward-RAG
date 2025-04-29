import os
import sys
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
sys.path.append("/data/track/mm-RewardRAG/")

from core.embeddings.jina_clip_v2 import JinaClipV2Embedder

valid_dataset = json.load(open("/data/track/mm-RewardRAG/eval_datasets/webqa/WebQA_data_first_release/WebQA_val.json", "r"))
output_dir = "/data/track/mm-RewardRAG/experiments/0-retrieval/results/webqa/hn/q2img"
retrieved_results_file = f"{output_dir}/retrieved_results_hard_neg.json"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(retrieved_results_file):
    embedder = JinaClipV2Embedder()
    
    # 加载index
    # indexer = FaissIndexer()
    # indexer.load("visual_index_gpu/webqa")
    
    bs = 512
    id_to_retrieved = {}
    valid_dataset_list = list(valid_dataset.items())
    image_dir = '/data/dataset/webqa/images'
    
    for i in tqdm(range(0, len(valid_dataset_list), bs)):
        batch = valid_dataset_list[i:i+bs]
        data_ids = [item[0] for item in batch]
        questions = [item[1]["Q"] for item in batch]
        
        # 批量获取问题嵌入向量
        query_vectors = embedder.embed_text(questions)
        
        for j, (data_id, query_vec) in tqdm(enumerate(zip(data_ids, query_vectors))):
            pos_facts = valid_dataset[data_id]["img_posFacts"]
            neg_facts = valid_dataset[data_id]["img_negFacts"]
            facts = pos_facts + neg_facts
            
            if len(facts) == 0:
                id_to_retrieved[data_id] = {
                    "data_id": data_id, 
                    "retrieved": []
                }
                continue
            
            # 批量加载图片
            valid_images = []
            valid_facts = []
            fact_to_index = {}
            
            for idx, fact in enumerate(facts):
                try:
                    image_path = os.path.join(image_dir, f'{fact["image_id"]}.jpg')
                    if os.path.exists(image_path):
                        image = Image.open(image_path).convert('RGB')
                        valid_images.append(image)
                        valid_facts.append(fact)
                        fact_to_index[len(valid_facts)-1] = idx
                except Exception as e:
                    continue
            
            if not valid_images:
                id_to_retrieved[data_id] = {
                    "data_id": data_id,
                    "retrieved": []
                }
                continue
            
            # 批量获取图像嵌入向量
            image_embeddings = embedder.embed_images(valid_images)
            
            # 计算相似度分数
            scores = (query_vec @ image_embeddings.T).tolist()
            
            # 获取分数最高的前5个
            top_k_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:len(facts)]
            
            id_to_retrieved[data_id] = {
                "data_id": data_id,
                "retrieved": [{
                    "image_id": valid_facts[idx]["image_id"], 
                    "title": valid_facts[idx]["title"],
                    "caption": valid_facts[idx]["caption"],
                    "score": scores[idx],
                    "url": valid_facts[idx]["url"],
                } for idx in top_k_indices]
            }
    
    with open(retrieved_results_file, "w") as f:
        json.dump(id_to_retrieved, f, indent=2)
else:
    id_to_retrieved = json.load(open(retrieved_results_file, "r"))

# 计算recall@k
def calculate_recall_at_k(id_to_retrieved, valid_dataset, k_values=[1, 3, 5]):
    recalls = {k: [] for k in k_values}
    max_k = max(k_values)
    
    for data_id, retrieved_data in tqdm(id_to_retrieved.items()):
        if data_id not in valid_dataset:
            continue
            
        gt_pos_facts = valid_dataset[data_id]["img_posFacts"]
        if not gt_pos_facts:
            continue
            
        gt_image_ids = set(fact["image_id"] for fact in gt_pos_facts)
        
        # 一次性获取所有需要的索引
        top_max_k_retrieved = retrieved_data["retrieved"][:max_k]
        retrieved_image_ids = [fact["image_id"] for fact in top_max_k_retrieved]
        
        # 对每个k值计算一次
        for k in k_values:
            ids_at_k = set(retrieved_image_ids[:k])
            correct_retrievals = len(gt_image_ids.intersection(ids_at_k))
            recall = correct_retrievals / len(gt_image_ids) if gt_image_ids else 0
            recalls[k].append(recall)
    
    avg_recalls = {f"recall@{k}": np.mean(recalls[k]) for k in k_values}
    return avg_recalls

# 定义k值
k_values = [1, 3, 5]

# 计算并打印结果
recall_results = calculate_recall_at_k(id_to_retrieved, valid_dataset, k_values)
print("图像检索 Recall@k结果:")
for k, recall in recall_results.items():
    print(f"{k}: {recall:.4f}")

# 保存结果
with open(f"{output_dir}/recall_results.json", "w") as f:
    json.dump(recall_results, f, indent=2)