file_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/evqa_train_1k_w_hard_negatives.json"
import json
import os
from PIL import Image
from tqdm import tqdm

select_n = 10
test_reranker = "openai" # option: jina, des, mono, openai

dataset = json.load(open(file_path, "r"))
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import random
import sys

# 这里先拿whole knowledge base来做，然后之后再做个更加细粒度的intra-knowledge的吧；

import sys
sys.path.append("/data/track/mm-RewardRAG")
from core.reranking.jina_reranker_m0 import JinaReranker
from core.reranking.des_reranker import DseReranker
from core.reranking.mono_reranker import MonoQwenVLReranker
from core.reranking.openai_reranker import VLLMReranker
from core.reranking.metrics import evaluate_reranker_dataset

if test_reranker == "jina":
    # support multiple metrics;
    reranker = JinaReranker()
elif test_reranker == "des":
    # query: only support text query
    reranker = DseReranker()
elif test_reranker == "mono":
    reranker = MonoQwenVLReranker()
elif test_reranker == "openai":
    reranker = VLLMReranker()
else:
    raise ValueError("reranker not supported")

text_corpus_path = "/data/track/mm-RewardRAG/indexs/bge/encyclopedia/whole/metadata.pkl"
import pickle
corpus = pickle.load(open(text_corpus_path, "rb"))

id_to_knowledge = { item["entity_id"]: item for item in corpus }

gt_entities = []
q2text_reranked_entites = []
img2img_reranked_entites = []

query2text = []
for item in tqdm(dataset, desc="Reranking"):
    ground_truth_id = item["entity_id"]
    gt_entities.append(ground_truth_id)
    q2text_hard_negative_entity_ids = item["hard_negatives"]["q2text"]
    q_img2img_hard_negative_entity_ids = item["hard_negatives"]["q_img2img"]
    if ground_truth_id in q2text_hard_negative_entity_ids:
        q2text_hard_negative_entity_ids.remove(ground_truth_id)
    if ground_truth_id in q_img2img_hard_negative_entity_ids:
        q_img2img_hard_negative_entity_ids.remove(ground_truth_id)
    # 这里分开来测试吧

    img2img_hard_negative_ids = q_img2img_hard_negative_entity_ids[:select_n]
    # q2text的hard negatives基本没啥意义，

    total_ids = [ground_truth_id] + img2img_hard_negative_ids

    total_contexts = [" ".join(id_to_knowledge[item]["section_texts"]) for item in total_ids]

    image = Image.open(item["image_paths"][0]).convert("RGB").resize((512, 512))
    question = item["question"]

    if test_reranker in ["jina"]:
        
        # 这里resize了一下，避免太大了
        # 这里就是把gt id放在第一个
        img2img_reranked_indices = reranker.compute_relevance(
            query=image,
            docs=total_contexts,
            query_type="image",
            doc_type="text" 
        )

        img2img_reranked_entites.append([total_ids[i] for i in img2img_reranked_indices])

        q2text_reranked_indices = reranker.compute_relevance(
            query=question,
            docs=total_contexts,
            query_type="text",
            doc_type="text"
        )

        # 根据这个indices得到新的entity ids
        q2text_reranked_entites.append([total_ids[i] for i in q2text_reranked_indices])

    elif test_reranker in ["openai"]:
        # 这里的格式应该是每个数据集是一个；
        query_dict = {
            "type": "ti",
            "text": question,
            "image": image,
        }
        document_dict_list = [
            {
                "type": "t",
                "text": context,
            } for context in total_contexts
        ]

        reranked_indices = reranker.compute_relevance(
            query=query_dict,
            docs=document_dict_list,
        )

        query2text.append([total_ids[i] for i in reranked_indices])



# q2text_results = evaluate_reranker_dataset(
#     all_ground_truth_ids=gt_entities,
#     all_reranked_ids=q2text_reranked_entites,
#     metrics=["ndcg", "ndcg@3", "ndcg@5", "ndcg@10", 
#              "map", "mrr", "p@1", "p@3", "p@5", "p@10"]
# )

# img2img_results = evaluate_reranker_dataset(
#     all_ground_truth_ids=gt_entities,
#     all_reranked_ids=img2img_reranked_entites,
#     metrics=["ndcg", "ndcg@3", "ndcg@5", "ndcg@10", 
#              "map", "mrr", "p@1", "p@3", "p@5", "p@10"]
# )

query2text_results = evaluate_reranker_dataset(
    all_ground_truth_ids=gt_entities,
    all_reranked_ids=query2text,
    metrics=["ndcg", "ndcg@3", "ndcg@5", "ndcg@10", 
             "map", "mrr", "p@1", "p@3", "p@5", "p@10"]
)

output_dir = f"/data/track/mm-RewardRAG/experiments/2-rerank/results/evqa/{test_reranker}"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "gt_entities.json"), "w") as f:
    json.dump(gt_entities, f, indent=4)
# with open(os.path.join(output_dir, "q2text_reranked_entities.json"), "w") as f:
#     json.dump(q2text_reranked_entites, f, indent=4)
# with open(os.path.join(output_dir, "img2img_reranked_entities.json"), "w") as f:
#     json.dump(img2img_reranked_entites, f, indent=4)
# with open(os.path.join(output_dir, "q2text_results.json"), "w") as f:
#     json.dump(q2text_results, f, indent=4)
# with open(os.path.join(output_dir, "img2img_results.json"), "w") as f:
#     json.dump(img2img_results, f, indent=4)

with open(os.path.join(output_dir, "query2text_reranked_entities.json"), "w") as f:
    json.dump(query2text, f, indent=4)
with open(os.path.join(output_dir, "query2text_results.json"), "w") as f:
    json.dump(query2text_results, f, indent=4)

