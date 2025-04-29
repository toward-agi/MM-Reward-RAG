file_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/webqa_train_1k_w_hard_negatives.json"
import json
import os
from PIL import Image
from tqdm import tqdm

select_n = 10
test_reranker = "openai" # option: jina, des, mono, openai
# test_reranker = "jina" # option: jina, des, mono, openai

dataset = json.load(open(file_path, "r"))
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import random
import sys

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

text_corpus_path = "/data/track/mm-RewardRAG/indexs/bge/webqa/metadata.pkl"
import pickle
text_corpus = pickle.load(open(text_corpus_path, "rb"))
"""
>>> data[0]
{'title': 'Xanadu Houses', 'fact': 'Construction of the Xanadu house in Kissimmee, Florida, began with the pouring of a concrete slab base and the erection of a tension ring 40 feet (12 m) in diameter to anchor the domed roof of what would become the "Great Room" of the house.', 'url': 'https://en.wikipedia.org/wiki/Xanadu_Houses', 'snippet_id': 'd5bbc6d80dba11ecb1e81171463288e9_6'}
>>> 
"""

image_corpus_path = "/data/track/mm-RewardRAG/indexs/clip-large/webqa/metadata.pkl"
image_corpus = pickle.load(open(image_corpus_path, "rb"))
"""
>>> data[0]
{'path': '/data/dataset/webqa/images/30264821.jpg', 'image_id': '30264821', 'title': 'Hastings Square, Cambridge, MA - IMG 5572', 'caption': 'Hastings Square, Cambridge, MA - IMG 5572', 'url': 'https://commons.wikimedia.org/wiki/File:Hastings_Square%2C_Cambridge%2C_MA_-_IMG_5572.JPG', 'imgUrl': 'https://upload.wikimedia.org/wikipedia/commons/0/00/Hastings_Square%2C_Cambridge%2C_MA_-_IMG_5572.JPG'}
>>> 
"""


# 这里直接使用之前的
id_to_text = { str(item["snippet_id"]): item for item in text_corpus }
id_to_image = { str(item["image_id"]): item for item in image_corpus }


# gt_entities = []
# q2text_reranked_entites = []
# img2img_reranked_entites = []

supporting_text_ids = []
supporting_image_ids = []
q2text_reranked_entites = []
q2img_reranked_entites = []
q2text_reranked_responses = []
q2img_reranked_responses = []


query2text = []
for item in tqdm(dataset[:10], desc="Reranking"):
    sub_supporting_text_ids = [id["snippet_id"] for id in item["txt_posFacts"]]
    sub_supporting_image_ids = [id["image_id"] for id in item["img_posFacts"]]

    if len(sub_supporting_text_ids) != 0:
        supporting_text_ids.append(sub_supporting_text_ids)
        original_text_hard_negatives = [id["snippet_id"] for id in item["txt_negFacts"]]
        for text_gt in supporting_text_ids:
            if text_gt in original_text_hard_negatives:
                original_text_hard_negatives.remove(text_gt)

        total_text_ids = sub_supporting_text_ids + original_text_hard_negatives
        total_text_corpus = [id_to_text[i]["fact"] for i in total_text_ids]

    if len(sub_supporting_image_ids) != 0:
        supporting_image_ids.append(sub_supporting_image_ids)
        original_image_hard_negatives = [id["image_id"] for id in item["img_negFacts"]]
        for image_gt in supporting_image_ids:
            if image_gt in original_image_hard_negatives:
                original_image_hard_negatives.remove(image_gt)

        total_image_ids = sub_supporting_image_ids + original_image_hard_negatives
    
    # total_image_paths = [id_to_image[i]["image_path"] for i in total_image_ids]
    # if len(total_text_ids) != 0:
    #     total_images = [Image.open(path).convert("RGB").resize((512, 512)) for path in total_image_paths]
    question = item["Q"]

    if test_reranker in ["jina"]:

        if len(sub_supporting_text_ids) != 0:
        
            q2text_reranked_indices = reranker.compute_relevance(
                query=question,
                docs=total_text_corpus,
                query_type="text",
                doc_type="text" 
            )

            q2text_reranked_entites.append([total_text_ids[i] for i in q2text_reranked_indices])

        if len(sub_supporting_image_ids) != 0:
            q2img_reranked_indices = reranker.compute_relevance(
                query=question,
                docs=[id_to_image[str(i)]["path"] for i in total_image_ids],
                query_type="text",
                doc_type="image"
            )
            q2img_reranked_entites.append([total_image_ids[i] for i in q2img_reranked_indices])

    elif test_reranker in ["openai"]:
        query_dict = {
            "type": "t",
            "text": question,
        }

        if len(sub_supporting_text_ids) != 0:
            text_document_dict_list = [
                {
                    "type": "t",
                    "text": context,
                } for context in total_text_corpus
            ]

            ranked_results = reranker.compute_relevance(
                query=query_dict,
                docs=text_document_dict_list,
            )

            q2text_reranked_entites.append([total_text_ids[i] for i in ranked_results["ranked_indices"]])
            q2text_reranked_responses.append(ranked_results["responses"])

        if len(sub_supporting_image_ids) != 0:

            image_document_dict_list = [
                {
                    "type": "ti",
                    "image": Image.open(id_to_image[str(i)]["path"]).convert("RGB").resize((512, 512)),
                    "text": id_to_image[str(i)]["caption"],
                } for i in total_image_ids
            ]

            ranked_results = reranker.compute_relevance(
                query=query_dict,
                docs=image_document_dict_list,
            )

            q2img_reranked_entites.append([total_image_ids[i] for i in ranked_results["ranked_indices"]])
            q2img_reranked_responses.append(ranked_results["responses"])


output_dir = f"/data/track/mm-RewardRAG/experiments/2-rerank/results/webqa/{test_reranker}"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "ids.json"), "w") as f:
    json.dump({
        "supporting_text_ids": supporting_text_ids,
        "supporting_image_ids": supporting_image_ids,
        "q2text_reranked_entites": q2text_reranked_entites,
        "q2img_reranked_entites": q2img_reranked_entites,
        "q2text_reranked_responses": q2text_reranked_responses,
        "q2img_reranked_responses": q2img_reranked_responses,
    }, f, indent=4)
            

q2text_results = evaluate_reranker_dataset(
    all_ground_truth_ids=supporting_text_ids,
    all_reranked_ids=q2text_reranked_entites,
    metrics=["ndcg", "ndcg@3", "ndcg@5", "ndcg@10", 
             "map", "mrr", "p@1", "p@3", "p@5", "p@10"]
)

q2img_results = evaluate_reranker_dataset(
    all_ground_truth_ids=supporting_image_ids,
    all_reranked_ids=q2img_reranked_entites,
    metrics=["ndcg", "ndcg@3", "ndcg@5", "ndcg@10",
            "map", "mrr", "p@1", "p@3", "p@5", "p@10"]
)



with open(os.path.join(output_dir, "q2text_results.json"), "w") as f:
    json.dump(q2text_results, f, indent=4)

with open(os.path.join(output_dir, "q2img_results.json"), "w") as f:
    json.dump(q2img_results, f, indent=4)
