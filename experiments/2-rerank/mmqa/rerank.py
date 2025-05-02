file_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/mmqa_train_1k_w_hard_negatives.json"
import json
import os
from PIL import Image
from tqdm import tqdm

select_n = 10
test_reranker = "mono" # option: jina, des, mono, openai
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

text_corpus_path = "/data/track/mm-RewardRAG/indexs/bge/mm_qa/metadata.pkl"
import pickle
text_corpus = pickle.load(open(text_corpus_path, "rb"))
"""
>>> data[0]
{'title': 'Monica Lewinsky', 'url': 'https://en.wikipedia.org/wiki/Monica_Lewinsky', 'id': '9dc9fabf3a930c7b479beb08e83fb1fb', 'text': 'Monica Samille Lewinsky (born July 23, 1973) is an American activist, television personality, fashion designer, and former White House intern. President Bill Clinton admitted to having had what he called an "inappropriate relationship" with Lewinsky while she worked at the White House in 1995–1996. The affair and its repercussions (which included Clinton\'s impeachment) became known later as the Clinton–Lewinsky scandal.'}
>>> 
"""

image_corpus_path = "/data/track/mm-RewardRAG/indexs/clip-large/mm_qa/metadata.pkl"
image_corpus = pickle.load(open(image_corpus_path, "rb"))
"""
>>> data[0]
{'metadata': {'title': 'Republican Party (United States)', 'url': 'https://en.wikipedia.org/wiki/Republican_Party_(United_States)', 'id': '35b31d9b4f723f806fd32662ef29edf7', 'path': '35b31d9b4f723f806fd32662ef29edf7.jpg'}, 'image_path': '/data/dataset/multimodal_qa/final_dataset_images/35b31d9b4f723f806fd32662ef29edf7.jpg'}
>>> 
"""


# 这里直接使用之前的
id_to_text = { str(item["id"]): item for item in text_corpus }
id_to_image = { str(item["metadata"]["id"]): item for item in image_corpus }


# gt_entities = []
# q2text_reranked_entites = []
# img2img_reranked_entites = []

supporting_text_ids = []
supporting_image_ids = []
q2text_reranked_entites = []
q2img_reranked_entites = []
q2text_responses = []
q2img_responses = []


query2text = []
for item in tqdm(dataset, desc="Reranking"):
    sub_supporting_text_ids = []
    sub_supporting_image_ids = []
    for context in item["supporting_context"]:
        if context["doc_part"] == "text":
            sub_supporting_text_ids.append(context["doc_id"])
        elif context["doc_part"] == "image":
            sub_supporting_image_ids.append(context["doc_id"])



    if len(sub_supporting_text_ids)!= 0:
        supporting_text_ids.append(sub_supporting_text_ids)

        original_text_hard_negatives = item["metadata"]["text_doc_ids"]

        for text_gt in sub_supporting_text_ids:
            if text_gt in original_text_hard_negatives:
                original_text_hard_negatives.remove(text_gt)

        total_text_ids = sub_supporting_text_ids + original_text_hard_negatives
        total_text_corpus = [id_to_text[i]["text"] for i in total_text_ids]

    if len(sub_supporting_image_ids)!= 0:
        supporting_image_ids.append(sub_supporting_image_ids)
        original_image_hard_negatives = item["metadata"]["image_doc_ids"]

        for image_gt in sub_supporting_image_ids:
            if image_gt in original_image_hard_negatives:
                original_image_hard_negatives.remove(image_gt)

        total_image_ids = sub_supporting_image_ids + original_image_hard_negatives


    question = item["question"]

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
            try:
                q2img_reranked_indices = reranker.compute_relevance(
                    query=question,
                    docs=[id_to_image[i]["image_path"] for i in total_image_ids],
                    query_type="text",
                    doc_type="image"
                )

                q2img_reranked_entites.append([total_image_ids[i] for i in q2img_reranked_indices])
            except Exception as e:
                q2img_reranked_entites.append(total_image_ids[::-1])
                print(f"Error in reranking image: {e}") # if error, just append the reverse order of image ids
            

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
            q2text_responses.append(ranked_results["responses"])


        if len(sub_supporting_image_ids) != 0:

            image_document_dict_list = [
                {
                    "type": "ti",
                    "image": Image.open(id_to_image[i]["image_path"]).convert("RGB").resize((512, 512)),
                    "text": id_to_image[i]["metadata"]["title"],
                } for i in total_image_ids
            ]

            ranked_results = reranker.compute_relevance(
                query=query_dict,
                docs=image_document_dict_list,
            )

            q2img_reranked_entites.append([total_image_ids[i] for i in ranked_results["ranked_indices"]])
            q2img_responses.append(ranked_results["responses"])

        
    elif test_reranker in ["des"]: # des的话，是支持text -> image和text-> text的，所以这里直接去做就好了；
        if len(sub_supporting_text_ids) != 0:
        
            q2text_reranked_indices = reranker.compute_relevance(
                query=question,
                docs=total_text_corpus,
                doc_type="text" 
            )

            q2text_reranked_entites.append([total_text_ids[i] for i in q2text_reranked_indices])

        if len(sub_supporting_image_ids) != 0:
            q2img_reranked_indices = reranker.compute_relevance(
                query=question,
                docs=[Image.open(id_to_image[i]["image_path"]).convert("RGB").resize((512, 512)) for i in total_image_ids],
                doc_type="image"
            )
            q2img_reranked_entites.append([total_image_ids[i] for i in q2img_reranked_indices])

    elif test_reranker in ["mono"]:
        """
        mono only support text to image, so here we just do text to image reranking;
        """
        if len(sub_supporting_image_ids) != 0:
            q2img_reranked_indices = reranker.compute_relevance(
                query=question,
                docs=[Image.open(id_to_image[i]["image_path"]).convert("RGB").resize((512, 512)) for i in total_image_ids],
                doc_type="image"
            )
            q2img_reranked_entites.append([total_image_ids[i] for i in q2img_reranked_indices])


output_dir = f"/data/track/mm-RewardRAG/experiments/2-rerank/results/mmqa/{test_reranker}"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "ids.json"), "w") as f:
    json.dump({
        "supporting_text_ids": supporting_text_ids,
        "supporting_image_ids": supporting_image_ids,
        "q2text_reranked_entites": q2text_reranked_entites,
        "q2img_reranked_entites": q2img_reranked_entites,
        "q2text_responses": q2text_responses,
        "q2img_responses": q2img_responses,
    }, f, indent=4)
            

# q2text_results = evaluate_reranker_dataset(
#     all_ground_truth_ids=supporting_text_ids,
#     all_reranked_ids=q2text_reranked_entites,
#     metrics=["ndcg", "ndcg@3", "ndcg@5", "ndcg@10", 
#              "map", "mrr", "p@1", "p@3", "p@5", "p@10"]
# )

q2img_results = evaluate_reranker_dataset(
    all_ground_truth_ids=supporting_image_ids,
    all_reranked_ids=q2img_reranked_entites,
    metrics=["ndcg", "ndcg@3", "ndcg@5", "ndcg@10",
            "map", "mrr", "p@1", "p@3", "p@5", "p@10"]
)

# with open(os.path.join(output_dir, "q2text_results.json"), "w") as f:
#     json.dump(q2text_results, f, indent=4)

with open(os.path.join(output_dir, "q2img_results.json"), "w") as f:
    json.dump(q2img_results, f, indent=4)
