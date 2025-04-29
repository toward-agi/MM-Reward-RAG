import json
from PIL import Image
import logging
from tqdm import tqdm

num_of_hard_negatives = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

evqa_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/evqa_train_1k_w_hard_negatives.json"
infoseek_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/infoseek_train_1k_w_hard_negatives.json"
mmqa_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/mmqa_train_1k_w_hard_negatives.json"
webqa_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/webqa_train_1k_w_hard_negatives.json"

evqa_kb_path = "/data/track/mm-RewardRAG/indexs/bge/encyclopedia/whole/metadata.pkl"
infoseek_kb_path = "/data/track/mm-RewardRAG/indexs/bge/infoseek/metadata.pkl"
mmqa_text_kb_path = "/data/track/mm-RewardRAG/indexs/bge/mm_qa/metadata.pkl"
mmqa_image_kb_path = "/data/track/mm-RewardRAG/indexs/clip-large/mm_qa/metadata.pkl"
webqa_text_kb_path = "/data/track/mm-RewardRAG/indexs/bge/webqa/metadata.pkl"
webqa_image_kb_path = "/data/track/mm-RewardRAG/indexs/clip-large/webqa/metadata.pkl"

import pickle
evqa_kb = pickle.load(open(evqa_kb_path, "rb"))
logger.info(f"Loaded {len(evqa_kb)} items from EVQA knowledge base.")
infoseek_kb = pickle.load(open(infoseek_kb_path, "rb"))
logger.info(f"Loaded {len(infoseek_kb)} items from InfoSeek knowledge base.")
mmqa_text_kb = pickle.load(open(mmqa_text_kb_path, "rb"))
logger.info(f"Loaded {len(mmqa_text_kb)} items from MMQA text knowledge base.")
mmqa_image_kb = pickle.load(open(mmqa_image_kb_path, "rb"))
logger.info(f"Loaded {len(mmqa_image_kb)} items from MMQA image knowledge base.")
webqa_text_kb = pickle.load(open(webqa_text_kb_path, "rb"))
logger.info(f"Loaded {len(webqa_text_kb)} items from WebQA text knowledge base.")
webqa_image_kb = pickle.load(open(webqa_image_kb_path, "rb"))
logger.info(f"Loaded {len(webqa_image_kb)} items from WebQA image knowledge base.")

output_dir = "/data/track/mm-RewardRAG/experiments/4-sample-rerank/results/3b"
import os
os.makedirs(output_dir, exist_ok=True)

with open(evqa_path, "r") as f:
    evqa_data = json.load(f)
with open(infoseek_path, "r") as f:
    infoseek_data = json.load(f)
with open(mmqa_path, "r") as f:
    mmqa_data = json.load(f)
with open(webqa_path, "r") as f:
    webqa_data = json.load(f)

evqa_queries = [
    {
        "type": "ti",
        "text": item["question"],
        "image": Image.open(item["image_paths"][0]).convert("RGB").resize((512, 512)),
    }
    for item in evqa_data
]

logger.info(f"Loaded {len(evqa_queries)} queries from EVQA dataset.")

infoseek_queries = [
    {
        "type": "ti",
        "text": item["question"],
        "image": Image.open(item["image_path"]).convert("RGB").resize((512, 512)),
    }
    for item in infoseek_data
]

logger.info(f"Loaded {len(infoseek_queries)} queries from InfoSeek dataset.")

mmqa_queries = [
    {
        "type": "t",
        "text": item["question"]
    }
    for item in mmqa_data
]

logger.info(f"Loaded {len(mmqa_queries)} queries from MMQA dataset.")

webqa_queries = [
    {
        "type": "t",
        "text": item["Q"]
    }
    for item in webqa_data
]

logger.info(f"Loaded {len(webqa_queries)} queries from WebQA dataset.")

import sys

sys.path.append("/data/track/mm-RewardRAG")
from core.reranking.sample_reranker import VLLMSampleReranker

ranker = VLLMSampleReranker(generator_name="vllm_sample_reranker_3b")

evqa_id_to_knowledge = {item["entity_id"]: item for item in evqa_kb}

evqa_output = []
for item in tqdm(evqa_data):
    query = {
        "type": "ti",
        "text": item["question"],
        "image": Image.open(item["image_paths"][0]).convert("RGB").resize((512, 512)),
    }
    
    ground_truth_id = item["entity_id"]
    q_img2img_hard_negative_entity_ids = item["hard_negatives"]["q_img2img"]
    if ground_truth_id in q_img2img_hard_negative_entity_ids:
        q_img2img_hard_negative_entity_ids.remove(ground_truth_id)

    img2img_hard_negative_ids = q_img2img_hard_negative_entity_ids[:num_of_hard_negatives]
    total_ids = [ground_truth_id] + img2img_hard_negative_ids
    total_contexts = [" ".join(evqa_id_to_knowledge[item]["section_texts"]) for item in total_ids]

    documents = [
        {
            "type": "t",
            "text": context,
        }
        for context in total_contexts
    ]

    sampled_rankings = ranker.compute_relevance(
        query=query,
        docs=documents,
    )

    evqa_output.append(sampled_rankings)

logger.info(f"Sampled {len(evqa_output)} rankings from EVQA dataset.")
with open(f"{output_dir}/evqa_sampled_rankings.json", "w") as f:
    json.dump(evqa_output, f, indent=4)

exit()

infoseek_id_to_knowledge = {item["entity_id"]: item for item in infoseek_kb}
infoseek_output = []

for item in tqdm(infoseek_data):
    query = {
        "type": "ti",
        "text": item["question"],
        "image": Image.open(item["image_path"]).convert("RGB").resize((512, 512)),
    }
    
    ground_truth_id = item["entity_id"]
    q_img2img_hard_negative_entity_ids = item["hard_negatives"]["q_image"]
    if ground_truth_id in q_img2img_hard_negative_entity_ids:
        q_img2img_hard_negative_entity_ids.remove(ground_truth_id)

    img2img_hard_negative_ids = q_img2img_hard_negative_entity_ids[:num_of_hard_negatives]
    total_ids = [ground_truth_id] + img2img_hard_negative_ids
    total_contexts = [infoseek_id_to_knowledge[item]['wikipedia_content'] for item in total_ids]

    documents = [
        {
            "type": "t",
            "text": context,
        }
        for context in total_contexts
    ]

    sampled_rankings = ranker.compute_relevance(
        query=query,
        docs=documents,
    )

    infoseek_output.append(sampled_rankings)

logger.info(f"Sampled {len(infoseek_output)} rankings from InfoSeek dataset.")
with open(f"{output_dir}/infoseek_sampled_rankings.json", "w") as f:
    json.dump(infoseek_output, f, indent=4)

mmqa_text_id_to_knowledge = {item["id"]: item for item in mmqa_text_kb}
mmqa_image_id_to_knowledge = {item["metadata"]["id"]: item for item in mmqa_image_kb}

mmqa_text_output = []
mmqa_image_output = []

for item in tqdm(mmqa_data):
    query = {
        "type": "t",
        "text": item["question"],
    }

    text_gt_ids = [i["doc_id"] for i in item["supporting_context"] if i["doc_part"] == "text"]
    image_gt_ids = [i["doc_id"] for i in item["supporting_context"] if i["doc_part"] == "image"]

    hard_text_ids = item["metadata"]["text_doc_ids"]
    for id in text_gt_ids:
        if id in hard_text_ids:
            hard_text_ids.remove(id)

    hard_image_ids = item["metadata"]["image_doc_ids"]
    for id in image_gt_ids:
        if id in hard_image_ids:
            hard_image_ids.remove(id)

    total_text_ids = text_gt_ids + hard_text_ids
    total_image_ids = image_gt_ids + hard_image_ids

    text_corpus = [mmqa_text_id_to_knowledge[i]["text"] for i in total_text_ids]
    text_docs = [
        {
            "type": "t",
            "text": context,
        }
        for context in text_corpus
    ]
    text_rankings = ranker.compute_relevance(
        query=query,
        docs=text_docs,
    )
    mmqa_text_output.append(text_rankings)

    # image_document_dict_list = [
    #             {
    #                 "type": "ti",
    #                 "image": Image.open(id_to_image[i]["image_path"]).convert("RGB").resize((512, 512)),
    #                 "text": id_to_image[i]["metadata"]["title"],
    #             } for i in total_image_ids
    #         ]
    image_docs = [
        {
            "type": "ti",
            "image": Image.open(mmqa_image_id_to_knowledge[i]["image_path"]).convert("RGB").resize((512, 512)),
            "text": mmqa_image_id_to_knowledge[i]["metadata"]["title"],
        }
        for i in total_image_ids
    ]

    image_rankings = ranker.compute_relevance(
        query=query,
        docs=image_docs,
    )

    mmqa_image_output.append(image_rankings)

logger.info(f"Sampled {len(mmqa_text_output)} rankings from MMQA dataset.")
with open(f"{output_dir}/mmqa_text_sampled_rankings.json", "w") as f:
    json.dump(mmqa_text_output, f, indent=4)
logger.info(f"Sampled {len(mmqa_image_output)} rankings from MMQA dataset.")
with open(f"{output_dir}/mmqa_image_sampled_rankings.json", "w") as f:
    json.dump(mmqa_image_output, f, indent=4)

webqa_text_output = []
webqa_image_output = []

webqa_text_id_to_knowledge = {item["snippet_id"]: item for item in webqa_text_kb}
webqa_image_id_to_knowledge = {item["image_id"]: item for item in webqa_image_kb}

for item in tqdm(webqa_data):
    query = {
        "type": "t",
        "text": item["Q"],
    }

    sub_supporting_text_ids = [id["snippet_id"] for id in item["txt_posFacts"]]
    sub_supporting_image_ids = [id["image_id"] for id in item["img_posFacts"]]

    original_text_hard_negatives = [id["snippet_id"] for id in item["txt_negFacts"]]
    for text_gt in sub_supporting_text_ids:
        if text_gt in original_text_hard_negatives:
            original_text_hard_negatives.remove(text_gt)

    original_image_hard_negatives = [id["image_id"] for id in item["img_negFacts"]]
    for image_gt in sub_supporting_image_ids:
        if image_gt in original_image_hard_negatives:
            original_image_hard_negatives.remove(image_gt)

    total_text_ids = sub_supporting_text_ids + original_text_hard_negatives
    total_image_ids = sub_supporting_image_ids + original_image_hard_negatives

    total_text_corpus = [webqa_text_id_to_knowledge[i]["fact"] for i in total_text_ids]
    text_docs = [
        {
            "type": "t",
            "text": context,
        }
        for context in total_text_corpus
    ]

    text_rankings = ranker.compute_relevance(
        query=query,
        docs=text_docs,
    )
    webqa_text_output.append(text_rankings)

    image_rankings = ranker.compute_relevance(
        query=query,
        docs=[
            {
                "type": "ti",
                "image": Image.open(webqa_image_id_to_knowledge[str(i)]["path"]).convert("RGB").resize((512, 512)),
                "text": webqa_image_id_to_knowledge[str(i)]["caption"],
            }
            for i in total_image_ids
        ],
    )

    webqa_image_output.append(image_rankings)


logger.info(f"Sampled {len(webqa_text_output)} rankings from WebQA dataset.")
with open(f"{output_dir}/webqa_text_sampled_rankings.json", "w") as f:
    json.dump(webqa_text_output, f, indent=4)
logger.info(f"Sampled {len(webqa_image_output)} rankings from WebQA dataset.")
with open(f"{output_dir}/webqa_image_sampled_rankings.json", "w") as f:
    json.dump(webqa_image_output, f, indent=4)

logger.info("All sampled rankings saved successfully.")
