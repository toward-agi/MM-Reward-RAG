import json
import sys
from tqdm import tqdm
sys.path.append("/data/track/mm-RewardRAG")
input_file_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/webqa_train_1k.json"
output_file_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/webqa_train_1k_w_hard_negatives.json"
import os

import pickle

input_file = json.load(open(input_file_path, "r"))

from core.embeddings.bge_text_embedding import BGETextEmbedder
from core.indexing.faiss_indexer import FaissIndexer

text_embedder = BGETextEmbedder()
text_indexer = FaissIndexer(mode="bge")
text_indexer.load("/data/track/mm-RewardRAG/indexs/bge/webqa")

caption_indexer = FaissIndexer(mode="bge")
caption_indexer.load("/data/track/mm-RewardRAG/indexs/bge/webqa_caption")

questions = [item["Q"] for item in input_file]
questions_embeddings = text_embedder.embed_text(questions)
questions_to_text_hn = text_indexer.search_batch(questions_embeddings, top_k=100)
questions_to_caption_hn = caption_indexer.search_batch(questions_embeddings, top_k=100)

for item, single_question_hn, single_caption_hn in tqdm(zip(input_file, questions_to_text_hn, questions_to_caption_hn)):
    item["hard_negatives"] = {}
    question_hn_ids = [item["snippet_id"] for item in single_question_hn[2]]
    caption_hn_ids = [item["image_id"] for item in single_caption_hn[2]]
    item["hard_negatives"]["q2text"] = question_hn_ids
    item["hard_negatives"]["q2caption"] = caption_hn_ids
    # if item["img_posFacts"] is not None:
    #     # TODO: Remove the positive facts from the hard negatives
    #     item["hard_negatives"]["q2text"].remove(item["img_posFacts"][0])
    #     item["hard_negatives"]["q2caption"].remove(item["img_posFacts"][0])

with open(output_file_path, "w") as f:
    json.dump(input_file, f, indent=4)
print(f"Saved the file with hard negatives to {output_file_path}")
    