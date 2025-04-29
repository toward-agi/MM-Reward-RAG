import json

input_file_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/mmqa_train_1k.json"
output_file_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/mmqa_train_1k_hard_negatives.json"

sampled_data = json.load(open(input_file_path, "r"))

import sys
sys.path.append("/data/track/mm-RewardRAG")
from core.embeddings.bge_text_embedding import BGETextEmbedder
from core.indexing.faiss_indexer import FaissIndexer

text_embedder = BGETextEmbedder()


"""
>>> data[0]
{'title': 'Monica Lewinsky', 'url': 'https://en.wikipedia.org/wiki/Monica_Lewinsky', 'id': '9dc9fabf3a930c7b479beb08e83fb1fb', 'text': 'Monica Samille Lewinsky (born July 23, 1973) is an American activist, television personality, fashion designer, and former White House intern. President Bill Clinton admitted to having had what he called an "inappropriate relationship" with Lewinsky while she worked at the White House in 1995–1996. The affair and its repercussions (which included Clinton\'s impeachment) became known later as the Clinton–Lewinsky scandal.'}
"""
text_indexer = FaissIndexer(mode="bge")
text_indexer.load("/data/track/mm-RewardRAG/indexs/bge/mm_qa")

"""
{'metadata': {'title': 'Republican Party (United States)', 'url': 'https://en.wikipedia.org/wiki/Republican_Party_(United_States)', 'id': '35b31d9b4f723f806fd32662ef29edf7', 'path': '35b31d9b4f723f806fd32662ef29edf7.jpg'}, 'image_path': '/data/dataset/multimodal_qa/final_dataset_images/35b31d9b4f723f806fd32662ef29edf7.jpg'}
"""

caption_indexer = FaissIndexer(mode="bge")
caption_indexer.load("/data/track/mm-RewardRAG/indexs/bge/mm_qa_title")

questions = [item["question"] for item in sampled_data]
questions_embeddings = text_embedder.embed_text(questions)

q2text_hn = text_indexer.search_batch(questions_embeddings,top_k=100)
q2caption_hn = caption_indexer.search_batch(questions_embeddings,top_k=100)

for item, single_q2text_hh, single_q2caption_hh in zip(sampled_data, q2text_hn, q2caption_hn):
    item["hard_negatives"] = {}
    # print(single_q2text_hh[2][0].keys())
    text_ids = [item["id"] for item in single_q2text_hh[2]]
    caption_ids = [item["metadata"]["id"] for item in single_q2caption_hh[2]]
    item["hard_negatives"]["text"] = text_ids
    item["hard_negatives"]["caption"] = caption_ids

with open(output_file_path, "w") as f:
    json.dump(sampled_data, f, indent=4)
print("Hard negatives added to the dataset.")





