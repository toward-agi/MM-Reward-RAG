import json
import random
import sys
import os
import pickle
sys.path.append("/data/track/mm-RewardRAG")

"""
{"data_id": "infoseek_train_00000000", "image_id": "oven_01963180", "question": "Which place is this animal endemic to?", "answer": ["People's Republic of China"], "answer_eval": ["cn", "People's Republic of China", "China", "Mainland China", "China PR", "PR China", "CHN", "CN", "PRC", "\ud83c\udde8\ud83c\uddf3"], "data_split": "train"}
{"data_id": "infoseek_train_00000001", "image_id": "oven_03952028", "question": "What is the mohs' hardness of this material?", "answer": ["7.6"], "answer_eval": [{"wikidata": 7.6, "range": [6.84, 8.36]}], "data_split": "train"}
"""

"""
{"data_id": "infoseek_train_00653800", "entity_id": "Q243381", "entity_text": "Macrolepiota procera"}
{"data_id": "infoseek_train_00688905", "entity_id": "Q9053", "entity_text": "Pudding"}
"""

# text_embedder = BGETextEmbedder()
# image_embedder = CLIPEmbedder()

# text_indexer = FaissIndexer(mode="bge")
# text_indexer.load("/data/track/mm-RewardRAG/indexs/bge/infoseek")

# image_indexer = FaissIndexer(mode="clip-large")
# image_indexer.load("/data/track/mm-RewardRAG/indexs/clip-large/infoseek")

infoseek_train_path = "/data/mingyan/infoseek/data/infoseek_train.jsonl"
kb_mapping_path = "/data/mingyan/infoseek/data/infoseek_train_withkb.jsonl"

output_dir = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results"
os.makedirs(output_dir, exist_ok=True)

train_dataset = []
with open(infoseek_train_path, "r") as f:
    for line in f:
        train_dataset.append(json.loads(line))

kb_mapping = {}
with open(kb_mapping_path, "r") as f:
    for line in f:
        item = json.loads(line)
        kb_mapping[item["data_id"]] = item["entity_id"]

# 这里应该过滤一下，就是都是有image和text的；
# 应该不用过滤了再；

filtered_dataset = train_dataset

samples_1k = random.sample(filtered_dataset, 1000)
questions = [item["question"] for item in samples_1k]
image_prefix = "/data/dataset/mm_RewardRAG/InfoSeek/oven/"
import os
i = 0
for item in samples_1k:
    image_path = os.path.join(image_prefix, item["image_id"][5:7], item["image_id"] + ".JPEG")
    if not os.path.exists(image_path):
        image_path = os.path.join(image_prefix, item["image_id"][5:7], item["image_id"] + ".jpg")
        if not os.path.exists(image_path):
            print(f"Image path {image_path} does not exist.")
            i += 1
            continue
    item["image_path"] = image_path
    item["entity_id"] = kb_mapping[item["data_id"]]

print(f"Total {i} images not found.")

# 这里先把gt的image和text拿出来，然后删除掉那俩内存；
image_metadata_path = "/data/track/mm-RewardRAG/indexs/clip-large/infoseek/metadata.pkl"
with open(image_metadata_path, "rb") as f:
    image_metadata = pickle.load(f)

entity2image = {}
for item in image_metadata:
    entity2image[item['metadata']["entity_id"]] = item['image_path']

# 删除image_metadata
del image_metadata

text_metadata_path = "/data/track/mm-RewardRAG/indexs/bge/infoseek/metadata.pkl"
with open(text_metadata_path, "rb") as f:
    text_metadata = pickle.load(f)
entity2text = {}
for item in text_metadata:
    entity2text[item["entity_id"]] = item

del text_metadata

for item in samples_1k:
    entity_id = item["entity_id"]
    if entity_id in entity2image:
        item["image_metadata"] = entity2image[entity_id]
    else:
        item["image_metadata"] = "NOT_FOUND"
        print(f"Entity {entity_id} not found in image metadata.")
    
    if entity_id in entity2text:
        item["text_metadata"] = entity2text[entity_id]
    else:
        print(f"Entity {entity_id} not found in text metadata.")

data_path = os.path.join(output_dir, "infoseek_train_1k.json")
with open(data_path, "w") as f:
    json.dump(samples_1k, f, indent=4)
print(f"Saved {len(samples_1k)} samples to {data_path}")

# question_embeddings = text_embedder.embed(questions)
# image_embeddings = image_embedder.embed(image_paths)


