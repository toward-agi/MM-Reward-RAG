import json
from pprint import pprint as print
mmqa_train_path = "/data/dataset/multimodal_qa/multimodalqa/dataset/MMQA_train.jsonl"

mmqa_train_file = []
with open(mmqa_train_path, "r") as f:
    for line in f:
        mmqa_train_file.append(json.loads(line))

filtered_dataset = []
for item in mmqa_train_file:
    if "TableQ" in item["metadata"]["type"]:
        continue
    filtered_dataset.append(item)

import random

samples_1k = random.sample(filtered_dataset, 1000)
with open("/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/mmqa_train_1k.json", "w") as f:
    json.dump(samples_1k, f, indent=4)
