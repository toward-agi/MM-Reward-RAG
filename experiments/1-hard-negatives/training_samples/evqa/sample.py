input_file_path = "/data/track/mm-RewardRAG/eval_datasets/encyclopedia/train.json"
output_file_path = "/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/evqa_train_1k.json"

import json

data = json.load(open(input_file_path, "r"))

import random
import numpy as np

sampled_data = random.sample(data, 1000)

with open(output_file_path, "w") as f:
    json.dump(sampled_data, f, indent=4)
print(f"Sampled {len(sampled_data)} items from {len(data)} total items.")
