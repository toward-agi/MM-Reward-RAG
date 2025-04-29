import json
train_val_path = "/data/dataset/webqa/WebQA_data_first_release/WebQA_train_val.json"

train_val_file = json.load(open(train_val_path, "r"))
# 这是一个字典，key是id，value是一个字典，里面有image和text的路径；

train_data = []

for key in train_val_file:
    item = train_val_file[key]
    if item["split"] == "train":
        train_data.append(item)
    
import random
samples_1k = random.sample(train_data, 1000)

with open("/data/track/mm-RewardRAG/experiments/1-hard-negatives/results/webqa_train_1k.json", "w") as f:
    json.dump(samples_1k, f, indent=4)
