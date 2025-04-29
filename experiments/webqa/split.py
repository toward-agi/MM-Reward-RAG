import json
from pathlib import Path
#split train and val
train_val = '/data/track/mm-RewardRAG/eval_datasets/webqa/WebQA_data_first_release/WebQA_train_val.json'
train_val_data = json.load(open(train_val, 'r'))
val_data = {}
for id, item in train_val_data.items():
    if item['split'] == 'val':
        val_data[id] = item
json.dump(val_data, open('/data/track/mm-RewardRAG/eval_datasets/webqa/WebQA_data_first_release/WebQA_val.json', 'w'))

