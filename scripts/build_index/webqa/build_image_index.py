from tqdm import tqdm
import numpy as np
import json
meta_data_file = "/data/dataset/webqa/webqa_images_meta.json"
meta_data = json.load(open(meta_data_file, "r"))
input_train_data = json.load(open("/data/dataset/webqa/webqa_train_val_img_facts.json", "r"))
input_test_data = json.load(open("/data/dataset/webqa/webqa_test_img_facts.json", "r"))
input_data = input_train_data + input_test_data
id_to_input_data = {str(item["image_id"]): item for item in input_data}

new_meta_data = []
for meta_item in meta_data:
    id = meta_item["image_id"]
    if str(id) not in id_to_input_data.keys():
        print(f"Image id {id} not found in input data.")
        continue
    caption = id_to_input_data[id]
    meta_item["title"] = caption["title"]
    meta_item["caption"] = caption["caption"]
    meta_item["url"] = caption["url"]
    meta_item["imgUrl"] = caption["imgUrl"]
    new_meta_data.append(meta_item)

# 保存这个新的metadata
# with open("/data/dataset/webqa/webqa_images_meta_with_caption.json", "w") as f:
#     json.dump(new_meta_data, f, indent=4)
# exit()

# 这里还得按照id把图片的caption加上
# 得到metadata之后

textual_feature_dir =  "/data/dataset/webqa/visual_features"

import os

batch_files = sorted([f for f in os.listdir(textual_feature_dir) if f.startswith("batch_")], 
                     key=lambda x: int(x.split("_")[1].split(".")[0]))

from core.indexing.faiss_indexer import FaissIndexer

indexer = FaissIndexer(mode="jina-clip-v2")

metadata_start = 0
# start adding vectors to the indexer
for batch_file in tqdm(batch_files):
    batch_path = os.path.join(textual_feature_dir, batch_file)
    textual_features = np.load(batch_path)
    indexer.add_vectors(textual_features, new_meta_data[metadata_start:metadata_start+textual_features.shape[0]])
    metadata_start += textual_features.shape[0]

save_dir = "/data/track/mm-RewardRAG/indexs/jina-clip-v2/visual_index_gpu/webqa"
os.makedirs(save_dir, exist_ok=True)
indexer.save(save_dir)


