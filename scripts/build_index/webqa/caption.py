from tqdm import tqdm
import numpy as np
import json

# 保存这个新的metadata
# with open("/data/dataset/webqa/webqa_images_meta_with_caption.json", "w") as f:
#     json.dump(new_meta_data, f, indent=4)
# exit()

# 这里还得按照id把图片的caption加上
# 得到metadata之后

new_meta_data = json.load(open("/data/dataset/webqa/webqa_images_meta_with_caption.json", "r"))

textual_feature_dir =  "/data/dataset/webqa/caption_features"

import os

batch_files = sorted([f for f in os.listdir(textual_feature_dir) if f.startswith("batch_")], 
                     key=lambda x: int(x.split("_")[1].split(".")[0]))

import sys
sys.path.append("/data/track/mm-RewardRAG/")

from core.indexing.faiss_indexer import FaissIndexer

indexer = FaissIndexer(mode="jina-clip-v2")

metadata_start = 0
# start adding vectors to the indexer
for batch_file in tqdm(batch_files):
    batch_path = os.path.join(textual_feature_dir, batch_file)
    textual_features = np.load(batch_path)
    indexer.add_vectors(textual_features, new_meta_data[metadata_start:metadata_start+textual_features.shape[0]])
    metadata_start += textual_features.shape[0]

save_dir = "/data/track/mm-RewardRAG/indexs/jina-clip-v2/textual_index_gpu/webqa_caption"
os.makedirs(save_dir, exist_ok=True)
indexer.save(save_dir)


