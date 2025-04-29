from tqdm import tqdm
import numpy as np
import json
input_data_file = "/data/dataset/mm_RewardRAG/encyclopedic/image.json"
input_data = json.load(open(input_data_file, "r"))

metadata = []
for img in tqdm(input_data):
    if img["source"] == "atomic":
        metadata.append(img)

textual_feature_dir = "/data/dataset/mm_RewardRAG/kb/encyclopedia/eva-clip/visual_features"

import os

batch_files = sorted([f for f in os.listdir(textual_feature_dir) if f.startswith("batch_")], 
                     key=lambda x: int(x.split("_")[1].split(".")[0]))

import sys
sys.path.append("/data/track/mm-RewardRAG")
from core.indexing.faiss_indexer import FaissIndexer

indexer = FaissIndexer(mode="eva-clip")

metadata_start = 0
# start adding vectors to the indexer
for batch_file in tqdm(batch_files):
    batch_path = os.path.join(textual_feature_dir, batch_file)
    textual_features = np.load(batch_path)
    indexer.add_vectors(textual_features, metadata[metadata_start:metadata_start+textual_features.shape[0]])
    metadata_start += textual_features.shape[0]

save_dir = "indexs/eva-vlip/encyclopedia/atomic"
os.makedirs(save_dir, exist_ok=True)
indexer.save(save_dir)


