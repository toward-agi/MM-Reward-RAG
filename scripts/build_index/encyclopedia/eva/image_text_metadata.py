import json
input_data_file = "/data/dataset/mm_RewardRAG/encyclopedic/wiki.json"

import os
import numpy as np

# step 0: load the textual feature

textual_feature_dir = "/data/dataset/mm_RewardRAG/kb/encyclopedia/eva-clip/meta_features/"

batch_files = sorted([f for f in os.listdir(textual_feature_dir) if f.startswith("batch_")], 
                     key=lambda x: int(x.split("_")[1].split(".")[0]))

# print(batch_files)
import sys
sys.path.append("/data/track/mm-RewardRAG")
from core.indexing.faiss_indexer import FaissIndexer
print("Building the index...")

metadata = json.load(open(input_data_file, "r"))
indexer = FaissIndexer(mode="eva-clip")

from tqdm import tqdm

metadata_start = 0

for batch_file in tqdm(batch_files):
    batch_path = os.path.join(textual_feature_dir, batch_file)
    textual_features = np.load(batch_path)
    indexer.add_vectors(textual_features, metadata[metadata_start:metadata_start+textual_features.shape[0]])
    metadata_start += textual_features.shape[0]

# all_textual_features = np.concatenate(all_textual_features, axis=0)

# print(all_image_features.shape)

# step 2: align the textual feature with the text metadata


# print(len(metadata))
print("Aligning the textual features with the metadata...")

# assert len(metadata) == all_textual_features.shape[0]

# step 3: build the faiss index and save it to disk


# indexer.add_vectors(all_textual_features, metadata)

save_dir = "indexs/eva-clip/encyclopedia/text_metadata"
os.makedirs(save_dir, exist_ok=True)
indexer.save(save_dir)
