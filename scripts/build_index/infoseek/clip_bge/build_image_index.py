import json
filtered_data_file = "/data/dataset/mm_RewardRAG/InfoSeek/valid_image_kb.json"

import os
import numpy as np

# step 0: load the textual feature

visual_feature_dir = "/data/dataset/mm_RewardRAG/InfoSeek/clip-large/visual_features"

batch_files = sorted([f for f in os.listdir(visual_feature_dir) if f.startswith("batch_")], 
                     key=lambda x: int(x.split("_")[1].split(".")[0]))

# print(batch_files)

all_textual_features = []

from tqdm import tqdm

for batch_file in tqdm(batch_files):
    batch_path = os.path.join(visual_feature_dir, batch_file)
    image_features = np.load(batch_path)
    all_textual_features.append(image_features)

all_textual_features = np.concatenate(all_textual_features, axis=0)

# print(all_image_features.shape)

# step 2: align the textual feature with the text metadata

metadata = json.load(open(filtered_data_file, "r"))
# metadata = ["metadata"] * all_textual_features.shape[0]
# print(len(metadata))
print("Aligning the textual features with the metadata...")

assert len(metadata) == all_textual_features.shape[0]

# step 3: build the faiss index and save it to disk
from core.indexing.faiss_indexer import FaissIndexer
print("Building the index...")
indexer = FaissIndexer(mode="clip-large")
indexer.add_vectors(all_textual_features, metadata)

save_dir = "indexs/clip-large/infoseek"
os.makedirs(save_dir, exist_ok=True)
indexer.save(save_dir)
