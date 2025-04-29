from core.embeddings.jina_clip_v2 import JinaClipV2Embedder
from core.indexing.faiss_indexer import FaissIndexer

import json
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

jina_embedder = JinaClipV2Embedder()
indexer = FaissIndexer(mode="jina-clip-v2")

input_data_file = "/data/dataset/mm_RewardRAG/InfoSeek/image_kb.json"
image_dir = "/data/dataset/mm_RewardRAG/InfoSeek"
filtered_data_file = "/data/dataset/mm_RewardRAG/InfoSeek/valid_image_kb.json"

if os.path.exists(filtered_data_file):
    valid_data = json.load(open(filtered_data_file, "r"))
    print(f"Loaded {len(valid_data)} valid images")
else:

    input_data = json.load(open(input_data_file, "r"))
    valid_data = []
    for item in tqdm(input_data):
        image_path = f"{image_dir}{item['path']}"
        try:
            Image.open(image_path)
            valid_data.append({
                "metadata": item,
                "image_path": image_path
            }
            )
        except Exception as e:
            print(f"Error loading {image_path}: {e}")

    print(f"Loaded {len(valid_data)} valid images")

    # save to disk
    with open("/data/dataset/mm_RewardRAG/InfoSeek/valid_image_kb.json", "w") as f:
        json.dump(valid_data, f)

# Embed images
batch_size = 1024

visual_feature_dir = "/data/dataset/mm_RewardRAG/InfoSeek/visual_features"
os.makedirs(visual_feature_dir, exist_ok=True)
for i in tqdm(range(0, len(valid_data), batch_size)):
    batch = valid_data[i:i + batch_size]
    images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
    image_features = jina_embedder.embed_images(images)

    batch_file = os.path.join(visual_feature_dir, f"batch_{i//batch_size}.npy")
    np.save(batch_file, image_features)
    # the feature has already been normalized

    


