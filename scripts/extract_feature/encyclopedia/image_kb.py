from core.embeddings.jina_clip_v2 import JinaClipV2Embedder
from core.indexing.faiss_indexer import FaissIndexer

import json
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

#TODO: 先跳过这里的comp，把atomic编码完

jina_embedder = JinaClipV2Embedder()
indexer = FaissIndexer(mode="jina-clip-v2")

input_data_file = "/data/dataset/mm_RewardRAG/encyclopedic/image.json"
image_file = "/data/dataset/AToMiC-Images-v0.2"
atomic_images = load_dataset(image_file)['train']

input_data = json.load(open(input_data_file, "r"))
# check image files first

# path: "path": "index_406282"
# for img in tqdm(input_data):
#     if img["source"] == "atomic":
#         index = int(img["path"].split("_")[-1])
#         pil_img = atomic_images[index]["image"]
#     elif img["source"] == "comp":
#         pil_img = Image.open(img["path"])

atomic_images_input = []
for img in tqdm(input_data):
    if img["source"] == "atomic":
        atomic_images_input.append(img)

print("All images are loaded successfully")


# Embed images
batch_size = 1024

visual_feature_dir = "/data/dataset/mm_RewardRAG/kb/encyclopedia/visual_features"
os.makedirs(visual_feature_dir, exist_ok=True)
for i in tqdm(range(0, len(atomic_images_input), batch_size)):
    batch = atomic_images_input[i:i + batch_size]
    images = [atomic_images[int(item["path"].split("_")[-1])]["image"] for item in batch]
    image_features = jina_embedder.embed_images(images)

    batch_file = os.path.join(visual_feature_dir, f"batch_{i//batch_size}.npy")
    np.save(batch_file, image_features)
    # the feature has already been normalized

    


