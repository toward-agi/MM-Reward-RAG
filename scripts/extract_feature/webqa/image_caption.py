import sys
sys.path.append("/data/track/mm-RewardRAG/")
from core.embeddings.jina_clip_v2 import JinaClipV2Embedder
# from core.embeddings.clip_embedder import CLIPEmbedder

import json
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

clip_embedder = JinaClipV2Embedder()

image_dir = "/data/dataset/webqa/images"
meta_data_file = "/data/dataset/webqa/webqa_images_meta_with_caption.json"

if os.path.exists(meta_data_file):
    meta_data = json.load(open(meta_data_file, "r"))
    print(f"Loaded {len(meta_data)} valid images")
else:

    images_path = os.listdir(image_dir)
    images_path = [os.path.join(image_dir, image_path) for image_path in images_path if image_path.endswith(".jpg")]
    meta_data = [{"path": image_path, "image_id": image_path.split("/")[-1].split(".")[0]} for image_path in images_path]

    print(f"Loaded {len(meta_data)} valid images")

    # save to disk
    with open(meta_data_file, "w") as f:
        json.dump(meta_data, f)

# Embed images
batch_size = 1024

visual_feature_dir = "/data/dataset/webqa/caption_features"
os.makedirs(visual_feature_dir, exist_ok=True)
for i in tqdm(range(0, len(meta_data), batch_size)):
    batch = meta_data[i:i + batch_size]
    text = [item["title"] + item["caption"] for item in batch]
    image_features = clip_embedder.embed_text(text)

    batch_file = os.path.join(visual_feature_dir, f"batch_{i//batch_size}.npy")
    np.save(batch_file, image_features)
    # the feature has already been normalized
