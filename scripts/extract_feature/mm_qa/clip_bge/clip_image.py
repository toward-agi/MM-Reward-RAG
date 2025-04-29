from core.embeddings.gme import GmeQwen2VLWrapper
from core.embeddings.clip_embedder import CLIPEmbedder

import json
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

embedder = GmeQwen2VLWrapper()
# embedder = CLIPEmbedder()


input_data_file = "/data/dataset/multimodal_qa/multimodalqa/dataset/MMQA_images.jsonl"
image_dir = "/data/dataset/multimodal_qa/final_dataset_images"
filtered_data_file = "/data/dataset/multimodal_qa/multimodalqa/dataset/MMQA_valid_images.json"

if os.path.exists(filtered_data_file):
    valid_data = json.load(open(filtered_data_file, "r"))
    print(f"Loaded {len(valid_data)} valid images")
else:

    input_data = [json.loads(line) for line in open(input_data_file, "r")]
    valid_data = []
    for item in tqdm(input_data):
        image_path = os.path.join(image_dir, item["path"])
        try:
            Image.open(image_path).convert("RGB")
            valid_data.append({
                "metadata": item,
                "image_path": image_path
            }
            )
        except Exception as e:
            print(f"Error loading {image_path}: {e}")

    print(f"Loaded {len(valid_data)} valid images")

    # save to disk
    with open("/data/dataset/multimodal_qa/multimodalqa/dataset/MMQA_valid_images.json", "w") as f:
        json.dump(valid_data, f)

# Embed images
batch_size = 8

visual_feature_dir = "/data/dataset/multimodal_qa/multimodalqa/gme-new/visual_features"
os.makedirs(visual_feature_dir, exist_ok=True)
for i in tqdm(range(0, len(valid_data), batch_size)):
    batch = valid_data[i:i + batch_size]
    images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
    image_features = embedder.embed_images(images)
    # print(image_features.shape)

    batch_file = os.path.join(visual_feature_dir, f"batch_{i//batch_size}.npy")
    np.save(batch_file, image_features)
    # the feature has already been normalized
