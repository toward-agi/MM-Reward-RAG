from core.embeddings.jina_clip_v2 import JinaClipV2Embedder

import json
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

image_dir = "/data/dataset/webqa/images"
# meta_data_file = "/data/dataset/webqa/webqa_images_meta.json"

input_train_data = json.load(open("/data/dataset/webqa/webqa_train_val_img_facts.json", "r"))
input_test_data = json.load(open("/data/dataset/webqa/webqa_test_img_facts.json", "r"))
input_data = input_train_data + input_test_data

i = 0
for item in input_data:
    item["path"] = os.path.join(image_dir, str(item["image_id"]) + ".jpg")
    if not os.path.exists(item["path"]):
        print(f"Image {item['path']} does not exist.")
        i += 1
        continue

print(f"Total missing images: {i} out of {len(input_data)}")
    
exit()

jina_embedder = JinaClipV2Embedder()
# Embed images
batch_size = 1024

meta_data = input_data

visual_feature_dir = "/data/dataset/webqa/visual_features"
os.makedirs(visual_feature_dir, exist_ok=True)
for i in tqdm(range(0, len(meta_data), batch_size)):
    batch = meta_data[i:i + batch_size]
    images = [Image.open(item["path"]).convert("RGB") for item in batch]
    image_features = jina_embedder.embed_images(images)

    batch_file = os.path.join(visual_feature_dir, f"batch_{i//batch_size}.npy")
    np.save(batch_file, image_features)
    # the feature has already been normalized
