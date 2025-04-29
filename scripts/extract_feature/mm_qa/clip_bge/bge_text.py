# from core.embeddings.jina_clip_v2 import JinaClipV2Embedder
# from core.embeddings.clip_embedder import CLIPEmbedder
from core.embeddings.bge_text_embedding import BGETextEmbedder

import json
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

embedder = BGETextEmbedder()
"""
{"title": "Monica Lewinsky", "url": "https://en.wikipedia.org/wiki/Monica_Lewinsky", "id": "9dc9fabf3a930c7b479beb08e83fb1fb", "text": "Monica Samille Lewinsky (born July 23, 1973) is an American activist, television personality, fashion designer, and former White House intern. President Bill Clinton admitted to having had what he called an \"inappropriate relationship\" with Lewinsky while she worked at the White House in 1995–1996. The affair and its repercussions (which included Clinton's impeachment) became known later as the Clinton–Lewinsky scandal."}
"""

input_data_file = "/data/dataset/multimodal_qa/multimodalqa/dataset/MMQA_texts.jsonl"
text_feature_dir = "/data/dataset/multimodal_qa/multimodalqa/bge/textual_features"

text_data = [json.loads(line) for line in open(input_data_file, "r")]

# Embed text
batch_size = 1024

os.makedirs(text_feature_dir, exist_ok=True)

for i in tqdm(range(0, len(text_data), batch_size)):
    batch = text_data[i:i + batch_size]
    texts = [item["text"] for item in batch]
    image_features = embedder.embed_text(texts)

    batch_file = os.path.join(text_feature_dir, f"batch_{i//batch_size}.npy")
    np.save(batch_file, image_features)
    # the feature has already been normalized

    


