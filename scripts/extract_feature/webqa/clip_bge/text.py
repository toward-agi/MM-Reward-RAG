# from core.embeddings.jina_clip_v2 import JinaClipV2Embedder
from core.embeddings.bge_text_embedding import BGETextEmbedder

import json
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

embedder = BGETextEmbedder()

"""
{
    "title": "Xanadu Houses",
    "fact": "Construction of the Xanadu house in Kissimmee, Florida, began with the pouring of a concrete slab base and the erection of a tension ring 40 feet (12 m) in diameter to anchor the domed roof of what would become the \"Great Room\" of the house.",
    "url": "https://en.wikipedia.org/wiki/Xanadu_Houses",
    "snippet_id": "d5bbc6d80dba11ecb1e81171463288e9_6"
  },
"""

input_data_file =  "/data/dataset/webqa/webqa_train_val_txt_facts.json"
text_feature_dir = "/data/dataset/webqa/bge/textual_feature"

text_data = json.load(open(input_data_file, "r"))

# Embed text
batch_size = 1024

os.makedirs(text_feature_dir, exist_ok=True)

for i in tqdm(range(0, len(text_data), batch_size)):
    batch = text_data[i:i + batch_size]
    texts = [item["fact"] for item in batch]
    image_features = embedder.embed_text(texts)

    batch_file = os.path.join(text_feature_dir, f"batch_{i//batch_size}.npy")
    np.save(batch_file, image_features)
    # the feature has already been normalized

    


